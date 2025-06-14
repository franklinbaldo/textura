import httpx
from httpx import AsyncBaseTransport, Request, Response
import asyncio
import os
import abc
import numpy as np
import logging
import time

# Configure basic logging for library-level warnings/errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # Placeholder
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logging.warning(
        "sentence_transformers library not found. LocalEmbedder will not function fully."
    )


class EmbedderBase(abc.ABC):
    """
    Abstract base class for embedders.
    """
    @abc.abstractmethod
    async def generate_embeddings(self, text_chunks: list[str]) -> list[np.ndarray]:
        """
        Generates embeddings for a list of text chunks.

        Args:
            text_chunks: A list of strings, where each string is a text chunk.

        Returns:
            A list of numpy arrays, where each array is the embedding for a chunk.
        """
        pass


class LocalEmbedder(EmbedderBase):
    """
    Generates text embeddings using local SentenceTransformer models.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the LocalEmbedder.

        Args:
            model_name: The name of the SentenceTransformer model to use.
                        Defaults to 'BAAI/bge-small-en-v1.5'.
        """
        self.model_name = model_name
        self.model = None
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logging.error(
                    f"Error initializing SentenceTransformer model '{self.model_name}': {e}"
                )
                logging.error(
                    "Please ensure the model name is correct and you have an internet connection "
                    "if the model needs to be downloaded."
                )
                logging.warning(
                    "LocalEmbedder model could not be loaded despite library being available."
                )
        else:
            logging.warning(
                "LocalEmbedder initialized without a SentenceTransformer model due to missing library."
            )

    async def generate_embeddings(self, text_chunks: list[str]) -> list[np.ndarray]:
        """
        Generates embeddings for a list of text chunks using a local model.

        Args:
            text_chunks: A list of strings, where each string is a text chunk.

        Returns:
            A list of numpy arrays, where each array is the embedding for a chunk.
            Returns an empty list if an error occurs or input is empty.
        """
        if not self.model:
            logging.error(
                "SentenceTransformer model is not available in LocalEmbedder. Cannot generate embeddings."
            )
            return []

        if not text_chunks:
            return []

        try:
            # Run synchronous model.encode in a thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,  # Uses the default ThreadPoolExecutor
                lambda: self.model.encode(
                    text_chunks, convert_to_numpy=True, show_progress_bar=False
                ),
            )
            return [np.array(emb) for emb in embeddings]
        except Exception as e:
            logging.error(f"Error generating embeddings with LocalEmbedder: {e}")
            return []


class RetryTransport(AsyncBaseTransport):
    """
    A custom httpx transport that adds retry logic for specific HTTP status codes
    and request errors.
    """
    def __init__(self, wrapped_transport: AsyncBaseTransport, max_retries: int = 3,
                 backoff_factor: float = 0.5,
                 status_forcelist: tuple = (429, 500, 502, 503, 504),
                 allowed_methods: tuple = ("HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST")):
        self.wrapped_transport = wrapped_transport
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.status_forcelist = status_forcelist
        self.allowed_methods = allowed_methods

    async def handle_async_request(self, request: Request) -> Response:
        retries = 0
        last_exception = None
        while retries <= self.max_retries:
            try:
                if retries > 0:
                    sleep_duration = self.backoff_factor * (2 ** (retries - 1))
                    logging.warning(
                        f"Retrying request {request.method} {request.url} (attempt {retries}/{self.max_retries}) "
                        f"after sleeping for {sleep_duration:.2f}s..."
                    )
                    await asyncio.sleep(sleep_duration)

                response = await self.wrapped_transport.handle_async_request(request)

                if response.status_code in self.status_forcelist and request.method in self.allowed_methods:
                    response.raise_for_status()  # Raise HTTPStatusError for retryable statuses

                return response
            except httpx.RequestError as e: # Includes ConnectTimeout, ReadTimeout, etc.
                last_exception = e
                logging.warning(
                    f"RequestError encountered for {request.method} {request.url}: {e}. Attempt {retries+1}/{self.max_retries+1}."
                )
            except httpx.HTTPStatusError as e: # Specifically for 5xx errors and 429
                last_exception = e
                if e.response.status_code in self.status_forcelist and request.method in self.allowed_methods:
                    logging.warning(
                        f"HTTPStatusError {e.response.status_code} encountered for {request.method} {request.url}. "
                        f"Attempt {retries+1}/{self.max_retries+1}."
                    )
                    if retries == self.max_retries: # If it's the last retry, re-raise
                        logging.error(f"Max retries reached for {request.method} {request.url}. Raising last exception.")
                        raise
                else: # If status is not in forcelist, or method not allowed, re-raise immediately
                    raise
            retries += 1

        # If loop finishes due to retries exhausted, raise the last known exception
        if last_exception:
            logging.error(f"Failed request {request.method} {request.url} after {self.max_retries} retries.")
            raise last_exception

        # Should not be reached if logic is correct, but as a fallback:
        raise RuntimeError("Retry loop exited unexpectedly.")


class RemoteEmbedder(EmbedderBase):
    """
    Generates text embeddings using a remote API (OpenAI or Gemini).
    """

    def __init__(self, api_key: str, model_name: str, api_provider: str,
                 timeout: int = 30, max_retries: int = 3):
        self.api_key = api_key
        self.model_name = model_name
        self.api_provider = api_provider.lower()

        if self.api_provider not in ("openai", "gemini"):
            raise ValueError(f"Unsupported api_provider: {api_provider}. Must be 'openai' or 'gemini'.")

        self.client = httpx.AsyncClient(
            transport=RetryTransport(
                wrapped_transport=httpx.AsyncHTTPTransport(),
                max_retries=max_retries,
                backoff_factor=0.5 # Adjust as needed
            ),
            timeout=timeout,
            headers={"User-Agent": "Textura/0.1.0"}
        )

    async def generate_embeddings(self, text_chunks: list[str]) -> list[np.ndarray]:
        if not text_chunks:
            return []

        headers = {"Content-Type": "application/json"}
        payload = {}
        url = ""

        if self.api_provider == "openai":
            url = "https://api.openai.com/v1/embeddings"
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {"input": text_chunks, "model": self.model_name}
        elif self.api_provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents"
            headers["x-goog-api-key"] = self.api_key
            payload = {
                "requests": [
                    {"model": f"models/{self.model_name}", "content": {"parts": [{"text": chunk}]}}
                    for chunk in text_chunks
                ]
            }

        try:
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Will raise HTTPStatusError for 4xx/5xx responses

            data = response.json()

            if self.api_provider == "openai":
                if not data.get("data") or not isinstance(data["data"], list):
                    logging.error(f"Invalid response structure from OpenAI: 'data' field missing or not a list. Response: {data}")
                    return []
                return [np.array(item['embedding']) for item in data['data']]
            elif self.api_provider == "gemini":
                if not data.get("embeddings") or not isinstance(data["embeddings"], list):
                    logging.error(f"Invalid response structure from Gemini: 'embeddings' field missing or not a list. Response: {data}")
                    return []
                return [np.array(item['values']) for item in data['embeddings']]

        except httpx.HTTPStatusError as e:
            logging.error(
                f"HTTP error occurred while calling {self.api_provider} API: {e.response.status_code} - {e.response.text}"
            )
            return []
        except httpx.RequestError as e: # Covers network errors, timeouts etc.
            logging.error(
                f"Request error occurred while calling {self.api_provider} API: {e}"
            )
            return []
        except Exception as e: # Catch-all for other unexpected errors (e.g., JSON parsing)
            logging.error(
                f"An unexpected error occurred while processing embeddings from {self.api_provider}: {e}"
            )
            return []
        # This final return [] should ideally be unreachable if all error cases lead to a return []
        # but it's here as a fallback.
        # Consider if specific error logging is needed if this point is reached.
        logging.error(f"Reached end of generate_embeddings for {self.api_provider} without returning embeddings. This indicates an issue.")
        return []


def get_embedder(config: dict) -> EmbedderBase:
    """
    Factory function to get an embedder instance based on configuration.

    Args:
        config: A dictionary containing the configuration for the embedder.
                Expected keys:
                - "provider": "local", "openai", or "gemini"
                - "model_name" (optional for local, openai, gemini - uses default or env var)
                - "api_key" (optional for openai, gemini - uses env var)
                - "timeout" (optional for remote, default 30)
                - "max_retries" (optional for remote, default 3)

    Returns:
        An instance of an EmbedderBase subclass.

    Raises:
        ValueError: If the provider is unknown or required configuration is missing.
    """
    provider = config.get("provider")

    if provider == "local":
        return LocalEmbedder(model_name=config.get("model_name", "BAAI/bge-small-en-v1.5"))
    elif provider in ("openai", "gemini"):
        api_key_env_var = f"{provider.upper()}_API_KEY"
        model_name_env_var = f"{provider.upper()}_MODEL_NAME"

        api_key = config.get("api_key") or os.getenv(api_key_env_var)
        # Default model names per provider if not specified in config or env var
        default_model_name = "text-embedding-ada-002" if provider == "openai" else "text-embedding-004" # or another gemini default like 'embedding-001'
        model_name = config.get("model_name") or os.getenv(model_name_env_var) or default_model_name


        if not api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Set it in config or as {api_key_env_var} environment variable."
            )
        if not model_name: # Should be covered by default, but good practice
            raise ValueError(
                f"Model name not found for {provider}. "
                f"Set it in config, as {model_name_env_var} environment variable, or rely on the default."
            )

        return RemoteEmbedder(
            api_key=api_key,
            model_name=model_name,
            api_provider=provider,
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3)
        )
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Embedder module script started.")
    if SENTENCE_TRANSFORMER_AVAILABLE:
        logging.info("sentence_transformers library IS available.")
    else:
        logging.warning(
            "sentence_transformers library IS NOT available. LocalEmbedder functionality will be limited."
        )

    async def main_example():
        # Ensure logging is configured at the start of main_example for script execution
        # If the module is imported, the initial basicConfig at module level will apply.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Running main_example for embedders...")

        sample_texts = [
            "Hello world from Textura embedder!",
            "This is a test sentence for embedding generation.",
            "Async Python is fun."
        ]

        # --- Local Embedder Example ---
        logging.info("\n--- Local Embedder Example (via get_embedder) ---")
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                local_config = {"provider": "local", "model_name": "BAAI/bge-small-en-v1.5"}
                local_embedder = get_embedder(local_config)
                logging.info(f"Using local embedder with model: {local_embedder.model_name}")

                embeddings = await local_embedder.generate_embeddings(sample_texts)
                if embeddings:
                    logging.info(f"Generated {len(embeddings)} local embeddings successfully.")
                    for i, emb in enumerate(embeddings):
                        logging.info(f"Local Embedding {i+1} shape: {emb.shape}, first 5 values: {emb[:5]}")
                else:
                    logging.warning("Local embedding generation failed or returned empty.")
            except Exception as e:
                logging.error(f"Error in Local Embedder example: {e}", exc_info=True)
        else:
            logging.warning("Skipping LocalEmbedder example as sentence_transformers is not available.")

        # --- OpenAI Embedder Example ---
        logging.info("\n--- OpenAI Embedder Example (via get_embedder) ---")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model_name = os.getenv("OPENAI_MODEL_NAME", "text-embedding-ada-002") # Default if not set
        if openai_api_key:
            try:
                openai_config = {
                    "provider": "openai",
                    "api_key": openai_api_key, # Explicitly pass for clarity, though get_embedder can use env
                    "model_name": openai_model_name
                }
                openai_embedder = get_embedder(openai_config)
                logging.info(f"Using OpenAI embedder with model: {openai_embedder.model_name}")

                embeddings = await openai_embedder.generate_embeddings(sample_texts)
                if embeddings:
                    logging.info(f"Generated {len(embeddings)} OpenAI embeddings successfully.")
                    for i, emb in enumerate(embeddings):
                        logging.info(f"OpenAI Embedding {i+1} shape: {emb.shape}, first 5 values: {emb[:5]}")
                else:
                    logging.warning("OpenAI embedding generation failed or returned empty.")
            except Exception as e:
                logging.error(f"Error in OpenAI Embedder example: {e}", exc_info=True)
        else:
            logging.warning("Skipping OpenAI Embedder example as OPENAI_API_KEY is not set.")

        # --- Gemini Embedder Example ---
        logging.info("\n--- Gemini Embedder Example (via get_embedder) ---")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "text-embedding-004") # Default model
        if gemini_api_key:
            try:
                gemini_config = {
                    "provider": "gemini",
                    "api_key": gemini_api_key,
                    "model_name": gemini_model_name
                }
                gemini_embedder = get_embedder(gemini_config)
                logging.info(f"Using Gemini embedder with model: {gemini_embedder.model_name}")

                # Gemini API might have stricter limits on requests per minute for free tiers
                # For the example, let's use fewer texts or expect potential rate limits if key is heavily used.
                # Also, some Gemini models might be specific like "models/embedding-001" vs just "embedding-001"
                # The RemoteEmbedder current constructs it as `models/{self.model_name}` for Gemini payload

                embeddings = await gemini_embedder.generate_embeddings(sample_texts[:2]) # Using fewer texts for Gemini example
                if embeddings:
                    logging.info(f"Generated {len(embeddings)} Gemini embeddings successfully.")
                    for i, emb in enumerate(embeddings):
                        logging.info(f"Gemini Embedding {i+1} shape: {emb.shape}, first 5 values: {emb[:5]}")
                else:
                    logging.warning("Gemini embedding generation failed or returned empty.")
            except Exception as e:
                logging.error(f"Error in Gemini Embedder example: {e}", exc_info=True)
        else:
            logging.warning("Skipping Gemini Embedder example as GEMINI_API_KEY is not set.")

    if os.getenv("TEXTURA_RUN_MAIN_EXAMPLES"): # Optionally control running main
        asyncio.run(main_example())
    else:
        logging.info("TEXTURA_RUN_MAIN_EXAMPLES environment variable not set. Skipping main_example execution.")
