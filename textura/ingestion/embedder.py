# from sentence_transformers import SentenceTransformer # Original import
import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SentenceTransformer = (
        None  # Placeholder class or type for type hinting if needed elsewhere
    )
    SENTENCE_TRANSFORMER_AVAILABLE = False  # Corrected typo
    print(
        "Warning: sentence_transformers library not found. Embedder will not function fully."
    )


class Embedder:
    """
    Generates text embeddings using SentenceTransformer models.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the Embedder.

        Args:
            model_name: The name of the SentenceTransformer model to use.
                        Defaults to 'BAAI/bge-small-en-v1.5'.

        """
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(
                    f"Error initializing SentenceTransformer model '{model_name}': {e}"
                )
                print(
                    "Please ensure the model name is correct and you have an internet connection "
                    "if the model needs to be downloaded."
                )
                self.model = None  # Fallback in case of initialization error
                print(
                    "Warning: Embedder model could not be loaded despite library being available."
                )
        else:
            self.model = None
            print(
                "Warning: Embedder initialized without a SentenceTransformer model due to missing library."
            )

    def generate_embeddings(self, text_chunks: list[str]) -> list[np.ndarray]:
        """
        Generates embeddings for a list of text chunks.

        Args:
            text_chunks: A list of strings, where each string is a text chunk.

        Returns:
            A list of numpy arrays, where each array is the embedding for a chunk.
            Returns an empty list if an error occurs or input is empty.

        """
        if not self.model:
            print(
                "Error: SentenceTransformer model is not available. Cannot generate embeddings."
            )
            return []

        if not text_chunks:
            return []

        try:
            # This part will only be reached if self.model is not None (i.e., SENTENCE_TRANSFORMER_AVAILABLE was True and model loaded)
            embeddings = self.model.encode(
                text_chunks, convert_to_numpy=True, show_progress_bar=False
            )
            return [emb for emb in embeddings]  # Ensure it's a list of arrays
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []


if __name__ == "__main__":
    # Example Usage
    print("Embedder module loaded.")
    if SENTENCE_TRANSFORMER_AVAILABLE:
        print("sentence_transformers library IS available.")
        # This will download the model if you don't have it cached and library is present.
        try:
            embedder = Embedder()
            if embedder.model:
                sample_chunks = [
                    "This is the first document.",
                    "This document is the second document.",
                    "And this is the third one.",
                    "Is this the first document?",
                ]

                print(f"Generating embeddings for {len(sample_chunks)} chunks...")
                embeddings_list = embedder.generate_embeddings(sample_chunks)

                if embeddings_list:
                    print(f"Successfully generated {len(embeddings_list)} embeddings.")
                    for i, emb in enumerate(embeddings_list):
                        print(
                            f"Embedding {i + 1} (shape: {emb.shape}):\n{emb[:5]}...\n---"
                        )
                else:
                    print("Embedding generation returned an empty list or failed.")
            else:
                print(
                    "Embedder initialized, but the model could not be loaded (e.g. model name error, network issue)."
                )

        except Exception as e:
            print(
                f"An error occurred during the Embedder example with library available: {e}"
            )
            print(
                "This might be due to model download issues or other runtime problems."
            )
    else:
        print(
            "sentence_transformers library IS NOT available. Embedder functionality will be limited."
        )
        # Demonstrate that Embedder can be instantiated but won't work
        embedder = Embedder()
        print(f"Embedder initialized: {embedder is not None}")
        print(f"Embedder model is None: {embedder.model is None}")
        embeddings_list = embedder.generate_embeddings(["test"])
        print(
            f"Attempting to generate embeddings without library: {embeddings_list} (should be empty list)"
        )
