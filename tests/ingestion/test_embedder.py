import unittest
import asyncio
import numpy as np
import os
import httpx  # For httpx.HTTPStatusError and httpx.RequestError
import logging

from unittest.mock import patch, AsyncMock, MagicMock, call

from textura.ingestion.embedder import (
    EmbedderBase,
    LocalEmbedder,
    RemoteEmbedder,
    get_embedder,
    SENTENCE_TRANSFORMER_AVAILABLE,
    RetryTransport, # Import to inspect or assist patching if needed
)

# Suppress most logging output during tests, enable for debugging if necessary
# logging.disable(logging.CRITICAL)


class TestLocalEmbedder(unittest.TestCase):
    def setUp(self):
        # Patch the SentenceTransformer constructor for all tests in this class
        # to avoid actual model loading/download.
        self.sentence_transformer_patcher = patch(
            "textura.ingestion.embedder.SentenceTransformer"
        )
        self.mock_sentence_transformer_class = (
            self.sentence_transformer_patcher.start()
        )
        # Configure the mock instance returned by the constructor
        self.mock_model_instance = MagicMock()
        self.mock_sentence_transformer_class.return_value = self.mock_model_instance

        # Reset SENTENCE_TRANSFORMER_AVAILABLE for each test if needed, or control via patch
        # For these tests, we assume it's available and model loading is mocked.
        # If testing the "unavailable" path, SENTENCE_TRANSFORMER_AVAILABLE would be False.
        self.embedder = LocalEmbedder(model_name="mock_model")
        # Ensure the mock model is assigned if SENTENCE_TRANSFORMER_AVAILABLE is true
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.embedder.model = self.mock_model_instance
        else:
            # If library is "not available", model would be None.
            # We might need a separate test class or setup for that scenario.
            # Forcing it here for this structure.
            self.embedder.model = None


    def tearDown(self):
        self.sentence_transformer_patcher.stop()

    @patch("textura.ingestion.embedder.logging.error")
    def test_generate_embeddings_success_or_library_unavailable(self, mock_logging_error):
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.embedder.model = self.mock_model_instance # Ensure model is set
            expected_embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
            self.mock_model_instance.encode.return_value = expected_embeddings

            chunks = ["text1", "text2"]
            result = asyncio.run(self.embedder.generate_embeddings(chunks))

            self.mock_model_instance.encode.assert_called_once_with(
                chunks, convert_to_numpy=True, show_progress_bar=False
            )
            self.assertEqual(len(result), len(expected_embeddings))
            for res_emb, exp_emb in zip(result, expected_embeddings):
                np.testing.assert_array_equal(res_emb, exp_emb)
        else:
            # Test the scenario where SENTENCE_TRANSFORMER_AVAILABLE is False
            self.embedder.model = None # Ensure model is None as it would be
            result = asyncio.run(self.embedder.generate_embeddings(["text1"]))
            self.assertEqual(result, [])
            mock_logging_error.assert_called_with(
                "SentenceTransformer model is not available in LocalEmbedder. Cannot generate embeddings."
            )

    def test_generate_embeddings_empty_chunks(self):
        result = asyncio.run(self.embedder.generate_embeddings([]))
        self.assertEqual(result, [])
        self.mock_model_instance.encode.assert_not_called()

    @patch("textura.ingestion.embedder.SENTENCE_TRANSFORMER_AVAILABLE", False)
    @patch("textura.ingestion.embedder.logging.error")
    def test_generate_embeddings_library_explicitly_unavailable(self, mock_logging_error, mock_st_available):
        # Re-initialize embedder with SENTENCE_TRANSFORMER_AVAILABLE patched to False
        embedder = LocalEmbedder(model_name="mock_model_unavailable")
        self.assertIsNone(embedder.model)

        result = asyncio.run(embedder.generate_embeddings(["text1"]))
        self.assertEqual(result, [])
        mock_logging_error.assert_called_with(
            "SentenceTransformer model is not available in LocalEmbedder. Cannot generate embeddings."
        )


class TestRemoteEmbedder(unittest.IsolatedAsyncioTestCase):
    def _create_mock_response(
        self, json_data, status_code=200, text_data=""
    ) -> MagicMock:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.json = MagicMock(return_value=json_data)
        mock_response.text = text_data # For error logging in HTTPStatusError

        if status_code >= 300:
            # Create a mock request object to pass to HTTPStatusError
            mock_request = MagicMock(spec=httpx.Request)
            mock_request.method = "POST" # Example method
            mock_request.url = "https://fake.url/api" # Example URL

            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"Mock HTTP error {status_code}",
                request=mock_request,
                response=mock_response, # Pass the mock_response itself
            )
        else:
            mock_response.raise_for_status = MagicMock()
        return mock_response

    @patch("textura.ingestion.embedder.httpx.AsyncClient")
    async def test_generate_embeddings_openai_success(self, MockAsyncClient):
        mock_client_instance = MockAsyncClient.return_value
        embedder = RemoteEmbedder(
            api_key="fake_key", model_name="openai_model", api_provider="openai"
        )

        expected_response_data = {
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        }
        mock_client_instance.post.return_value = self._create_mock_response(
            expected_response_data
        )

        chunks = ["text1", "text2"]
        result = await embedder.generate_embeddings(chunks)

        mock_client_instance.post.assert_called_once()
        args, kwargs = mock_client_instance.post.call_args
        self.assertEqual(args[0], "https://api.openai.com/v1/embeddings")
        self.assertEqual(kwargs["json"]["input"], chunks)
        self.assertEqual(kwargs["json"]["model"], "openai_model")
        self.assertIn("Bearer fake_key", kwargs["headers"]["Authorization"])

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], np.array([0.1, 0.2]))
        np.testing.assert_array_equal(result[1], np.array([0.3, 0.4]))

    @patch("textura.ingestion.embedder.httpx.AsyncClient")
    async def test_generate_embeddings_gemini_success(self, MockAsyncClient):
        mock_client_instance = MockAsyncClient.return_value
        embedder = RemoteEmbedder(
            api_key="fake_key", model_name="gemini_model", api_provider="gemini"
        )

        expected_response_data = {
            "embeddings": [{"values": [0.5, 0.6]}, {"values": [0.7, 0.8]}]
        }
        mock_client_instance.post.return_value = self._create_mock_response(
            expected_response_data
        )

        chunks = ["text1", "text2"]
        result = await embedder.generate_embeddings(chunks)

        mock_client_instance.post.assert_called_once()
        args, kwargs = mock_client_instance.post.call_args
        self.assertEqual(
            args[0],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini_model:batchEmbedContents",
        )
        self.assertEqual(len(kwargs["json"]["requests"]), 2)
        self.assertEqual(kwargs["json"]["requests"][0]["content"]["parts"][0]["text"], "text1")
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "fake_key")

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], np.array([0.5, 0.6]))
        np.testing.assert_array_equal(result[1], np.array([0.7, 0.8]))

    async def test_generate_embeddings_empty_chunks(self):
        embedder = RemoteEmbedder(
            api_key="fake", model_name="fake", api_provider="openai"
        )
        result = await embedder.generate_embeddings([])
        self.assertEqual(result, [])

    @patch("textura.ingestion.embedder.logging.error")
    @patch("textura.ingestion.embedder.httpx.AsyncClient")
    async def test_generate_embeddings_api_error_returns_empty(
        self, MockAsyncClient, mock_logging_error
    ):
        mock_client_instance = MockAsyncClient.return_value
        embedder = RemoteEmbedder(
            api_key="fake", model_name="fake", api_provider="openai", max_retries=0 # No retries for this specific test
        )
        # Configure the client mock itself, not its transport for this direct error test
        mock_client_instance.post.side_effect = httpx.RequestError("API down")

        result = await embedder.generate_embeddings(["text1"])

        self.assertEqual(result, [])
        mock_logging_error.assert_called_once()
        self.assertIn("Request error occurred", mock_logging_error.call_args[0][0])


    @patch("textura.ingestion.embedder.logging.warning")
    @patch("textura.ingestion.embedder.httpx.AsyncHTTPTransport.handle_async_request", new_callable=AsyncMock)
    async def test_retries_on_server_error(self, mock_transport_request, mock_logging_warning):
        embedder = RemoteEmbedder(
            api_key="fake_key", model_name="any_model", api_provider="openai", max_retries=1
        )

        # Simulate one server error, then a success
        mock_transport_request.side_effect = [
            self._create_mock_response({}, 500, text_data="Server Error"), # Retried
            self._create_mock_response({"data": [{"embedding": [0.1, 0.2]}]}, 200) # Success
        ]

        chunks = ["text1"]
        result = await embedder.generate_embeddings(chunks)

        self.assertEqual(mock_transport_request.call_count, 2)
        self.assertTrue(len(result) == 1)
        np.testing.assert_array_equal(result[0], np.array([0.1,0.2]))

        # Check if logging.warning was called for the retry attempt
        # It should be called by RetryTransport
        found_retry_log = False
        for call_arg in mock_logging_warning.call_args_list:
            if "Retrying request" in call_arg[0][0]:
                found_retry_log = True
                break
        self.assertTrue(found_retry_log, "logging.warning with retry message not found")


    @patch("textura.ingestion.embedder.logging.error")
    @patch("textura.ingestion.embedder.logging.warning")
    @patch("textura.ingestion.embedder.httpx.AsyncHTTPTransport.handle_async_request", new_callable=AsyncMock)
    async def test_max_retries_exceeded(self, mock_transport_request, mock_logging_warning, mock_logging_error):
        embedder = RemoteEmbedder(
            api_key="fake_key", model_name="any_model", api_provider="openai", max_retries=1 # 1 initial + 1 retry
        )

        # Simulate persistent server errors
        mock_transport_request.side_effect = [
            self._create_mock_response({}, 500, text_data="Server Error attempt 1"),
            self._create_mock_response({}, 500, text_data="Server Error attempt 2")
        ]

        chunks = ["text1"]
        result = await embedder.generate_embeddings(chunks)

        self.assertEqual(result, [])
        self.assertEqual(mock_transport_request.call_count, 2) # 1 initial + 1 retry

        # Check for retry warning
        found_retry_log = False
        for call_arg in mock_logging_warning.call_args_list:
            if "Retrying request" in call_arg[0][0]: # From RetryTransport
                found_retry_log = True
                break
        self.assertTrue(found_retry_log, "logging.warning with retry message not found")

        # Check for final error log from RemoteEmbedder or RetryTransport
        # RemoteEmbedder logs HTTPStatusError or RequestError. RetryTransport logs if max retries reached.
        # In this case, RetryTransport raises HTTPStatusError after max retries, RemoteEmbedder catches and logs it.
        found_final_error_log = False
        for call_arg in mock_logging_error.call_args_list:
            if "HTTP error occurred" in call_arg[0][0] or "Failed request" in call_arg[0][0]:
                found_final_error_log = True
                break
        self.assertTrue(found_final_error_log, "logging.error with final failure message not found")


class TestGetEmbedder(unittest.TestCase):
    def test_get_local_embedder(self):
        config = {"provider": "local", "model_name": "test_model"}
        with patch("textura.ingestion.embedder.SENTENCE_TRANSFORMER_AVAILABLE", True): # Assume available for this test
             with patch("textura.ingestion.embedder.SentenceTransformer"): # Mock ST
                embedder = get_embedder(config)
        self.assertIsInstance(embedder, LocalEmbedder)
        self.assertEqual(embedder.model_name, "test_model")

    def test_get_openai_embedder(self):
        config = {
            "provider": "openai",
            "api_key": "oa_key",
            "model_name": "oa_model",
        }
        embedder = get_embedder(config)
        self.assertIsInstance(embedder, RemoteEmbedder)
        self.assertEqual(embedder.api_provider, "openai")
        self.assertEqual(embedder.api_key, "oa_key")
        self.assertEqual(embedder.model_name, "oa_model")

    @patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "env_gem_key", "GEMINI_MODEL_NAME": "env_gem_model"},
    )
    def test_get_gemini_embedder_from_env(self):
        config = {"provider": "gemini"}
        embedder = get_embedder(config)
        self.assertIsInstance(embedder, RemoteEmbedder)
        self.assertEqual(embedder.api_key, "env_gem_key")
        self.assertEqual(embedder.model_name, "env_gem_model")
        self.assertEqual(embedder.api_provider, "gemini")

    @patch.dict(os.environ, clear=True) # Ensure env vars are not interfering
    def test_get_embedder_missing_key_raises_error(self):
        config = {"provider": "openai", "model_name": "model"}
        with self.assertRaisesRegex(ValueError, "API key not found for openai"):
            get_embedder(config)

    def test_get_embedder_unknown_provider_raises_error(self):
        config = {"provider": "unknown_provider"}
        with self.assertRaisesRegex(ValueError, "Unknown embedder provider: unknown_provider"):
            get_embedder(config)

    def test_get_remote_embedder_default_model(self):
        # Test OpenAI default model
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake_key"}, clear=True):
            config_openai = {"provider": "openai"}
            embedder_openai = get_embedder(config_openai)
            self.assertEqual(embedder_openai.model_name, "text-embedding-ada-002")

        # Test Gemini default model
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}, clear=True):
            config_gemini = {"provider": "gemini"}
            embedder_gemini = get_embedder(config_gemini)
            self.assertEqual(embedder_gemini.model_name, "text-embedding-004")


if __name__ == "__main__":
    unittest.main()
