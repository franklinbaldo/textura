# Textura LLM Clients

This package contains clients for interacting with various Large Language Models.

## GeminiClient (`gemini_client.py`)

The `GeminiClient` interfaces with Google's Gemini models.

### Configuration

The client requires the following environment variables to be set for live API calls:

- **`GOOGLE_API_KEY`**: Your Google API key for accessing the Gemini API. This is essential for authentication.
- **`TEXTURA_LLM_MODEL`**: (Optional) Specifies which Gemini model to use (e.g., `gemini-1.5-pro-latest`, `gemini-pro`). If not set, the client defaults to `gemini-1.5-pro-latest`.

You can set these variables in your shell environment before running Textura commands that invoke the LLM, for example:

```bash
export GOOGLE_API_KEY="your_api_key_here"
export TEXTURA_LLM_MODEL="gemini-1.5-pro-latest"
textura extract --workspace /path/to/your/workspace
```

Alternatively, Google's client libraries might implicitly pick up credentials from other standard Google Cloud environment variables (e.g., `GOOGLE_APPLICATION_CREDENTIALS`) if `GOOGLE_API_KEY` is not directly available to the `google.generativeai` configuration step used in the client.
