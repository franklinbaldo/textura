import unittest
from unittest.mock import patch, MagicMock

# Attempt to import GeminiClient, handle potential ImportError if google.generativeai is not installed
# In a CI environment, google.generativeai might not be installed unless specified.
# For local dev, it should be part of requirements.
try:
    from textura.llm_clients.gemini_client import GeminiClient
    from textura.llm_clients.types import (
        Tool as TexturaTool,
        FunctionDeclaration as TexturaFunctionDeclaration,
        FunctionParameters as TexturaFunctionParameters,
        FunctionParameterProperty as TexturaFunctionParameterProperty,
        LLMResponse,
        FunctionCall
    )
    from google.generativeai.types import Type as GeminiSDKType
    google_generativeai_available = True
except ImportError:
    google_generativeai_available = False
    GeminiClient = None # Placeholder if import fails
    # Define placeholders for types if import failed, to allow class structure to be parsed
    TexturaTool, TexturaFunctionDeclaration, TexturaFunctionParameters, TexturaFunctionParameterProperty, LLMResponse, FunctionCall, GeminiSDKType = (None,)*7


# Conditionally skip tests if google.generativeai is not available and GeminiClient could not be imported
# This is more relevant if these tests were to run in an environment without all dependencies.
# For now, we'll assume it's available for structuring the tests.
# A better way for CI is to ensure the dependency is installed.

@unittest.skipIf(not google_generativeai_available, "google.generativeai library not available, skipping GeminiClient tests")
class TestGeminiClient(unittest.TestCase):

    @patch.dict('os.environ', {}, clear=True) # Start with a clean environment
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_init_with_api_key_and_model_params(self, mock_generative_model, mock_configure):
        """Test initialization with direct API key and model name parameters."""
        api_key = "test_api_key_param"
        model_name = "test_model_param"

        client = GeminiClient(api_key=api_key, model_name=model_name)

        mock_configure.assert_called_once_with(api_key=api_key)
        mock_generative_model.assert_called_once_with(model_name)
        self.assertEqual(client.model_name, model_name)
        self.assertIsNotNone(client.model)

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "test_api_key_env", "TEXTURA_LLM_MODEL": "test_model_env"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_init_with_env_vars(self, mock_generative_model, mock_configure):
        """Test initialization using environment variables."""
        client = GeminiClient()

        mock_configure.assert_called_once_with(api_key="test_api_key_env")
        mock_generative_model.assert_called_once_with("test_model_env")
        self.assertEqual(client.model_name, "test_model_env")

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "test_api_key_env"}) # TEXTURA_LLM_MODEL not set
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_init_with_default_model_name(self, mock_generative_model, mock_configure):
        """Test initialization uses default model name if TEXTURA_LLM_MODEL is not set."""
        client = GeminiClient()

        mock_configure.assert_called_once_with(api_key="test_api_key_env")
        # Default model name is "gemini-1.5-pro-latest" in GeminiClient
        mock_generative_model.assert_called_once_with("gemini-1.5-pro-latest")
        self.assertEqual(client.model_name, "gemini-1.5-pro-latest")

    @patch.dict('os.environ', {}, clear=True) # No API key
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_init_without_api_key_logs_warning(self, mock_generative_model, mock_configure):
        """Test that a warning is (conceptually) logged if no API key is found."""
        # The client currently prints a warning. We can capture stdout or check logs if more formal.
        # For now, just ensure it attempts to configure with None or an empty string if that's the flow.
        # Or, if it raises an error, test for that. Current client prints a warning.
        with patch('builtins.print') as mock_print:
            GeminiClient() # api_key=None by default if not in env
            mock_configure.assert_called_once_with(api_key=None) # Or whatever it defaults to
            # Check if warning print was called. This depends on exact warning message.
            self.assertTrue(any("GOOGLE_API_KEY not found" in call.args[0] for call in mock_print.call_args_list))

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_calls_generate_content(self, mock_generative_model_class, mock_configure):
        """Test that predict method calls model.generate_content."""
        mock_model_instance = MagicMock()
        # Simulate the response structure from Gemini
        mock_gemini_response = MagicMock()
        # mock_gemini_response.text = "Test LLM response" # For simple text response
        # If response can be multi-part:
        mock_part = MagicMock()
        mock_part.text = "Test LLM response"
        mock_gemini_response.parts = [mock_part]
        # mock_gemini_response.text = None # if .text is not present for multi-part and parts is used

        mock_model_instance.generate_content.return_value = mock_gemini_response
        mock_generative_model_class.return_value = mock_model_instance # mock_generative_model_class is the class, .return_value is the instance

        client = GeminiClient(api_key="fake_key")
        prompt = "Hello, world!"
        response = client.predict(prompt)

        mock_model_instance.generate_content.assert_called_once_with(prompt)
        self.assertEqual(response, "Test LLM response")

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_calls_generate_content_with_simple_text_response(self, mock_generative_model_class, mock_configure):
        """Test that predict method calls model.generate_content and handles simple .text response."""
        mock_model_instance = MagicMock()
        mock_gemini_response = MagicMock()
        mock_gemini_response.text = "Simple text response"
        mock_gemini_response.parts = [] # Explicitly empty or None if .text is primary

        mock_model_instance.generate_content.return_value = mock_gemini_response
        mock_generative_model_class.return_value = mock_model_instance

        client = GeminiClient(api_key="fake_key")
        prompt = "Hello again!"
        response = client.predict(prompt)

        mock_model_instance.generate_content.assert_called_once_with(prompt)
        self.assertEqual(response, "Simple text response")


    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_handles_api_error(self, mock_generative_model_class, mock_configure):
        """Test predict method handles exceptions from API call."""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_generative_model_class.return_value = mock_model_instance

        client = GeminiClient(api_key="fake_key")
        response = client.predict("test prompt")

        self.assertTrue("Error during API call: API Error" in response)

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_handles_model_not_initialized(self, mock_gm_class, mock_configure):
        # Simulate model initialization failure
        mock_gm_class.side_effect = Exception("Failed to init model")

        client = GeminiClient(api_key="fake_key")
        # Ensure client.model is None
        self.assertIsNone(client.model)
        response = client.predict("test")
        self.assertEqual(response, "Error: Model not initialized.")

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_with_tools_constructs_gemini_tools_correctly(self, mock_generative_model_class, mock_configure):
        """Test that Textura Tool definitions are correctly converted to Gemini format."""
        mock_model_instance = MagicMock()
        mock_generative_model_class.return_value = mock_model_instance

        # Mock the response from generate_content to be a simple text response
        mock_gemini_response = MagicMock()
        mock_text_part = MagicMock()
        mock_text_part.text = "Text response"
        mock_text_part.function_call = None # Explicitly
        mock_gemini_response.candidates = [MagicMock(content=MagicMock(parts=[mock_text_part]))]
        mock_model_instance.generate_content.return_value = mock_gemini_response

        client = GeminiClient(api_key="fake_key")

        textura_tools = [
            TexturaTool(function_declarations=[
                TexturaFunctionDeclaration(
                    name="get_weather",
                    description="Get current weather in a location",
                    parameters=TexturaFunctionParameters(
                        properties={
                            "location": TexturaFunctionParameterProperty(type="string", description="City and state")
                        },
                        required=["location"]
                    )
                )
            ])
        ]

        client.predict_with_tools("What's the weather in Boston?", tools=textura_tools)

        args, kwargs = mock_model_instance.generate_content.call_args

        # Check that generation_config was passed in kwargs
        self.assertIn('generation_config', kwargs)
        gen_config = kwargs['generation_config']

        # Check tools in gen_config
        self.assertTrue(hasattr(gen_config, 'tools'))
        self.assertEqual(len(gen_config.tools), 1)
        gemini_tool = gen_config.tools[0]
        self.assertEqual(len(gemini_tool.function_declarations), 1)
        gemini_func_decl = gemini_tool.function_declarations[0]

        self.assertEqual(gemini_func_decl.name, "get_weather")
        self.assertEqual(gemini_func_decl.description, "Get current weather in a location")
        self.assertIsNotNone(gemini_func_decl.parameters)
        self.assertEqual(gemini_func_decl.parameters.type, GeminiSDKType.OBJECT)
        self.assertIn("location", gemini_func_decl.parameters.properties)
        self.assertEqual(gemini_func_decl.parameters.properties["location"].type, GeminiSDKType.STRING)
        self.assertEqual(gemini_func_decl.parameters.required, ["location"])

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_with_tools_parses_function_call_response(self, mock_generative_model_class, mock_configure):
        """Test parsing of a function call response from Gemini."""
        mock_model_instance = MagicMock()
        mock_generative_model_class.return_value = mock_model_instance

        # Mock Gemini's FunctionCall and parts structure
        mock_gemini_fc = MagicMock()
        mock_gemini_fc.name = "get_weather"
        # Gemini's FunctionCall.args is often a Struct/dict-like object
        mock_gemini_fc.args = {"location": "Boston"}

        mock_fc_part = MagicMock()
        mock_fc_part.function_call = mock_gemini_fc
        mock_fc_part.text = None # Explicitly

        mock_gemini_response = MagicMock()
        mock_gemini_response.candidates = [MagicMock(content=MagicMock(parts=[mock_fc_part]))]
        mock_gemini_response.text = None # No direct text if function call is made

        mock_model_instance.generate_content.return_value = mock_gemini_response

        client = GeminiClient(api_key="fake_key")
        # Simplified tool for this test, details tested in constructs_gemini_tools_correctly
        textura_tools = [TexturaTool(function_declarations=[TexturaFunctionDeclaration(name="get_weather", description="...", parameters=TexturaFunctionParameters(properties={"location": TexturaFunctionParameterProperty(type="string")}))])]

        llm_response = client.predict_with_tools("What's the weather?", tools=textura_tools)

        self.assertIsNotNone(llm_response.function_calls)
        self.assertEqual(len(llm_response.function_calls), 1)
        self.assertIsNone(llm_response.text)

        function_call = llm_response.function_calls[0]
        self.assertEqual(function_call.name, "get_weather")
        self.assertEqual(function_call.arguments, {"location": "Boston"})

    @patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_key"})
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_predict_with_tools_parses_text_response_when_tools_provided(self, mock_generative_model_class, mock_configure):
        """Test parsing of a text response when tools are provided but not used by LLM."""
        mock_model_instance = MagicMock()
        mock_generative_model_class.return_value = mock_model_instance

        mock_text_part = MagicMock()
        mock_text_part.text = "I cannot call tools right now."
        mock_text_part.function_call = None

        mock_gemini_response = MagicMock()
        mock_gemini_response.candidates = [MagicMock(content=MagicMock(parts=[mock_text_part]))]

        mock_model_instance.generate_content.return_value = mock_gemini_response

        client = GeminiClient(api_key="fake_key")
        textura_tools = [TexturaTool(function_declarations=[TexturaFunctionDeclaration(name="get_weather", description="...", parameters=TexturaFunctionParameters(properties={"location": TexturaFunctionParameterProperty(type="string")}))])]

        llm_response = client.predict_with_tools("Some prompt", tools=textura_tools)

        self.assertIsNone(llm_response.function_calls)
        self.assertIsNotNone(llm_response.text)
        self.assertEqual(llm_response.text, "I cannot call tools right now.")


if __name__ == '__main__':
    unittest.main()
