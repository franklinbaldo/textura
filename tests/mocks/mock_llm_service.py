import json
from typing import Dict, List, Any, Union, Optional
from textura.llm_clients.base import BaseLLMClient
from textura.llm_clients.types import Tool, LLMResponse, FunctionCall

class MockLLMService(BaseLLMClient):
    """
    A mock LLM service for testing, implementing BaseLLMClient.
    It can be configured with specific LLMResponse objects for given prompt keys or cycle through default LLMResponse objects.
    """
    def __init__(self):
        self.responses: Dict[str, LLMResponse] = {}
        # Convert old default_responses to LLMResponse objects
        _old_default_responses: List[Union[str, Dict[str, Any]]] = [
            {"extractions": [{"type": "event", "data": {"timestamp": "Default Event Time", "description": "Default event description."}}]},
            {"extractions": [{"type": "mystery", "data": {"question": "Default mystery question?", "context": "Default context."}}]},
            '{"extractions": [{"type": "event", "data": {"timestamp": "Malformed JSON response"}}', # Malformed
        ]
        self.default_responses: List[LLMResponse] = []
        for item in _old_default_responses:
            if isinstance(item, dict):
                # Assuming these dicts were meant to be JSON strings for the ExtractorAgent's simple mock
                # For LLMResponse, they should be actual text or function calls.
                # For simplicity, let's assume they become text responses.
                self.default_responses.append(LLMResponse(text=json.dumps(item)))
            elif isinstance(item, str):
                 # If it's a string (like malformed JSON), wrap it in LLMResponse.text
                self.default_responses.append(LLMResponse(text=item))

        self.call_count = 0
        self.requests_log: List[Dict[str, Any]] = [] # Store more structured log data

    def set_response(self, prompt_key: str, response: LLMResponse):
        """
        Set a specific LLMResponse for a given prompt key.
        """
        self.responses[prompt_key] = response

    def add_default_response(self, response: LLMResponse): # Now expects LLMResponse
        """Add an LLMResponse to the default cycle list."""
        self.default_responses.insert(0, response)

    def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Returns the text part of a configured or default LLMResponse.
        This adapts the old predict method.
        Logs the prompt request.
        """
        self.call_count += 1
        self.requests_log.append({"prompt": prompt, "kwargs": kwargs, "method": "predict"})

        if prompt in self.responses:
            llm_response = self.responses[prompt]
            if llm_response.text is not None:
                return llm_response.text
            elif llm_response.function_calls: # If only function calls, return a representation or error
                return f"Error: LLMResponse for '{prompt}' contained function calls, not text."

        # Fallback to a very simple default string if no specific LLMResponse with text found
        return "MockLLMService: Simple text response from old predict method."

    def predict_with_tools(self, prompt: str, tools: Optional[List[Tool]] = None, **kwargs: Any) -> LLMResponse:
        """
        Returns a configured or default LLMResponse based on the prompt.
        Logs the prompt, tools info, and kwargs.
        The 'tools' argument is logged but not used for response selection in this mock.
        """
        self.call_count += 1
        self.requests_log.append({
            "prompt": prompt,
            "tools_provided_count": len(tools) if tools else 0,
            "kwargs": kwargs,
            "method": "predict_with_tools"
        })

        if prompt in self.responses:
            return self.responses[prompt]
        elif self.default_responses:
            response_index = (self.call_count - 1) % len(self.default_responses)
            return self.default_responses[response_index]

        # Fallback if no specific or default response is found
        print(f"MockLLMService: No specific or default LLMResponse for prompt: {prompt}")
        return LLMResponse(text="MockLLMService: No response configured for this prompt.", function_calls=None)

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        return self.requests_log[-1] if self.requests_log else None

    def get_request_count(self) -> int: # This now just returns length of structured log
        return len(self.requests_log)


if __name__ == '__main__':
    mock_service = MockLLMService()

    # Example for predict_with_tools
    prompt1 = "Find user by ID 123"
    fc1 = FunctionCall(name="getUserById", arguments={"id": 123})
    llm_response1 = LLMResponse(function_calls=[fc1])
    mock_service.set_response(prompt1, llm_response1)

    response1 = mock_service.predict_with_tools(prompt1)
    print(f"Response for '{prompt1}':")
    if response1.function_calls:
        for fc in response1.function_calls:
            print(f"  Function Call: {fc.name}, Args: {fc.arguments}")
    if response1.text:
        print(f"  Text: {response1.text}")

    # Example for old predict method (if a response with text is set)
    prompt2 = "Give me some text."
    llm_response2 = LLMResponse(text="This is a text response for the old predict method.")
    mock_service.set_response(prompt2, llm_response2)
    text_response = mock_service.predict(prompt2)
    print(f"\nOld predict response for '{prompt2}': {text_response}")

    # Example of predict_with_tools using a default response
    prompt3 = "Any default response?"
    default_tool_response = mock_service.predict_with_tools(prompt3)
    print(f"\nDefault response for '{prompt3}':")
    if default_tool_response.function_calls:
        for fc in default_tool_response.function_calls:
            print(f"  Function Call: {fc.name}, Args: {fc.arguments}")
    if default_tool_response.text:
        print(f"  Text: {default_tool_response.text}")


    print(f"\nLLM Service Call count: {mock_service.call_count}")
    print(f"Last request made: {mock_service.get_last_request()}")
    # print(f"All requests: {mock_service.requests_log}")
