import json
from typing import Dict, List, Any, Union, Optional
from textura.llm_clients.base import BaseLLMClient

class MockLLMService(BaseLLMClient):
    """
    A mock LLM service for testing, implementing BaseLLMClient.
    It can be configured with specific responses for given prompt keys or cycle through default responses.
    """
    def __init__(self):
        self.responses: Dict[str, str] = {} # Stores prompt_key -> JSON string response
        self.default_responses: List[Union[str, Dict[str, Any]]] = [
            # Default cycle of responses
            {"extractions": [{"type": "event", "data": {"timestamp": "Default Event Time", "description": "Default event description."}}]},
            {"extractions": [{"type": "mystery", "data": {"question": "Default mystery question?", "context": "Default context."}}]},
            '{"extractions": [{"type": "event", "data": {"timestamp": "Malformed JSON response"}}', # Malformed
        ]
        self.call_count = 0
        self.requests_log: List[str] = []

    def set_response(self, prompt_key: str, response_data: Union[str, Dict[str, Any]]):
        """
        Set a specific response for a given prompt key.
        If response_data is a dict, it will be converted to a JSON string.
        """
        if isinstance(response_data, dict):
            self.responses[prompt_key] = json.dumps(response_data)
        else: # It's already a string (e.g., pre-formatted JSON or malformed JSON string)
            self.responses[prompt_key] = response_data

    def add_default_response(self, response: Union[str, Dict[str, Any]]):
        """Add a response to the default cycle list."""
        self.default_responses.insert(0, response)

    def predict(self, prompt: str, **kwargs: Any) -> str:
        """
        Returns a configured or default response based on the prompt.
        Logs the prompt request.
        """
        self.requests_log.append(prompt)
        self.call_count += 1

        if prompt in self.responses:
            # self.responses stores already stringified JSON
            return self.responses[prompt]
        elif self.default_responses:
            default_response_item = self.default_responses[(self.call_count -1) % len(self.default_responses)]
            if isinstance(default_response_item, dict):
                return json.dumps(default_response_item)
            return str(default_response_item) # Already a string
        else:
            # Fallback if no specific or default responses are set
            return json.dumps({"extractions": [], "error": f"Mock response not set for prompt: {prompt}"})

    def get_last_request(self) -> Optional[str]:
        return self.requests_log[-1] if self.requests_log else None

    def get_request_count(self) -> int:
        return len(self.requests_log)

if __name__ == '__main__':
    # Example Usage
    mock_service = MockLLMService()

    # Configure a specific response
    test_prompt1 = "Tell me about the meeting."
    mock_service.set_response(test_prompt1, {
        "extractions": [
            {"type": "event", "data": {"timestamp": "2024-07-01", "description": "Meeting was productive."}}
        ]
    })
    print(f"Response for '{test_prompt1}': {mock_service.predict(test_prompt1)}")

    # Use default responses
    test_prompt2 = "Any mysteries here?"
    print(f"Response for '{test_prompt2}' (default 1): {mock_service.predict(test_prompt2)}")
    test_prompt3 = "Another query."
    print(f"Response for '{test_prompt3}' (default 2): {mock_service.predict(test_prompt3)}")
    test_prompt4 = "And another." # Will cycle to malformed
    print(f"Response for '{test_prompt4}' (default 3): {mock_service.predict(test_prompt4)}")
    test_prompt5 = "Back to default 1."
    print(f"Response for '{test_prompt5}' (default 1 again): {mock_service.predict(test_prompt5)}")

    print(f"\nLLM Service Call count: {mock_service.call_count}") # Changed from get_request_count()
    print(f"Last request made: {mock_service.get_last_request()}")
