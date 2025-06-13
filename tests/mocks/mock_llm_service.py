import json
from typing import Any


class MockLLMService:
    """
    A mock LLM service for testing the ExtractorAgent.
    It can be configured with specific responses for given inputs or cycle through default responses.
    """

    def __init__(self):
        self.responses: dict[str, str | dict[str, Any]] = {}
        self.default_responses: list[str | dict[str, Any]] = [
            # Default cycle of responses, similar to the original mock_llm_client
            {
                "extractions": [
                    {
                        "type": "event",
                        "data": {
                            "timestamp": "Default Event Time",
                            "description": "Default event description.",
                        },
                    }
                ]
            },
            {
                "extractions": [
                    {
                        "type": "mystery",
                        "data": {
                            "question": "Default mystery question?",
                            "context": "Default context.",
                        },
                    }
                ]
            },
            '{"extractions": [{"type": "event", "data": {"timestamp": "Malformed JSON response"}}',  # Malformed
        ]
        self.call_count = 0
        self.requests_log: list[str] = []

    def set_response(self, input_text_key: str, response: str | dict[str, Any]):
        """
        Set a specific response for a given input text key.
        If response is a dict, it will be converted to a JSON string.
        """
        if isinstance(response, dict):
            self.responses[input_text_key] = json.dumps(response)
        else:
            self.responses[input_text_key] = response

    def add_default_response(self, response: str | dict[str, Any]):
        """Add a response to the default cycle list."""
        # Prepend so the newest response is returned first
        self.default_responses.insert(0, response)

    def __call__(self, chunk_text: str) -> str:
        """
        Mimics an LLM client call.
        Logs the request and returns a configured or default response.
        """
        self.requests_log.append(chunk_text)

        if chunk_text in self.responses:
            response_data = self.responses[chunk_text]
        elif self.default_responses:
            response_data = self.default_responses[
                self.call_count % len(self.default_responses)
            ]
        else:
            # Fallback if no specific or default responses are set
            response_data = {
                "extractions": [
                    {
                        "type": "event",
                        "data": {
                            "description": "Fallback response: No specific mock found."
                        },
                    }
                ]
            }

        self.call_count += 1

        if isinstance(response_data, dict):
            return json.dumps(response_data)
        return str(response_data)  # If it's already a string (e.g. malformed JSON)

    def get_last_request(self) -> str | None:
        return self.requests_log[-1] if self.requests_log else None

    def get_request_count(self) -> int:
        return len(self.requests_log)


if __name__ == "__main__":
    # Example Usage
    mock_service = MockLLMService()

    # Configure a specific response
    test_input1 = "Tell me about the meeting."
    mock_service.set_response(
        test_input1,
        {
            "extractions": [
                {
                    "type": "event",
                    "data": {
                        "timestamp": "2024-07-01",
                        "description": "Meeting was productive.",
                    },
                },
            ],
        },
    )
    print(f"Response for '{test_input1}': {mock_service(test_input1)}")

    # Use default responses
    test_input2 = "Any mysteries here?"
    print(f"Response for '{test_input2}' (default 1): {mock_service(test_input2)}")
    test_input3 = "Another query."
    print(f"Response for '{test_input3}' (default 2): {mock_service(test_input3)}")
    test_input4 = "And another."  # Will cycle to malformed
    print(f"Response for '{test_input4}' (default 3): {mock_service(test_input4)}")
    test_input5 = "Back to default 1."
    print(
        f"Response for '{test_input5}' (default 1 again): {mock_service(test_input5)}"
    )

    print(f"\nLLM Service Call count: {mock_service.get_request_count()}")
    print(f"Last request made: {mock_service.get_last_request()}")
