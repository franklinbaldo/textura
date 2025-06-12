import json
from typing import List, Dict, Any, Union, Type, Callable, Optional # Added Optional
from pydantic import ValidationError

from textura.extraction.models import EventV1, MysteryV1, ExtractionItem
from textura.logging.metacog import Metacog

# --- Mock LLM Implementation ---
# [[[POC_MOCK_LLM_RESPONSES_START]]]
MOCK_LLM_RESPONSES = [
    # Valid Event
    {
        "extractions": [
            {
                "type": "event",
                "data": {
                    "timestamp": "2024-06-15T14:30:00Z",
                    "description": "The team meeting started with a review of last week's progress.",
                }
            }
        ]
    },
    # Valid Mystery
    {
        "extractions": [
            {
                "type": "mystery",
                "data": {
                    "question": "What is the 'Project Chimera' mentioned in the document?",
                    "context": "The document refers to 'Project Chimera' multiple times but provides no definition."
                }
            }
        ]
    },
    # Mixed valid Event and valid Mystery
    {
        "extractions": [
            {
                "type": "event",
                "data": {
                    "timestamp": "Yesterday afternoon",
                    "description": "Alice finished her part of the report."
                }
            },
            {
                "type": "mystery",
                "data": {
                    "question": "Why was Bob absent from the critical meeting?",
                    "context": "Meeting minutes show Alice, Carol, and Dave present, but not Bob."
                }
            }
        ]
    },
    # Invalid Event (missing description)
    {
        "extractions": [
            {
                "type": "event",
                "data": {
                    "timestamp": "2024-06-15"
                }
            }
        ]
    },
    # Invalid Mystery (question is not a string)
    {
        "extractions": [
            {
                "type": "mystery",
                "data": {
                    "question": 12345,
                    "context": "A number was found instead of a question."
                }
            }
        ]
    },
    # Invalid type field
    {
        "extractions": [
            {
                "type": "unknown_type",
                "data": {
                    "info": "This type is not recognized."
                }
            }
        ]
    },
    # Malformed JSON (will be returned as a string by mock_llm_client)
    '{"extractions": [{"type": "event", "data": {"timestamp": "today", "description": "System backup completed."}}',
    # Empty extractions list
    {
        "extractions": []
    },
    # No "extractions" key
    {
        "other_key": "some data"
    }
]
_mock_llm_call_count = 0
# [[[POC_MOCK_LLM_RESPONSES_END]]]

def mock_llm_client(chunk_text: str) -> str:
    """
    A mock LLM client that cycles through predefined responses.
    Some responses are valid JSON, others are malformed or cause validation errors.
    """
    global _mock_llm_call_count
    response_index = _mock_llm_call_count % len(MOCK_LLM_RESPONSES)
    _mock_llm_call_count += 1

    response = MOCK_LLM_RESPONSES[response_index]
    if isinstance(response, str): # For testing malformed JSON
        return response
    return json.dumps(response)

# --- ExtractorAgent Implementation ---

class ExtractorAgent:
    """
    Extracts structured information (Events, Mysteries) from text chunks
    using an LLM (currently mocked) and validates against Pydantic models.
    """
    def __init__(
        self,
        metacog_logger: Metacog,
        llm_client: Optional[Callable[[str], str]] = None, # Added llm_client parameter
        event_model: Type[EventV1] = EventV1,
        mystery_model: Type[MysteryV1] = MysteryV1
    ):
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = mock_llm_client # Default to the built-in mock if no client is provided
        self.metacog_logger = metacog_logger
        self.event_model = event_model
        self.mystery_model = mystery_model
        self.model_map: Dict[str, Type[Union[EventV1, MysteryV1]]] = {
            "event": self.event_model,
            "mystery": self.mystery_model,
        }

    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        source_file: str
    ) -> List[ExtractionItem]:
        """
        Uses an LLM (mocked) to extract structured data from a text chunk,
        validates it, and logs the process.

        Args:
            chunk_text: The text content of the chunk.
            chunk_id: The unique identifier for the chunk.
            source_file: The original file name the chunk belongs to.

        Returns:
            A list of validated Pydantic model instances (EventV1 or MysteryV1).
        """
        raw_llm_output = self.llm_client(chunk_text)

        validated_extractions: List[ExtractionItem] = []
        errors: List[Dict[str, Any]] = []

        try:
            parsed_llm_response = json.loads(raw_llm_output)
            if not isinstance(parsed_llm_response, dict) or "extractions" not in parsed_llm_response:
                errors.append({
                    "error_type": "FormatError",
                    "message": "LLM response is missing 'extractions' key or is not a dictionary.",
                    "raw_output_snippet": raw_llm_output[:200]
                })
            elif not isinstance(parsed_llm_response["extractions"], list):
                errors.append({
                    "error_type": "FormatError",
                    "message": "'extractions' key is not a list.",
                    "raw_output_snippet": raw_llm_output[:200]
                })
            else:
                for item_data in parsed_llm_response["extractions"]:
                    if not isinstance(item_data, dict) or "type" not in item_data or "data" not in item_data:
                        errors.append({
                            "error_type": "FormatError",
                            "message": "Extraction item is not a dict or missing 'type'/'data' keys.",
                            "item_data": item_data
                        })
                        continue

                    extraction_type_str = item_data.get("type")
                    model_class = self.model_map.get(extraction_type_str)

                    if not model_class:
                        errors.append({
                            "error_type": "UnsupportedType",
                            "message": f"Unknown extraction type: {extraction_type_str}",
                            "item_data": item_data,
                        })
                        continue

                    payload = item_data.get("data", {})
                    # Add source_file and chunk_id from context, not from LLM
                    payload["source_file"] = source_file
                    payload["chunk_id"] = chunk_id

                    try:
                        validated_item = model_class(**payload)
                        validated_extractions.append(validated_item)
                    except ValidationError as e:
                        errors.append({
                            "error_type": "ValidationError",
                            "model_type": extraction_type_str,
                            "message": str(e),
                            "problematic_data": payload,
                            "pydantic_errors": e.errors()
                        })
                    except Exception as e: # Catch any other unexpected error during instantiation
                        errors.append({
                            "error_type": "InstantiationError",
                            "model_type": extraction_type_str,
                            "message": str(e),
                            "problematic_data": payload,
                        })

        except json.JSONDecodeError:
            errors.append({
                "error_type": "JSONDecodeError",
                "message": "LLM output was not valid JSON.",
                "raw_output_snippet": raw_llm_output[:200] # Log a snippet
            })
        except Exception as e: # Catch any other unexpected error during parsing
             errors.append({
                "error_type": "GenericParsingError",
                "message": str(e),
                "raw_output_snippet": raw_llm_output[:200]
            })


        self.metacog_logger.log_extraction(
            chunk_id=chunk_id,
            source_file=source_file,
            raw_llm_output=raw_llm_output,
            validated_extractions=validated_extractions,
            errors=errors
        )
        return validated_extractions

if __name__ == '__main__':
    # Example Usage
    from pathlib import Path
    dummy_workspace_path = Path("_test_workspace_extractor")
    dummy_workspace_path.mkdir(exist_ok=True)
    (dummy_workspace_path / "logs").mkdir(exist_ok=True) # Ensure logs dir exists

    metacog = Metacog(workspace_path=str(dummy_workspace_path))
    agent = ExtractorAgent(metacog_logger=metacog)

    sample_texts = [
        "The meeting happened at noon. We discussed Project X.",
        "There's a question about the budget for Y.",
        "Alice reported success. Bob's status is unknown.",
        "This chunk is deliberately malformed by the mock LLM.",
        "This one will have a missing field.",
        "This one will have a type error.",
        "This will be an unknown type.",
        "This will be empty extractions",
        "This will be missing extractions key"
    ]

    all_extracted_items: List[ExtractionItem] = []

    print(f"--- Running ExtractorAgent with Mock LLM ({len(MOCK_LLM_RESPONSES)} predefined responses) ---")
    for i, text in enumerate(sample_texts):
        print(f"\nProcessing chunk {i+1}...")
        chunk_id = f"chunk_id_{i:03d}"
        source_file = f"source_file_{i%2}.txt"

        extracted = agent.extract_from_chunk(text, chunk_id, source_file)
        if extracted:
            print(f"  Validated extractions ({len(extracted)}):")
            for item in extracted:
                print(f"    - {item.model_dump_json(indent=4)}")
                all_extracted_items.append(item)
        else:
            print("  No items validated or extracted for this chunk.")

    print(f"\n--- Metacog Log Contents ({metacog.log_file_path.resolve()}) ---")
    if metacog.log_file_path.exists():
        with open(metacog.log_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                print(f"Log Entry {line_num+1}: {line.strip()}")
    else:
        print("Metacog log file not found.")

    # Clean up
    # import shutil
    # shutil.rmtree(dummy_workspace_path)
