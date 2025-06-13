import json
from typing import List, Dict, Any, Union, Type, Callable, Optional # Added Optional
from pydantic import ValidationError

from textura.extraction.models import EventV1, MysteryV1, PersonV1, LocationV1, OrganizationV1, ExtractionItem
from textura.extraction.prompts import SCHEMA_FIRST_EXTRACTION_PROMPT
from textura.logging.metacog import Metacog
from textura.llm_clients.base import BaseLLMClient

# --- Mock LLM Implementation (Internal Fallback) ---
# [[[POC_MOCK_LLM_RESPONSES_START]]]
MOCK_LLM_RESPONSES = [
    # Valid Event
    {"extractions": [{"type": "event", "data": {"timestamp": "2024-06-15T14:30:00Z", "description": "Team meeting for Project Alpha."}}]},
    # Valid Mystery
    {"extractions": [{"type": "mystery", "data": {"question": "What is 'Project Chimera'?", "context": "Document mentions 'Project Chimera' without definition."}}]},
    # Valid Person
    {"extractions": [{"type": "person", "data": {"name": "Dr. Evelyn Reed", "title": "Chief Scientist", "role": "Lead Investigator"}}]},
    # Valid Location
    {"extractions": [{"type": "location", "data": {"name": "Area 51", "type": "Research Facility"}}]},
    # Valid Organization
    {"extractions": [{"type": "organization", "data": {"name": "BioFuture Inc.", "type": "Biotechnology Company", "industry": "Healthcare"}}]},
    # Mixed valid items
    {"extractions": [
        {"type": "event", "data": {"timestamp": "Yesterday", "description": "Alice completed her report for Globex."}},
        {"type": "person", "data": {"name": "Alice Wonderland", "role": "Reporter"}},
        {"type": "organization", "data": {"name": "Globex Corporation"}}
    ]},
    # Invalid Event (missing description)
    {"extractions": [{"type": "event", "data": {"timestamp": "2024-06-15"}}]},
    # Invalid Mystery (question is not a string)
    {"extractions": [{"type": "mystery", "data": {"question": 12345, "context": "Invalid question type."}}]},
    # Invalid Person (missing name)
    {"extractions": [{"type": "person", "data": {"title": "CEO"}}]},
    # Invalid Location (name is not a string)
    {"extractions": [{"type": "location", "data": {"name": True, "type": "Restricted Area"}}]},
    # Invalid Organization (type is a number)
    {"extractions": [{"type": "organization", "data": {"name": "DataSys", "type": 123}}]},
    # Invalid type field
    {"extractions": [{"type": "unknown_entity", "data": {"info": "This type is not recognized."}}]},
    # Malformed JSON string
    '{"extractions": [{"type": "event", "data": {"timestamp": "today", "description": "System backup completed."}}',
    # Empty extractions list
    {"extractions": []},
    # No "extractions" key
    {"other_key": "some data"},
    # Multiple extractions, one invalid
    {"extractions": [
        {"type": "person", "data": {"name": "Valid Person"}},
        {"type": "event", "data": {}} # Invalid event, missing timestamp and description
    ]}
]
_mock_llm_call_count = 0
# [[[POC_MOCK_LLM_RESPONSES_END]]]

def mock_llm_client(prompt_string: str) -> str: # Parameter changed to prompt_string, but it's not used by mock
    """
    A mock LLM client that cycles through predefined responses.
    It IGNORES the prompt_string and just returns the next mock response.
    Some responses are valid JSON, others are malformed or cause validation errors.
    """
    global _mock_llm_call_count
    response_index = _mock_llm_call_count % len(MOCK_LLM_RESPONSES)
    _mock_llm_call_count += 1
    print(f"MockLLM: Called with prompt (first 50 chars): '{prompt_string[:50]}...'. Returning response index: {response_index}") # For debugging

    response = MOCK_LLM_RESPONSES[response_index]
    if isinstance(response, str): # For testing malformed JSON
        return response
    return json.dumps(response)

# --- ExtractorAgent Implementation ---

class ExtractorAgent:
    """
    Extracts structured information from text chunks using an LLM client
    and validates against Pydantic models.
    """
    def __init__(
        self,
        metacog_logger: Metacog,
        llm_client: Optional[BaseLLMClient] = None, # Updated type hint
    ):
        self.metacog_logger = metacog_logger
        self.llm_client_instance = llm_client # Store the BaseLLMClient instance

        # Model map setup
        self.model_map: Dict[str, Type[ExtractionItem]] = {
            "event": EventV1, # Assumes EventV1 is imported
            "mystery": MysteryV1, # Assumes MysteryV1 is imported
            "person": PersonV1,
            "location": LocationV1,
            "organization": OrganizationV1,
        }

    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        source_file: str
    ) -> List[ExtractionItem]:
        """
        Uses an LLM to extract structured data from a text chunk by formatting
        a detailed prompt, then validates the LLM's response, and logs the process.

        Args:
            chunk_text: The text content of the chunk.
            chunk_id: The unique identifier for the chunk.
            source_file: The original file name the chunk belongs to.

        Returns:
            A list of validated Pydantic model instances (subtypes of ExtractionItem).
        """
        formatted_prompt = SCHEMA_FIRST_EXTRACTION_PROMPT.format(text_chunk_content=chunk_text)
        raw_llm_output: str

        if self.llm_client_instance:
            raw_llm_output = self.llm_client_instance.predict(formatted_prompt)
        else:
            # Fallback to internal mock if no client instance was provided
            # This maintains behavior for the existing if __name__ == '__main__' block
            # and for tests that don't pass an llm_client.
            # The internal mock_llm_client's signature takes `prompt_string` (previously `chunk_text`)
            # which matches the `formatted_prompt` here, so it should still work.
            # However, the *behavior* of the internal mock is to cycle through MOCK_LLM_RESPONSES
            # ignoring the actual prompt content. This is acceptable for a fallback/demo.
            print("ExtractorAgent: No LLM client instance provided, using internal mock_llm_client.")
            raw_llm_output = mock_llm_client(formatted_prompt) # Pass the formatted_prompt

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

    # More diverse sample texts to trigger various mock responses
    sample_texts = [
        "A meeting about Project Alpha was held. Dr. Evelyn Reed attended.", # Event, Person
        "Where is Area 51 located? BioFuture Inc. might know.", # Mystery, Location, Organization
        "Alice from Globex called yesterday.", # Person, Organization, Event
        "The financial report for Q2 is out.", # Potential Event
        "Dr. Smith visited the New York office of OmniCorp.", # Person, Location, Organization
        "What happened to the server in London?", # Mystery, Location
        "The CEO of MacroHard, Bill G., announced a new product.", # Person, Organization, Event
        "This text should trigger a malformed JSON response from the mock.", # Malformed JSON
        "This text should trigger an empty extractions list from the mock.", # Empty list
        "This text should trigger a response missing the 'extractions' key.", # Missing key
        "This should trigger an invalid event.", # Invalid event
        "This should trigger an invalid mystery.", # Invalid mystery
        "This should trigger an invalid person.", # Invalid person
        "This should trigger an invalid location.", # Invalid location
        "This should trigger an invalid organization.", # Invalid organization
        "This should trigger an unknown type.", # Unknown type
        "Valid person Dr. Valid and invalid event." # Mixed valid/invalid
    ]

    all_extracted_items: List[ExtractionItem] = []

    # Ensure MOCK_LLM_RESPONSES has enough variety for the sample texts
    print(f"--- Running ExtractorAgent with Mock LLM ({len(MOCK_LLM_RESPONSES)} predefined responses for {len(sample_texts)} text chunks) ---")
    if len(sample_texts) > len(MOCK_LLM_RESPONSES):
        print("Warning: More sample texts than mock responses. Responses will repeat.")

    for i, text in enumerate(sample_texts):
        print(f"\nProcessing chunk {i+1}: '{text}'")
        chunk_id = f"chunk_id_{i:03d}"
        source_file = f"source_file_{i%3}.txt" # Cycle through 3 source files

        # The mock_llm_client will ignore 'text' and cycle through MOCK_LLM_RESPONSES
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
