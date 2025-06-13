import json
from typing import List, Dict, Any, Union, Type, Optional # Callable removed, get_type_hints etc added later
from pydantic import ValidationError, BaseModel
import inspect # To check for BaseModel

from textura.extraction.models import EventV1, MysteryV1, PersonV1, LocationV1, OrganizationV1, ExtractionItem
from textura.extraction.prompts import SCHEMA_FIRST_EXTRACTION_PROMPT # This prompt will need updating for tool use
from textura.logging.metacog import Metacog
from textura.llm_clients.base import BaseLLMClient
from textura.llm_clients.types import (
    Tool as TexturaTool,
    FunctionDeclaration as TexturaFunctionDeclaration,
    FunctionParameters as TexturaFunctionParameters,
    FunctionParameterProperty as TexturaFunctionParameterProperty,
    LLMResponse,
    FunctionCall
)

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

# This internal mock_llm_client is now less aligned with the new LLMResponse-based flow.
# It's kept for basic fallback if no llm_client_instance is provided, but will not support tool use.
def mock_llm_client(prompt_string: str) -> str:
    """
    A mock LLM client that cycles through predefined JSON string responses.
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

    def _generate_tools_from_pydantic_models(self) -> List[TexturaTool]:
        tools = []
        function_declarations = []

        for model_name_str, model_class in self.model_map.items():
            if not inspect.isclass(model_class) or not issubclass(model_class, BaseModel):
                continue

            schema = model_class.model_json_schema()
            properties = {}
            # Pydantic v2 model_json_schema() might put required fields inside 'components' or 'defs'
            # and then reference them. For simple flat models, 'required' might be top-level.
            # This logic might need adjustment based on actual schema structure for complex models.
            required_fields = schema.get('required', [])

            for field_name, field_props in schema.get('properties', {}).items():
                if field_name in ["source_file", "chunk_id"]:
                    continue

                json_type = field_props.get('type', 'string')
                # Handle Optional fields (e.g. {'anyOf': [{'type': 'string'}, {'type': 'null'}]})
                # or {'type': ['string', 'null']} in newer JSON schema drafts
                if 'anyOf' in field_props:
                    types_without_null = [t['type'] for t in field_props['anyOf'] if t.get('type') != 'null']
                    if len(types_without_null) == 1:
                        json_type = types_without_null[0]
                elif isinstance(field_props.get('type'), list): # e.g. "type": ["string", "null"]
                    types_without_null = [t for t in field_props['type'] if t != 'null']
                    if len(types_without_null) == 1:
                        json_type = types_without_null[0]

                properties[field_name] = TexturaFunctionParameterProperty(
                    type=json_type,
                    description=field_props.get('description', field_props.get('title', ''))
                )

            function_declarations.append(
                TexturaFunctionDeclaration(
                    name=model_name_str,
                    description=schema.get('description', f"Extracts a {model_name_str} entity."),
                    parameters=TexturaFunctionParameters(
                        properties=properties,
                        required=[rf for rf in required_fields if rf not in ["source_file", "chunk_id"]]
                    )
                )
            )

        if function_declarations:
            tools.append(TexturaTool(function_declarations=function_declarations))
        return tools

    def extract_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        source_file: str
    ) -> List[ExtractionItem]:

        # TODO: The SCHEMA_FIRST_EXTRACTION_PROMPT needs to be updated to be more suitable for tool use.
        # It currently instructs the LLM to format as JSON with an "extractions" key, which is
        # different from how function calling/tool use works.
        # For now, we'll use it, but this is a key area for future improvement.
        formatted_prompt = SCHEMA_FIRST_EXTRACTION_PROMPT.format(text_chunk_content=chunk_text)

        tools_for_llm = self._generate_tools_from_pydantic_models()

        llm_response_obj: Optional[LLMResponse] = None
        raw_llm_output_for_log: str = ""
        errors: List[Dict[str, Any]] = []
        validated_extractions: List[ExtractionItem] = []

        if self.llm_client_instance:
            try:
                llm_response_obj = self.llm_client_instance.predict_with_tools(
                    formatted_prompt,
                    tools=tools_for_llm
                )
                if llm_response_obj:
                    raw_llm_output_for_log = llm_response_obj.model_dump_json()
            except Exception as e:
                errors.append({
                    "error_type": "LLMClientError",
                    "message": f"Error calling LLM client's predict_with_tools: {str(e)}",
                })
                raw_llm_output_for_log = f"LLMClientError: {str(e)}"
        else:
            print("ExtractorAgent: No LLM client instance provided, using internal mock_llm_client (tool use not simulated).")
            raw_text_output = mock_llm_client(formatted_prompt)
            raw_llm_output_for_log = raw_text_output
            # The internal mock returns a JSON string that looks like the old direct extraction format.
            # To make it somewhat compatible with the new flow, we'll try to parse it
            # and if it contains "extractions", we'll process them as if they were function calls.
            # This is a temporary bridge for the internal mock.
            try:
                parsed_mock_json = json.loads(raw_text_output)
                if "extractions" in parsed_mock_json and isinstance(parsed_mock_json["extractions"], list):
                    mock_function_calls = []
                    for item in parsed_mock_json["extractions"]:
                        if isinstance(item, dict) and "type" in item and "data" in item:
                            mock_function_calls.append(FunctionCall(name=item["type"], arguments=item["data"]))
                    if mock_function_calls:
                        llm_response_obj = LLMResponse(function_calls=mock_function_calls)
                    else:
                        llm_response_obj = LLMResponse(text=raw_text_output) # No valid mock "extractions"
                else:
                    llm_response_obj = LLMResponse(text=raw_text_output) # Not the expected mock structure
            except json.JSONDecodeError:
                llm_response_obj = LLMResponse(text=raw_text_output) # Malformed JSON from mock


        if llm_response_obj and llm_response_obj.function_calls:
            for func_call in llm_response_obj.function_calls:
                model_class = self.model_map.get(func_call.name)
                if not model_class:
                    errors.append({
                        "error_type": "UnsupportedFunctionCall",
                        "message": f"LLM called unknown function: {func_call.name}",
                        "function_call_name": func_call.name,
                        "arguments": func_call.arguments
                    })
                    continue

                payload = func_call.arguments.copy() if func_call.arguments else {}
                payload["source_file"] = source_file
                payload["chunk_id"] = chunk_id

                try:
                    validated_item = model_class(**payload)
                    validated_extractions.append(validated_item)
                except ValidationError as e:
                    errors.append({
                        "error_type": "ValidationErrorFromFunctionCall",
                        "model_type": func_call.name,
                        "message": str(e),
                        "problematic_data": payload,
                        "pydantic_errors": e.errors()
                    })
                except Exception as e:
                    errors.append({
                        "error_type": "InstantiationErrorFromFunctionCall",
                        "model_type": func_call.name,
                        "message": str(e),
                        "problematic_data": payload
                    })
        elif llm_response_obj and llm_response_obj.text:
            errors.append({
                "error_type": "TextResponseFromLLM", # Renamed for clarity
                "message": "LLM returned a text response. If tools were provided, this means no suitable tool was called.",
                "text_response": llm_response_obj.text[:500]
            })
        elif not llm_response_obj: # Only if client returned None and no error caught before
             errors.append({
                "error_type": "NoLLMResponse",
                "message": "LLM client returned no response object.",
            })
        # If llm_response_obj exists but has neither text nor function_calls (should be rare with Pydantic model)
        elif not llm_response_obj.text and not llm_response_obj.function_calls:
             errors.append({
                "error_type": "EmptyLLMResponse",
                "message": "LLMResponse object had neither text nor function calls.",
            })

        self.metacog_logger.log_extraction(
            chunk_id=chunk_id,
            source_file=source_file,
            raw_llm_output=raw_llm_output_for_log,
            validated_extractions=validated_extractions,
            errors=errors,
            prompt_sent=formatted_prompt
            # TODO: Add tools_sent=tools_for_llm to log (need to serialize TexturaTool list)
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
