import pytest
from pathlib import Path
from pydantic import ValidationError

from textura.extraction.extractor_agent import ExtractorAgent
from textura.extraction.models import EventV1, MysteryV1, PersonV1, LocationV1, OrganizationV1, ExtractionItem
from textura.logging.metacog import Metacog
from tests.mocks.mock_llm_service import MockLLMService

# --- Direct Pydantic Model Validation Tests ---

def test_person_model_validation():
    """Test direct Pydantic validation for PersonV1."""
    # Valid data
    person = PersonV1(name="John Doe", title="Dr.", role="Lead", source_file="s.txt", chunk_id="c1")
    assert person.name == "John Doe"
    assert person.title == "Dr."

    # Missing name
    with pytest.raises(ValidationError) as excinfo:
        PersonV1(title="Mr.", source_file="s.txt", chunk_id="c1")
    assert "name" in str(excinfo.value).lower()
    assert "field required" in str(excinfo.value).lower()

    # Name not a string
    with pytest.raises(ValidationError) as excinfo:
        PersonV1(name=123, source_file="s.txt", chunk_id="c1")
    assert "name" in str(excinfo.value).lower()
    assert "string type expected" in str(excinfo.value).lower()

def test_location_model_validation():
    """Test direct Pydantic validation for LocationV1."""
    location = LocationV1(name="Area 51", type="Military Base", source_file="s.txt", chunk_id="c1")
    assert location.name == "Area 51"

    with pytest.raises(ValidationError) as excinfo:
        LocationV1(type="Unknown", source_file="s.txt", chunk_id="c1")
    assert "name" in str(excinfo.value).lower()
    assert "field required" in str(excinfo.value).lower()

def test_organization_model_validation():
    """Test direct Pydantic validation for OrganizationV1."""
    org = OrganizationV1(name="Acme Corp", industry="Manufacturing", source_file="s.txt", chunk_id="c1")
    assert org.name == "Acme Corp"

    with pytest.raises(ValidationError) as excinfo:
        OrganizationV1(industry="Tech", source_file="s.txt", chunk_id="c1")
    assert "name" in str(excinfo.value).lower()
    assert "field required" in str(excinfo.value).lower()


# --- ExtractorAgent Tests ---

@pytest.fixture
def metacog_logger(test_workspace: Path) -> Metacog:
    """Fixture to provide a Metacog logger instance for tests."""
    return Metacog(workspace_path=str(test_workspace))

@pytest.fixture
def mock_llm() -> MockLLMService:
    """Fixture to provide a MockLLMService instance."""
    return MockLLMService()

# Helper function to check Metacog logs
def check_metacog_log(metacog_logger: Metacog, expected_strings: list[str], not_expected_strings: list[str] = None):
    log_file = metacog_logger.log_file_path
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
    for s in expected_strings:
        assert s in log_content
    if not_expected_strings:
        for s in not_expected_strings:
            assert s not in log_content

def test_extractor_agent_extracts_event(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test that ExtractorAgent can extract a valid EventV1."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)

    chunk_text = "An event happened on July 4th."
    chunk_id = "test_chunk_001"
    source_file = "test_doc.txt"

    # Configure mock LLM to return a specific valid event
    mock_llm.set_response(chunk_text, {
        "extractions": [
            {
                "type": "event",
                "data": {
                    "timestamp": "July 4th",
                    "description": "Fireworks display observed."
                }
            }
        ]
    })

    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)

    assert len(extractions) == 1
    extracted_item = extractions[0]
    assert isinstance(extracted_item, EventV1)
    assert extracted_item.description == "Fireworks display observed."
    assert extracted_item.timestamp == "July 4th"
    assert extracted_item.source_file == source_file
    assert extracted_item.chunk_id == chunk_id

    # Check if Metacog logged something (basic check)
    log_file = metacog_logger.log_file_path
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert "Fireworks display observed." in log_content
    check_metacog_log(metacog_logger, ["Fireworks display observed.", "errors\": []"])


def test_extractor_agent_extracts_person(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent can extract a valid PersonV1."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "Dr. John Doe was present."
    chunk_id = "person_chunk_001"
    source_file = "person_doc.txt"

    mock_llm.set_response(chunk_text, {
        "extractions": [{
            "type": "person",
            "data": {"name": "Dr. John Doe", "title": "Dr.", "role": "Attendee"}
        }]
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 1
    item = extractions[0]
    assert isinstance(item, PersonV1)
    assert item.name == "Dr. John Doe"
    assert item.title == "Dr."
    assert item.role == "Attendee"
    assert item.source_file == source_file
    assert item.chunk_id == chunk_id
    check_metacog_log(metacog_logger, ["Dr. John Doe", "errors\": []"])

def test_extractor_agent_extracts_location(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent can extract a valid LocationV1."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "The meeting was in Building A."
    chunk_id = "loc_chunk_001"
    source_file = "loc_doc.txt"

    mock_llm.set_response(chunk_text, {
        "extractions": [{
            "type": "location",
            "data": {"name": "Building A", "type": "Office Building"}
        }]
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 1
    item = extractions[0]
    assert isinstance(item, LocationV1)
    assert item.name == "Building A"
    assert item.type == "Office Building"
    check_metacog_log(metacog_logger, ["Building A", "errors\": []"])


def test_extractor_agent_extracts_organization(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent can extract a valid OrganizationV1."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "Acme Corp announced profits."
    chunk_id = "org_chunk_001"
    source_file = "org_doc.txt"

    mock_llm.set_response(chunk_text, {
        "extractions": [{
            "type": "organization",
            "data": {"name": "Acme Corp", "industry": "Manufacturing"}
        }]
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 1
    item = extractions[0]
    assert isinstance(item, OrganizationV1)
    assert item.name == "Acme Corp"
    assert item.industry == "Manufacturing"
    check_metacog_log(metacog_logger, ["Acme Corp", "errors\": []"])


def test_extractor_agent_extracts_mixed_entities(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent can extract multiple different valid entities from one chunk."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "Event at Acme Corp involving Dr. Doe in the Main Hall."
    chunk_id = "mixed_chunk_001"
    source_file = "mixed_doc.txt"

    mock_llm.set_response(chunk_text, {
        "extractions": [
            {"type": "event", "data": {"timestamp": "today", "description": "Keynote"}},
            {"type": "person", "data": {"name": "Dr. Doe", "role": "Speaker"}},
            {"type": "location", "data": {"name": "Main Hall", "type": "Venue"}},
            {"type": "organization", "data": {"name": "Acme Corp"}}
        ]
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 4
    assert any(isinstance(item, EventV1) and item.description == "Keynote" for item in extractions)
    assert any(isinstance(item, PersonV1) and item.name == "Dr. Doe" for item in extractions)
    assert any(isinstance(item, LocationV1) and item.name == "Main Hall" for item in extractions)
    assert any(isinstance(item, OrganizationV1) and item.name == "Acme Corp" for item in extractions)
    check_metacog_log(metacog_logger, ["Keynote", "Dr. Doe", "Main Hall", "Acme Corp", "errors\": []"])


def test_extractor_agent_handles_validation_error(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent's handling of data that causes a Pydantic validation error for Event."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)

    chunk_text = "This data will be invalid for an event."
    chunk_id = "test_chunk_002"
    source_file = "test_doc_invalid.txt"

    # Configure mock LLM to return data missing a required field for EventV1 (description)
    mock_llm.set_response(chunk_text, {
        "extractions": [{"type": "event", "data": {"timestamp": "Tomorrow"}}]
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 0 # No valid extractions
    check_metacog_log(metacog_logger, ["ValidationError", "description", "Field required"])


def test_extractor_agent_handles_person_validation_error(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent's handling of data that causes a Pydantic validation error for PersonV1."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "This data will be invalid for a person."
    chunk_id = "invalid_person_chunk"
    source_file = "invalid_person_doc.txt"

    mock_llm.set_response(chunk_text, {
        "extractions": [{"type": "person", "data": {"title": "The Nameless"}}] # Missing 'name'
    })
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 0
    check_metacog_log(metacog_logger, ["ValidationError", "name", "Field required", "person"])

# It might be good to add similar validation error tests for LocationV1 and OrganizationV1 if they
# have more complex validation rules beyond 'name' being required, or for completeness.

def test_extractor_agent_handles_malformed_json(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent's handling of malformed JSON from LLM."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)
    chunk_text = "This will be malformed JSON."
    chunk_id = "test_chunk_003"
    source_file = "test_doc_malformed.txt"

    mock_llm.set_response(chunk_text, '{"extractions": [{"type": "event", "data": {"timestamp": "today", "description": "System backup completed."}}') # Intentionally malformed
    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)
    assert len(extractions) == 0
    check_metacog_log(metacog_logger, ["JSONDecodeError", "LLM output was not valid JSON"])
