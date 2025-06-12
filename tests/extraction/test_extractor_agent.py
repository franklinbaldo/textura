import pytest
from pathlib import Path

from textura.extraction.extractor_agent import ExtractorAgent
from textura.extraction.models import EventV1, MysteryV1
from textura.logging.metacog import Metacog
from tests.mocks.mock_llm_service import MockLLMService # Adjusted import path

@pytest.fixture
def metacog_logger(test_workspace: Path) -> Metacog:
    """Fixture to provide a Metacog logger instance for tests."""
    return Metacog(workspace_path=str(test_workspace))

@pytest.fixture
def mock_llm() -> MockLLMService:
    """Fixture to provide a MockLLMService instance."""
    return MockLLMService()

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
    assert "errors\": []" in log_content # No errors expected

def test_extractor_agent_handles_validation_error(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent's handling of data that causes a Pydantic validation error."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)

    chunk_text = "This data will be invalid."
    chunk_id = "test_chunk_002"
    source_file = "test_doc_invalid.txt"

    # Configure mock LLM to return data missing a required field for EventV1 (description)
    mock_llm.set_response(chunk_text, {
        "extractions": [
            {
                "type": "event",
                "data": {
                    "timestamp": "Tomorrow"
                    # Description is missing
                }
            }
        ]
    })

    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)

    assert len(extractions) == 0 # No valid extractions

    # Check Metacog for error logging
    log_file = metacog_logger.log_file_path
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert "ValidationError" in log_content
    assert "description" in log_content # Error message should mention the missing field
    assert "Field required" in log_content

def test_extractor_agent_handles_malformed_json(metacog_logger: Metacog, mock_llm: MockLLMService):
    """Test ExtractorAgent's handling of malformed JSON from LLM."""
    agent = ExtractorAgent(metacog_logger=metacog_logger, llm_client=mock_llm)

    chunk_text = "This will be malformed JSON."
    chunk_id = "test_chunk_003"
    source_file = "test_doc_malformed.txt"

    mock_llm.set_response(chunk_text, '{"extractions": [{"type": "event", "data": {"timestamp": "today", "description": "System backup completed."}}') # Intentionally malformed

    extractions = agent.extract_from_chunk(chunk_text, chunk_id, source_file)

    assert len(extractions) == 0

    log_file = metacog_logger.log_file_path
    assert log_file.exists()
    with open(log_file, 'r') as f:
        log_content = f.read()
    assert "JSONDecodeError" in log_content
    assert "LLM output was not valid JSON" in log_content
