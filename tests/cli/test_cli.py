import pytest
from click.testing import CliRunner
from pathlib import Path
import json

from textura.cli import textura_cli
from tests.mocks.mock_llm_service import MockLLMService # For patching ExtractorAgent's LLM
from textura.extraction import extractor_agent # To allow patching its DEFAULT mock_llm_client

# Store the original mock_llm_client from extractor_agent
_original_mock_llm_client = extractor_agent.mock_llm_client

@pytest.fixture(scope="module")
def runner():
    return CliRunner()

def test_cli_init_command(runner: CliRunner, test_workspace: Path):
    """Test the 'textura init' command."""
    result = runner.invoke(textura_cli, ['init', '--workspace', str(test_workspace)])
    assert result.exit_code == 0
    assert f"Initializing Textura workspace at {test_workspace.resolve()}" in result.output
    assert (test_workspace / "manifest.json").exists()
    assert (test_workspace / "vault" / "notes").is_dir()
    assert (test_workspace / "data" / "processed").is_dir()
    assert (test_workspace / "logs").is_dir()
    assert (test_workspace / "index").is_dir()

def test_cli_e2e_ingest_extract_weave(runner: CliRunner, test_workspace: Path, test_source_dir: Path):
    """
    Basic end-to-end test for ingest, extract, and weave commands.
    Uses a small sample file and the MockLLMService for extraction.
    """
    # --- Setup Mock LLM for ExtractorAgent ---
    # This is a bit tricky as ExtractorAgent is instantiated within the CLI command.
    # We need to monkeypatch the default mock_llm_client used by ExtractorAgent
    # if no client is passed (which is the case when called from CLI).

    mock_llm_for_e2e = MockLLMService()
    # Configure a simple, reliable response for any text it might receive
    mock_llm_for_e2e.add_default_response({
        "extractions": [{
            "type": "event",
            "data": {"timestamp": "2024-07-15", "description": "E2E test event"}
        }]
    })
    # Monkeypatch the default mock client in the extractor_agent module
    extractor_agent.mock_llm_client = mock_llm_for_e2e

    # 1. Init
    result_init = runner.invoke(textura_cli, ['init', '--workspace', str(test_workspace)])
    assert result_init.exit_code == 0

    # 2. Ingest
    sample_txt_file = test_source_dir / "sample_e2e.txt"
    sample_txt_file.write_text("This is a simple text for E2E test.")

    result_ingest = runner.invoke(textura_cli, [
        'ingest', '--workspace', str(test_workspace), '--source', str(sample_txt_file)
    ])
    assert result_ingest.exit_code == 0
    assert "Successfully processed and ingested 1 file(s)" in result_ingest.output
    assert (test_workspace / "index" / "docs.jsonl").exists()

    # Verify docs.jsonl content (should have one doc)
    with open(test_workspace / "index" / "docs.jsonl", 'r') as f:
        docs = [json.loads(line) for line in f if line.strip()]
    assert len(docs) == 1
    assert docs[0]['text'] == "This is a simple text for E2E test."

    # 3. Extract
    result_extract = runner.invoke(textura_cli, ['extract', '--workspace', str(test_workspace)])
    assert result_extract.exit_code == 0
    assert "Successfully extracted 1 items" in result_extract.output # Based on mock_llm_for_e2e
    extractions_file = test_workspace / "data" / "processed" / "extractions.jsonl"
    assert extractions_file.exists()
    with open(extractions_file, 'r') as f:
        ext_data = [json.loads(line) for line in f if line.strip()]
    assert len(ext_data) == 1
    assert ext_data[0]['description'] == "E2E test event"

    # 4. Weave
    result_weave = runner.invoke(textura_cli, ['weave', '--workspace', str(test_workspace), '--stats'])
    assert result_weave.exit_code == 0
    assert "Wrote 1 items to the vault" in result_weave.output # 1 event
    assert "Building timelines from 1 events" in result_weave.output
    assert "Textura Run Stats" in result_weave.output

    # Verify vault and timeline file creation (basic checks)
    event_notes_path = test_workspace / "vault" / "notes" / "events"
    timeline_path = test_workspace / "vault" / "timelines"

    event_files = list(event_notes_path.glob("*.md"))
    assert len(event_files) == 1
    assert "e2e-test-event" in event_files[0].name # From slugify

    timeline_year_file = timeline_path / "2024.md"
    timeline_month_file = timeline_path / "2024-07.md"
    assert timeline_year_file.exists()
    assert timeline_month_file.exists()

    # --- Teardown Mock LLM ---
    # Restore the original mock_llm_client to avoid interference with other tests
    # (though with pytest's test isolation, this might be overly cautious for module-level mocks)
    extractor_agent.mock_llm_client = _original_mock_llm_client

    print("E2E test completed successfully.") # For visibility in pytest output if needed

# Add more CLI tests, e.g., for error conditions, different options, etc.
