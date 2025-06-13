import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Assuming ExtractionItem is defined in textura.extraction.models
# If not, you might need a more generic way to serialize or define it here.
# For now, we'll assume it has a .model_dump() method if it's a Pydantic model.
from textura.extraction.models import ExtractionItem


class Metacog:
    """
    A simple logger for metacognitive information related to LLM extractions.
    Logs raw LLM outputs, validated extractions, and errors to a JSONL file.
    """

    def __init__(self, workspace_path: str, log_filename: str = "metacog.jsonl"):
        self.log_dir = Path(workspace_path) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / log_filename

    def log_extraction(
        self,
        chunk_id: str,
        source_file: str,  # Added source_file for better context
        raw_llm_output: str,
        validated_extractions: list[ExtractionItem],  # List of Pydantic models
        errors: list[
            dict[str, Any]
        ],  # List of dictionaries representing validation errors
    ):
        """
        Logs the details of an extraction attempt.

        Args:
            chunk_id: The ID of the chunk being processed.
            source_file: The source file of the chunk.
            raw_llm_output: The raw string output from the LLM.
            validated_extractions: A list of Pydantic models that passed validation.
            errors: A list of dictionaries, each detailing a validation error.

        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_id": chunk_id,
            "source_file": source_file,
            "raw_llm_output": raw_llm_output,
            # Serialize Pydantic models to dictionaries for JSON logging
            "validated_extractions": [
                item.model_dump() for item in validated_extractions
            ],
            "errors": errors,
        }

        try:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error writing to metacog log: {e}")


if __name__ == "__main__":
    # Example Usage
    dummy_workspace = Path("_test_workspace_metacog")
    dummy_workspace.mkdir(exist_ok=True)

    metacog_logger = Metacog(workspace_path=str(dummy_workspace))

    # Dummy data (assuming EventV1 and MysteryV1 are Pydantic models from models.py)
    from textura.extraction.models import EventV1

    valid_event = EventV1(
        timestamp="2023-01-01T12:00:00Z",
        description="Test event",
        source_file="test.txt",
        chunk_id="chunk_test_001",
    )

    validation_error_example = {
        "error_type": "ValidationError",
        "field": "description",
        "message": "Field required",
        "problematic_data": {
            "timestamp": "now",
            "source_file": "test.txt",
            "chunk_id": "chunk_test_002",
        },
    }

    metacog_logger.log_extraction(
        chunk_id="chunk_test_001",
        source_file="test.txt",
        raw_llm_output='{"event": {"timestamp": "2023-01-01T12:00:00Z", "description": "Test event"}}',
        validated_extractions=[valid_event],
        errors=[],
    )

    metacog_logger.log_extraction(
        chunk_id="chunk_test_002",
        source_file="test.txt",
        raw_llm_output='{"event": {"timestamp": "now"}}',  # Missing description
        validated_extractions=[],
        errors=[validation_error_example],
    )

    print(
        f"Metacog logs written to: {(dummy_workspace / 'logs' / 'metacog.jsonl').resolve()}"
    )
    print("Contents:")
    with open(dummy_workspace / "logs" / "metacog.jsonl") as f:
        for line in f:
            print(line.strip())

    # Clean up
    # import shutil
    # shutil.rmtree(dummy_workspace)
