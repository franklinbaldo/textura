from pydantic import BaseModel, Field
from typing import Optional, Union

class EventV1(BaseModel):
    """
    Represents a significant event extracted from text.
    """
    timestamp: str  # Can be ISO format, date, or natural language like "yesterday evening"
    description: str = Field(..., min_length=1)
    source_file: str
    chunk_id: str # ID of the text chunk from which this event was extracted
    file_id: Optional[str] = None # To store the vault filename (without extension)
    # model_version: str = Field(default="EventV1", frozen=True) # For future model versioning

class MysteryV1(BaseModel):
    """
    Represents a question or point of confusion encountered in the text.
    """
    question: str = Field(..., min_length=1) # The question or mystery.
    context: str # Surrounding text or summary that provides context to the mystery.
    source_file: str
    chunk_id: str # ID of the text chunk
    status: str = Field(default="NEW") # e.g., NEW, ADDRESSED, IGNORED
    file_id: Optional[str] = None # To store the vault filename (without extension)
    # model_version: str = Field(default="MysteryV1", frozen=True)

# For type hinting purposes in the ExtractorAgent
ExtractionItem = Union[EventV1, MysteryV1]

if __name__ == '__main__':
    # Example Usage & Validation Test
    valid_event_data = {
        "timestamp": "2024-06-12T10:00:00Z",
        "description": "A critical meeting was held to discuss project milestones.",
        "source_file": "project_notes.txt",
        "chunk_id": "chunk_001"
    }
    event = EventV1(**valid_event_data)
    print("Valid EventV1:", event.model_dump_json(indent=2))

    valid_mystery_data = {
        "question": "Why was the budget for Q3 suddenly reduced?",
        "context": "The financial report for Q2 showed strong performance, yet the Q3 budget was cut.",
        "source_file": "financial_reports.pdf",
        "chunk_id": "chunk_042",
        "status": "INVESTIGATING"
    }
    mystery = MysteryV1(**valid_mystery_data)
    print("\nValid MysteryV1:", mystery.model_dump_json(indent=2))

    # Example of data that might come from an LLM (could be invalid)
    llm_event_data_invalid = {
        "timestamp": "yesterday",
        # "description": "A minor issue was reported.", # Missing description
        "source_file": "daily_log.txt",
        "chunk_id": "chunk_002"
    }
    try:
        EventV1(**llm_event_data_invalid)
    except Exception as e:
        print(f"\nError creating EventV1 from invalid data: {e}")

    llm_mystery_data_invalid_type = {
        "question": "What is the core component?",
        "context": "The document mentions a 'central processing unit' repeatedly but doesn't define it.",
        "source_file": "tech_spec.doc",
        "chunk_id": 123, # chunk_id should be a string
        "status": "NEW"
    }
    try:
        MysteryV1(**llm_mystery_data_invalid_type)
    except Exception as e:
        print(f"\nError creating MysteryV1 from data with invalid type: {e}")
