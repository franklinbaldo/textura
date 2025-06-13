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


class PersonV1(BaseModel):
    """
    Represents a person entity extracted from text.
    """
    name: str = Field(..., min_length=1)
    title: Optional[str] = None
    role: Optional[str] = None
    source_file: str
    chunk_id: str
    file_id: Optional[str] = None
    # model_version: str = Field(default="PersonV1", frozen=True)

class LocationV1(BaseModel):
    """
    Represents a location entity extracted from text.
    """
    name: str = Field(..., min_length=1)
    type: Optional[str] = None # e.g., City, Country, Building, Landmark
    address: Optional[str] = None
    source_file: str
    chunk_id: str
    file_id: Optional[str] = None
    # model_version: str = Field(default="LocationV1", frozen=True)

class OrganizationV1(BaseModel):
    """
    Represents an organization entity extracted from text.
    """
    name: str = Field(..., min_length=1)
    type: Optional[str] = None # e.g., Company, NGO, Government Agency
    industry: Optional[str] = None
    source_file: str
    chunk_id: str
    file_id: Optional[str] = None
    # model_version: str = Field(default="OrganizationV1", frozen=True)

# For type hinting purposes in the ExtractorAgent
ExtractionItem = Union[EventV1, MysteryV1, PersonV1, LocationV1, OrganizationV1]

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

    valid_person_data = {
        "name": "Dr. Eleanor Vance",
        "title": "Lead Researcher",
        "role": "Principal Investigator",
        "source_file": "research_proposal.docx",
        "chunk_id": "chunk_007"
    }
    person = PersonV1(**valid_person_data)
    print("\nValid PersonV1:", person.model_dump_json(indent=2))

    valid_location_data = {
        "name": "Building 7G",
        "type": "Laboratory",
        "address": "123 Innovation Drive, Tech Park",
        "source_file": "facility_map.png",
        "chunk_id": "chunk_map_001"
    }
    location = LocationV1(**valid_location_data)
    print("\nValid LocationV1:", location.model_dump_json(indent=2))

    valid_organization_data = {
        "name": "FutureTech Corp.",
        "type": "Multinational Conglomerate",
        "industry": "Artificial Intelligence",
        "source_file": "annual_report_2023.pdf",
        "chunk_id": "chunk_intro_002"
    }
    organization = OrganizationV1(**valid_organization_data)
    print("\nValid OrganizationV1:", organization.model_dump_json(indent=2))

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
