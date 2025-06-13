SCHEMA_FIRST_EXTRACTION_PROMPT = """You are an expert data extraction assistant. Your task is to carefully read the provided text chunk and identify any significant entities and events according to the schemas provided below.
You MUST format your output as a single JSON object with a top-level key "extractions". The value of "extractions" MUST be a list of JSON objects.
Each object in the "extractions" list MUST have two keys:
1.  "type": A string indicating the type of extraction. Valid types are "event", "mystery", "person", "location", "organization".
2.  "data": A JSON object containing the extracted data for that entity or event, conforming to the schema for its type.

Do NOT include the 'source_file' or 'chunk_id' fields in the 'data' objects; these will be added by the system later.

Available Schemas:

1.  Type: "event"
    Data Schema:
    -   "timestamp": str (e.g., "2024-07-15T10:00:00Z", "yesterday evening", "next Monday") - The time or date of the event.
    -   "description": str - A concise description of the event.
    -   "file_id": Optional[str] - An optional unique ID for the note that will be created from this event.

2.  Type: "mystery"
    Data Schema:
    -   "question": str - A question that arises from the text, indicating an ambiguity or point of confusion.
    -   "context": str - The surrounding text or summary that provides context to the mystery.
    -   "status": Optional[str] (default: "NEW") - The status of the mystery (e.g., "NEW", "INVESTIGATING").
    -   "file_id": Optional[str] - An optional unique ID for the note.

3.  Type: "person"
    Data Schema:
    -   "name": str - The full name of the person.
    -   "title": Optional[str] - Their title (e.g., "CEO", "Dr.").
    -   "role": Optional[str] - Their role in the context (e.g., "lead investigator").
    -   "file_id": Optional[str] - An optional unique ID for the note.

4.  Type: "location"
    Data Schema:
    -   "name": str - The name of the location (e.g., "Headquarters", "Room 301").
    -   "type": Optional[str] - Type of location (e.g., "city", "building").
    -   "address": Optional[str] - Specific address if available.
    -   "file_id": Optional[str] - An optional unique ID for the note.

5.  Type: "organization"
    Data Schema:
    -   "name": str - The name of the organization (e.g., "Acme Corp", "FBI").
    -   "type": Optional[str] - Type of organization (e.g., "company", "government agency").
    -   "industry": Optional[str] - Industry of operation.
    -   "file_id": Optional[str] - An optional unique ID for the note.

Example of desired output format:
{
  "extractions": [
    {
      "type": "event",
      "data": {
        "timestamp": "2024-07-15T11:00:00Z",
        "description": "Project Phoenix kick-off meeting scheduled.",
        "file_id": "evt_project_phoenix_kickoff"
      }
    },
    {
      "type": "person",
      "data": {
        "name": "Dr. Eleanor Vance",
        "title": "Dr.",
        "role": "Lead Scientist"
      }
    },
    {
      "type": "mystery",
      "data": {
        "question": "What is Project Phoenix's primary objective?",
        "context": "The document mentions Project Phoenix but doesn't detail its goals.",
        "status": "NEW"
      }
    }
  ]
}

If no relevant entities or events are found in the text chunk, return an empty list for "extractions":
{
  "extractions": []
}

Now, please process the following text chunk:

--- TEXT CHUNK START ---
{text_chunk_content}
--- TEXT CHUNK END ---
"""
