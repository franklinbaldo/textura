import json
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Union, Tuple # Import Tuple if using capitalized version

from textura.extraction.models import EventV1, MysteryV1

def generate_metadata_comment(metadata: Dict[str, Any]) -> str:
    """Generates a metadata comment string."""
    # Sort keys for consistent output, helpful for testing or diffing
    metadata_json_string = json.dumps(metadata, sort_keys=True)
    return f"<!-- TEXTURA:_v1 {metadata_json_string} -->"

def slugify(text: str) -> str:
    """
    Convert a string into a filename-safe slug.
    Replaces non-alphanumeric characters with underscores.
    """
    text = re.sub(r'[^\w\s-]', '', text.lower()) # Remove special chars except whitespace and hyphens
    text = re.sub(r'[-\s]+', '-', text).strip('-_') # Replace whitespace/hyphens with single hyphen
    return text[:75] # Limit length for safety

class VaultWriter:
    """
    Writes extracted Pydantic models (Events, Mysteries) to Markdown files
    in a structured vault.
    """
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.vault_path = self.workspace_path / "vault"
        self.notes_path = self.vault_path / "notes"
        self.events_path = self.notes_path / "events"
        self.mysteries_path = self.notes_path / "mysteries"

        # Ensure directories exist
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.mysteries_path.mkdir(parents=True, exist_ok=True)

    def _generate_unique_id(self) -> str:
        """Generates a unique ID for an item."""
        return str(uuid.uuid4())

    def write_event(self, event: EventV1) -> tuple[Path, str]: # Changed to lowercase 'tuple'
        """
        Writes an EventV1 object to a Markdown file.

        Args:
            event: The EventV1 object to write.

        Returns:
            A tuple of (Path to the created Markdown file, generated_event_id).
        """
        event_id = f"event_{slugify(event.description)}_{self._generate_unique_id()[:8]}"

        metadata = {
            "id": event_id, # Use the generated filename ID as the unique ID
            "type": "EventV1",
            "source_file": event.source_file,
            "chunk_id": event.chunk_id,
            "timestamp": event.timestamp # Store original timestamp for reference
        }
        metadata_comment = generate_metadata_comment(metadata)

        content = f"# Event: {event.description}\n\n"
        content += f"**Timestamp:** {event.timestamp}\n\n"
        content += f"**Source:** [[{event.source_file}]] (Chunk: {event.chunk_id})\n\n"
        content += metadata_comment

        file_path = self.events_path / f"{event_id}.md"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # print(f"Event written to: {file_path}")
        return file_path, event_id

    def write_mystery(self, mystery: MysteryV1) -> tuple[Path, str]: # Changed to lowercase 'tuple'
        """
        Writes a MysteryV1 object to a Markdown file.

        Args:
            mystery: The MysteryV1 object to write.

        Returns:
            A tuple of (Path to the created Markdown file, generated_mystery_id).
        """
        mystery_id = f"mystery_{slugify(mystery.question)}_{self._generate_unique_id()[:8]}"

        metadata = {
            "id": mystery_id,
            "type": "MysteryV1",
            "source_file": mystery.source_file,
            "chunk_id": mystery.chunk_id,
            "status": mystery.status
        }
        metadata_comment = generate_metadata_comment(metadata)

        content = f"# Mystery: {mystery.question}\n\n"
        content += f"**Status:** {mystery.status}\n\n"
        content += f"**Context:**\n{mystery.context}\n\n"
        content += f"**Source:** [[{mystery.source_file}]] (Chunk: {mystery.chunk_id})\n\n"
        content += metadata_comment

        file_path = self.mysteries_path / f"{mystery_id}.md"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # print(f"Mystery written to: {file_path}")
        return file_path, mystery_id

if __name__ == '__main__':
    # Example Usage
    dummy_workspace = Path("_test_workspace_vault_writer")
    dummy_workspace.mkdir(exist_ok=True)

    vault_writer = VaultWriter(workspace_path=str(dummy_workspace))

    sample_event = EventV1(
        timestamp="2024-06-12 An important meeting was held.",
        description="Project Phoenix kickoff meeting to discuss initial strategy and resource allocation.",
        source_file="meeting_notes/2024-06-12_phoenix_kickoff.txt",
        chunk_id="chunk_001__proj_phoenix"
    )
    event_file = vault_writer.write_event(sample_event)
    print(f"Sample event written to: {event_file}")
    with open(event_file, 'r') as f: print(f.read())


    sample_mystery = MysteryV1(
        question="What is the 'Oracle Chamber' mentioned in the ancient texts?",
        context="The texts refer to the Oracle Chamber as a source of great power but do not specify its location or nature.",
        source_file="ancient_scrolls/volume_3.pdf",
        chunk_id="chunk_105_scroll_vol3",
        status="RESEARCHING"
    )
    mystery_file = vault_writer.write_mystery(sample_mystery)
    print(f"\nSample mystery written to: {mystery_file}")
    with open(mystery_file, 'r') as f: print(f.read())

    # Test slugify
    print(f"\nSlugify test: 'This is a Test! 123.' -> '{slugify('This is a Test! 123.')}'")
    print(f"Slugify test: 'long description that exceeds seventy five characters limit to check truncation' -> '{slugify('long description that exceeds seventy five characters limit to check truncation')}'")


    # Clean up
    # import shutil
    # shutil.rmtree(dummy_workspace)
