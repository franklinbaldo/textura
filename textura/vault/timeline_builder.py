import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dateutil import parser as date_parser

from textura.extraction.models import EventV1
from textura.vault.vault_writer import generate_metadata_comment, slugify


class TimelineBuilder:
    """
    Builds yearly and monthly timeline Markdown notes from EventV1 objects.
    """
    def __init__(self, workspace_path: str, events: List[EventV1]):
        self.workspace_path = Path(workspace_path)
        self.events = events
        self.timelines_path = self.workspace_path / "vault" / "timelines"
        self.notes_events_path = self.workspace_path / "vault" / "notes" / "events" # For relative links

        self.timelines_path.mkdir(parents=True, exist_ok=True)

    def _parse_timestamp(self, timestamp_str: str) -> Optional[Tuple[int, int]]:
        """
        Attempts to parse a timestamp string to extract year and month.
        Focuses on "YYYY-MM-DD", "YYYY-MM", "YYYY".
        Uses dateutil.parser for more flexible parsing but extracts only Y/M.
        Returns (year, month) or None if parsing fails.
        """
        try:
            # Try to parse the full timestamp
            dt = date_parser.parse(timestamp_str)
            return dt.year, dt.month
        except (ValueError, OverflowError, TypeError):
            # If flexible parsing fails, try specific regex for YYYY or YYYY-MM patterns
            # This is a simplified approach for this pass.
            match_yyyy_mm = re.match(r'(\d{4})-(\d{1,2})', timestamp_str)
            if match_yyyy_mm:
                return int(match_yyyy_mm.group(1)), int(match_yyyy_mm.group(2))
            match_yyyy = re.match(r'(\d{4})', timestamp_str)
            if match_yyyy:
                return int(match_yyyy.group(1)), 1 # Default to January if only year found
            return None


    def build_timelines(self):
        """
        Generates yearly and monthly timeline Markdown files.
        """
        if not self.events:
            print("No events provided to TimelineBuilder. Skipping timeline generation.")
            return

        # Group events by year, then by month
        # Dict[year, Dict[month, List[EventV1]]]
        grouped_events: Dict[int, Dict[int, List[EventV1]]] = defaultdict(lambda: defaultdict(list))

        for event in self.events:
            parsed_date_tuple = self._parse_timestamp(event.timestamp)
            if parsed_date_tuple:
                year, month = parsed_date_tuple
                grouped_events[year][month].append(event)
            else:
                print(f"Could not parse timestamp for event: '{event.description[:50]}...' (Timestamp: {event.timestamp}). Skipping for timeline.")

        if not grouped_events:
            print("No events could be reliably dated. Skipping timeline generation.")
            return

        # Create yearly notes
        for year, monthly_events_dict in sorted(grouped_events.items()):
            year_file_id = f"timeline_{year}"
            year_metadata = {"id": year_file_id, "type": "TimelineYearly", "year": year}
            year_metadata_comment = generate_metadata_comment(year_metadata)

            year_content = f"# Timeline: {year}\n\n"
            year_content += "## Months\n"
            for month in sorted(monthly_events_dict.keys()):
                year_content += f"- [[{year}-{month:02d}]]\n"
            year_content += f"\n{year_metadata_comment}\n"

            year_file_path = self.timelines_path / f"{year}.md"
            with open(year_file_path, 'w', encoding='utf-8') as f:
                f.write(year_content)
            # print(f"Yearly timeline written: {year_file_path}")

            # Create monthly notes for this year
            for month, events_in_month in sorted(monthly_events_dict.items()):
                month_file_id = f"timeline_{year}-{month:02d}"
                month_metadata = {"id": month_file_id, "type": "TimelineMonthly", "year": year, "month": month}
                month_metadata_comment = generate_metadata_comment(month_metadata)

                month_content = f"# Timeline: {year}-{month:02d}\n\n"
                month_content += "## Events\n"

                # Sort events in month (e.g., by original timestamp string, or later by more precise date)
                # For now, simple sort by description for consistency if timestamps are identical or non-standard
                events_in_month.sort(key=lambda e: (e.timestamp, e.description))

                for event in events_in_month:
                    # Construct relative path to the event note
                    # Event note ID needs to be known or reconstructed.
                    # Assuming event_id used for filename in VaultWriter is recoverable or stored.
                    # For now, we rely on a slug of description + unique part.
                    # This linking is fragile if VaultWriter's ID generation changes.
                    # A more robust system would store the generated event_file_id on EventV1 or pass it.
                    # VaultWriter currently generates event_id like: f"event_{slugify(event.description)}_{uuid_part}"
                    # We need to find that file or have a consistent ID.
                    # Let's assume for now we can reconstruct a reference or use description.

                    # To make links work, we need event_id as used in VaultWriter.
                    # This requires either storing the generated ID on the event object
                    # or having VaultWriter return it and storing it with the event before passing to TimelineBuilder.
                    # For this pass, we'll use a placeholder link structure if the exact ID isn't on the EventV1 model.
                    # Let's assume event.id is populated by VaultWriter or a prior step.
                    # If not, we make a "best guess" link.
                    if not event.file_id:
                        print(f"Warning: Event '{event.description[:30]}...' is missing file_id for timeline linking.")
                        # Fallback or skip, for now, create a potentially broken link
                        event_file_id_for_link = f"event_{slugify(event.description)}_MISSING_ID"
                    else:
                        event_file_id_for_link = event.file_id

                    # Obsidian style link: [[../../notes/events/event_slug_id|Description]]
                    # The path needs to be relative from `timelines/YYYY-MM.md` to `notes/events/event_id.md`
                    relative_event_path = f"../../notes/events/{event_file_id_for_link}"

                    # Use a display name for the link, e.g., the event's description or timestamp
                    link_display_name = event.description[:80] # Truncate for display
                    if event.timestamp:
                        link_display_name = f"{event.timestamp}: {link_display_name}"

                    month_content += f"- [[{relative_event_path}|{link_display_name}]]\n"

                month_content += f"\n{month_metadata_comment}\n"

                month_file_path = self.timelines_path / f"{year}-{month:02d}.md"
                with open(month_file_path, 'w', encoding='utf-8') as f:
                    f.write(month_content)
                # print(f"Monthly timeline written: {month_file_path}")

        print(f"Timeline generation complete. Files are in {self.timelines_path.resolve()}")


# Need to import re for the _parse_timestamp method if using regex there.
import re

if __name__ == '__main__':
    # Example Usage
    dummy_workspace_tl = Path("_test_workspace_timeline")
    dummy_workspace_tl.mkdir(parents=True, exist_ok=True)

    # Create dummy event files (as if VaultWriter created them)
    # For robust linking, the event objects should have their file IDs assigned.
    event_notes_path = dummy_workspace_tl / "vault" / "notes" / "events"
    event_notes_path.mkdir(parents=True, exist_ok=True)

    def create_dummy_event_file(event_id_val, content=""):
        with open(event_notes_path / f"{event_id_val}.md", "w") as f:
            f.write(f"# Dummy Event: {event_id_val}\n{content}")

    events_data = [
        EventV1(timestamp="2023-04-15 Meeting", description="Strategy meeting for Q2", source_file="doc1.txt", chunk_id="c1", file_id="event_strategy-meeting-for-q2_abc1"),
        EventV1(timestamp="2023-04-20 Release", description="Alpha version released", source_file="doc2.txt", chunk_id="c2", file_id="event_alpha-version-released_def2"),
        EventV1(timestamp="2023-05-10 Conference", description="Attended industry conference", source_file="doc3.txt", chunk_id="c3", file_id="event_attended-industry-confer_ghi3"),
        EventV1(timestamp="2024-01-05 Planning", description="Yearly planning session", source_file="doc4.txt", chunk_id="c4", file_id="event_yearly-planning-session_jkl4"),
        EventV1(timestamp="Late 2023", description="Initial project discussions", source_file="doc5.txt", chunk_id="c5", file_id="event_initial-project-discussi_mno5"), # Parsable by dateutil
        EventV1(timestamp="Sometime last year", description="Concept phase", source_file="doc6.txt", chunk_id="c6", file_id="event_concept-phase_pqr6"), # Might not parse well
        EventV1(timestamp="2023", description="Project conception year", source_file="doc7.txt", chunk_id="c7", file_id="event_project-conception-year_stu7")
    ]
    # Create dummy files for these events
    for e in events_data:
        if hasattr(e, 'file_id'):
            create_dummy_event_file(e.file_id, e.description)


    builder = TimelineBuilder(workspace_path=str(dummy_workspace_tl), events=events_data)
    builder.build_timelines()

    print("\nGenerated timeline files:")
    for f in (dummy_workspace_tl / "vault" / "timelines").glob("*.md"):
        print(f"--- {f.name} ---")
        print(f.read_text())
        print("---------------------\n")

    # Clean up
    # import shutil
    # shutil.rmtree(dummy_workspace_tl)
