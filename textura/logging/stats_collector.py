import json
from pathlib import Path
from typing import Dict


class StatsCollector:
    """Collects basic run statistics from the metacog log."""

    def __init__(self, workspace_path: str, log_filename: str = "metacog.jsonl"):
        self.log_file = Path(workspace_path) / "logs" / log_filename

    def collect(self) -> Dict[str, int]:
        stats = {"log_entries": 0, "extractions": 0, "errors": 0}
        if not self.log_file.exists():
            return stats
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["log_entries"] += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    stats["errors"] += 1
                    continue
                stats["extractions"] += len(entry.get("validated_extractions", []))
                stats["errors"] += len(entry.get("errors", []))
        return stats

    def format_summary(self, stats: Dict[str, int]) -> str:
        lines = [
            "--- Textura Run Stats ---",
            f"Log Entries: {stats['log_entries']}",
            f"Validated Extractions: {stats['extractions']}",
            f"Errors: {stats['errors']}",
        ]
        return "\n".join(lines)
