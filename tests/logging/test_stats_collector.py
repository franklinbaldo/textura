import json
from pathlib import Path

from textura.logging.stats_collector import StatsCollector


def create_metacog_log(workspace: Path, entries: list[dict]) -> Path:
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "metacog.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return log_file


def test_stats_collector_counts_extractions_and_errors(tmp_path: Path):
    entries = [
        {
            "validated_extractions": [
                {"type": "event", "description": "d", "timestamp": "t"},
            ],
            "errors": [],
        },
        {
            "validated_extractions": [],
            "errors": [{"msg": "oops"}],
        },
    ]
    create_metacog_log(tmp_path, entries)
    collector = StatsCollector(workspace_path=str(tmp_path))
    stats = collector.collect()
    assert stats["log_entries"] == 2
    assert stats["extractions"] == 1
    assert stats["errors"] == 1


def test_stats_collector_summary_format(tmp_path: Path):
    entries = [
        {"validated_extractions": [], "errors": []},
    ]
    create_metacog_log(tmp_path, entries)
    collector = StatsCollector(workspace_path=str(tmp_path))
    summary = collector.format_summary(collector.collect())
    assert "Textura Run Stats" in summary
    assert "Log Entries: 1" in summary
