import json
import os
from pathlib import Path


class SourceWatcher:
    """
    Monitors a source directory for new files and tracks ingested files
    using a manifest file.
    """

    def __init__(self, source_dir: str, workspace_path: str):
        self.source_dir = Path(source_dir)
        self.workspace_path = Path(workspace_path)
        self.manifest_path = self.workspace_path / "manifest.json"
        self.ingested_files: set[str] = self._load_manifest()

    def _load_manifest(self) -> set[str]:
        """Loads the set of ingested file paths from manifest.json."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                try:
                    data = json.load(f)
                    return set(data.get("ingested_files", []))
                except json.JSONDecodeError:
                    # Handle empty or invalid manifest
                    return set()
        else:
            # Create an empty manifest if it doesn't exist
            with open(self.manifest_path, "w") as f:
                json.dump({"ingested_files": []}, f)
            return set()

    def _save_manifest(self):
        """Saves the current set of ingested file paths to manifest.json."""
        with open(self.manifest_path, "w") as f:
            json.dump(
                {"ingested_files": sorted(list(self.ingested_files))}, f, indent=4
            )

    def get_new_files(self) -> list[Path]:
        """
        Scans the source directory and returns a list of absolute file paths
        that have not been ingested yet.
        """
        new_files: list[Path] = []
        if not self.source_dir.is_dir():
            print(
                f"Error: Source directory '{self.source_dir}' not found or is not a directory."
            )
            return new_files

        for filepath in self.source_dir.rglob("*"):  # Recursively find all files
            if filepath.is_file():
                # Store relative path from source_dir to ensure consistency
                relative_filepath_str = str(filepath.relative_to(self.source_dir))
                if relative_filepath_str not in self.ingested_files:
                    new_files.append(filepath.resolve())  # Return absolute path
        return new_files

    def mark_as_ingested(self, file_path: Path):
        """
        Marks a file as ingested by adding its relative path to the manifest.
        The path stored in the manifest is relative to the source_dir.
        """
        if self.source_dir not in file_path.parents:
            # Fallback for paths not directly under source_dir (e.g. single file ingestion)
            relative_filepath_str = file_path.name
        else:
            relative_filepath_str = str(file_path.relative_to(self.source_dir))

        self.ingested_files.add(relative_filepath_str)
        self._save_manifest()

    def get_ingested_files_count(self) -> int:
        """Returns the number of files marked as ingested."""
        return len(self.ingested_files)


if __name__ == "__main__":
    # Example Usage
    # Create dummy workspace and source directories for testing
    test_workspace = Path("_test_workspace_sw")
    test_source = Path("_test_source_sw")

    os.makedirs(test_workspace, exist_ok=True)
    os.makedirs(test_source / "subdir", exist_ok=True)

    # Create some dummy files
    (test_source / "file1.txt").write_text("Content of file1")
    (test_source / "file2.pdf").write_text("Content of file2")
    (test_source / "subdir" / "file3.txt").write_text("Content of file3")

    watcher = SourceWatcher(
        source_dir=str(test_source), workspace_path=str(test_workspace)
    )

    print("Initial check for new files:")
    new_files_list = watcher.get_new_files()
    for f in new_files_list:
        print(f"Found new file: {f}")

    if new_files_list:
        watcher.mark_as_ingested(new_files_list[0])
        print(f"\nMarked '{new_files_list[0]}' as ingested.")
        watcher.mark_as_ingested(new_files_list[1])  # Mark another for testing
        print(f"Marked '{new_files_list[1]}' as ingested.")

    print("\nSecond check for new files:")
    new_files_list_after = watcher.get_new_files()
    for f in new_files_list_after:
        print(f"Found new file: {f}")
    if not new_files_list_after:
        print("No new files found, as expected.")

    print(f"\nTotal ingested files: {watcher.get_ingested_files_count()}")
    print(f"Manifest content at: {watcher.manifest_path.resolve()}")

    # Clean up dummy directories
    # import shutil
    # shutil.rmtree(test_workspace)
    # shutil.rmtree(test_source)
