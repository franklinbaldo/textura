import pytest
import tempfile
from pathlib import Path
import shutil

@pytest.fixture(scope="function") # "function" scope means it runs once per test function
def test_workspace():
    """Creates a temporary workspace directory for a test function."""
    temp_dir = tempfile.mkdtemp(prefix="textura_test_ws_")
    workspace_path = Path(temp_dir)

    # Yield the path to the test
    yield workspace_path

    # Teardown: remove the temporary directory after the test
    # Use shutil.rmtree to remove non-empty directories
    try:
        shutil.rmtree(workspace_path)
    except OSError as e:
        print(f"Error removing temporary workspace {workspace_path}: {e}")

@pytest.fixture(scope="function")
def test_source_dir():
    """Creates a temporary source directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="textura_test_source_")
    source_path = Path(temp_dir)
    yield source_path
    try:
        shutil.rmtree(source_path)
    except OSError as e:
        print(f"Error removing temporary source directory {source_path}: {e}")
