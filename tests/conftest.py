from pathlib import Path

import pytest


@pytest.fixture(scope="function")  # Create a new dir for each test function
def output_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for saving test results."""
    test_output_path = tmp_path / "test_results"
    test_output_path.mkdir()
    print(f"\nCreated temp output dir: {test_output_path}")
    return test_output_path


# You can add other fixtures here if needed later
