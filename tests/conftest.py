"""This file is run by PyTest as first file.
Define testing "fixtures" here.
"""
import pytest, os, sys
import typing as ty
from pathlib import Path
try:
    import pyrootutils  # type: ignore
except Exception:
    pyrootutils = None

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
if pyrootutils is not None:
    root = pyrootutils.setup_root(
        search_from=".",
        indicator=[".git", "pyproject.toml"],
        pythonpath=True,
        dotenv=True,
        cwd=True,
    )
else:
    # Fallback: locate project root by searching for pyproject.toml, ensure it and src/ are on sys.path
    here = Path(__file__).resolve()
    cand = here
    root = None
    for _ in range(5):
        if (cand / "pyproject.toml").exists() or (cand / ".git").exists():
            root = cand
            break
        cand = cand.parent
    if root is None:
        root = here.parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

# Example of a fixture, which are values we can pass to all tests
@pytest.fixture(scope="session")
def data_path() -> str:
    """Path where to find data. Reading this value from an environment variable if defined."""
    return os.environ.get("DATA_LOC", ".data")

# Example of a fixture, which are values we can pass to all tests
@pytest.fixture(scope="session")
def resources_path() -> str:
    """Path where to resources for the tests."""
    return os.environ.get("RESOURCES_LOC", "tests/res")
