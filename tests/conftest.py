"""This file is run by PyTest as first file.
Define testing "fixtures" here.
"""
import pytest, os
import typing as ty
import pyrootutils

# Using pyrootutils, we find the root directory of this project and make sure it is our working directory
root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

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
