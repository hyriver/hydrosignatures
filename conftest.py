"""Configuration for pytest."""

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    """Add hydrosignatures namespace for doctest."""
    import hydrosignatures as hs

    doctest_namespace["hs"] = hs
