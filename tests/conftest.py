"""Keep upstream network regression tests explicit without modifying them."""
import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify the preserved Hugging Face regression suite as slow."""
    for item in items:
        if item.path.name == "test_kronos_regression.py":
            item.add_marker(pytest.mark.slow)