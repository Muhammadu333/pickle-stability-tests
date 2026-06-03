"""
pytest configuration and shared fixtures.

Fixtures are protocol-parametrised so every test that accepts `protocol`
automatically runs against all supported pickle protocols (0-5 on CPython 3.8+).
"""

import pickle
import platform
import sys

import pytest

from utils import ALL_PROTOCOLS, platform_info


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "stability: tests that check hash-identical determinism"
    )
    config.addinivalue_line(
        "markers", "correctness: tests that verify roundtrip fidelity"
    )
    config.addinivalue_line(
        "markers", "boundary: boundary-value analysis tests"
    )
    config.addinivalue_line(
        "markers", "fuzzing: property-based / randomised tests"
    )
    config.addinivalue_line(
        "markers", "whitebox: white-box / structural coverage tests"
    )


@pytest.fixture(params=ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
def protocol(request):
    """Parametrise over every supported pickle protocol."""
    return request.param


@pytest.fixture(scope="session", autouse=True)
def print_platform(request):
    """Print platform metadata once per session for reproducibility reports."""
    info = platform_info()
    print("\n--- Platform metadata ---")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("-------------------------\n")


@pytest.fixture
def fixed_seed():
    """Provide a fixed random seed so fuzzing tests are reproducible."""
    import random
    rng = random.Random(42)
    return rng
