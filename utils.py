"""Shared utilities for the pickle stability test suite."""

import hashlib
import pickle
import platform
import sys


ALL_PROTOCOLS = list(range(pickle.HIGHEST_PROTOCOL + 1))


def pickle_hash(obj, protocol):
    """Return SHA-256 hex digest of pickle.dumps(obj, protocol)."""
    data = pickle.dumps(obj, protocol=protocol)
    return hashlib.sha256(data).hexdigest()


def pickle_hash_default(obj):
    """Return SHA-256 hex digest using the default protocol."""
    data = pickle.dumps(obj)
    return hashlib.sha256(data).hexdigest()


def assert_stable(obj, protocol, runs=10):
    """Assert that pickle output is hash-identical across multiple calls."""
    hashes = {pickle_hash(obj, protocol) for _ in range(runs)}
    assert len(hashes) == 1, (
        f"Non-deterministic pickle for protocol={protocol}: "
        f"got {len(hashes)} distinct hashes over {runs} runs."
    )
    return hashes.pop()


def roundtrip(obj, protocol):
    """Pickle then unpickle; return the reconstructed object."""
    return pickle.loads(pickle.dumps(obj, protocol=protocol))


def platform_info():
    """Return a dict of relevant platform metadata for reproducibility."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pickle_highest_protocol": pickle.HIGHEST_PROTOCOL,
        "byteorder": sys.byteorder,
    }
