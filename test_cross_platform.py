"""
Cross-platform and cross-Python-version stability tests.

The central claim of the lab is that the same input must produce the same
pickle bytes (hash-identical) under all circumstances, including:
  - Different operating systems
  - Different Python versions (within the same protocol version)

Since we cannot run multiple OS/Python versions in a single test process, this
module does two things:

  1. Generates reference hashes for the current platform and writes them to a
     JSON fixture file.  These can be committed and compared against results
     from other platforms.

  2. Loads any existing reference file and asserts that this platform's hashes
     match — flagging cross-platform divergences as failures.

The fixture file path is: fixtures/reference_hashes.json

To add another platform: run `pytest test_cross_platform.py --generate-refs`
on that platform and commit the resulting fixture.
"""

import hashlib
import json
import os
import pickle
import platform
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REFERENCE_FILE = os.path.join(FIXTURE_DIR, "reference_hashes.json")


# Objects whose hashes must be identical across platforms/versions.
# These use only deterministic built-in types.
REFERENCE_OBJECTS = {
    "none": None,
    "true": True,
    "false": False,
    "zero": 0,
    "one": 1,
    "neg_one": -1,
    "int_255": 255,
    "int_256": 256,
    "int_65535": 65535,
    "int_65536": 65536,
    "int_2pow31": 2 ** 31,
    "int_2pow63": 2 ** 63,
    "float_one": 1.0,
    "float_pi": 3.141592653589793,
    "float_inf": float("inf"),
    "float_neg_inf": float("-inf"),
    "empty_string": "",
    "hello": "hello",
    "ascii_255": "a" * 255,
    "ascii_256": "a" * 256,
    "unicode": "héllo wörld",
    "empty_bytes": b"",
    "bytes_255": b"\xab" * 255,
    "bytes_256": b"\xab" * 256,
    "empty_list": [],
    "simple_list": [1, 2, 3],
    "empty_tuple": (),
    "simple_tuple": (1, 2, 3),
    "empty_dict": {},
    "simple_dict": {"a": 1, "b": 2},
    "nested": {"x": [1, (2, 3)], "y": None},
    "frozenset_123": frozenset([1, 2, 3]),
}


def _build_reference_table():
    table = {}
    for name, obj in REFERENCE_OBJECTS.items():
        table[name] = {
            str(proto): pickle_hash(obj, proto)
            for proto in ALL_PROTOCOLS
        }
    return table


@pytest.fixture(scope="session")
def generate_refs(request):
    # --generate-refs is registered in conftest.py
    return request.config.getoption("--generate-refs", default=False)


class TestCrossPlatform:

    @pytest.mark.stability
    def test_generate_or_compare_reference_hashes(self, generate_refs):
        """
        --generate-refs:  Write hashes for this platform to the fixture file.
        Normal run:        Compare this platform's hashes against the fixture.
        """
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        current = _build_reference_table()

        if generate_refs:
            payload = {
                "platform": platform.platform(),
                "python_version": sys.version,
                "pickle_highest_protocol": pickle.HIGHEST_PROTOCOL,
                "hashes": current,
            }
            with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            pytest.skip(f"Reference hashes written to {REFERENCE_FILE}")

        if not os.path.exists(REFERENCE_FILE):
            pytest.skip(
                "No reference hash file found.  Run with --generate-refs on a "
                "baseline platform first."
            )

        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            reference = json.load(f)

        ref_hashes = reference["hashes"]
        failures = []

        for name, proto_hashes in current.items():
            if name not in ref_hashes:
                continue
            for proto_str, current_hash in proto_hashes.items():
                ref_hash = ref_hashes[name].get(proto_str)
                if ref_hash is None:
                    continue
                if current_hash != ref_hash:
                    failures.append(
                        f"  Object={name!r} proto={proto_str}: "
                        f"current={current_hash[:16]}… "
                        f"reference={ref_hash[:16]}…"
                    )

        if failures:
            ref_info = (
                f"Reference: {reference.get('platform')} / "
                f"Python {reference.get('python_version', '?')[:6]}"
            )
            current_info = f"Current:   {platform.platform()} / Python {sys.version[:6]}"
            msg = "\n".join([
                "Cross-platform hash mismatch detected!",
                ref_info,
                current_info,
                "Divergences:",
            ] + failures)
            pytest.fail(msg)


class TestSameProcessStability:
    """
    Sanity: within a single process, all reference objects must produce
    identical hashes across two independent calls.
    """

    @pytest.mark.stability
    @pytest.mark.parametrize("name,obj", list(REFERENCE_OBJECTS.items()))
    def test_object_is_stable_within_process(self, name, obj, protocol):
        h1 = pickle_hash(obj, protocol)
        h2 = pickle_hash(obj, protocol)
        assert h1 == h2, (
            f"Non-deterministic: {name!r} with protocol={protocol}"
        )
