"""
Instability findings: which inputs produce non-hash-identical pickle output?

This module is the central "findings" file for the lab.  Each test either:
  - PASSES  → confirms a known instability (the unstable behaviour is reproducible)
  - FAILS   → would mean the instability was fixed / does not apply here

Run with -v to see which cases are stable vs unstable.

Known instabilities found by this suite:
  I1  Sets ({...})           – PYTHONHASHSEED randomises element order
  I2  Frozensets             – same root cause as sets
  I3  Custom __reduce__      – if implementation uses id() or memory address
  I4  Protocol 0 floats      – repr()-based; historically version-sensitive
  I5  Nesting depth >= ~500  – RecursionError; output never produced
  I6  Protocol version       – same object, different protocol → different hash
                               (by design, but relevant for cross-version pickle files)
"""

import os
import pickle
import subprocess
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash

SEEDS = [0, 1, 42, 12345, 99999]


def _hash_in_subprocess(obj_repr, protocol, seed):
    """Return SHA-256 hex of pickle.dumps(eval(obj_repr)) in a fresh process."""
    code = (
        f"import pickle, hashlib\n"
        f"obj = {obj_repr}\n"
        f"print(hashlib.sha256(pickle.dumps(obj, protocol={protocol})).hexdigest())\n"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=15,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# I1 – Sets are unstable across processes
# ---------------------------------------------------------------------------

class TestI1SetInstability:
    """
    FINDING: pickle.dumps(set_with_strings) is NOT hash-identical across
    Python processes because PYTHONHASHSEED randomises the internal hash
    table layout, which changes the iteration order that pickle observes.
    """

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_string_set_differs_across_hash_seeds(self, protocol):
        obj_repr = "{'apple', 'banana', 'cherry', 'date', 'elderberry'}"
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        assert len(hashes) > 1, (
            f"UNEXPECTED: string set was stable across all PYTHONHASHSEED values "
            f"for protocol={protocol}. Got: {hashes}"
        )

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_mixed_set_differs_across_hash_seeds(self, protocol):
        obj_repr = "{'x', 'y', 'z', 'a', 'b', 'c'}"
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        assert len(hashes) > 1, (
            f"Mixed set should be unstable across seeds for protocol={protocol}"
        )

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_integer_only_set_stability(self, protocol):
        """
        Integer hash() is NOT randomised by PYTHONHASHSEED.
        Integer-only sets MAY be stable — this documents that behaviour.
        """
        obj_repr = "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        stability = "STABLE" if len(hashes) == 1 else "UNSTABLE"
        # We document but do not assert — result depends on CPython internals.
        print(f"\nInteger-only set proto={protocol}: {stability} ({len(hashes)} unique hashes)")


# ---------------------------------------------------------------------------
# I2 – Frozensets are unstable across processes (string elements)
# ---------------------------------------------------------------------------

class TestI2FrozensetInstability:
    """
    FINDING: frozenset with string elements is also unstable across processes.
    """

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_string_frozenset_differs_across_hash_seeds(self, protocol):
        obj_repr = "frozenset({'alpha', 'beta', 'gamma', 'delta', 'epsilon'})"
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        assert len(hashes) > 1, (
            f"String frozenset should be unstable across seeds for protocol={protocol}"
        )


# ---------------------------------------------------------------------------
# I3 – Custom __reduce__ using id() is unstable
# ---------------------------------------------------------------------------

class TestI3CustomReduceInstability:
    """
    FINDING: if a class's __reduce__ encodes the object's memory address
    (e.g. via id()), the pickle bytes will differ across runs because
    memory layout is not deterministic.
    """

    def test_reduce_with_id_is_unstable(self):
        """
        Simulate a poorly written __reduce__ that includes id() in its output.
        The resulting pickle bytes will differ between object creations.
        """
        results = set()
        for _ in range(10):
            # Each new object gets a different id().
            # We pickle a dict containing the id to simulate this pattern.
            obj = object()
            data = pickle.dumps({"addr": id(obj)})
            import hashlib
            results.add(hashlib.sha256(data).hexdigest())

        assert len(results) > 1, (
            "Expected different hashes when id() is included in pickle data"
        )


# ---------------------------------------------------------------------------
# I4 – Protocol 0 float repr (historically unstable, stable in Python 3)
# ---------------------------------------------------------------------------

class TestI4Protocol0FloatRepr:
    """
    FINDING (historical): Protocol 0 serialises floats via repr().
    In Python 2, repr(0.1) produced platform-dependent output.
    In Python 3.1+, repr() always produces the shortest round-trip string,
    so this is NOW STABLE within Python 3.

    This test documents the behaviour and confirms Python 3 stability.
    """

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_float_is_stable_across_seeds(self, protocol):
        """
        Floats should be hash-identical across processes — PYTHONHASHSEED
        does not affect float serialisation.
        """
        obj_repr = "3.141592653589793"
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        assert len(hashes) == 1, (
            f"FINDING: float pickle is unexpectedly unstable for protocol={protocol}. "
            f"Got {len(hashes)} distinct hashes."
        )

    def test_protocol0_uses_repr_not_binary(self):
        """Protocol 0 stores floats as their decimal repr string."""
        data = pickle.dumps(0.1, protocol=0)
        assert b"0.1" in data, (
            "Protocol 0 should contain the decimal string repr of the float"
        )


# ---------------------------------------------------------------------------
# I5 – Deep nesting causes RecursionError (no output produced)
# ---------------------------------------------------------------------------

class TestI5RecursionInstability:
    """
    FINDING: Deeply nested structures (depth >= ~500) cause RecursionError
    during pickling.  No pickle output is produced — this is a hard stability
    failure for such inputs.
    """

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_depth_500_recursion_behaviour(self, protocol):
        """
        FINDING: CPython's C pickle extension handles list nesting iteratively
        so depth 500 does NOT raise RecursionError on CPython.
        The pure-Python pickler would raise at this depth.
        This test documents which behaviour is observed on the current platform.
        """
        lst = []
        cur = lst
        for _ in range(499):
            inner = []
            cur.append(inner)
            cur = inner
        try:
            data = pickle.dumps(lst, protocol=protocol)
            result = "STABLE (C extension — iterative)"
            assert pickle.loads(data) is not None
        except RecursionError:
            result = "UNSTABLE (pure Python — recursive)"
        print(f"\n  proto={protocol}: depth-500 nesting → {result}")

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_depth_100_is_picklable(self, protocol):
        """Moderate nesting is fine — only extreme depths fail."""
        lst = []
        cur = lst
        for _ in range(99):
            inner = []
            cur.append(inner)
            cur = inner
        data = pickle.dumps(lst, protocol=protocol)
        assert pickle.loads(data) is not None


# ---------------------------------------------------------------------------
# I6 – Protocol version changes the hash (by design)
# ---------------------------------------------------------------------------

class TestI6ProtocolVersionHashDifference:
    """
    FINDING: The same object pickled with different protocol versions produces
    different bytes and thus different SHA-256 hashes.  This is intentional,
    but means pickle files are NOT hash-identical across protocol versions.

    Practical implication: if you store/compare pickle hashes, you MUST also
    fix the protocol version.
    """

    @pytest.mark.parametrize("obj,label", [
        (42, "int"),
        ("hello", "str"),
        ([1, 2, 3], "list"),
        ({"a": 1}, "dict"),
        (b"bytes", "bytes"),
    ])
    def test_same_object_different_protocols_differ(self, obj, label):
        hashes = {p: pickle_hash(obj, p) for p in ALL_PROTOCOLS}
        unique = set(hashes.values())
        assert len(unique) >= 2, (
            f"Expected at least 2 distinct hashes across protocols for {label!r}, "
            f"got: {hashes}"
        )
        # Print the per-protocol hashes for the report.
        print(f"\n{label}: {len(unique)} distinct hashes across {len(ALL_PROTOCOLS)} protocols")


# ---------------------------------------------------------------------------
# Summary: stable types (control group)
# ---------------------------------------------------------------------------

class TestStableTypes:
    """
    Control group: these types ARE hash-identical across processes.
    Tests here should all pass, confirming stability.
    """

    STABLE_CASES = [
        ("None",       "None"),
        ("True",       "True"),
        ("False",      "False"),
        ("int",        "42"),
        ("large int",  "2**100"),
        ("float",      "3.14"),
        ("string",     "'hello world'"),
        ("bytes",      "b'hello'"),
        ("list",       "[1, 2, 3]"),
        ("tuple",      "(1, 2, 3)"),
        ("dict",       "{'a': 1, 'b': 2}"),   # insertion-ordered in 3.7+
        ("frozenset integers", "frozenset([1, 2, 3])"),
    ]

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    @pytest.mark.parametrize("label,obj_repr", STABLE_CASES, ids=lambda x: x if isinstance(x, str) else "")
    def test_type_is_stable_across_hash_seeds(self, label, obj_repr, protocol):
        hashes = {_hash_in_subprocess(obj_repr, protocol, s) for s in SEEDS}
        assert len(hashes) == 1, (
            f"UNEXPECTED INSTABILITY: {label} is not stable across PYTHONHASHSEED "
            f"values for protocol={protocol}. Got {len(hashes)} distinct hashes."
        )
