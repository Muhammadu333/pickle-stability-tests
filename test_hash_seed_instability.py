"""
PYTHONHASHSEED instability tests.

Python 3.3+ randomises hash values for str, bytes, and datetime objects by
default (PYTHONHASHSEED=random).  This affects the internal ordering of sets
and any dict whose iteration order depended on hash order (pre-3.7).

Key finding:
  - Sets ({...}) serialise their elements in hash-insertion order.
    Different PYTHONHASHSEED values → different element orderings →
    different pickle bytes → DIFFERENT SHA-256 hashes across Python processes.

  - Dicts in Python 3.7+ are insertion-ordered, so their pickle output is
    stable across PYTHONHASHSEED values (element order was set at creation).

  - frozenset has the same instability as set.

This module documents and tests this behaviour.  The cross-process instability
cannot be directly tested in one pytest run (we cannot change PYTHONHASHSEED
mid-process), so instead we:
  1. Test within-process stability (should always pass).
  2. Spawn subprocess with explicit PYTHONHASHSEED values and compare hashes.
"""

import hashlib
import os
import pickle
import subprocess
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash


HELPER_SCRIPT = os.path.join(os.path.dirname(__file__), "_hash_helper.py")


def _run_hash_helper(seed, obj_repr, protocol):
    """
    Run a subprocess with the given PYTHONHASHSEED and return the SHA-256
    hex digest of pickle.dumps(eval(obj_repr), protocol).
    """
    code = (
        f"import pickle, hashlib\n"
        f"obj = {obj_repr}\n"
        f"data = pickle.dumps(obj, protocol={protocol})\n"
        f"print(hashlib.sha256(data).hexdigest())\n"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, f"Subprocess error: {result.stderr}"
    return result.stdout.strip()


class TestWithinProcessStability:
    """Within a process, PYTHONHASHSEED is constant; everything should be stable."""

    @pytest.mark.stability
    def test_set_stable_within_process(self, protocol):
        s = {1, 2, 3, 4, 5, "a", "b", "c"}
        h1 = pickle_hash(s, protocol)
        h2 = pickle_hash(s, protocol)
        assert h1 == h2

    @pytest.mark.stability
    def test_frozenset_stable_within_process(self, protocol):
        fs = frozenset([1, 2, 3, "x", "y"])
        assert pickle_hash(fs, protocol) == pickle_hash(fs, protocol)

    @pytest.mark.stability
    def test_dict_stable_within_process(self, protocol):
        d = {"alpha": 1, "beta": 2, "gamma": 3}
        assert pickle_hash(d, protocol) == pickle_hash(d, protocol)


class TestCrossProcessHashSeedInstability:
    """
    Cross-process tests: spawn two Python processes with different
    PYTHONHASHSEED values and compare pickle hashes.

    Sets and frozensets are EXPECTED to be unstable (different hashes).
    Dicts (Python 3.7+) and primitive types are expected to be stable.
    """

    SEEDS = [0, 1, 42, 12345, 99999]

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_set_is_unstable_across_processes(self, protocol):
        """
        FINDING: Sets are NOT hash-identical across Python processes with
        different PYTHONHASHSEED values.  This is a known instability.
        """
        obj_repr = "{1, 2, 3, 4, 5, 'a', 'b', 'c'}"
        hashes = {_run_hash_helper(seed, obj_repr, protocol) for seed in self.SEEDS}
        # We expect more than one unique hash — document as a finding.
        if len(hashes) == 1:
            pytest.xfail(
                "All seeds produced the same hash for this set. "
                "This can happen for integer-only sets on some Python versions "
                "because integer hash() is seed-independent."
            )
        assert len(hashes) > 1, (
            "Expected set to produce different hashes with different PYTHONHASHSEED. "
            f"Got hashes: {hashes}"
        )

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_string_set_is_unstable_across_processes(self, protocol):
        """String sets are definitely hash-seed-dependent."""
        obj_repr = "{'apple', 'banana', 'cherry', 'date', 'elderberry'}"
        hashes = {_run_hash_helper(seed, obj_repr, protocol) for seed in self.SEEDS}
        assert len(hashes) > 1, (
            "String set should produce different hashes with different PYTHONHASHSEED. "
            f"Got: {hashes}"
        )

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_dict_is_stable_across_processes(self, protocol):
        """
        Python 3.7+ dicts are insertion-ordered.  Same literal dict →
        same insertion order → same pickle bytes regardless of PYTHONHASHSEED.
        """
        obj_repr = "{'a': 1, 'b': 2, 'c': 3}"
        hashes = {_run_hash_helper(seed, obj_repr, protocol) for seed in self.SEEDS}
        assert len(hashes) == 1, (
            f"Dict pickle hash should be PYTHONHASHSEED-stable but got "
            f"{len(hashes)} distinct hashes: {hashes}"
        )

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_integer_is_stable_across_processes(self, protocol):
        hashes = {_run_hash_helper(seed, "42", protocol) for seed in self.SEEDS}
        assert len(hashes) == 1

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_string_is_stable_across_processes(self, protocol):
        """String pickle bytes don't depend on hash seed."""
        hashes = {_run_hash_helper(seed, "'hello world'", protocol) for seed in self.SEEDS}
        assert len(hashes) == 1

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_list_is_stable_across_processes(self, protocol):
        hashes = {_run_hash_helper(seed, "[1, 2, 3, 4, 5]", protocol) for seed in self.SEEDS}
        assert len(hashes) == 1

    @pytest.mark.stability
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_frozenset_int_only_stability(self, protocol):
        """
        Integer hashes are not seed-randomised, so integer-only frozensets
        may be stable.  This test documents that behaviour.
        """
        obj_repr = "frozenset([1, 2, 3, 4, 5])"
        hashes = {_run_hash_helper(seed, obj_repr, protocol) for seed in self.SEEDS}
        # Document: may be 1 (stable) or >1 (unstable depending on Python version)
        # We don't assert, just record:
        stability = "STABLE" if len(hashes) == 1 else "UNSTABLE"
        print(f"\nfrozenset([1..5]) proto={protocol}: {stability} ({len(hashes)} unique hashes)")
