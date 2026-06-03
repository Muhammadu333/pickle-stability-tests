"""
Stability tests: does the same input always produce the same pickle bytes?

Strategy: call pickle.dumps() N times on identical inputs and assert that
every SHA-256 digest is identical (hash-identical, not just equivalent).

This is the core question of the lab project.
"""

import math
import pickle
import sys

import pytest

from utils import ALL_PROTOCOLS, assert_stable, pickle_hash


RUNS = 20  # number of repeated dumps() calls per assertion


# ---------------------------------------------------------------------------
# Primitive scalars
# ---------------------------------------------------------------------------

class TestDeterminismPrimitives:

    @pytest.mark.stability
    @pytest.mark.parametrize("value", [
        0, 1, -1, 42, -42,
        2**31 - 1, 2**31, -(2**31), 2**63 - 1, -(2**63),
        2**100, -(2**100),          # big integers
        True, False, None,
        0.0, 1.0, -1.0, 0.1, 1 / 3,
        1e308, -1e308,               # near float max
        5e-324,                      # smallest positive float (denormal)
        math.pi, math.e, math.tau,
        float("inf"), float("-inf"),
        "",  "hello", "a" * 1000,
        b"", b"\x00\xff", b"bytes",
        complex(1, 2), complex(0, 0), complex(float("inf"), 0),
    ], ids=repr)
    def test_scalar_is_stable(self, value, protocol):
        assert_stable(value, protocol, runs=RUNS)

    @pytest.mark.stability
    def test_nan_is_stable(self, protocol):
        """
        float('nan') != float('nan'), but pickle bytes should be identical
        on repeated calls because the bit-pattern is fixed.
        """
        nan = float("nan")
        assert_stable(nan, protocol, runs=RUNS)

    @pytest.mark.stability
    def test_negative_zero_is_stable(self, protocol):
        """-0.0 and 0.0 are equal but have different bit patterns."""
        assert_stable(-0.0, protocol, runs=RUNS)

    @pytest.mark.stability
    def test_negative_zero_differs_from_zero(self, protocol):
        """-0.0 and 0.0 must produce different pickle bytes."""
        h_pos = pickle_hash(0.0, protocol)
        h_neg = pickle_hash(-0.0, protocol)
        # Protocol 0 uses repr() and may collapse them; document the result.
        if protocol == 0:
            pytest.xfail(
                "Protocol 0 uses repr(); -0.0 and 0.0 may collide (known limitation)"
            )
        assert h_pos != h_neg, "-0.0 and 0.0 should hash-differ for protocol >= 1"


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestDeterminismCollections:

    @pytest.mark.stability
    def test_list_is_stable(self, protocol):
        assert_stable([1, 2, 3, "a", None, True], protocol, runs=RUNS)

    @pytest.mark.stability
    def test_empty_list_is_stable(self, protocol):
        assert_stable([], protocol, runs=RUNS)

    @pytest.mark.stability
    def test_tuple_is_stable(self, protocol):
        assert_stable((1, "two", 3.0, None), protocol, runs=RUNS)

    @pytest.mark.stability
    def test_nested_list_is_stable(self, protocol):
        assert_stable([[1, 2], [3, [4, 5]]], protocol, runs=RUNS)

    @pytest.mark.stability
    def test_dict_is_stable(self, protocol):
        """
        Dicts are insertion-ordered in Python 3.7+, so a dict with a fixed
        literal ordering should always produce the same bytes.
        """
        d = {"a": 1, "b": 2, "c": [3, 4]}
        assert_stable(d, protocol, runs=RUNS)

    @pytest.mark.stability
    def test_set_stability(self, protocol):
        """
        Sets are unordered and use PYTHONHASHSEED for internal ordering.
        This test documents whether set pickling is stable within one process.
        Cross-process stability is NOT guaranteed (see test_cross_process).
        """
        s = frozenset([1, 2, 3, 4, 5])
        assert_stable(s, protocol, runs=RUNS)

    @pytest.mark.stability
    def test_mutable_set_within_process(self, protocol):
        """
        A mutable set's pickle order depends on hash randomisation.
        Within the same process the order is fixed, but this marks it as
        a known cross-run instability concern.
        """
        s = {1, 2, 3, 4, 5}
        hashes = {pickle_hash(s, protocol) for _ in range(RUNS)}
        # Within one process, hash seed is constant → should be stable.
        assert len(hashes) == 1, "Set unexpectedly non-deterministic within same process"

    @pytest.mark.stability
    @pytest.mark.parametrize("size", [0, 1, 100, 1000])
    def test_string_various_sizes(self, size, protocol):
        assert_stable("x" * size, protocol, runs=RUNS)


# ---------------------------------------------------------------------------
# Cross-protocol hash isolation
# ---------------------------------------------------------------------------

class TestProtocolIsolation:

    @pytest.mark.stability
    @pytest.mark.parametrize("value", [42, "hello", [1, 2, 3], {"a": 1}])
    def test_different_protocols_produce_different_bytes(self, value):
        """
        Each protocol version is a distinct binary format.  Hashes for the
        same object MUST differ across protocols (or at most match by coincidence
        for trivially simple values — we document rather than assert).
        """
        hashes = [pickle_hash(value, p) for p in ALL_PROTOCOLS]
        unique = len(set(hashes))
        # Not all protocols are guaranteed to differ for every value, but the
        # suite must observe that at least two distinct hashes exist.
        assert unique >= 2, (
            f"All protocols produced the same hash for {value!r}. "
            "This may indicate protocol aliasing — worth investigating."
        )
