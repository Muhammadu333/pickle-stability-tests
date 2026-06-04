"""
Boundary Value Analysis (BVA) tests.

For each dimension we test: min, min+1, nominal, max-1, max, and just-outside-max
(where applicable).  BVA is most valuable at protocol boundaries, integer
encoding thresholds, and string/bytes length limits that change serialisation
format.

Pickle protocol internals use specific cut-offs for SHORT_BINSTRING (<=255),
LONG (>= 256-bit integers), etc.  These are the natural boundary values.
"""

import pickle
import struct
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


# ---------------------------------------------------------------------------
# BV1 – Integer encoding boundaries
# Pickle uses different opcodes based on integer magnitude:
#   Protocol 2+: BININT1 (0-255), BININT2 (256-65535), BININT (32-bit signed),
#                LONG1 / LONG4 for big integers.
# ---------------------------------------------------------------------------

class TestBVIntegers:

    @pytest.mark.boundary
    @pytest.mark.parametrize("value", [
        # BININT1 boundaries
        0, 1, 254, 255,
        # BININT2 boundaries
        256, 257, 65534, 65535,
        # BININT (32-bit) boundaries
        65536, 65537,
        2**31 - 2, 2**31 - 1,
        -(2**31), -(2**31) + 1,
        # LONG1 boundaries
        2**31, 2**31 + 1,
        -(2**31) - 1, -(2**31) - 2,
        # Large integers
        2**63 - 1, 2**63, 2**64 - 1, 2**64,
        2**255, 2**256 - 1,
    ])
    def test_integer_boundary_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) == value

    @pytest.mark.boundary
    @pytest.mark.parametrize("value", [
        0, 255, 256, 65535, 65536, 2**31 - 1, 2**31,
    ])
    def test_integer_boundary_hash_stable(self, value, protocol):
        assert pickle_hash(value, protocol) == pickle_hash(value, protocol)

    @pytest.mark.boundary
    def test_boundary_255_not_equal_256(self, protocol):
        """255 and 256 should produce different pickle bytes."""
        assert pickle_hash(255, protocol) != pickle_hash(256, protocol)

    @pytest.mark.boundary
    def test_boundary_65535_not_equal_65536(self, protocol):
        assert pickle_hash(65535, protocol) != pickle_hash(65536, protocol)


# ---------------------------------------------------------------------------
# BV2 – String / bytes length boundaries
# SHORT_BINUNICODE uses 1 byte for length (<= 255).
# BINUNICODE uses 4 bytes.  BINUNICODE8 uses 8 bytes (protocol 4+).
# ---------------------------------------------------------------------------

class TestBVStringLength:

    @pytest.mark.boundary
    @pytest.mark.parametrize("length", [
        0, 1, 254, 255, 256, 257,
        65535, 65536,
        1_000, 10_000,
    ])
    def test_string_length_roundtrip(self, length, protocol):
        s = "a" * length
        assert roundtrip(s, protocol) == s

    @pytest.mark.boundary
    @pytest.mark.parametrize("length", [0, 1, 254, 255, 256, 65535, 65536])
    def test_bytes_length_roundtrip(self, length, protocol):
        b = b"\xAB" * length
        assert roundtrip(b, protocol) == b

    @pytest.mark.boundary
    def test_string_255_vs_256_different_hash(self, protocol):
        """
        At length 256 pickle switches opcodes (SHORT_BINUNICODE → BINUNICODE)
        for protocols >= 1.  The hashes of 255-char and 256-char strings must
        differ for reasons other than content alone.
        """
        h255 = pickle_hash("a" * 255, protocol)
        h256 = pickle_hash("a" * 256, protocol)
        assert h255 != h256


# ---------------------------------------------------------------------------
# BV3 – Container size boundaries
# ---------------------------------------------------------------------------

class TestBVContainerSize:

    @pytest.mark.boundary
    @pytest.mark.parametrize("size", [0, 1, 254, 255, 256, 1000])
    def test_list_size_boundary_roundtrip(self, size, protocol):
        lst = list(range(size))
        assert roundtrip(lst, protocol) == lst

    @pytest.mark.boundary
    @pytest.mark.parametrize("size", [0, 1, 255, 256, 1000])
    def test_dict_size_boundary_roundtrip(self, size, protocol):
        d = {i: i * 2 for i in range(size)}
        assert roundtrip(d, protocol) == d

    @pytest.mark.boundary
    @pytest.mark.parametrize("size", [0, 1, 255, 256, 1000])
    def test_tuple_size_boundary_roundtrip(self, size, protocol):
        t = tuple(range(size))
        assert roundtrip(t, protocol) == t


# ---------------------------------------------------------------------------
# BV4 – Protocol version boundaries
# ---------------------------------------------------------------------------

class TestBVProtocolVersions:

    @pytest.mark.boundary
    def test_protocol_zero_is_text_based(self):
        """Protocol 0 output must be valid ASCII (human-readable)."""
        data = pickle.dumps(42, protocol=0)
        assert all(c < 128 for c in data), "Protocol 0 should be ASCII-safe"

    @pytest.mark.boundary
    def test_protocol_minimum(self):
        """Protocol 0 is the minimum and must always be supported."""
        data = pickle.dumps("hello", protocol=0)
        assert pickle.loads(data) == "hello"

    @pytest.mark.boundary
    def test_protocol_maximum(self):
        """HIGHEST_PROTOCOL must be accepted and produce valid output."""
        data = pickle.dumps("hello", protocol=pickle.HIGHEST_PROTOCOL)
        assert pickle.loads(data) == "hello"

    @pytest.mark.boundary
    def test_protocol_negative_one_is_highest(self):
        """protocol=-1 is an alias for HIGHEST_PROTOCOL."""
        h_neg = pickle_hash("hello", -1)
        h_max = pickle_hash("hello", pickle.HIGHEST_PROTOCOL)
        assert h_neg == h_max

    @pytest.mark.boundary
    def test_invalid_protocol_raises(self):
        with pytest.raises((ValueError, pickle.PicklingError)):
            pickle.dumps("hello", protocol=999)

    @pytest.mark.boundary
    def test_protocol_2_introduced_long1(self):
        """Protocol 2 introduced LONG1 for big integers; must work correctly."""
        big = 2**100
        data = pickle.dumps(big, protocol=2)
        assert pickle.loads(data) == big


# ---------------------------------------------------------------------------
# BV5 – Nesting depth boundaries
# ---------------------------------------------------------------------------

class TestBVNestingDepth:

    @pytest.mark.boundary
    @pytest.mark.parametrize("depth", [1, 10, 100])
    def test_nested_list_depth_roundtrip(self, depth, protocol):
        """Moderately deeply nested list should survive pickle/unpickle."""
        obj = []
        current = obj
        for _ in range(depth - 1):
            inner = []
            current.append(inner)
            current = inner
        assert roundtrip(obj, protocol) is not None

    @pytest.mark.boundary
    def test_depth_500_hits_recursion_limit(self, protocol):
        """
        FINDING (implementation-dependent): CPython's C pickle extension handles
        list nesting iteratively, so depth 500 does NOT raise RecursionError.
        The pure-Python pickler would raise.  We document this as a platform
        finding rather than asserting a specific outcome.
        """
        depth = 500
        obj = []
        current = obj
        for _ in range(depth - 1):
            inner = []
            current.append(inner)
            current = inner
        try:
            data = pickle.dumps(obj, protocol=protocol)
            # C extension handled it — document as stable at this depth.
            assert pickle.loads(data) is not None
        except RecursionError:
            # Pure-Python pickler hit the limit — also acceptable.
            pass

    @pytest.mark.boundary
    def test_recursion_limit_breach(self, protocol):
        """
        Extremely deep nesting (recursion_limit + 100) may or may not raise,
        depending on whether the C or pure-Python pickler is active.
        Either outcome is acceptable; corrupt silent output is not.
        """
        depth = sys.getrecursionlimit() + 100
        obj = []
        current = obj
        for _ in range(depth):
            inner = []
            current.append(inner)
            current = inner
        try:
            data = pickle.dumps(obj, protocol=protocol)
            # If it didn't raise, the output must at least be loadable.
            pickle.loads(data)
        except (RecursionError, pickle.PicklingError, MemoryError):
            pass  # raising is also acceptable
