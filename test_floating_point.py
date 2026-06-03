"""
Floating-point stability tests.

Floats are IEEE 754 double-precision values.  Their pickle representation
should be bit-for-bit identical on any conforming platform.  However, several
edge cases exist:
  - Negative zero (-0.0 == 0.0 but different bit pattern)
  - NaN (not equal to itself; multiple NaN bit patterns exist)
  - Infinity
  - Subnormal (denormal) numbers
  - Platform-specific float repr in protocol 0

The lab explicitly calls out floating-point accuracy as a possible instability
vector.
"""

import math
import pickle
import struct
import sys

import pytest

from utils import ALL_PROTOCOLS, assert_stable, pickle_hash, roundtrip

# Bit-level NaN variants (quiet NaN and signalling NaN)
_QNaN = struct.unpack("d", b"\x00\x00\x00\x00\x00\x00\xf8\x7f")[0]
_SNaN = struct.unpack("d", b"\x01\x00\x00\x00\x00\x00\xf0\x7f")[0]  # may normalise
_NEG_QNaN = struct.unpack("d", b"\x00\x00\x00\x00\x00\x00\xf8\xff")[0]


class TestFloatEdgeCases:

    @pytest.mark.stability
    def test_pos_zero_stable(self, protocol):
        assert_stable(0.0, protocol)

    @pytest.mark.stability
    def test_neg_zero_stable(self, protocol):
        assert_stable(-0.0, protocol)

    @pytest.mark.stability
    def test_nan_stable(self, protocol):
        """NaN bytes should be consistent within a Python process."""
        assert_stable(float("nan"), protocol)

    @pytest.mark.stability
    def test_inf_stable(self, protocol):
        assert_stable(float("inf"), protocol)

    @pytest.mark.stability
    def test_neg_inf_stable(self, protocol):
        assert_stable(float("-inf"), protocol)

    @pytest.mark.stability
    def test_subnormal_stable(self, protocol):
        """Smallest positive subnormal float."""
        subnormal = 5e-324
        assert_stable(subnormal, protocol)

    @pytest.mark.stability
    @pytest.mark.parametrize("value", [
        1 / 3,
        2 / 3,
        math.pi,
        math.e,
        math.tau,
        math.sqrt(2),
        1.0000000000000002,   # just above 1.0 (machine epsilon boundary)
        0.1 + 0.2,            # classic floating-point imprecision
    ])
    def test_irrational_approximation_stable(self, value, protocol):
        assert_stable(value, protocol)


class TestFloatRoundtrip:

    @pytest.mark.correctness
    @pytest.mark.parametrize("value", [
        0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
        sys.float_info.max, sys.float_info.min,
        5e-324,  # subnormal
        1.7976931348623157e+308,  # near max
    ])
    def test_normal_float_roundtrip(self, value, protocol):
        result = roundtrip(value, protocol)
        # Use bit-level comparison for -0.0 preservation
        assert struct.pack("d", result) == struct.pack("d", value)

    @pytest.mark.correctness
    def test_inf_roundtrip(self, protocol):
        assert roundtrip(float("inf"), protocol) == float("inf")
        assert roundtrip(float("-inf"), protocol) == float("-inf")

    @pytest.mark.correctness
    def test_nan_roundtrip(self, protocol):
        result = roundtrip(float("nan"), protocol)
        assert math.isnan(result)

    @pytest.mark.correctness
    def test_neg_zero_roundtrip_bit_identical(self, protocol):
        """
        -0.0 must survive the roundtrip as -0.0 (not 0.0).
        Protocol 0 uses repr() which may lose the sign — documented below.
        """
        result = roundtrip(-0.0, protocol)
        bits_in = struct.pack("d", -0.0)
        bits_out = struct.pack("d", result)
        if protocol == 0:
            # Protocol 0 round-trips via repr(); -0.0 repr is "-0.0" in
            # Python 3 so it should be preserved, but we document this risk.
            pass
        assert bits_in == bits_out, (
            f"Protocol {protocol}: -0.0 bit pattern not preserved on roundtrip"
        )


class TestFloatHashDistinction:

    @pytest.mark.stability
    def test_neg_zero_hash_distinct_from_pos_zero(self, protocol):
        """
        -0.0 and 0.0 are equal (==) but must produce different pickle bytes
        for protocols that preserve the IEEE 754 sign bit (protocols >= 1).
        """
        if protocol == 0:
            pytest.xfail(
                "Protocol 0 (ASCII) may not distinguish -0.0 from 0.0 "
                "depending on Python version repr behaviour."
            )
        h_pos = pickle_hash(0.0, protocol)
        h_neg = pickle_hash(-0.0, protocol)
        assert h_pos != h_neg

    @pytest.mark.stability
    def test_one_vs_neg_one(self, protocol):
        assert pickle_hash(1.0, protocol) != pickle_hash(-1.0, protocol)

    @pytest.mark.stability
    def test_inf_vs_large_float(self, protocol):
        """inf and sys.float_info.max must produce different hashes."""
        assert pickle_hash(float("inf"), protocol) != pickle_hash(
            sys.float_info.max, protocol
        )


class TestFloatProtocol0Special:
    """
    Protocol 0 serialises floats as ASCII repr strings.  This exposes
    platform/version differences if repr() behaviour changed.
    """

    def test_protocol0_float_is_ascii(self):
        data = pickle.dumps(3.14, protocol=0)
        assert all(c < 128 for c in data)

    def test_protocol0_float_roundtrip(self):
        for value in [0.1, 1 / 3, math.pi, sys.float_info.max]:
            result = roundtrip(value, protocol=0)
            assert result == value, f"Protocol 0 roundtrip failed for {value!r}"

    def test_protocol0_nan_roundtrip(self):
        result = roundtrip(float("nan"), protocol=0)
        assert math.isnan(result)

    def test_protocol0_inf_roundtrip(self):
        assert roundtrip(float("inf"), protocol=0) == float("inf")
        assert roundtrip(float("-inf"), protocol=0) == float("-inf")
