"""
Correctness / roundtrip tests.

These tests verify that unpickle(pickle(x)) == x (or is equivalent for
special values like NaN).  This is distinct from stability tests: stability
checks if pickle bytes are identical; correctness checks if the deserialised
value is semantically correct.
"""

import datetime
import decimal
import enum
import fractions
import math
import pickle
import re
import uuid

import pytest

from utils import ALL_PROTOCOLS, roundtrip


# ---------------------------------------------------------------------------
# Standard library types
# ---------------------------------------------------------------------------

class TestStdlibTypes:

    @pytest.mark.correctness
    def test_datetime_roundtrip(self, protocol):
        dt = datetime.datetime(2024, 1, 15, 12, 30, 45, 123456)
        assert roundtrip(dt, protocol) == dt

    @pytest.mark.correctness
    def test_date_roundtrip(self, protocol):
        d = datetime.date(2024, 6, 1)
        assert roundtrip(d, protocol) == d

    @pytest.mark.correctness
    def test_time_roundtrip(self, protocol):
        t = datetime.time(14, 30, 0, 500)
        assert roundtrip(t, protocol) == t

    @pytest.mark.correctness
    def test_timedelta_roundtrip(self, protocol):
        td = datetime.timedelta(days=3, hours=4, minutes=5, seconds=6)
        assert roundtrip(td, protocol) == td

    @pytest.mark.correctness
    def test_decimal_roundtrip(self, protocol):
        for value in ["3.14", "0", "-1", "1e50", "Infinity", "-Infinity", "NaN"]:
            d = decimal.Decimal(value)
            result = roundtrip(d, protocol)
            if d.is_nan():
                assert result.is_nan()
            else:
                assert result == d

    @pytest.mark.correctness
    def test_fraction_roundtrip(self, protocol):
        for value in [
            fractions.Fraction(1, 3),
            fractions.Fraction(22, 7),
            fractions.Fraction(0),
            fractions.Fraction(-5, 6),
        ]:
            assert roundtrip(value, protocol) == value

    @pytest.mark.correctness
    def test_uuid_roundtrip(self, protocol):
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        assert roundtrip(u, protocol) == u

    @pytest.mark.correctness
    def test_regex_roundtrip(self, protocol):
        pattern = re.compile(r"^\d+\.\d+$", re.MULTILINE)
        result = roundtrip(pattern, protocol)
        assert result.pattern == pattern.pattern
        assert result.flags == pattern.flags

    @pytest.mark.correctness
    def test_range_roundtrip(self, protocol):
        for r in [range(10), range(1, 100, 3), range(0, -10, -1)]:
            assert roundtrip(r, protocol) == r


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Flag(enum.IntFlag):
    READ = 1
    WRITE = 2
    EXEC = 4


class TestEnums:

    @pytest.mark.correctness
    def test_enum_roundtrip(self, protocol):
        for member in Color:
            assert roundtrip(member, protocol) is member

    @pytest.mark.correctness
    def test_intflag_roundtrip(self, protocol):
        combined = Flag.READ | Flag.WRITE
        result = roundtrip(combined, protocol)
        assert result == combined


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

try:
    import dataclasses

    @dataclasses.dataclass
    class Point3D:
        x: float
        y: float
        z: float

    @dataclasses.dataclass
    class Node:
        value: int
        children: list = dataclasses.field(default_factory=list)

    class TestDataclasses:

        @pytest.mark.correctness
        def test_dataclass_roundtrip(self, protocol):
            p = Point3D(1.0, 2.5, -3.14)
            assert roundtrip(p, protocol) == p

        @pytest.mark.correctness
        def test_nested_dataclass_roundtrip(self, protocol):
            root = Node(1, [Node(2), Node(3, [Node(4)])])
            assert roundtrip(root, protocol) == root

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Type identity after roundtrip
# ---------------------------------------------------------------------------

class TestTypeIdentity:

    @pytest.mark.correctness
    def test_int_stays_int(self, protocol):
        assert type(roundtrip(42, protocol)) is int

    @pytest.mark.correctness
    def test_bool_stays_bool(self, protocol):
        assert type(roundtrip(True, protocol)) is bool

    @pytest.mark.correctness
    def test_float_stays_float(self, protocol):
        assert type(roundtrip(1.0, protocol)) is float

    @pytest.mark.correctness
    def test_str_stays_str(self, protocol):
        assert type(roundtrip("hello", protocol)) is str

    @pytest.mark.correctness
    def test_bytes_stays_bytes(self, protocol):
        assert type(roundtrip(b"data", protocol)) is bytes

    @pytest.mark.correctness
    def test_list_stays_list(self, protocol):
        assert type(roundtrip([1, 2], protocol)) is list

    @pytest.mark.correctness
    def test_tuple_stays_tuple(self, protocol):
        assert type(roundtrip((1, 2), protocol)) is tuple

    @pytest.mark.correctness
    def test_dict_stays_dict(self, protocol):
        assert type(roundtrip({"a": 1}, protocol)) is dict

    @pytest.mark.correctness
    def test_set_stays_set(self, protocol):
        assert type(roundtrip({1, 2, 3}, protocol)) is set

    @pytest.mark.correctness
    def test_frozenset_stays_frozenset(self, protocol):
        assert type(roundtrip(frozenset([1, 2]), protocol)) is frozenset


# ---------------------------------------------------------------------------
# Pickle protocol version header
# ---------------------------------------------------------------------------

class TestProtocolHeader:

    @pytest.mark.correctness
    @pytest.mark.parametrize("proto", [2, 3, 4, 5])
    def test_protocol_header_present(self, proto):
        """Protocols >= 2 must start with the PROTO opcode (0x80)."""
        data = pickle.dumps("test", protocol=proto)
        assert data[0:1] == b"\x80", "Missing PROTO opcode at start"
        assert data[1] == proto, f"Protocol byte should be {proto}"

    @pytest.mark.correctness
    def test_protocol_0_no_proto_opcode(self):
        data = pickle.dumps("test", protocol=0)
        assert data[0:1] != b"\x80", "Protocol 0 should not start with PROTO opcode"
