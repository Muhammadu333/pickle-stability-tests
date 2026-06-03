"""
Equivalence Partitioning tests.

We partition the input space of pickle into equivalence classes where objects
within a class are expected to behave the same way with respect to pickling.
Each partition is represented by at least one test case.

Partitions:
  EP1  – None / singleton values
  EP2  – Booleans
  EP3  – Integers (small, large, negative, zero)
  EP4  – Floats (normal, subnormal, special)
  EP5  – Complex numbers
  EP6  – Strings (empty, ASCII, Unicode, long)
  EP7  – Bytes / bytearray
  EP8  – Sequences (list, tuple)
  EP9  – Mappings (dict)
  EP10 – Sets (set, frozenset)
  EP11 – User-defined classes (simple, with __reduce__, with __getstate__)
  EP12 – Nested / composite objects
  EP13 – Un-picklable objects (should raise, not silently corrupt)
"""

import pickle
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


# ---------------------------------------------------------------------------
# EP1 – Singletons
# ---------------------------------------------------------------------------

class TestEP1Singletons:

    @pytest.mark.parametrize("value", [None, ..., NotImplemented])
    def test_singleton_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) is value

    def test_none_hash_stable(self, protocol):
        h1 = pickle_hash(None, protocol)
        h2 = pickle_hash(None, protocol)
        assert h1 == h2


# ---------------------------------------------------------------------------
# EP2 – Booleans
# ---------------------------------------------------------------------------

class TestEP2Booleans:

    def test_true_roundtrip(self, protocol):
        result = roundtrip(True, protocol)
        assert result is True

    def test_false_roundtrip(self, protocol):
        result = roundtrip(False, protocol)
        assert result is False

    def test_bool_not_int_alias(self, protocol):
        """True pickles to a different byte sequence than 1."""
        assert pickle_hash(True, protocol) != pickle_hash(1, protocol)

    def test_false_not_zero_alias(self, protocol):
        assert pickle_hash(False, protocol) != pickle_hash(0, protocol)


# ---------------------------------------------------------------------------
# EP3 – Integers
# ---------------------------------------------------------------------------

class TestEP3Integers:

    @pytest.mark.parametrize("value", [
        0, 1, -1,
        127, 128, 255, 256,
        32767, 32768, 65535, 65536,
        2**31 - 1, 2**31, -(2**31) - 1,
        2**63 - 1, 2**63, -(2**63),
        2**100, 2**255, 10**50,
    ])
    def test_integer_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) == value

    @pytest.mark.parametrize("value", [0, 1, -1, 2**63])
    def test_integer_stable(self, value, protocol):
        assert pickle_hash(value, protocol) == pickle_hash(value, protocol)


# ---------------------------------------------------------------------------
# EP4 – Floats
# ---------------------------------------------------------------------------

class TestEP4Floats:

    @pytest.mark.parametrize("value", [
        0.0, -0.0, 1.0, -1.0,
        0.1, 0.2, 0.1 + 0.2,     # known float imprecision result
        1 / 3, 2 / 3,
        sys.float_info.max,
        sys.float_info.min,
        5e-324,                   # smallest positive subnormal
        float("inf"), float("-inf"),
    ])
    def test_float_roundtrip(self, value, protocol):
        import math
        result = roundtrip(value, protocol)
        if math.isinf(value):
            assert math.isinf(result) and (result > 0) == (value > 0)
        else:
            assert result == value or (math.isnan(result) and math.isnan(value))

    def test_nan_roundtrip(self, protocol):
        import math
        result = roundtrip(float("nan"), protocol)
        assert math.isnan(result)

    def test_float_hash_stable(self, protocol):
        assert pickle_hash(3.14, protocol) == pickle_hash(3.14, protocol)

    def test_negative_zero_hash_stable(self, protocol):
        assert pickle_hash(-0.0, protocol) == pickle_hash(-0.0, protocol)


# ---------------------------------------------------------------------------
# EP5 – Complex numbers
# ---------------------------------------------------------------------------

class TestEP5Complex:

    @pytest.mark.parametrize("value", [
        complex(0, 0), complex(1, 1), complex(-1, -1),
        complex(float("inf"), 0), complex(0, float("inf")),
        complex(float("nan"), 0),
    ])
    def test_complex_roundtrip(self, value, protocol):
        result = roundtrip(value, protocol)
        import cmath
        if cmath.isnan(value):
            assert cmath.isnan(result)
        elif cmath.isinf(value):
            assert cmath.isinf(result)
        else:
            assert result == value


# ---------------------------------------------------------------------------
# EP6 – Strings
# ---------------------------------------------------------------------------

class TestEP6Strings:

    @pytest.mark.parametrize("value", [
        "",
        "hello",
        "Hello, World!",
        "\x00\x01\x02",           # null and control bytes
        "\n\r\t",
        "a" * 10000,
        "éàü",     # Latin extended
        "中文",           # CJK
        "\U0001f600",             # emoji (surrogate pair territory)
        "line1\nline2\nline3",
    ])
    def test_string_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) == value

    def test_string_hash_stable(self, protocol):
        assert pickle_hash("stable", protocol) == pickle_hash("stable", protocol)


# ---------------------------------------------------------------------------
# EP7 – Bytes / bytearray
# ---------------------------------------------------------------------------

class TestEP7Bytes:

    @pytest.mark.parametrize("value", [
        b"", b"\x00", b"\xff", b"hello", bytes(range(256)),
    ])
    def test_bytes_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) == value

    @pytest.mark.parametrize("value", [
        bytearray(), bytearray(b"hello"), bytearray(range(256)),
    ])
    def test_bytearray_roundtrip(self, value, protocol):
        assert roundtrip(value, protocol) == value


# ---------------------------------------------------------------------------
# EP8 – Sequences
# ---------------------------------------------------------------------------

class TestEP8Sequences:

    def test_empty_list(self, protocol):
        assert roundtrip([], protocol) == []

    def test_nested_list(self, protocol):
        obj = [[1, 2], [3, [4, 5]], []]
        assert roundtrip(obj, protocol) == obj

    def test_empty_tuple(self, protocol):
        assert roundtrip((), protocol) == ()

    def test_single_element_tuple(self, protocol):
        assert roundtrip((42,), protocol) == (42,)

    def test_mixed_types_list(self, protocol):
        obj = [1, "two", 3.0, None, True, b"five"]
        assert roundtrip(obj, protocol) == obj


# ---------------------------------------------------------------------------
# EP9 – Mappings
# ---------------------------------------------------------------------------

class TestEP9Dicts:

    def test_empty_dict(self, protocol):
        assert roundtrip({}, protocol) == {}

    def test_string_keyed_dict(self, protocol):
        d = {"a": 1, "b": 2, "c": 3}
        assert roundtrip(d, protocol) == d

    def test_int_keyed_dict(self, protocol):
        d = {0: "zero", 1: "one", -1: "neg_one"}
        assert roundtrip(d, protocol) == d

    def test_nested_dict(self, protocol):
        d = {"outer": {"inner": [1, 2, 3]}}
        assert roundtrip(d, protocol) == d

    def test_dict_insertion_order_preserved(self, protocol):
        """Python 3.7+ guarantees insertion order; pickle must maintain it."""
        d = {"z": 1, "a": 2, "m": 3}
        result = roundtrip(d, protocol)
        assert list(result.keys()) == ["z", "a", "m"]


# ---------------------------------------------------------------------------
# EP10 – Sets
# ---------------------------------------------------------------------------

class TestEP10Sets:

    def test_empty_set_roundtrip(self, protocol):
        assert roundtrip(set(), protocol) == set()

    def test_set_roundtrip(self, protocol):
        s = {1, 2, 3}
        assert roundtrip(s, protocol) == s

    def test_frozenset_roundtrip(self, protocol):
        fs = frozenset([1, 2, 3])
        assert roundtrip(fs, protocol) == fs

    def test_frozenset_hash_stable(self, protocol):
        fs = frozenset([1, 2, 3])
        assert pickle_hash(fs, protocol) == pickle_hash(fs, protocol)


# ---------------------------------------------------------------------------
# EP11 – User-defined classes
# ---------------------------------------------------------------------------

class SimplePoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, SimplePoint) and self.x == other.x and self.y == other.y


class PointWithGetstate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        return {"x": self.x, "y": self.y}

    def __setstate__(self, state):
        self.x = state["x"]
        self.y = state["y"]

    def __eq__(self, other):
        return isinstance(other, PointWithGetstate) and self.x == other.x and self.y == other.y


class TestEP11UserClasses:

    def test_simple_class_roundtrip(self, protocol):
        p = SimplePoint(1, 2)
        assert roundtrip(p, protocol) == p

    def test_simple_class_hash_stable(self, protocol):
        p = SimplePoint(1, 2)
        assert pickle_hash(p, protocol) == pickle_hash(p, protocol)

    def test_getstate_class_roundtrip(self, protocol):
        p = PointWithGetstate(3, 4)
        assert roundtrip(p, protocol) == p

    def test_lambda_is_not_picklable(self):
        with pytest.raises(AttributeError):
            pickle.dumps(lambda x: x)

    def test_local_class_not_picklable(self):
        class LocalClass:
            pass

        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(LocalClass())


# ---------------------------------------------------------------------------
# EP12 – Nested / composite
# ---------------------------------------------------------------------------

class TestEP12Composite:

    def test_list_of_dicts(self, protocol):
        obj = [{"a": 1}, {"b": 2}]
        assert roundtrip(obj, protocol) == obj

    def test_dict_of_lists(self, protocol):
        obj = {"x": [1, 2], "y": [3, 4]}
        assert roundtrip(obj, protocol) == obj

    def test_tuple_of_mixed(self, protocol):
        obj = (1, "two", [3, 4], {"five": 5})
        assert roundtrip(obj, protocol) == obj


# ---------------------------------------------------------------------------
# EP13 – Un-picklable objects (error partition)
# ---------------------------------------------------------------------------

class TestEP13Unpicklable:

    def test_file_object_raises(self):
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fname = f.name
        try:
            with open(fname, "rb") as fh:
                with pytest.raises((TypeError, pickle.PicklingError)):
                    pickle.dumps(fh)
        finally:
            os.unlink(fname)

    def test_generator_raises(self):
        def gen():
            yield 1

        with pytest.raises((TypeError, AttributeError)):
            pickle.dumps(gen())

    def test_lock_raises(self):
        import threading
        lock = threading.Lock()
        with pytest.raises((TypeError, pickle.PicklingError)):
            pickle.dumps(lock)
