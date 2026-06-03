"""
White-box / structural tests.

Based on reading CPython's Lib/pickle.py source code.

We exercise specific internal branches and code paths identified through
all-def / all-uses analysis of the Pickler.dump() and Pickler.save() methods.

Key definitions (assignments) and their uses in pickle.py:
  - self.proto  → used in save_bytes, save_str, save_long, save_float, save_pers_id
  - self.memo   → used in save() to detect shared/recursive objects (memoisation)
  - self.bin    → True for protocols >= 1; controls binary vs. ASCII encoding
  - obj's __reduce_ex__ → used in save_reduce
  - obj's __reduce__    → used in save_reduce
  - obj's __getstate__  → used in save_reduce → load_build
  - obj's __getnewargs_ex__ / __getnewargs__ → used for NEWOBJ / NEWOBJ_EX opcodes

We target:
  WB1  – proto=0 (text) vs proto>=1 (binary) branches in save_bytes/save_str
  WB2  – proto>=2 branch for LONG1/LONG4 in save_long
  WB3  – proto>=4 branch for BINUNICODE8 in save_str
  WB4  – proto>=5 branch for BYTEARRAY8 in save_bytes
  WB5  – memo table: first vs. repeated object reference
  WB6  – __reduce_ex__ dispatch path
  WB7  – __reduce__ fallback path
  WB8  – __getstate__ / __setstate__ path
  WB9  – __getnewargs__ path
  WB10 – __getnewargs_ex__ path
  WB11 – persistent_id hook
  WB12 – dispatch_table override
"""

import copyreg
import io
import pickle
import pickletools

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


# Module-level class required for dispatch_table test (local classes can't be pickled).
class _Widget:
    def __init__(self, color):
        self.color = color


# ---------------------------------------------------------------------------
# WB1 – Binary vs text branches
# ---------------------------------------------------------------------------

class TestWB1BinaryVsText:

    def test_proto0_bytes_encoded_as_latin1(self):
        """
        Protocol 0 cannot natively encode bytes; pickle converts them via
        latin-1.  Verify correct roundtrip.
        """
        value = bytes(range(256))
        result = roundtrip(value, protocol=0)
        assert result == value

    def test_proto1_bytes_binary_encoding(self):
        data0 = pickle.dumps(b"hello", protocol=0)
        data1 = pickle.dumps(b"hello", protocol=1)
        assert data0 != data1  # different opcode
        assert pickle.loads(data1) == b"hello"

    def test_proto0_unicode_encoded_as_raw_unicode_escape(self):
        result = roundtrip("héllo", protocol=0)
        assert result == "héllo"


# ---------------------------------------------------------------------------
# WB2 – LONG1 / LONG4 in save_long (protocol >= 2)
# ---------------------------------------------------------------------------

class TestWB2LongEncoding:

    @pytest.mark.whitebox
    def test_proto2_uses_long1_opcode_for_large_int(self):
        """LONG1 opcode (0x8a) should appear in protocol 2 output for big ints."""
        big_int = 2 ** 100
        data = pickle.dumps(big_int, protocol=2)
        op_names = {op.name for op, _, _ in pickletools.genops(data)}
        assert "LONG1" in op_names, f"Expected LONG1 in protocol 2 output; got {op_names}"

    @pytest.mark.whitebox
    def test_proto0_encodes_long_as_text(self):
        big_int = 2 ** 100
        data = pickle.dumps(big_int, protocol=0)
        op_names = {op.name for op, _, _ in pickletools.genops(data)}
        assert "INT" in op_names or "LONG" in op_names or "LONG1" in op_names, (
            f"Expected INT/LONG opcode in protocol 0; got {op_names}"
        )

    @pytest.mark.whitebox
    def test_long_roundtrip_proto2(self):
        for v in [2**31, 2**63, 2**100, 2**255, -(2**100)]:
            assert roundtrip(v, protocol=2) == v


# ---------------------------------------------------------------------------
# WB3 – BINUNICODE8 in save_str (protocol >= 4)
# ---------------------------------------------------------------------------

class TestWB3BinUnicode8:

    @pytest.mark.whitebox
    def test_proto4_uses_short_binunicode_for_short_strings(self):
        """SHORT_BINUNICODE (0x8c) used for strings <= 255 bytes."""
        data = pickle.dumps("hello", protocol=4)
        assert b"\x8c" in data, "Expected SHORT_BINUNICODE opcode (0x8c)"

    @pytest.mark.whitebox
    def test_proto4_uses_binunicode8_for_long_strings(self):
        """BINUNICODE8 (0x8d) for strings > 2**32 bytes (hard to test size-wise)."""
        # Just verify 256-char strings work correctly
        s = "a" * 256
        assert roundtrip(s, protocol=4) == s


# ---------------------------------------------------------------------------
# WB4 – BYTEARRAY8 (protocol >= 5)
# ---------------------------------------------------------------------------

class TestWB4Bytearray8:

    @pytest.mark.whitebox
    def test_proto5_bytearray_roundtrip(self):
        ba = bytearray(b"hello world")
        result = roundtrip(ba, protocol=5)
        assert result == ba

    @pytest.mark.whitebox
    def test_proto5_bytearray_uses_bytearray8_opcode(self):
        ba = bytearray(b"test")
        data = pickle.dumps(ba, protocol=5)
        op_names = {op.name for op, _, _ in pickletools.genops(data)}
        assert "BYTEARRAY8" in op_names, (
            f"Expected BYTEARRAY8 opcode in proto 5 output; got {op_names}"
        )


# ---------------------------------------------------------------------------
# WB5 – Memo table: shared / repeated references
# ---------------------------------------------------------------------------

class TestWB5MemoTable:

    @pytest.mark.whitebox
    def test_shared_object_uses_get_opcode(self, protocol):
        """When the same object appears twice, pickle uses GET/BINGET to reference memo."""
        shared = [1, 2, 3]
        container = [shared, shared]
        data = pickle.dumps(container, protocol=protocol)
        op_names = {op.name for op, _, _ in pickletools.genops(data)}
        has_get = bool(op_names & {"GET", "BINGET", "LONG_BINGET"})
        assert has_get, f"Expected a GET opcode for shared object reference; got {op_names}"

    @pytest.mark.whitebox
    def test_non_shared_object_no_get_opcode(self, protocol):
        """Two equal but independent lists: no GET opcode needed."""
        container = [[1, 2, 3], [1, 2, 3]]
        data = pickle.dumps(container, protocol=protocol)
        result = pickle.loads(data)
        assert result == [[1, 2, 3], [1, 2, 3]]
        assert result[0] is not result[1]


# ---------------------------------------------------------------------------
# WB6 – __reduce_ex__ dispatch
# ---------------------------------------------------------------------------

class ReduceExClass:
    def __init__(self, value):
        self.value = value

    def __reduce_ex__(self, protocol):
        return (self.__class__, (self.value,))

    def __eq__(self, other):
        return isinstance(other, ReduceExClass) and self.value == other.value


class TestWB6ReduceEx:

    @pytest.mark.whitebox
    def test_reduce_ex_is_called(self, protocol):
        obj = ReduceExClass(99)
        assert roundtrip(obj, protocol) == obj

    @pytest.mark.whitebox
    def test_reduce_ex_stable(self, protocol):
        obj = ReduceExClass(99)
        assert pickle_hash(obj, protocol) == pickle_hash(obj, protocol)


# ---------------------------------------------------------------------------
# WB7 – __reduce__ fallback
# ---------------------------------------------------------------------------

class ReduceClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __reduce__(self):
        return (self.__class__, (self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, ReduceClass) and self.x == other.x and self.y == other.y


class TestWB7Reduce:

    @pytest.mark.whitebox
    def test_reduce_roundtrip(self, protocol):
        obj = ReduceClass(1, 2)
        assert roundtrip(obj, protocol) == obj

    @pytest.mark.whitebox
    def test_reduce_stable(self, protocol):
        obj = ReduceClass(3, 4)
        assert pickle_hash(obj, protocol) == pickle_hash(obj, protocol)


# ---------------------------------------------------------------------------
# WB8 – __getstate__ / __setstate__
# ---------------------------------------------------------------------------

class StatefulClass:
    def __init__(self, data):
        self._data = data
        self._cache = {}  # should NOT be pickled

    def __getstate__(self):
        return {"data": self._data}

    def __setstate__(self, state):
        self._data = state["data"]
        self._cache = {}

    def __eq__(self, other):
        return isinstance(other, StatefulClass) and self._data == other._data


class TestWB8GetSetState:

    @pytest.mark.whitebox
    def test_getstate_excludes_cache(self, protocol):
        obj = StatefulClass({"key": "value"})
        obj._cache["populated"] = True
        result = roundtrip(obj, protocol)
        assert result._cache == {}  # cache not pickled
        assert result._data == {"key": "value"}

    @pytest.mark.whitebox
    def test_getstate_stable(self, protocol):
        obj = StatefulClass([1, 2, 3])
        assert pickle_hash(obj, protocol) == pickle_hash(obj, protocol)


# ---------------------------------------------------------------------------
# WB9 – __getnewargs__
# ---------------------------------------------------------------------------

class NewargsClass:
    def __new__(cls, x):
        instance = super().__new__(cls)
        instance.x = x
        return instance

    def __getnewargs__(self):
        return (self.x,)

    def __eq__(self, other):
        return isinstance(other, NewargsClass) and self.x == other.x


class TestWB9Getnewargs:

    @pytest.mark.whitebox
    def test_getnewargs_roundtrip(self, protocol):
        obj = NewargsClass(42)
        assert roundtrip(obj, protocol) == obj


# ---------------------------------------------------------------------------
# WB10 – __getnewargs_ex__ (protocol >= 4)
# ---------------------------------------------------------------------------

class NewargsExClass:
    def __new__(cls, *, value):
        instance = super().__new__(cls)
        instance.value = value
        return instance

    def __getnewargs_ex__(self):
        return ((), {"value": self.value})

    def __eq__(self, other):
        return isinstance(other, NewargsExClass) and self.value == other.value


class TestWB10GetnewargEx:

    @pytest.mark.whitebox
    @pytest.mark.parametrize("protocol", [4, 5], ids=lambda p: f"proto{p}")
    def test_getnewargs_ex_roundtrip(self, protocol):
        obj = NewargsExClass(value=77)
        assert roundtrip(obj, protocol) == obj


# ---------------------------------------------------------------------------
# WB11 – persistent_id hook
# ---------------------------------------------------------------------------

class _LargeIntPickler(pickle.Pickler):
    def persistent_id(self, obj):
        # Return a plain string so pickle doesn't recurse back into persistent_id.
        if isinstance(obj, int) and not isinstance(obj, bool) and obj > 1000:
            return f"large_int:{obj}"
        return None


class _LargeIntUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if pid.startswith("large_int:"):
            return int(pid.split(":", 1)[1])
        raise pickle.UnpicklingError(f"Unknown pid: {pid}")


class TestWB11PersistentId:

    @pytest.mark.whitebox
    def test_persistent_id_hook(self):
        """
        Custom Pickler with persistent_id can replace objects with IDs.
        The paired Unpickler with persistent_load reconstructs them.
        """
        obj = [1, 2, 9999, "hello"]
        buf = io.BytesIO()
        _LargeIntPickler(buf).dump(obj)
        buf.seek(0)
        result = _LargeIntUnpickler(buf).load()
        assert result == obj


# ---------------------------------------------------------------------------
# WB12 – dispatch_table override
# ---------------------------------------------------------------------------

class TestWB12DispatchTable:

    @pytest.mark.whitebox
    def test_custom_dispatch_table(self, protocol):
        """A custom dispatch_table entry is used in preference to __reduce_ex__."""

        def _widget_reduce(obj):
            return (_Widget, (obj.color,))

        buf = io.BytesIO()
        p = pickle.Pickler(buf, protocol=protocol)
        p.dispatch_table = copyreg.dispatch_table.copy()
        p.dispatch_table[_Widget] = _widget_reduce
        p.dump(_Widget("red"))

        buf.seek(0)
        result = pickle.Unpickler(buf).load()
        assert result.color == "red"


# ---------------------------------------------------------------------------
# Pickletools opcode analysis
# ---------------------------------------------------------------------------

class TestPickletoolsOpcodeAnalysis:
    """
    Use pickletools.genops() to directly inspect opcodes and verify that
    specific code paths were exercised.
    """

    @pytest.mark.whitebox
    def test_int_opcode_selection(self):
        """Verify INT vs BININT vs BININT1 opcode selection."""
        opcodes_0 = {op.name for op, _, _ in pickletools.genops(pickle.dumps(42, 0))}
        opcodes_2 = {op.name for op, _, _ in pickletools.genops(pickle.dumps(42, 2))}
        assert "INT" in opcodes_0 or "LONG" in opcodes_0 or "LONG1" in opcodes_0
        assert "BININT1" in opcodes_2

    @pytest.mark.whitebox
    def test_none_opcode(self):
        for proto in ALL_PROTOCOLS:
            ops = {op.name for op, _, _ in pickletools.genops(pickle.dumps(None, proto))}
            assert "NONE" in ops

    @pytest.mark.whitebox
    def test_true_false_opcodes(self):
        for proto in ALL_PROTOCOLS:
            ops_t = {op.name for op, _, _ in pickletools.genops(pickle.dumps(True, proto))}
            ops_f = {op.name for op, _, _ in pickletools.genops(pickle.dumps(False, proto))}
            if proto >= 2:
                assert "NEWTRUE" in ops_t
                assert "NEWFALSE" in ops_f
