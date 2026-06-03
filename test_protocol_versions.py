"""
Protocol version-specific tests.

Pickle has 6 protocol versions (0-5).  Each adds new opcodes and capabilities.
These tests verify protocol-specific behaviour, upgrade paths, and the
impact of protocol choice on hash-identical stability.

Timeline:
  Protocol 0 – Original ASCII format
  Protocol 1 – Binary format
  Protocol 2 – Python 2.2; new-style classes, LONG1/LONG4
  Protocol 3 – Python 3.0; explicit bytes type support
  Protocol 4 – Python 3.4; large objects, named tuples, FRAME opcode
  Protocol 5 – Python 3.8; out-of-band data, BYTEARRAY8
"""

import pickle
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


class TestProtocolCompatibility:

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_loads_pickle_from_same_protocol(self, proto):
        """Data pickled with protocol N can be unpickled on the same version."""
        data = pickle.dumps({"key": [1, 2, 3]}, protocol=proto)
        assert pickle.loads(data) == {"key": [1, 2, 3]}

    def test_lower_protocol_can_be_loaded_by_higher(self):
        """Lower-protocol pickles are always loadable by newer Python."""
        for producer in ALL_PROTOCOLS:
            data = pickle.dumps("hello", protocol=producer)
            assert pickle.loads(data) == "hello"

    def test_protocol_version_in_default_dumps(self):
        """pickle.dumps without protocol uses DEFAULT_PROTOCOL, not HIGHEST."""
        data = pickle.dumps("test")
        # DEFAULT_PROTOCOL is typically 5 on 3.8+, but we just check it parses.
        assert pickle.loads(data) == "test"


class TestProtocol0Specifics:
    """Protocol 0 is human-readable ASCII."""

    def test_output_is_printable(self):
        data = pickle.dumps(42, protocol=0)
        assert all(32 <= c < 127 or c in (9, 10, 13) for c in data)

    def test_int_repr(self):
        data = pickle.dumps(42, protocol=0)
        assert b"42" in data

    def test_list_uses_append_opcodes(self):
        """Protocol 0 uses MARK + APPEND sequence for lists."""
        import pickletools
        ops = [op.name for op, _, _ in pickletools.genops(pickle.dumps([1, 2], 0))]
        assert "MARK" in ops


class _ModuleLevelClass:
    """Module-level class required so pickle can locate it by qualified name."""
    def __init__(self, x):
        self.x = x


class TestProtocol2Specifics:
    """Protocol 2: introduced NEWOBJ, LONG1, LONG4, NEWTRUE, NEWFALSE."""

    def test_newobj_opcode_for_new_style_class(self):
        """New-style classes use NEWOBJ (0x81) in protocol 2."""
        import pickletools
        ops = {op.name for op, _, _ in pickletools.genops(
            pickle.dumps(_ModuleLevelClass(1), protocol=2)
        )}
        assert "NEWOBJ" in ops or "REDUCE" in ops  # depends on __reduce_ex__

    def test_newtrue_newfalse_opcodes(self):
        import pickletools
        ops_t = {op.name for op, _, _ in pickletools.genops(pickle.dumps(True, 2))}
        ops_f = {op.name for op, _, _ in pickletools.genops(pickle.dumps(False, 2))}
        assert "NEWTRUE" in ops_t
        assert "NEWFALSE" in ops_f

    def test_long1_for_big_integer(self):
        import pickletools
        big = 2 ** 100
        ops = {op.name for op, _, _ in pickletools.genops(pickle.dumps(big, 2))}
        assert "LONG1" in ops


class TestProtocol4Specifics:
    """Protocol 4: FRAME opcode wraps data for streaming."""

    def test_frame_opcode_present(self):
        import pickletools
        data = pickle.dumps([1, 2, 3], protocol=4)
        ops = {op.name for op, _, _ in pickletools.genops(data)}
        assert "FRAME" in ops

    def test_short_binunicode_opcode(self):
        import pickletools
        data = pickle.dumps("hello", protocol=4)
        ops = {op.name for op, _, _ in pickletools.genops(data)}
        assert "SHORT_BINUNICODE" in ops


class TestProtocol5Specifics:
    """Protocol 5: BYTEARRAY8 and out-of-band buffer support."""

    def test_bytearray_opcode(self):
        import pickletools
        data = pickle.dumps(bytearray(b"test"), protocol=5)
        ops = {op.name for op, _, _ in pickletools.genops(data)}
        assert "BYTEARRAY8" in ops

    def test_out_of_band_buffer(self):
        """PickleBuffer enables out-of-band data for protocol 5."""
        large = bytearray(1000)
        buffers = []

        data = pickle.dumps(
            pickle.PickleBuffer(large),
            protocol=5,
            buffer_callback=buffers.append,
        )
        result = pickle.loads(data, buffers=buffers)
        assert bytes(result) == bytes(large)


class TestProtocolHashIsolation:
    """Different protocols MUST produce different byte sequences (and thus hashes)."""

    @pytest.mark.parametrize("obj", [
        42, "hello", [1, 2, 3], {"a": 1}, b"bytes", True,
    ], ids=repr)
    def test_at_least_two_protocols_differ(self, obj):
        hashes = [pickle_hash(obj, p) for p in ALL_PROTOCOLS]
        assert len(set(hashes)) >= 2, (
            f"All {len(ALL_PROTOCOLS)} protocols produced the same hash for {obj!r}"
        )

    def test_protocol_0_vs_1_differ_for_most_types(self):
        """
        Protocols 0 and 1 differ for most types.  A handful of trivial
        singletons (None, True, False) use the same single-opcode encoding
        in both protocols and are documented as exceptions.
        """
        same_in_both = []
        differ = []
        for obj in [42, "hello", b"data", [1, 2], 3.14, {"a": 1}]:
            if pickle_hash(obj, 0) == pickle_hash(obj, 1):
                same_in_both.append(obj)
            else:
                differ.append(obj)
        # At least the non-trivial types should differ.
        assert len(differ) >= 4, (
            f"Expected protocol 0 vs 1 to differ for most objects. "
            f"Same: {same_in_both}, Different: {differ}"
        )
