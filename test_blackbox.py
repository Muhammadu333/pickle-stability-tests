"""
Black-box tests for pickle stability and correctness.

Black-box testing treats pickle as a sealed box: we only interact through its
public API (pickle.dumps / pickle.loads) and observe inputs vs outputs.
We do NOT read the source code or rely on knowledge of internal opcodes.

Techniques applied:
  BB1 – Input/output contracts (what goes in must come out)
  BB2 – Hash-identity contract (same input → same SHA-256 digest)
  BB3 – Cross-process stability (spawn subprocess, compare hashes)
  BB4 – Symmetry: pickle then unpickle is the identity function
  BB5 – Asymmetry: different inputs must not produce the same pickle bytes
  BB6 – Output is bytes (type contract)
  BB7 – Size monotonicity: larger inputs produce larger (or equal) output
  BB8 – Protocol contract: output header matches requested protocol
  BB9 – Error contract: invalid inputs raise, not silently corrupt
"""

import hashlib
import os
import pickle
import subprocess
import sys

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


# ---------------------------------------------------------------------------
# BB1 – Input/output contracts
# ---------------------------------------------------------------------------

class TestBB1InputOutputContract:
    """
    Treating pickle purely as a black box:
    given any supported input, loads(dumps(x)) must equal x.
    """

    @pytest.mark.parametrize("obj", [
        None, True, False,
        0, 1, -1, 2**64, -(2**64),
        0.0, -0.0, 3.14, float("inf"), float("-inf"), float("nan"),
        "", "hello", "unicode: 日本語", "a" * 10_000,
        b"", b"\x00\xff", bytes(range(256)),
        [], [1, 2, 3], [[1], [2, [3]]],
        (), (1,), (1, 2, 3),
        {}, {"a": 1}, {"nested": {"x": [1, 2]}},
        set(), frozenset(), frozenset([1, 2, 3]),
    ], ids=repr)
    def test_roundtrip_contract(self, obj, protocol):
        import math
        result = roundtrip(obj, protocol)
        if isinstance(obj, float) and math.isnan(obj):
            assert math.isnan(result)
        else:
            assert result == obj


# ---------------------------------------------------------------------------
# BB2 – Hash-identity contract (within process)
# ---------------------------------------------------------------------------

class TestBB2HashIdentityContract:
    """
    Core lab requirement: same input → same SHA-256 digest.
    Tested purely from the outside — call dumps() twice, hash both.
    """

    @pytest.mark.parametrize("obj", [
        None, True, 42, 3.14, "hello", b"bytes",
        [1, 2, 3], (4, 5), {"a": 1}, frozenset([1, 2]),
    ], ids=repr)
    def test_hash_identical_on_repeated_call(self, obj, protocol):
        h1 = pickle_hash(obj, protocol)
        h2 = pickle_hash(obj, protocol)
        assert h1 == h2, (
            f"Non-deterministic: {obj!r} produced two different hashes "
            f"with protocol={protocol}"
        )

    def test_hash_is_sha256_length(self, protocol):
        """SHA-256 digest is always 64 hex characters."""
        h = pickle_hash(42, protocol)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# BB3 – Cross-process hash stability
# ---------------------------------------------------------------------------

def _subprocess_hash(obj_repr, protocol, seed=0):
    code = (
        f"import pickle, hashlib\n"
        f"obj = {obj_repr}\n"
        f"print(hashlib.sha256(pickle.dumps(obj, {protocol})).hexdigest())\n"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=15,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


SEEDS = [0, 1, 42, 999]


class TestBB3CrossProcessStability:
    """
    Spawn fresh Python processes with different PYTHONHASHSEED values.
    Types that are stable should produce identical hashes across all processes.
    Types that are unstable should produce different hashes.
    """

    # --- STABLE types (should pass) ---

    @pytest.mark.parametrize("label,expr", [
        ("None",        "None"),
        ("True",        "True"),
        ("False",       "False"),
        ("int",         "12345"),
        ("big int",     "2**100"),
        ("float",       "3.141592653589793"),
        ("string",      "'hello world'"),
        ("bytes",       "b'\\xff\\x00\\xab'"),
        ("empty list",  "[]"),
        ("list",        "[1, 2, 3, 'a', None]"),
        ("tuple",       "(1, 2, 3)"),
        ("dict",        "{'z': 3, 'a': 1, 'm': 2}"),
        ("nested",      "{'x': [1, (2, 3)], 'y': None}"),
        ("frozenset int", "frozenset([1, 2, 3, 4, 5])"),
    ])
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_stable_type_cross_process(self, label, expr, protocol):
        hashes = {_subprocess_hash(expr, protocol, s) for s in SEEDS}
        assert len(hashes) == 1, (
            f"INSTABILITY: {label} (proto={protocol}) produced "
            f"{len(hashes)} different hashes across processes: {hashes}"
        )

    # --- UNSTABLE types (should confirm instability) ---

    @pytest.mark.parametrize("label,expr", [
        ("string set",    "{'apple', 'banana', 'cherry', 'date', 'elderberry'}"),
        ("string frozenset", "frozenset({'alpha', 'beta', 'gamma', 'delta'})"),
    ])
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_unstable_type_cross_process(self, label, expr, protocol):
        hashes = {_subprocess_hash(expr, protocol, s) for s in SEEDS}
        assert len(hashes) > 1, (
            f"Expected instability for {label} (proto={protocol}) "
            f"but got identical hashes: {hashes}"
        )


# ---------------------------------------------------------------------------
# BB4 – Symmetry: pickle is the left-inverse of unpickle
# ---------------------------------------------------------------------------

class TestBB4Symmetry:
    """
    pickle.loads(pickle.dumps(x)) == x
    Treated as a black-box algebraic identity.
    """

    @pytest.mark.parametrize("obj", [
        42, "hello", [1, 2], {"a": 1}, (3, 4), frozenset([5]),
        True, None, b"data", 2**200,
    ], ids=repr)
    def test_loads_is_left_inverse_of_dumps(self, obj, protocol):
        assert pickle.loads(pickle.dumps(obj, protocol)) == obj

    def test_double_roundtrip_is_identity(self, protocol):
        """pickle(unpickle(pickle(x))) == pickle(x)"""
        obj = {"key": [1, 2, 3], "other": (True, None)}
        once = pickle.dumps(obj, protocol)
        restored = pickle.loads(once)
        twice = pickle.dumps(restored, protocol)
        assert once == twice


# ---------------------------------------------------------------------------
# BB5 – Asymmetry: distinct inputs must produce distinct hashes
# ---------------------------------------------------------------------------

class TestBB5Asymmetry:
    """
    Two inputs that are not equal must not produce the same pickle bytes.
    (Collision resistance — verifiable from the outside.)
    """

    @pytest.mark.parametrize("a,b", [
        (0, 1),
        (0, False),         # int 0 vs bool False — equal value, different type
        (1, True),          # int 1 vs bool True
        ("", b""),          # empty str vs empty bytes
        ([], ()),           # empty list vs empty tuple
        (0.0, -0.0),        # positive vs negative zero (proto >= 1)
        (1, 1.0),           # int vs float with same numeric value
        (None, False),
        ([1, 2], [2, 1]),   # same elements, different order
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),  # same dict, different insertion order
    ], ids=lambda x: repr(x)[:20])
    def test_distinct_inputs_produce_distinct_hashes(self, a, b, protocol):
        if (a, b) == (0.0, -0.0) and protocol == 0:
            pytest.xfail("Protocol 0 may not distinguish -0.0 from 0.0 (repr-based)")
        ha = pickle_hash(a, protocol)
        hb = pickle_hash(b, protocol)
        assert ha != hb, (
            f"Collision: pickle({a!r}) == pickle({b!r}) with protocol={protocol}"
        )


# ---------------------------------------------------------------------------
# BB6 – Output type contract
# ---------------------------------------------------------------------------

class TestBB6OutputTypeContract:
    """
    pickle.dumps() must always return bytes, regardless of input or protocol.
    """

    @pytest.mark.parametrize("obj", [None, 42, "hello", [1, 2], {"a": 1}])
    def test_dumps_always_returns_bytes(self, obj, protocol):
        result = pickle.dumps(obj, protocol)
        assert isinstance(result, bytes), (
            f"pickle.dumps returned {type(result)}, expected bytes"
        )

    @pytest.mark.parametrize("obj", [None, 42, "hello", [1, 2]])
    def test_loads_does_not_return_bytes(self, obj, protocol):
        """loads() must return the original type, not raw bytes."""
        result = pickle.loads(pickle.dumps(obj, protocol))
        assert type(result) is type(obj)


# ---------------------------------------------------------------------------
# BB7 – Size monotonicity
# ---------------------------------------------------------------------------

class TestBB7SizeMonotonicity:
    """
    Larger inputs should produce larger (or equal) pickle output.
    Tested purely by measuring len(pickle.dumps(x)).
    """

    def test_longer_string_produces_larger_output(self, protocol):
        small = pickle.dumps("a", protocol)
        large = pickle.dumps("a" * 1000, protocol)
        assert len(large) > len(small)

    def test_longer_list_produces_larger_output(self, protocol):
        small = pickle.dumps([1], protocol)
        large = pickle.dumps(list(range(1000)), protocol)
        assert len(large) > len(small)

    def test_empty_vs_nonempty_dict(self, protocol):
        assert len(pickle.dumps({"a": 1}, protocol)) > len(pickle.dumps({}, protocol))

    def test_larger_int_not_necessarily_larger_output(self, protocol):
        """
        Boundary note: small integers may use fixed-size opcodes,
        so 255 and 256 might produce the same output length even though
        the value is larger.  We just document — not assert.
        """
        size_255 = len(pickle.dumps(255, protocol))
        size_256 = len(pickle.dumps(256, protocol))
        # No assertion — document the opcode boundary behaviour.
        print(f"\nproto={protocol}: len(pickle(255))={size_255}, len(pickle(256))={size_256}")


# ---------------------------------------------------------------------------
# BB8 – Protocol header contract
# ---------------------------------------------------------------------------

class TestBB8ProtocolHeaderContract:
    """
    From the outside: the first two bytes of the output tell us the protocol.
    We verify this contract without looking at the source.
    """

    @pytest.mark.parametrize("proto", [2, 3, 4, 5])
    def test_output_starts_with_protocol_version(self, proto):
        data = pickle.dumps(42, protocol=proto)
        assert data[0] == 0x80, "First byte must be PROTO opcode (0x80)"
        assert data[1] == proto, f"Second byte must be protocol number {proto}"

    def test_protocol0_is_human_readable(self):
        data = pickle.dumps(42, protocol=0)
        assert all(b < 128 for b in data), "Protocol 0 output must be ASCII"

    def test_protocol1_is_binary(self):
        # Protocol 1 is binary but has no PROTO header
        data = pickle.dumps(42, protocol=1)
        assert isinstance(data, bytes)
        assert pickle.loads(data) == 42

    @pytest.mark.parametrize("proto", ALL_PROTOCOLS)
    def test_output_ends_with_stop_opcode(self, proto):
        """Every pickle stream must end with the STOP opcode (0x2e = '.')."""
        data = pickle.dumps({"key": [1, 2, 3]}, protocol=proto)
        assert data[-1] == ord("."), (
            f"Protocol {proto} output must end with STOP opcode '.'"
        )


# ---------------------------------------------------------------------------
# BB9 – Error contract
# ---------------------------------------------------------------------------

class TestBB9ErrorContract:
    """
    Black-box: call dumps() on things that shouldn't work.
    The contract is: raise a clear exception, never silently produce garbage.
    """

    def test_unpicklable_lambda_raises(self):
        with pytest.raises((AttributeError, pickle.PicklingError, TypeError)):
            pickle.dumps(lambda x: x)

    def test_unpicklable_generator_raises(self):
        def gen():
            yield 1
        with pytest.raises((AttributeError, pickle.PicklingError, TypeError)):
            pickle.dumps(gen())

    def test_corrupted_stream_raises_on_load(self):
        data = pickle.dumps({"key": "value"}, protocol=2)
        corrupted = data[:5] + b"\xff\xff" + data[7:]
        with pytest.raises(Exception):  # UnpicklingError, ValueError, etc.
            pickle.loads(corrupted)

    def test_empty_bytes_raises_on_load(self):
        with pytest.raises(Exception):
            pickle.loads(b"")

    def test_truncated_stream_raises_on_load(self):
        data = pickle.dumps([1, 2, 3, 4, 5], protocol=2)
        with pytest.raises(Exception):
            pickle.loads(data[:3])

    def test_invalid_protocol_raises_on_dump(self):
        with pytest.raises((ValueError, pickle.PicklingError)):
            pickle.dumps(42, protocol=999)

    def test_loads_wrong_type_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            pickle.loads("this is a string not bytes")
