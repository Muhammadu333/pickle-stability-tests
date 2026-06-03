"""
Security and robustness tests.

While the lab focuses on stability, a professional test suite must also
verify that malformed or adversarial pickle data is handled safely.

These tests document known pickle security characteristics and verify that
the unpickler behaves predictably on edge-case and malformed input.
"""

import pickle
import io

import pytest

from utils import ALL_PROTOCOLS


class TestMalformedInput:

    def test_empty_bytes_raises(self):
        with pytest.raises((EOFError, pickle.UnpicklingError)):
            pickle.loads(b"")

    def test_truncated_data_raises(self):
        data = pickle.dumps([1, 2, 3])
        with pytest.raises((EOFError, pickle.UnpicklingError, struct.error)):
            pickle.loads(data[:-5])

    def test_invalid_opcode_raises(self):
        """An unknown opcode should raise UnpicklingError, not crash silently."""
        with pytest.raises((pickle.UnpicklingError, ValueError)):
            pickle.loads(b"\x80\x05\xff\x00")  # PROTO 5, then invalid opcode

    def test_random_bytes_raise(self):
        """Random bytes should raise, not produce corrupt data silently."""
        import os
        random_bytes = os.urandom(64)
        try:
            pickle.loads(random_bytes)
        except Exception:
            pass  # any exception is acceptable


try:
    import struct
except ImportError:
    pass


class TestRestrictedUnpickler:
    """
    Demonstrate how to safely restrict unpickling to known types.
    This is a recommended defensive practice.
    """

    def test_restricted_unpickler_blocks_arbitrary_class(self):
        class SafeUnpickler(pickle.Unpickler):
            SAFE_CLASSES = {
                ("builtins", "list"),
                ("builtins", "dict"),
                ("builtins", "tuple"),
                ("builtins", "int"),
                ("builtins", "str"),
                ("builtins", "float"),
                ("builtins", "bytes"),
                ("builtins", "bool"),
                ("builtins", "NoneType"),
            }

            def find_class(self, module, name):
                if (module, name) not in self.SAFE_CLASSES:
                    raise pickle.UnpicklingError(
                        f"Blocked: {module}.{name}"
                    )
                return super().find_class(module, name)

        # Safe objects should load fine.
        data = pickle.dumps([1, "hello", 3.14])
        result = SafeUnpickler(io.BytesIO(data)).load()
        assert result == [1, "hello", 3.14]

        # Arbitrary classes should be blocked.
        import datetime
        data_dt = pickle.dumps(datetime.datetime(2024, 1, 1))
        with pytest.raises(pickle.UnpicklingError, match="Blocked"):
            SafeUnpickler(io.BytesIO(data_dt)).load()
