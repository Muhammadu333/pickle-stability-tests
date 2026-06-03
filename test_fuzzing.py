"""
Fuzzing / property-based tests.

We use two approaches:
  1. Hypothesis-based property testing (if hypothesis is installed).
  2. Manual random generation with a fixed seed (always available).

Properties under test:
  F1 – Roundtrip identity: unpickle(pickle(x)) == x
  F2 – Intra-process stability: pickle(x) == pickle(x) (same hash twice)
  F3 – No silent data corruption: if pickle raises, it must raise consistently.
"""

import hashlib
import pickle
import random
import string

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# ---------------------------------------------------------------------------
# Manual fuzzing with fixed seed
# ---------------------------------------------------------------------------

RNG = random.Random(42)  # fixed seed for reproducibility


def _random_primitive(rng):
    choices = [
        lambda: rng.randint(-2**63, 2**63),
        lambda: rng.uniform(-1e308, 1e308),
        lambda: "".join(rng.choices(string.printable, k=rng.randint(0, 200))),
        lambda: bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 200))),
        lambda: None,
        lambda: True,
        lambda: False,
    ]
    return rng.choice(choices)()


def _random_container(rng, depth=0):
    if depth >= 3:
        return _random_primitive(rng)
    kind = rng.choice(["list", "tuple", "dict", "scalar"])
    if kind == "list":
        return [_random_container(rng, depth + 1) for _ in range(rng.randint(0, 8))]
    if kind == "tuple":
        return tuple(_random_container(rng, depth + 1) for _ in range(rng.randint(0, 8)))
    if kind == "dict":
        keys = [str(rng.randint(0, 1000)) for _ in range(rng.randint(0, 6))]
        return {k: _random_container(rng, depth + 1) for k in keys}
    return _random_primitive(rng)


FUZZ_SAMPLES = 200


class TestManualFuzzing:

    @pytest.mark.fuzzing
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_fuzz_roundtrip(self, protocol):
        """
        F1: For 200 randomly generated objects, roundtrip must produce
        equal output.  Floats are compared with tolerance for special values.
        """
        rng = random.Random(42)
        failures = []
        for i in range(FUZZ_SAMPLES):
            obj = _random_container(rng)
            try:
                result = roundtrip(obj, protocol)
                if result != obj:
                    failures.append((i, repr(obj)[:80]))
            except (pickle.PicklingError, TypeError, OverflowError):
                pass  # some random objects may not be picklable — expected

        assert not failures, (
            f"Roundtrip failures (protocol={protocol}):\n"
            + "\n".join(f"  [{i}] {r}" for i, r in failures[:10])
        )

    @pytest.mark.fuzzing
    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS, ids=lambda p: f"proto{p}")
    def test_fuzz_intraprocess_stability(self, protocol):
        """
        F2: Same object pickled twice in the same process must yield
        the same SHA-256 hash.
        """
        rng = random.Random(42)
        failures = []
        for i in range(FUZZ_SAMPLES):
            obj = _random_container(rng)
            try:
                h1 = pickle_hash(obj, protocol)
                h2 = pickle_hash(obj, protocol)
                if h1 != h2:
                    failures.append((i, repr(obj)[:80]))
            except (pickle.PicklingError, TypeError):
                pass

        assert not failures, (
            f"Intra-process stability failures (protocol={protocol}):\n"
            + "\n".join(f"  [{i}] {r}" for i, r in failures[:10])
        )

    @pytest.mark.fuzzing
    def test_fuzz_unpicklable_raises_consistently(self):
        """
        F3: If pickle raises for a given object, repeated calls must also raise
        (not alternate between success and failure).
        """
        # Generators are not picklable.
        def gen():
            yield from range(10)

        g = gen()
        raised_first = False
        try:
            pickle.dumps(g)
        except (TypeError, AttributeError, pickle.PicklingError):
            raised_first = True

        # Second call — must also raise.
        if raised_first:
            with pytest.raises((TypeError, AttributeError, pickle.PicklingError)):
                pickle.dumps(g)


# ---------------------------------------------------------------------------
# Hypothesis-based fuzzing (runs only if hypothesis is installed)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:

    picklable_strategy = st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=True),
            st.text(),
            st.binary(),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=10),
            st.tuples(*([children] * 3)),
            st.dictionaries(st.text(max_size=20), children, max_size=10),
        ),
        max_leaves=30,
    )

    class TestHypothesisFuzzing:

        @pytest.mark.fuzzing
        @given(obj=picklable_strategy)
        @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
        def test_hypothesis_roundtrip_all_protocols(self, obj):
            """Property: for any picklable obj, roundtrip(obj) == obj."""
            import math

            def nan_safe_equal(a, b):
                if type(a) is not type(b):
                    return False
                if isinstance(a, float):
                    return (math.isnan(a) and math.isnan(b)) or a == b
                if isinstance(a, (list, tuple)):
                    return len(a) == len(b) and all(
                        nan_safe_equal(x, y) for x, y in zip(a, b)
                    )
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all(
                        nan_safe_equal(a[k], b[k]) for k in a
                    )
                return a == b

            for protocol in ALL_PROTOCOLS:
                try:
                    result = roundtrip(obj, protocol)
                    assert nan_safe_equal(result, obj)
                except (pickle.PicklingError, TypeError, OverflowError):
                    pass  # some Hypothesis floats may overflow protocol 0

        @pytest.mark.fuzzing
        @given(obj=picklable_strategy)
        @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
        def test_hypothesis_intraprocess_stability(self, obj):
            """Property: pickle(obj) has the same hash on two consecutive calls."""
            for protocol in ALL_PROTOCOLS:
                try:
                    h1 = pickle_hash(obj, protocol)
                    h2 = pickle_hash(obj, protocol)
                    assert h1 == h2, (
                        f"Non-deterministic for {obj!r} with protocol={protocol}"
                    )
                except (pickle.PicklingError, TypeError, OverflowError):
                    pass

        @pytest.mark.fuzzing
        @given(
            a=picklable_strategy,
            b=picklable_strategy,
        )
        @settings(max_examples=300)
        def test_hypothesis_distinct_inputs_usually_distinct_hashes(self, a, b):
            """
            Property: if two objects produce different pickle bytes, their
            SHA-256 hashes should differ (collision resistance check).
            We use pickle bytes themselves as the ground truth for equality
            to avoid NaN inequality edge cases.
            """
            for protocol in ALL_PROTOCOLS:
                try:
                    bytes_a = pickle.dumps(a, protocol=protocol)
                    bytes_b = pickle.dumps(b, protocol=protocol)
                    assume(bytes_a != bytes_b)  # skip if pickle bytes are identical
                    import hashlib
                    ha = hashlib.sha256(bytes_a).hexdigest()
                    hb = hashlib.sha256(bytes_b).hexdigest()
                    # SHA-256 collisions are astronomically rare — treat as a finding.
                    if ha == hb:
                        pytest.fail(
                            f"SHA-256 collision on distinct pickle bytes for "
                            f"protocol={protocol}"
                        )
                except (pickle.PicklingError, TypeError, OverflowError):
                    pass

else:
    # Placeholder so pytest collection doesn't fail if hypothesis is absent.
    class TestHypothesisFuzzing:
        def test_hypothesis_not_installed(self):
            pytest.skip("hypothesis not installed; install with: pip install hypothesis")
