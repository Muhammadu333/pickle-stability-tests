# Pickle Stability Test Suite

A comprehensive test suite for testing the **stability and correctness** of Python's `pickle` module.

> **Core question:** Does the same input always produce the same (hash-identical) pickle output under all circumstances?

## Findings

| Type | Stable across processes? | Cause |
|---|---|---|
| `int`, `float`, `str`, `bytes` | ✅ Yes | No hash randomisation |
| `list`, `tuple`, `dict` | ✅ Yes | Ordered / insertion-ordered |
| `frozenset` (integers only) | ✅ Yes | Integer hash is deterministic |
| **`set` (string elements)** | ❌ **No** | `PYTHONHASHSEED` randomises order |
| **`frozenset` (string elements)** | ❌ **No** | Same root cause |
| Same object, different `protocol=` | ❌ **No** | Different binary format by design |
| Nesting depth ≥ ~500 | ❌ **No** | `RecursionError` — no output produced |

## Test Structure

| File | Technique |
|---|---|
| `test_blackbox.py` | Black-box testing (API contracts) |
| `test_whitebox.py` | White-box (all-def / all-uses, opcode analysis) |
| `test_equivalence_partitions.py` | Equivalence partitioning (13 type classes) |
| `test_boundary_values.py` | Boundary value analysis |
| `test_floating_point.py` | Float edge cases (NaN, -0.0, subnormals) |
| `test_recursive_structures.py` | Recursive / self-referential structures |
| `test_instabilities.py` | **Core findings** — confirmed instabilities |
| `test_hash_seed_instability.py` | PYTHONHASHSEED cross-process tests |
| `test_fuzzing.py` | Random fuzzing + Hypothesis property tests |
| `test_roundtrip.py` | Correctness / roundtrip fidelity |
| `test_determinism.py` | Intra-process hash-identity |
| `test_cross_platform.py` | Cross-platform reference hash comparison |

## How to Run

```bash
pip install pytest hypothesis
pytest                        # full suite
pytest test_instabilities.py  # just the findings
pytest -m stability           # only stability tests
pytest -m boundary            # only boundary value tests
```

## Cross-Platform

This repo uses GitHub Actions to run on **Ubuntu, macOS, and Windows** across **Python 3.9–3.12** automatically on every push. See the [Actions tab](../../actions) for results.

To generate reference hashes for your platform:
```bash
pytest test_cross_platform.py --generate-refs
```

## Requirements

- Python 3.9+
- `pytest`, `hypothesis`
