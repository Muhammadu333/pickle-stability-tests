"""
Recursive and self-referential data structure tests.

Pickle handles recursive structures via a memo table.  The lab description
explicitly calls out recursive data structures (citing JSON) as a possible
instability vector.

We test:
  RS1 – Self-referential lists (list that contains itself)
  RS2 – Self-referential dicts
  RS3 – Mutually referential objects
  RS4 – Shared references (same object referenced twice)
  RS5 – Stability of shared-reference output across runs
"""

import pickle

import pytest

from utils import ALL_PROTOCOLS, pickle_hash, roundtrip


def _make_self_ref_list():
    lst = []
    lst.append(lst)
    return lst


def _make_self_ref_dict():
    d = {}
    d["self"] = d
    return d


class A:
    pass


class B:
    pass


def _make_mutual_ref():
    a = A()
    b = B()
    a.other = b
    b.other = a
    return a, b


class TestSelfReferential:

    @pytest.mark.correctness
    def test_self_referential_list_roundtrip(self, protocol):
        """pickle must handle and reconstruct a list that contains itself."""
        lst = _make_self_ref_list()
        data = pickle.dumps(lst, protocol=protocol)
        result = pickle.loads(data)
        assert result[0] is result  # the inner element is the list itself

    @pytest.mark.correctness
    def test_self_referential_dict_roundtrip(self, protocol):
        d = _make_self_ref_dict()
        data = pickle.dumps(d, protocol=protocol)
        result = pickle.loads(data)
        assert result["self"] is result

    @pytest.mark.stability
    def test_self_referential_list_stable(self, protocol):
        """Repeated dumps of a self-referential list must be hash-identical."""
        lst = _make_self_ref_list()
        h1 = pickle_hash(lst, protocol)
        h2 = pickle_hash(lst, protocol)
        assert h1 == h2

    @pytest.mark.stability
    def test_self_referential_dict_stable(self, protocol):
        d = _make_self_ref_dict()
        h1 = pickle_hash(d, protocol)
        h2 = pickle_hash(d, protocol)
        assert h1 == h2


class TestMutualReferences:

    @pytest.mark.correctness
    def test_mutual_ref_roundtrip(self, protocol):
        a, b = _make_mutual_ref()
        data = pickle.dumps((a, b), protocol=protocol)
        a2, b2 = pickle.loads(data)
        assert a2.other is b2
        assert b2.other is a2

    @pytest.mark.stability
    def test_mutual_ref_stable(self, protocol):
        a, b = _make_mutual_ref()
        h1 = pickle_hash((a, b), protocol)
        h2 = pickle_hash((a, b), protocol)
        assert h1 == h2


class TestSharedReferences:

    @pytest.mark.correctness
    def test_shared_reference_preserved(self, protocol):
        """
        When the same object appears twice in a container, pickle should
        reconstruct it as the same object (not two independent copies),
        using the memo mechanism.
        """
        shared = [1, 2, 3]
        container = [shared, shared]
        result = roundtrip(container, protocol)
        assert result[0] is result[1], (
            "Shared reference not preserved after unpickling"
        )

    @pytest.mark.stability
    def test_shared_reference_hash_stable(self, protocol):
        shared = [1, 2, 3]
        container = [shared, shared]
        h1 = pickle_hash(container, protocol)
        h2 = pickle_hash(container, protocol)
        assert h1 == h2

    @pytest.mark.correctness
    def test_shared_reference_vs_copy_differ(self, protocol):
        """
        A container with two references to the same object and a container
        with two independent equal objects may or may not have the same
        pickle bytes.  We document this protocol-specific behaviour.
        """
        shared = [1, 2, 3]
        independent = [1, 2, 3]

        container_shared = [shared, shared]
        container_copy = [independent, list(independent)]

        h_shared = pickle_hash(container_shared, protocol)
        h_copy = pickle_hash(container_copy, protocol)

        # Not asserting equality or inequality — just verifying no crash,
        # and documenting behaviour:
        if h_shared == h_copy:
            # Protocol did not encode identity — two equal lists look the same.
            pass
        else:
            # Protocol encoded the backreference — shared identity is visible.
            pass

    @pytest.mark.correctness
    def test_large_shared_dag(self, protocol):
        """
        A DAG with many shared nodes should be reconstructed correctly and
        not cause exponential blowup.
        """
        # Build a diamond DAG: root → (left, right) → shared_leaf
        leaf = {"data": list(range(100))}
        left = {"leaf": leaf, "tag": "left"}
        right = {"leaf": leaf, "tag": "right"}
        root = {"left": left, "right": right, "direct_leaf": leaf}

        result = roundtrip(root, protocol)
        # All three references should resolve to the same reconstructed object.
        assert result["left"]["leaf"] is result["right"]["leaf"]
        assert result["left"]["leaf"] is result["direct_leaf"]
