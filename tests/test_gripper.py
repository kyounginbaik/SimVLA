import pytest

from simvla.gripper import resolve_pad_joint_indices


def test_interleaved_joint_order_keeps_each_pad_with_its_own_jaw():
    assert resolve_pad_joint_indices(["r1", "l1", "r2", "l2"],
                                     (("r1", "r2"), ("l1", "l2"))) == ((0, 2), (1, 3))


@pytest.mark.parametrize("names,groups", [
    (["a", "b"], ((), ())),
    (["a", "b"], (("a",), ("a",))),
    (["a", "b"], (("a",), ("c",))),
    (["a", "b"], (("a",),)),
    (["a", "a"], (("a",), ("b",))),
])
def test_invalid_or_partial_pad_mapping_is_rejected(names, groups):
    with pytest.raises(ValueError):
        resolve_pad_joint_indices(names, groups)
