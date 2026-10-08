import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "planner_diagnostics", Path(__file__).resolve().parents[1]
    / "scripts/simvla/planner_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def reshape(self, *shape):
        return Tensor(self.value.reshape(*shape))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value.tolist()


def snapshot(position):
    spheres = Tensor([[0., 0., 0., .1], [1., 0., 0., .2]])
    kin = SimpleNamespace(
        kinematics_config=SimpleNamespace(
            link_name_to_idx_map={"hand": 7, "elbow": 3},
            link_sphere_idx_map=Tensor([3, 7])),
        get_state=lambda q: SimpleNamespace(get_link_spheres=lambda: spheres))
    joints = SimpleNamespace(position=Tensor(position), joint_names=["a", "b"])
    return module.failure_snapshot(kin, joints, arm="l", step=5, status="collision")


def test_failure_snapshot_preserves_joint_and_sphere_order():
    result = json.loads(snapshot([.1, .2]))
    assert result["joint_position"] == [.1, .2]
    assert result["sphere_links"] == ["elbow", "hand"]
    assert result["spheres_base_frame"][1] == [1., 0., 0., .2]
    assert result["step"] == 5


def test_failure_snapshot_rejects_nonfinite_start():
    with pytest.raises(ValueError):
        snapshot([float("nan"), 0.])


def test_overlap_ignores_are_symmetric_and_disabled_spheres_are_excluded():
    payload = {"sphere_links": ["a", "b", "c", "a"],
               "spheres_base_frame": [[0, 0, 0, .2], [.1, 0, 0, .2],
                                      [0, 0, 0, -1], [0, 0, 0, .2]]}
    overlaps = module.overlapping_link_pairs(payload, {})
    assert len(overlaps) == 1
    assert overlaps[0]["links"] == ["a", "b"]
    assert overlaps[0]["penetration_m"] == pytest.approx(.3)
    assert module.overlapping_link_pairs(payload, {"b": ["a"]}) == []
