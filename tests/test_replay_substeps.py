import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/simvla"))
from replay_substeps import load_joint_substeps


@pytest.fixture
def dataset(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    np.save(meta / "motors.npy", np.arange(12, dtype=np.float32).reshape(2, 3, 2))
    manifest = {"schema_version": 1, "control_fps": 20, "joint_names": ["left", "right"],
                "episodes": [{"file": "motors.npy", "frames": 2, "substeps": 3}]}
    (meta / "joint_substeps.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def load(root):
    return load_joint_substeps(root, 0, ["right", "left"], frames=2, decimation=3, control_fps=20)


def test_named_motor_order_and_substep_order(dataset):
    root, _ = dataset
    assert load(root).tolist() == [[[1, 0], [3, 2], [5, 4]], [[7, 6], [9, 8], [11, 10]]]


@pytest.mark.parametrize("change", ["fps", "names", "frames", "substeps", "escape", "missing"])
def test_bad_metadata_rejected(dataset, change):
    root, manifest = dataset
    if change == "fps": manifest["control_fps"] = 30
    elif change == "names": manifest["joint_names"] = ["left", "left"]
    elif change == "frames": manifest["episodes"][0]["frames"] = 3
    elif change == "substeps": manifest["episodes"][0]["substeps"] = 6
    elif change == "escape": manifest["episodes"][0]["file"] = "../../motors.npy"
    else: manifest["episodes"][0] = None
    (root / "meta/joint_substeps.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load(root)


@pytest.mark.parametrize("bad", [np.full((2, 3, 2), np.nan), np.zeros((2, 2, 2)), np.zeros((2, 3, 2), dtype=int)])
def test_bad_array_rejected(dataset, bad):
    root, _ = dataset
    np.save(root / "meta/motors.npy", bad)
    with pytest.raises(ValueError): load(root)


@pytest.mark.parametrize("high_rate", [False, True])
@pytest.mark.parametrize("settling", [False, True])
def test_replay_motor_dispatch_preserves_simvqa_target_map(high_rate, settling):
    """Exercise the actual dispatch block: motor tensors must not shadow VQA targets."""
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock

    torch = pytest.importorskip("torch")
    source = Path(__file__).parents[1] / "scripts/simvla/simvla_replay.py"
    tree = ast.parse(source.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "joint_targets is not None"
                  and any(isinstance(child, ast.Name) and child.id == "recorded_idx"
                          for child in ast.walk(node)))
    setter = Mock()
    target_map = {0: "mug"}
    namespace = dict(
        targets=target_map, joint_targets=torch.zeros(2, 4),
        motor_substeps=torch.arange(48).reshape(2, 6, 4).float() if high_rate else None,
        recorded_idx=0, settling=settling, substep_start=0, num_envs=1, replay_joint_ids=[0, 1, 2, 3],
        env=SimpleNamespace(action_manager=SimpleNamespace(set_joint_position_replay_targets=setter)),
    )
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["targets"] is target_map
    assert namespace["targets"].get(0) == "mug"
    assert setter.call_args.args[0].shape == ((1, 6, 4) if high_rate and not settling else (1, 4))
    if high_rate and settling:
        assert torch.equal(setter.call_args.args[0], torch.tensor([[20., 21., 22., 23.]]))
