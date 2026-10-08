import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/simvla"))
from replay_initial_step import load_initial_step


@pytest.fixture
def dataset(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    step = {"schema_version": 1, "initial_state_phase": "before_first_action",
            "raw_action": [0., 1., -1.], "subtask_index": 0,
            "motor_targets": [10., 20.], "motor_substeps": [[1., 2.], [3., 4.]]}
    states = {"schema_version": 1, "frame": "environment",
              "episodes": [{"initial": {"articulation": {}}, "initial_step": step}]}
    (meta / "scene_states.json").write_text(json.dumps(states))
    (meta / "info.json").write_text(json.dumps({"features": {
        "action.joint": {"shape": [2], "names": {"action.joint": ["a", "b"]}}}}))
    return tmp_path, states


def load(root):
    return load_initial_step(root, 0, action_dim=3, joint_names=["b", "a"], decimation=2)


def test_prelude_preserves_raw_control_and_orders_motor_columns(dataset):
    root, _ = dataset
    assert load(root) == {"raw_action": [0., 1., -1.], "subtask_index": 0,
                          "motor_targets": [20., 10.], "motor_substeps": [[2., 1.], [4., 3.]]}


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("initial_state_phase", "after_first_action"),
    ("raw_action", [0., 1.]), ("raw_action", [0., float("nan"), 1.]),
    ("raw_action", [True, 1., 0.]), ("subtask_index", -1), ("subtask_index", True),
    ("motor_targets", [0.]), ("motor_substeps", [[1., 2.]]),
    ("motor_substeps", [[1., 2.], [3., float("inf")]]),
])
def test_corrupt_prelude_is_not_replayed(dataset, field, value):
    root, states = dataset
    states["episodes"][0]["initial_step"][field] = value
    (root / "meta/scene_states.json").write_text(json.dumps(states))
    with pytest.raises(ValueError):
        load(root)


@pytest.mark.parametrize("missing", ["initial", "initial_step"])
def test_old_or_sliced_episode_requires_recollection(dataset, missing):
    root, states = dataset
    del states["episodes"][0][missing]
    (root / "meta/scene_states.json").write_text(json.dumps(states))
    with pytest.raises(ValueError, match="recollect"):
        load(root)


def test_named_motor_contract_is_checked(dataset):
    root, _ = dataset
    with pytest.raises(ValueError, match="joint names"):
        load_initial_step(root, 0, action_dim=3, joint_names=["a", "a"], decimation=2)


def test_replay_runs_prelude_after_each_reset_and_before_exported_actions():
    source = (Path(__file__).parents[1] / "scripts/simvla/simvla_replay.py").read_text()
    assert source.count("    run_initial_step()") == 2
    assert source.index("    run_initial_step()") < source.index("    action_data =")
    last_reset = source.rindex("                env.reset()")
    assert source.index("                run_initial_step()", last_reset) > source.index(
        'env.action_manager.get_term("armR_action")._ik_controller.reset()', last_reset)


@pytest.mark.parametrize("motor_mode", ["none", "control", "physics"])
def test_actual_prelude_dispatch_steps_physics_once(dataset, motor_mode):
    import ast
    from types import SimpleNamespace
    from unittest.mock import Mock
    torch = pytest.importorskip("torch")
    root, _ = dataset
    path = Path(__file__).parents[1] / "scripts/simvla/simvla_replay.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "run_initial_step")
    method.decorator_list = []
    manager = SimpleNamespace(set_joint_position_replay_targets=Mock())
    shared = SimpleNamespace(env_goal_indices=torch.tensor([-1]))
    def check_step(action, goal):
        assert torch.equal(shared.env_goal_indices, goal)
        assert shared.env_goal_indices.device == action.device
    env = SimpleNamespace(action_manager=manager, step=Mock(side_effect=check_step),
                          reset_buf=torch.tensor([False]))
    namespace = dict(initial_step=load(root), torch=torch, device="cpu", num_envs=1,
                     replay_robot=object(), replay_lift_ids=None, hold_uncontrolled_lift=Mock(),
                     joint_targets=None if motor_mode == "none" else object(),
                     substep_replay=motor_mode == "physics", replay_joint_ids=[0], env=env,
                     variable=shared)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    namespace["run_initial_step"]()
    env.step.assert_called_once()
    action, index = env.step.call_args.args
    assert action.tolist() == [[0., 1., -1.]] and index.tolist() == [0]
    if motor_mode == "none":
        manager.set_joint_position_replay_targets.assert_not_called()
    else:
        args, kwargs = manager.set_joint_position_replay_targets.call_args
        assert kwargs == {"joint_ids": [0]}
        assert args[0].tolist() == ([[[2., 1.], [4., 3.]]] if motor_mode == "physics"
                                    else [[20., 10.]])
    env.reset_buf[:] = True
    with pytest.raises(RuntimeError, match="prelude"):
        namespace["run_initial_step"]()
