import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_initial_snapshot_after_manual_retry_is_not_duplicated():
    path = Path(__file__).parents[1] / "scripts/simvla/recording_state.py"
    spec = importlib.util.spec_from_file_location("recording_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episodes = {0: SimpleNamespace(data={"initial_state": {}}),
                1: SimpleNamespace(data={})}
    calls = []

    def record(ids):
        calls.append(ids)
        for index in ids:
            episodes[index].data["initial_state"] = {}

    manager = SimpleNamespace(get_episode=episodes.__getitem__, record_post_reset=record)
    module.ensure_initial_snapshot(manager, [0, 1])
    module.ensure_initial_snapshot(manager, [0, 1])
    module.ensure_initial_snapshot(manager, [])
    assert calls == [[1]]


def test_snapshot_capture_follows_mid_iteration_retry_and_precedes_physics():
    path = Path(__file__).parents[1] / "scripts/simvla/simvla_gen.py"
    source = path.read_text()
    assert source.index("if post_lift_failed_envs:") < source.index(
        "ensure_initial_snapshot(env.recorder_manager") < source.index("obv = env.step(")
    assert "torch.where(env.episode_length_buf == 0)" in source


def test_capture_uses_physics_episode_boundary_not_script_counter():
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[1] / "scripts/simvla/simvla_gen.py"
    tree = ast.parse(path.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_initial_env_ids" for t in n.targets))
    capture = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name) and n.target.id == "_env_id")
    obj = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=torch.tensor([[[1., 2., 3.]], [[11., 2., 3.]], [[21., 2., 3.]]]),
        body_quat_w=torch.tensor([[[1., 0., 0., 0.]]] * 3)))
    saved = {1: "unchanged"}
    namespace = dict(torch=torch, timestep=torch.tensor([0, 1, 9]),
                     env=SimpleNamespace(episode_length_buf=torch.tensor([0, 4, 0]),
                         scene=SimpleNamespace(rigid_objects={"mug0": obj},
                             env_origins=torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]]))),
                     initial_objects_by_env=saved)
    exec(compile(ast.Module(body=[assignment, capture], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_initial_env_ids"] == [0, 2]
    assert saved == {0: {"mug0": [1., 2., 3., 1., 0., 0., 0.]},
                     1: "unchanged", 2: {"mug0": [1., 2., 3., 1., 0., 0., 0.]}}
