"""Check capture placement without requiring an Isaac Sim process."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recorder_class():
    torch = pytest.importorskip("torch")
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "LeRobotPostStepActionsRecorder")
    class Base:
        def __init__(self, cfg, env):
            self._env = env
    namespace = {"RecorderTerm": Base, "os": os, "torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[cls.name], torch


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_opt_in_records_cpu_substeps_without_changing_final_targets(recorder_class, monkeypatch, enabled):
    cls, torch = recorder_class
    monkeypatch.setenv("SIMVLA_RECORD_JOINT_SUBSTEPS", enabled)
    targets = torch.arange(10.).reshape(2, 5)
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=3),
                          action_manager=SimpleNamespace(action_absolute={"joint_pos": targets}),
                          _recorded_joint_substeps=[targets + i for i in range(3)])
    recorder = cls(None, env)
    key, result = recorder.record_post_step()
    assert key == "actions"
    assert torch.equal(result["joint_pos"], targets)
    if enabled == "1":
        assert result["joint_substeps"].shape == (2, 3, 5)
        assert result["joint_substeps"].device.type == "cpu"
        assert torch.equal(result["joint_substeps"][:, 2], targets + 2)
    else:
        assert "joint_substeps" not in result


def test_recorder_rejects_incomplete_stream(recorder_class, monkeypatch):
    cls, _ = recorder_class
    monkeypatch.setenv("SIMVLA_RECORD_JOINT_SUBSTEPS", "1")
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=3),
                          action_manager=SimpleNamespace(action_absolute={}),
                          _recorded_joint_substeps=[])
    with pytest.raises(RuntimeError, match="Incomplete"):
        cls(None, env).record_post_step()


def test_recorder_rejects_invalid_flag(recorder_class, monkeypatch):
    cls, _ = recorder_class
    monkeypatch.setenv("SIMVLA_RECORD_JOINT_SUBSTEPS", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        cls(None, SimpleNamespace())


def test_substep_capture_is_after_actuator_write_and_before_simulation():
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/envs/manager_based_rl_env.py"
    source = path.read_text()
    start = source.index("for _ in range(self.cfg.decimation)")
    write = source.index("self.scene.write_data_to_sim()", start)
    capture = source.index("self._recorded_joint_substeps.append", start)
    simulate = source.index("self.sim.step(render=False)", start)
    assert write < capture < simulate
    ast.parse(source)


def test_substep_cursor_rewinds_at_each_control_step():
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/managers/action_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActionManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "process_action")
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "self._joint_position_replay_substep = 0"
               for n in ast.walk(method))
