"""Exercise the isolated skill example from declaration through a v2 goal."""

import importlib.util
import json
from pathlib import Path
import sys

from simvla import skills as _skills  # Populate built-in registry before the temporary addition.
from simvla.goal_validate import validate_goal
from simvla.skill_contract import REGISTRY, SKILL_ID
from simvla.task_template import TaskTemplate, TemplateStep, validate_template
from simvla.task_validate import validate_sequence


ROOT = Path(__file__).resolve().parents[1]


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_new_planned_skill_through_template_and_goal(tmp_path):
    before = dict(REGISTRY)
    try:
        extension = _load_file("simvla_example_gripper_open",
                               ROOT / "examples/skills/gripper_open.py")
        spec = REGISTRY["gripper.open"]
        assert spec.actions == ("G_r", "G_l")
        assert not spec.is_runtime
        assert "gripper.open" in SKILL_ID()

        task = TaskTemplate(
            name="open_right_gripper",
            language="Open right gripper.",
            roles=[],
            steps=[TemplateStep("gripper.open", "G_r", {}, "Open right gripper")],
            success={"gripper_open": {"arm": "right", "gap": 0.079}},
        )
        validate_template(task)
        validate_sequence(task)

        output = tmp_path / "goal.json"
        goal = extension.GripperOpen().plan(None, "G_r", {})
        output.write_text(json.dumps({
            "version": 2,
            "task_name": task.name,
            "goals": [[{
                "skill": "gripper.open", "action": "G_r", "params": {},
                "goal": goal, "language": "Open right gripper",
            }]],
        }))
        assert json.loads(output.read_text())["goals"][0][0]["goal"] is False
        assert validate_goal(output)["version"] == 2
    finally:
        REGISTRY.clear()
        REGISTRY.update(before)
