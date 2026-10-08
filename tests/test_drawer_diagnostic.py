import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("drawer_diagnostic", ROOT / "scripts/tools/make_anubis_drawer_diagnostic.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_waypoint_preserves_grasps_and_remaps_export_groups():
    original = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    result = module.with_drawer_waypoint(original)
    old = original["goals"][0]
    new = result["goals"][0]
    assert len(old) == 12 and len(new) == 13
    assert new[:4] == old[:4]
    assert new[5:] == old[4:]
    assert new[4][0] == "N_s"
    assert new[4][1][0] == pytest.approx(old[0][1][0] + .12)
    assert new[4][1][1:] == old[0][1][1:]
    assert result["run_config"]["export_groups"] == "4,5,6,7,8;9,10,11;12"
    with pytest.raises(ValueError, match="unmodified"):
        module.with_drawer_waypoint(result)


def test_generated_goal_loads_in_actual_executor(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/simvla"))
    import simvla.skills  # populate the real executor registry
    from executor_dispatch import load_script
    original = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    path = tmp_path / "goal.json"
    path.write_text(json.dumps(module.with_drawer_waypoint(original)))
    steps, _, _ = load_script(path)
    assert len(steps) == 13
    assert steps[4].action == "N_s"


def test_swapping_drawer_jaws_preserves_position_and_object_grasp():
    original = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    snapshot = json.dumps(original)
    result = module.with_drawer_waypoint(original, swap_drawer_jaws=True)
    assert json.dumps(original) == snapshot
    assert result["goals"][0][:4] == original["goals"][0][:4]
    for old_index, new_index in ((4, 5), (5, 6)):
        before = original["goals"][0][old_index][1]
        after = result["goals"][0][new_index][1]
        assert after[:3] == before[:3]
        w, x, y, z = before[3:]
        assert after[3:] == [-z, y, -x, w]
        # Tool-Z direction is invariant under a local-Z half-turn.
        def tool_z(q):
            w, x, y, z = q
            return (2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y))
        assert tool_z(after[3:]) == pytest.approx(tool_z(before[3:]))


def test_backoff_changes_only_added_base_waypoint():
    original = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    result = module.with_drawer_waypoint(original, .17, backoff_m=.05)
    old = original["goals"][0]
    new = result["goals"][0]
    assert new[:4] == old[:4] and new[5:] == old[4:]
    assert new[4][1][:2] == pytest.approx([old[0][1][0]+.17, old[0][1][1]-.05])


def test_metadata_upgrade_preserves_every_executor_payload(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/simvla"))
    import simvla.skills
    from executor_dispatch import load_script
    original = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.json").read_text())
    twin = json.loads((ROOT / "examples/goals/Isaac-Kitchen-v813-00.reloadable.json").read_text())
    legacy = module.with_drawer_waypoint(original, .17, swap_drawer_jaws=True, backoff_m=.05)
    upgraded = module.add_step_metadata(legacy, twin)
    scripts = []
    for name, goal in (("legacy", legacy), ("metadata", upgraded)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(goal))
        scripts.append(load_script(path)[0])
    for before, after in zip(*scripts):
        assert (before.action, before.skill, before.skill_id, before.spec) == (
            after.action, after.skill, after.skill_id, after.spec)
        assert after.language.strip()
    assert scripts[1][1].params["prim_path"] == "/world/bowl0"
