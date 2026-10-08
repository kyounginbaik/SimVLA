"""Grasp-only template generation. Pure. Run: cd scripts/simvla && pytest test_grasp_templates.py -v"""
import json

import pytest

from grasp_templates import build_grasp_template, write_grasp_template


def _skills(t):
    return [(s["skill"], s["action"]) for s in t["steps"]]


def test_single_graspable_uses_arm_grasp_then_close_then_lift():
    t = build_grasp_template("mug", "single")
    assert _skills(t) == [("nav.to_prim", "N_s"), ("arm.grasp", "A_r"),
                          ("gripper.set", "G_r"), ("arm.reset", "A_r")]


def test_single_clutter_uses_arm_squeeze_ar_like_vase_neck():
    t = build_grasp_template("cup", "single")
    assert _skills(t) == [("nav.to_prim", "N_s"), ("arm.squeeze", "A_r"),
                          ("gripper.set", "G_r"), ("arm.reset", "A_r")]


def test_bimanual_uses_arm_squeeze_ab_and_no_gripper_close():
    t = build_grasp_template("vase", "bimanual")
    assert _skills(t) == [("nav.to_prim", "N_s"), ("arm.squeeze", "A_b"), ("arm.reset", "A_b")]
    assert all(s["skill"] != "gripper.set" for s in t["steps"])


def test_target_role_and_scene_object_are_the_requested_type():
    t = build_grasp_template("jar", "single")
    target = next(r for r in t["roles"] if r["name"] == "target")
    assert target["object_type"] == "jar"
    assert any(o["object_type"] == "jar" for o in t["scene"])


def test_write_emits_a_named_file(tmp_path):
    p = write_grasp_template("vase", "bimanual", str(tmp_path))
    assert p.endswith("graspcamp_vase_bimanual.json")
    assert json.loads(open(p).read())["name"] == "graspcamp_vase_bimanual"


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_grasp_template("vase", "sideways")
