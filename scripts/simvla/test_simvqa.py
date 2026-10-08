"""SimVQA: the per-frame object sampling, the exact-name segmentation lookup, and the Q/A expander.

Two modules under test:

- ``scripts/simvla/simvqa.py`` (the expander) imports cleanly.
- ``source/isaaclab/isaaclab/simvla/simvqa.py`` (capture helpers) only needs json/os/numpy/torch/
  PIL, but ``import isaaclab.simvla.simvqa`` drags ``isaaclab/__init__`` -> isaacsim in. So it is
  loaded straight from its file path, bypassing the package.
"""

from __future__ import annotations

import collections
import importlib.util
import math
import json
import random
from pathlib import Path

import pytest
import torch

import simvqa as expander

_LIB = Path(__file__).resolve().parents[2] / "source/isaaclab/isaaclab/simvla/simvqa.py"
_spec = importlib.util.spec_from_file_location("simvqa_lib", _LIB)
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)


@pytest.mark.parametrize(
    ("robot", "right_gripper", "left_gripper", "closed_travel"),
    [
        ("anubis", "gripper1R_joint", "gripper2R_joint", 0.002),
        ("rby1", "gripper_finger_r1", "gripper_finger_l1", 0.002),
        ("aiworker", "gripper_r_joint1", "gripper_l_joint1", 0.02),
    ],
)
def test_replay_vqa_calibration_matches_each_collection_robot(
    robot, right_gripper, left_gripper, closed_travel
):
    calibration = json.loads(
        (Path(__file__).parent / "calibration" / f"{robot}.json").read_text()
    )["vqa"]
    assert calibration["base_body"] == "base_link"
    assert calibration["ee_links"] == {"left": "ee_link2", "right": "ee_link1"}
    assert calibration["gripper_joints"] == {
        "left": left_gripper, "right": right_gripper,
    }
    assert calibration["gripper_closed_travel"] == pytest.approx(closed_travel)


# ----------------------------------------------------------------------------- capture helpers

def test_find_instance_key_matches_a_whole_prim_segment_not_a_substring():
    info = {"instance_id_segmentation_fast": {"idToLabels": {
        "(1,0,0,255)": "/World/envs/env_0/Kitchen/sodacan0/geometry",
        "(2,0,0,255)": "/World/envs/env_0/Kitchen/can0/geometry",
    }}}
    key, _ = lib.find_instance_key_by_name(info, "can0", eid=0)
    assert key == "(2,0,0,255)"


def test_find_instance_key_does_not_confuse_env_1_with_env_10():
    info = {"instance_id_segmentation_fast": {"idToLabels": {
        "(1,0,0,255)": "/World/envs/env_10/Kitchen/mug0/geometry",
        "(2,0,0,255)": "/World/envs/env_1/Kitchen/mug0/geometry",
    }}}
    key, _ = lib.find_instance_key_by_name(info, "mug0", eid=1)
    assert key == "(2,0,0,255)"


def test_find_instance_key_returns_none_when_absent():
    info = {"instance_id_segmentation_fast": {"idToLabels": {
        "(1,0,0,255)": "/World/envs/env_0/Kitchen/mug0/geometry",
    }}}
    assert lib.find_instance_key_by_name(info, "apple0", eid=0) == (None, None)


def test_sample_frame_objects_puts_the_target_first_and_returns_k_unique_entries():
    rng = random.Random(0)
    out = lib.sample_frame_objects("mug0", ["mug0", "bottle0", "jar0", "Countertop"],
                                   vocab=("mug", "bottle", "jar", "apple", "bowl"), k=3, rng=rng)
    assert len(out) == 3
    assert out[0] == {"name": "mug0", "present": True}
    assert len({o["name"] for o in out}) == 3
    assert all(o["name"] != "Countertop" for o in out)   # not a vocab type -> not an object


def test_sample_frame_objects_fills_with_absent_types_when_the_scene_is_sparse():
    rng = random.Random(0)
    out = lib.sample_frame_objects("mug0", ["mug0"], vocab=("mug", "apple", "bowl"), k=3, rng=rng)
    absent = [o for o in out[1:] if not o["present"]]
    assert len(absent) == 2
    assert {o["name"] for o in absent} == {"apple", "bowl"}   # never the target's own type


def test_sample_frame_objects_mixes_present_and_absent_over_many_draws():
    seen_present = seen_absent = False
    for seed in range(50):
        out = lib.sample_frame_objects("mug0", ["mug0", "bottle0", "jar0", "bowl0"],
                                       vocab=("mug", "bottle", "jar", "bowl", "apple", "can"),
                                       k=3, rng=random.Random(seed))
        for o in out[1:]:
            seen_present |= o["present"]
            seen_absent |= not o["present"]
    assert seen_present and seen_absent


def test_sample_subtask_frames_always_includes_the_last_frame_of_the_subtask():
    for seed in range(20):
        picked = lib.sample_subtask_frames([10, 11, 12, 13, 14, 15], 2, rng=random.Random(seed))
        assert 15 in picked
        assert len(picked) == len(set(picked)) == 2


def test_sample_subtask_frames_caps_at_the_candidate_count():
    assert sorted(lib.sample_subtask_frames([3, 4], 5, rng=random.Random(0))) == [3, 4]
    assert lib.sample_subtask_frames([], 2, rng=random.Random(0)) == []


def test_flush_writes_the_objects_list_alongside_the_legacy_fields(tmp_path):
    img = torch.zeros((4, 4, 4), dtype=torch.uint8)
    rec = {
        "t": 7, "goal_idx": 1, "obj_name": "mug0", "subtask": "reach for the mug",
        "images": {c: img for c in ("front", "wrist_left", "wrist_right")},
        "seg_images": {c: img for c in ("front", "wrist_left", "wrist_right")},
        "bboxes": {"front": "<loc0001><loc0002><loc0003><loc0004>", "wrist_left": None, "wrist_right": None},
        "distance_base": torch.tensor([0.1, 0.2, 0.3]),
        "goal_state_mobile": None, "goal_state_gripper": None,
        "left_gripper": False, "right_gripper": True, "gripper": None,
        "objects": [
            {"name": "mug0", "present": True,
             "bboxes": {"front": "<loc0001><loc0002><loc0003><loc0004>", "wrist_left": None, "wrist_right": None},
             "distance_base": torch.tensor([0.1, 0.2, 0.3])},
            {"name": "apple", "present": False, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
             "distance_base": None},
        ],
    }
    out = tmp_path / "vqa.jsonl"
    lib.flush_simvqa_on_success(0, {"env_00": [rec]}, str(out), str(tmp_path / "img"), str(tmp_path / "seg"), "Put mug to sink.")
    row = json.loads(out.read_text().splitlines()[0])
    assert row["front_bbox"] == "<loc0001><loc0002><loc0003><loc0004>"
    assert row["meta"]["obj_name"] == "mug0"
    assert row["objects"][0]["name"] == "mug0"
    assert row["objects"][0]["distance_base"] == pytest.approx([0.1, 0.2, 0.3])
    assert row["objects"][1] == {"name": "apple", "present": False,
                                 "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
                                 "distance_base": None}


# ----------------------------------------------------------------------------- expander

SUBTASKS = ["move to the mug", "reach for the mug", "grasp the mug"]
BB = "<loc0100><loc0200><loc0300><loc0400>"
BBW = "<loc0010><loc0020><loc0030><loc0040>"


def _row(**meta):
    m = {"goal_idx": 0, "obj_name": "mug0", "gripper": None, "r_gripper": False, "l_gripper": False,
         "goal_state_gripper": None, "goal_state_mobile": None, "distance_base": [0.5, 0.1, -0.2],
         "grasp_target": True}
    m.update(meta)
    return {"id": "env_00_t0007", "prompt": "Put mug to sink.", "images": {}, "seg_images": {},
            "target_text": "Subtask: move to the mug", "front_bbox": BB, "wrist_left_bbox": None,
            "wrist_right_bbox": BBW, "meta": m}


def _qa(examples, fn):
    return [(e["cotrain_prompt"], e["cotrain_answer"]) for e in examples if e["vqa_function"] == fn]


def test_high_level_subtask_question_uses_the_template_wording():
    ex = expander.build_examples_for_row(_row(), SUBTASKS, Path("/x"))
    (q, a), = _qa(ex, "high_level_subtask")
    assert q == 'Question: To complete the task "Put mug to sink.", what should the robot do now?\nAnswer:'
    assert a == "move to the mug"


def test_object_reachable_answers_yes_inside_the_nav_tolerance():
    ex = expander.build_examples_for_row(_row(goal_state_mobile=[0.01, -0.005, 0.05]), SUBTASKS, Path("/x"))
    (_, a), = _qa(ex, "object_reachable")
    assert a == "yes"


def test_object_reachable_answers_no_with_a_move_outside_the_nav_tolerance():
    ex = expander.build_examples_for_row(_row(goal_state_mobile=[0.4, -0.1, 0.05]), SUBTASKS, Path("/x"))
    (_, a), = _qa(ex, "object_reachable")
    assert a == "no. move base dx 0.4 dy -0.1 dyaw 0.05"


def test_object_graspable_answers_yes_inside_the_grasp_tolerance():
    ex = expander.build_examples_for_row(
        _row(goal_idx=1, gripper="left", goal_state_gripper=[0.005, 0.0, -0.01, 0.9995, 0.03, 0.0, 0.0]),
        SUBTASKS, Path("/x"))
    (q, a), = _qa(ex, "object_graspable")
    assert "left gripper" in q
    assert a == "yes"


def test_object_graspable_answers_no_when_the_rotation_alone_is_off():
    # 0.3 rad about z: qw = cos(0.15), qz = sin(0.15)
    ex = expander.build_examples_for_row(
        _row(goal_idx=1, gripper="left", goal_state_gripper=[0.0, 0.0, 0.0, 0.98877, 0.0, 0.0, 0.14944]),
        SUBTASKS, Path("/x"))
    (_, a), = _qa(ex, "object_graspable")
    assert a.startswith("no. move gripper dx 0 dy 0 dz 0 dquat 0.9888 0 0 0.1494")


def test_per_object_rows_ask_detection_about_every_sampled_object():
    row = _row()
    row["objects"] = [
        {"name": "mug0", "present": True, "bboxes": {"front": BB, "wrist_left": None, "wrist_right": BBW},
         "distance_base": [0.5, 0.1, -0.2]},
        {"name": "bottle0", "present": True, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
         "distance_base": [1.0, 0.0, 0.0]},
        {"name": "apple", "present": False, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
         "distance_base": None},
    ]
    ex = expander.build_examples_for_row(row, SUBTASKS, Path("/x"))
    det = dict(_qa(ex, "object_detection"))
    assert det == {
        "Question: Is the mug visible in the front camera?\nAnswer:": f"yes. bbox: {BB}",
        "Question: Is the bottle visible in the front camera?\nAnswer:": "no",
        "Question: Is the apple visible in the front camera?\nAnswer:": "no",
    }
    ids = [e["id"] for e in ex if e["vqa_function"] == "object_detection"]
    assert len(ids) == len(set(ids)) == 3


def test_per_object_rows_give_3d_info_only_for_present_objects():
    row = _row()
    row["objects"] = [
        {"name": "mug0", "present": True, "bboxes": {"front": BB, "wrist_left": None, "wrist_right": BBW},
         "distance_base": [0.5, 0.1, -0.2]},
        {"name": "bottle0", "present": True, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
         "distance_base": [1.0, 0.0, 0.0]},
        {"name": "apple", "present": False, "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
         "distance_base": None},
    ]
    ex = expander.build_examples_for_row(row, SUBTASKS, Path("/x"))
    info = dict(_qa(ex, "object_3d_information"))
    assert info == {
        "Question: How far is the mug from the robot base?\nAnswer:": "dx 0.5 dy 0.1 dz -0.2",
        "Question: How far is the bottle from the robot base?\nAnswer:": "dx 1 dy 0 dz 0",
    }
    corr = _qa(ex, "view_correspondence")
    assert corr == [(f"Question: Given front bbox {BB}, what are the wrist camera bboxes?\nAnswer:",
                     f"left none right {BBW}")]


def test_reachable_and_graspable_are_asked_only_about_the_target():
    row = _row(goal_state_mobile=[0.4, 0.0, 0.0])
    row["objects"] = [
        {"name": "mug0", "present": True, "bboxes": {"front": BB, "wrist_left": None, "wrist_right": None},
         "distance_base": [0.5, 0.1, -0.2]},
        {"name": "bottle0", "present": True, "bboxes": {"front": BB, "wrist_left": None, "wrist_right": None},
         "distance_base": [1.0, 0.0, 0.0]},
    ]
    ex = expander.build_examples_for_row(row, SUBTASKS, Path("/x"))
    (q, _), = _qa(ex, "object_reachable")
    assert "the mug" in q


def test_legacy_rows_without_objects_still_expand_from_the_top_level_fields():
    ex = expander.build_examples_for_row(_row(), SUBTASKS, Path("/x"))
    assert dict(_qa(ex, "object_detection")) == {
        "Question: Is the mug visible in the front camera?\nAnswer:": f"yes. bbox: {BB}"}
    assert len(_qa(ex, "object_3d_information")) == 1
    assert len(_qa(ex, "view_correspondence")) == 1


# ----------------------------------------------------------------------------- task maps from the goals file

V2 = {"version": 2, "task_name": "Isaac-Kitchen-v9-00",
      "run_config": {"obj_name": "fooditem0", "task_language": "Put the food item in the bowl."},
      "goals": [[
          {"skill": "nav.to_prim", "action": "N_s", "params": {"prim_path": "/world/fooditem0"},
           "goal": [1.0, 2.0, 0.5], "language": "Move to the food item"},
          {"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "/world/fooditem0"},
           "goal": None, "language": "Right hand grasps the food item"},
          {"skill": "gripper.set", "action": "G_r", "params": {"grasp": True}, "goal": [1.0],
           "language": "Close the gripper"},
          {"skill": "nav.to_prim", "action": "N_s", "params": {"prim_path": "/world/bowl0"},
           "goal": [3.0, 4.0, 0.0], "language": "Carry the food item to the bowl"},
          {"skill": "arm.reset", "action": "A_r", "params": {}, "goal": None,
           "language": "Return the arm home"},
      ]]}

V1 = {"task_name": "Isaac-Kitchen-v01-01",
      "goals": [[["N_s", [0.77, -1.74, 0.0]], ["A_l", [[1.28, -1.74, 0.91, 0.46, 0.53, 0.65, 0.27]]],
                 ["G_l", [1.0]]]]}
V1_TWIN = {"goals": [[
    {"language": "Move to bowl", "action": "N_s", "parameters": {"prim_path": "/world/bowl0"}},
    {"language": "left arm to bowl", "action": "A_l", "parameters": {"prim_path": "/world/bowl0"}},
    {"language": "close", "action": "G_l", "parameters": {}},
]]}


def test_describe_steps_reads_language_and_prim_from_a_v2_file():
    d = lib.describe_steps(V2)
    assert [s["language"] for s in d] == ["Move to the food item", "Right hand grasps the food item",
                                          "Close the gripper", "Carry the food item to the bowl",
                                          "Return the arm home"]
    assert [s["prim_path"] for s in d] == ["/world/fooditem0", "/world/fooditem0", None, "/world/bowl0", None]
    assert [s["action"] for s in d] == ["N_s", "A_r", "G_r", "N_s", "A_r"]
    assert d[0]["goal"] == [1.0, 2.0, 0.5] and d[1]["goal"] is None


def test_describe_steps_takes_language_and_prim_from_the_v1_reloadable_twin():
    d = lib.describe_steps(V1, V1_TWIN)
    assert [s["language"] for s in d] == ["Move to bowl", "left arm to bowl", "close"]
    assert [s["prim_path"] for s in d] == ["/world/bowl0", "/world/bowl0", None]
    assert d[0]["action"] == "N_s" and d[0]["goal"] == [0.77, -1.74, 0.0]
    assert d[1]["goal"] is None       # arm goals come from the demo, never the file


def test_describe_steps_v1_without_a_twin_has_no_language():
    d = lib.describe_steps(V1, None)
    assert [s["language"] for s in d] == ["", "", ""]
    assert all(s["prim_path"] is None for s in d)


def test_vqa_task_maps_nav_steps_target_their_destination_and_arm_steps_the_manipuland():
    subtasks, targets = lib.vqa_task_maps(lib.describe_steps(V2), vocab=("fooditem", "bowl"))
    assert subtasks == {0: "Move to the food item", 1: "Right hand grasps the food item",
                        2: "Close the gripper", 3: "Carry the food item to the bowl",
                        4: "Return the arm home"}
    # 3 is the nav to the bowl (reachable <-> bowl); 4 is the arm again, still about the food.
    assert targets == {0: "fooditem0", 1: "fooditem0", 2: "fooditem0", 3: "bowl0", 4: "fooditem0"}


def test_vqa_task_maps_arm_steps_after_a_nav_to_a_fixture_keep_the_held_object():
    steps = [{"language": "", "prim_path": "/world/mug0", "action": "N_s", "goal": [0, 0, 0]},
             {"language": "", "prim_path": "/world/mug0", "action": "A_r", "goal": None},
             {"language": "", "prim_path": None, "action": "G_r", "goal": None},
             {"language": "", "prim_path": "/world/sink_cabinet", "action": "N_s", "goal": [1, 1, 0]},
             {"language": "", "prim_path": "/world/sink", "action": "A_r", "goal": None},
             {"language": "", "prim_path": None, "action": "G_r", "goal": None}]
    _, targets = lib.vqa_task_maps(steps, vocab=("mug",))
    assert targets == {0: "mug0", 1: "mug0", 2: "mug0", 3: "sink_cabinet", 4: "mug0", 5: "mug0"}


def test_vqa_task_maps_backfills_leading_steps_and_handles_handle_prims():
    steps = [{"language": "a", "prim_path": None, "action": "A_r", "goal": None},
             {"language": "b", "prim_path": "/world/base_cabinet/door_0_0/door_handle", "action": "N_s", "goal": [0, 0, 0]}]
    _, targets = lib.vqa_task_maps(steps)
    assert targets == {0: "door_handle", 1: "door_handle"}


def test_vqa_task_maps_with_no_prims_anywhere_has_no_target():
    _, targets = lib.vqa_task_maps(lib.describe_steps(V1, None))
    assert targets == {0: None, 1: None, 2: None}


def test_dual_arm_drawer_targets_do_not_overwrite_carried_bowl():
    steps = [{"language": "", "prim_path": path, "action": action, "goal": None}
             for action, path in [
                 ("A_r", "/world/bowl0"), ("G_r", None),
                 ("A_l", "/world/base_cabinet/drawer_0_0/door_handle"),
                 ("G_l", None), ("N", None), ("A_r", None),
                 ("G_r", None), ("N", None)]]
    _, targets = lib.vqa_task_maps(steps, vocab=("bowl",))
    assert targets == {0: "bowl0", 1: "bowl0", 2: "door_handle", 3: "door_handle",
                       4: "door_handle", 5: "bowl0", 6: "bowl0", 7: "door_handle"}
    path = lib.target_prim_path(steps, 7, targets[7])
    assert path == "/world/base_cabinet/drawer_0_0/door_handle"
    info = {"instance_id_segmentation_fast": {"idToLabels": {
        "1": "/World/envs/env_0/Kitchen/base_cabinet/drawer_0_0/door_handle/mesh",
        "2": "/World/envs/env_0/Kitchen/base_cabinet/drawer_0_1/door_handle/mesh",
        "3": "/World/envs/env_1/Kitchen/base_cabinet/drawer_0_0/door_handle/mesh"}}}
    assert lib.find_instance_keys_by_name(info, path) == ["1"]


def test_task_language_of_prefers_the_run_config_then_the_fallback():
    assert lib.task_language_of(V2, "cli") == "Put the food item in the bowl."
    assert lib.task_language_of(V1, "cli") == "cli"


def test_uniform_sample_plan_covers_every_subtask_in_the_episode():
    assert lib.uniform_sample_plan([0, 0, 1, 1, 1, 3, -1], 2) == {0: 2, 1: 2, 3: 2}


def test_rel_pos_in_base_rotates_the_world_offset_into_the_base_frame():
    yaw = math.pi / 2
    q = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])   # wxyz, +90 deg about z
    out = lib.rel_pos_in_base(torch.tensor([1.0, 0.0, 0.5]), torch.tensor([0.0, 0.0, 0.2]), q)
    assert out.tolist() == pytest.approx([0.0, -1.0, 0.3], abs=1e-6)


def test_sample_frame_objects_without_a_target_returns_k_sampled_objects():
    out = lib.sample_frame_objects(None, ["mug0", "bowl0"], vocab=("mug", "bowl", "apple"), k=3,
                                   rng=random.Random(0))
    assert len(out) == 3 and {o["name"] for o in out} == {"mug0", "bowl0", "apple"}


def test_flush_writes_the_task_name_into_each_row(tmp_path):
    img = torch.zeros((4, 4, 4), dtype=torch.uint8)
    rec = {"t": 1, "goal_idx": 0, "obj_name": None, "subtask": "x",
           "images": {c: img for c in ("front", "wrist_left", "wrist_right")},
           "seg_images": {c: img for c in ("front", "wrist_left", "wrist_right")},
           "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
           "distance_base": None, "goal_state_mobile": None, "goal_state_gripper": None,
           "left_gripper": False, "right_gripper": False, "gripper": None, "objects": []}
    out = tmp_path / "vqa.jsonl"
    lib.flush_simvqa_on_success(0, {"env_00": [rec]}, str(out), str(tmp_path / "i"), str(tmp_path / "s"),
                                "Open the cabinet door.", task="Isaac-Kitchen-v1220-01")
    row = json.loads(out.read_text().splitlines()[0])
    assert row["task"] == "Isaac-Kitchen-v1220-01"
    assert row["meta"]["obj_name"] is None


# ----------------------------------------------------------------------------- expander: multi-task roots

def test_clean_obj_name_strips_the_index_and_underscores():
    assert expander.clean_obj_name("mug0") == "mug"
    assert expander.clean_obj_name("door_handle") == "door handle"
    assert expander.clean_obj_name("fooditem12") == "fooditem"


def test_load_subtask_list_accepts_a_goal_idx_dict(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"1": "b", "0": "a"}))
    assert expander.load_subtask_list(p) == ["a", "b"]


def _write_root(tmp_path, name, rows, meta=None):
    root = tmp_path / name
    root.mkdir()
    (root / "vqa.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if meta is not None:
        (root / "vqa_meta.json").write_text(json.dumps(meta))
    return root


def test_convert_root_uses_the_roots_own_vqa_meta_and_prefixes_ids_with_the_task(tmp_path):
    row = _row()
    root = _write_root(tmp_path, "Isaac-Kitchen-v9-00", [row],
                       meta={"task": "Isaac-Kitchen-v9-00", "task_language": "Put mug to sink.",
                             "subtasks": {"0": "go", "1": "reach"}, "targets": {"0": "mug0", "1": "mug0"}})
    import io
    buf = io.StringIO()
    n = expander.convert_root(root, None, buf)
    rows = [json.loads(l) for l in buf.getvalue().splitlines()]
    assert n == len(rows) > 0
    hl = [r for r in rows if r["vqa_function"] == "high_level_subtask"]
    assert hl[0]["cotrain_answer"] == "go"
    assert all(r["id"].startswith("Isaac-Kitchen-v9-00/env_00_t0007::") for r in rows)
    nxt = [r for r in rows if r["vqa_function"] == "next_subtask"]
    assert nxt[0]["cotrain_answer"] == "reach"


def test_convert_root_without_meta_and_without_a_list_is_an_error(tmp_path):
    root = _write_root(tmp_path, "Isaac-Kitchen-v9-01", [_row()])
    import io
    with pytest.raises(FileNotFoundError):
        expander.convert_root(root, None, io.StringIO())


def test_convert_root_falls_back_to_the_cli_subtask_list_for_old_roots(tmp_path):
    root = _write_root(tmp_path, "Isaac-Kitchen-v9-02", [_row()])
    import io
    buf = io.StringIO()
    expander.convert_root(root, ["legacy-0", "legacy-1"], buf)
    rows = [json.loads(l) for l in buf.getvalue().splitlines()]
    hl = [r for r in rows if r["vqa_function"] == "high_level_subtask"]
    assert hl[0]["cotrain_answer"] == "legacy-0"
    assert rows[0]["id"].startswith("Isaac-Kitchen-v9-02/")


# ----------------------------------------------------------------------------- round 3 (from the live run)

def test_sample_frame_objects_always_includes_a_present_other_when_the_scene_has_one():
    for seed in range(30):
        out = lib.sample_frame_objects("mug0", ["mug0", "bowl0"], vocab=("mug", "bowl", "apple", "jar", "can", "vase"),
                                       k=3, rng=random.Random(seed))
        names = [o["name"] for o in out]
        assert names[0] == "mug0" and "bowl0" in names and len(names) == 3
        assert sum(not o["present"] for o in out) == 1


def test_sample_frame_objects_two_present_others_fill_one_present_and_one_absent_slot():
    seen = collections.Counter()
    for seed in range(60):
        out = lib.sample_frame_objects("mug0", ["mug0", "bowl0", "jar0"], vocab=("mug", "bowl", "jar", "apple"),
                                       k=3, rng=random.Random(seed))
        assert [o["present"] for o in out[1:]].count(True) == 1
        seen[out[1]["name"] if out[1]["present"] else out[2]["name"]] += 1
    assert set(seen) == {"bowl0", "jar0"}


def test_subtask_end_does_not_override_failed_reach_measurement():
    row = _row(goal_state_mobile=[-0.036, 0.024, 0.0]); row["subtask_end"] = True
    (_, a), = _qa(expander.build_examples_for_row(row, SUBTASKS, Path("/x")), "object_reachable")
    assert a.startswith("no. move base")


def test_subtask_end_does_not_override_failed_grasp_measurement():
    row = _row(goal_idx=1, gripper="right", goal_state_gripper=[0.001, 0.014, -0.02, 0.9995, 0.03, 0.0, 0.0])
    row["subtask_end"] = True
    (_, a), = _qa(expander.build_examples_for_row(row, SUBTASKS, Path("/x")), "object_graspable")
    assert a.startswith("no. move gripper")


@pytest.mark.parametrize("flag", [False, None])
def test_reset_place_and_unannotated_records_do_not_claim_graspability(flag):
    row = _row(gripper="right", grasp_target=flag,
               goal_state_gripper=[0, 0, 0, 1, 0, 0, 0])
    row["subtask_end"] = True
    assert not _qa(expander.build_examples_for_row(row, SUBTASKS, Path("/x")), "object_graspable")


def test_grasp_annotation_requires_a_same_arm_close_next():
    goal = {"goals": [[
        {"action": "A_r", "skill": "arm.grasp"},
        {"action": "G_r", "skill": "gripper.set", "params": {"grasp": True}},
        {"action": "A_r", "skill": "arm.bowl_place"},
        {"action": "G_r", "skill": "gripper.set", "params": {"grasp": False}},
        {"action": "A_r", "skill": "arm.reset"},
        {"action": "A_l", "skill": "arm.pose"},
        {"action": "G_l", "skill": "gripper.set", "params": {"grasp": True}}]]}
    assert [s["grasp_target"] for s in lib.describe_steps(goal)] == [True, False, False, False, False, True, False]


def test_replay_sampling_includes_frame_zero_and_keeps_terminal_alignment():
    assert lib.sample_subtask_ticks([0], 2) == [1]
    assert lib.sample_subtask_ticks([0, 1, 2], 3) == [1, 2, 3]
    assert lib.sample_subtask_ticks([17, 18, 19], 1) == [20]
    assert lib.sample_subtask_ticks([], 2) == []


def test_flush_writes_subtask_end_and_gripper_joints(tmp_path):
    img = torch.zeros((4, 4, 4), dtype=torch.uint8)
    rec = {"t": 1, "goal_idx": 0, "obj_name": None, "subtask": "x", "subtask_end": True,
           "images": {c: img for c in ("front", "wrist_left", "wrist_right")},
           "seg_images": {c: img for c in ("front", "wrist_left", "wrist_right")},
           "bboxes": {"front": None, "wrist_left": None, "wrist_right": None},
           "distance_base": None, "goal_state_mobile": None, "goal_state_gripper": None,
           "left_gripper": False, "right_gripper": True, "gripper": None, "objects": [],
           "gripper_joints": {"left": 0.04, "right": 0.036, "left_open": 0.04, "right_open": 0.04}}
    out = tmp_path / "vqa.jsonl"
    lib.flush_simvqa_on_success(0, {"env_00": [rec]}, str(out), str(tmp_path / "i"), str(tmp_path / "s"), "p")
    row = json.loads(out.read_text().splitlines()[0])
    assert row["subtask_end"] is True
    assert row["meta"]["gripper_joints"] == {"left": 0.04, "right": 0.036, "left_open": 0.04, "right_open": 0.04}


def test_gripper_closed_is_travel_from_the_open_reading():
    assert lib.gripper_closed(torch.tensor(0.036), torch.tensor(0.040), travel=0.002) is True
    assert lib.gripper_closed(torch.tensor(0.0395), torch.tensor(0.040), travel=0.002) is False
    assert lib.gripper_closed(torch.tensor(0.004), torch.tensor(0.000), travel=0.002) is True   # either sign


def test_find_instance_keys_returns_every_label_under_the_prim():
    info = {"instance_id_segmentation_fast": {"idToLabels": {
        "(1,0,0,255)": "/World/envs/env_0/Kitchen/sink_cabinet/door_0/mesh",
        "(2,0,0,255)": "/World/envs/env_0/Kitchen/sink_cabinet/basin/mesh",
        "(3,0,0,255)": "/World/envs/env_0/Kitchen/mug0/mesh",
        "(4,0,0,255)": "/World/envs/env_1/Kitchen/sink_cabinet/basin/mesh",
    }}}
    assert lib.find_instance_keys_by_name(info, "sink_cabinet", eid=0) == ["(1,0,0,255)", "(2,0,0,255)"]
    assert lib.find_instance_keys_by_name(info, "apple0", eid=0) == []


def test_bbox_from_rgba_seg_unions_several_keys():
    seg = torch.zeros((10, 10, 4), dtype=torch.int64)
    seg[2, 2] = torch.tensor([1, 0, 0, 255]); seg[7, 8] = torch.tensor([2, 0, 0, 255])
    assert lib.bbox_from_rgba_seg(seg, [(1, 0, 0, 255), (2, 0, 0, 255)]) == (2, 2, 8, 7)
    assert lib.bbox_from_rgba_seg(seg, [(1, 0, 0, 255)]) == (2, 2, 2, 2)
    assert lib.bbox_from_rgba_seg(seg, []) is None
