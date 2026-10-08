from simvla.tasks import load_template


def test_mug_to_sink_is_a_complete_supported_skill_sequence():
    task = load_template("mug_to_sink")

    assert task.language == "Grasp the mug and put it in the sink."
    assert [step.skill for step in task.steps] == [
        "nav.to_prim", "arm.grasp", "gripper.set", "arm.reset",
        "nav.to_prim", "arm.bowl_place", "gripper.set", "arm.reset",
    ]
    assert task.steps[0].params["safety"] == 0.28
    assert task.steps[4].params == {
        "prim_path": "/world/sink_cabinet", "which_arm": "Right", "safety": 0.15,
    }
    assert task.subtask_groups == [[3], [5, 6, 7]]
    near_sink = task.success["all"][0]["obj_near_prim"]
    assert near_sink == {
        "role": "@target", "target_role": "sink_cabinet", "radius": 0.12,
        "anchor": "door_midpoint", "z_override": 0.88,
    }
    assert task.success["all"][1] == {"eef_home": {"arm": "both", "radius": 0.12}}
