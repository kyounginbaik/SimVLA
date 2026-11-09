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


def test_mug_is_released_over_the_basin_not_the_counter_lip():
    # nav.to_prim parks 0.15 m of clearance (~0.38 m base-to-cabinet front); the skill default
    # forward_m=0.23 then released the mug ~8 cm past the front edge, on the countertop, short of
    # the door_midpoint anchor. 0.38 put it in the basin (Anubis, kitchen 813, job 2731063).
    for name in ("mug_to_sink", "mug_to_sink_left"):
        place = load_template(name).steps[5]
        assert place.skill == "arm.bowl_place"
        assert place.params["forward_m"] == 0.38
