from retry_diagnostics import describe_retry_union


SPEC = {
    "any": [
        {"obj_z": {"role": "mug0", "hi": 0.3}},
        {"robot_fell": {"z": -0.1}},
        {"base_floor_collision": {"depth_m": 0.02, "frames": 80}},
    ]
}


def test_infers_the_single_stateful_branch_from_a_true_union():
    lines = describe_retry_union(
        SPEC, [1], object_z={"mug0": [0.2, 0.95]}, base_z=[0.0, 0.01]
    )
    assert lines == [
        "env=1 retry=true branches=obj_z=false(z=0.9500) "
        "robot_fell=false(z=0.0100) base_floor_collision=true(inferred_from_union)"
    ]


def test_reports_a_dropped_object_without_guessing_the_stateful_branch():
    line = describe_retry_union(
        SPEC, [0], object_z={"mug0": [0.2]}, base_z=[0.01]
    )[0]
    assert "obj_z=true(z=0.2000)" in line
    assert "base_floor_collision=unknown" in line


def test_reports_a_fallen_robot_without_guessing_the_stateful_branch():
    line = describe_retry_union(
        SPEC, [0], object_z={"mug0": [0.9]}, base_z=[-0.2]
    )[0]
    assert "robot_fell=true(z=-0.2000)" in line
    assert "base_floor_collision=unknown" in line
