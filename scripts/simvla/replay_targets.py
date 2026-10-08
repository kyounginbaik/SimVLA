"""Match collector targets for robot joints not owned by the ActionManager."""

import math


def replay_settle_steps(value, *, joint_replay):
    """Explicit, bounded end-of-recording motor hold; disabled by default."""
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError("SIMVLA_REPLAY_SETTLE_STEPS must be an integer from 0 to 200")
    steps = int(value)
    if not 0 <= steps <= 200:
        raise ValueError("SIMVLA_REPLAY_SETTLE_STEPS must be an integer from 0 to 200")
    if steps and not joint_replay:
        raise ValueError("Terminal settling requires recorded joint motor targets")
    return steps


def joint_target_order(feature, live_names):
    """Map named recorded motor targets to the live articulation, failing closed."""
    names = feature.get("names")
    if isinstance(names, dict):
        names = names.get("action.joint")
    if (not isinstance(names, list) or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or len(set(live_names)) != len(live_names)
            or set(names) != set(live_names) or feature.get("shape") != [len(names)]):
        raise ValueError("action.joint metadata must name every live joint exactly once")
    return [names.index(name) for name in live_names]


def base_tracking_correction(reference, actual):
    """Bounded world-frame velocity feedback for a recorded base trajectory.

    Poses are [x, y, qw, qx, qy, qz], the dataset's ``initial_pose`` feature.
    This is physical trajectory tracking, not a pose write or action-only replay.
    Limits: 5 cm/s translation and 0.1 rad/s yaw; proportional gain 2/s.
    """
    for pose in (reference, actual):
        if len(pose) != 6 or not all(math.isfinite(float(v)) for v in pose):
            raise ValueError("base tracking requires finite six-component poses")
        if abs(sum(float(v)**2 for v in pose[2:]) - 1.) > 1e-3:
            raise ValueError("base tracking requires unit wxyz quaternions")
    def yaw(pose):
        w, x, y, z = pose[2:]
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    dx, dy = 2*(reference[0]-actual[0]), 2*(reference[1]-actual[1])
    scale = min(1., .05 / max(math.hypot(dx, dy), 1e-12))
    angle = (yaw(reference)-yaw(actual)+math.pi) % (2*math.pi)-math.pi
    return [dx*scale, dy*scale, max(-.1, min(.1, 2*angle))]


def hold_uncontrolled_lift(robot, joint_ids):
    """Match collection: no ActionManager term owns AI Worker's lift joint."""
    if joint_ids is not None:
        robot.set_joint_position_target(robot.data.default_joint_pos[:, joint_ids], joint_ids=joint_ids)


def non_gripper_joint_ids(joint_count, gripper_groups):
    """Let live binary/contact controllers own fingers during optional hybrid replay."""
    excluded = set()
    for group in gripper_groups:
        if isinstance(group, slice):
            group = range(*group.indices(joint_count))
        for index in group:
            if type(index) is not int or not 0 <= index < joint_count:
                raise ValueError("gripper joint index outside articulation")
            excluded.add(index)
    selected = [i for i in range(joint_count) if i not in excluded]
    if not excluded or not selected:
        raise ValueError("hybrid replay requires both gripper and non-gripper joints")
    return selected
