"""Small, simulator-free choices for embodiment-specific collection behavior."""

from __future__ import annotations

import math
import os
import random
from typing import Sequence


def goal_noise_standard_deviations(environ=None) -> tuple[float, float, float]:
    """Explicit grasp-search target augmentation, in metres/radians/metres.

    Zero disables only the named target perturbation, not scene/spawn randomness.
    Keep historical defaults so existing collection recipes remain unchanged.
    """
    environ = os.environ if environ is None else environ
    result = []
    for name, default, maximum in (
        ("SIMVLA_GOAL_NAV_XY_STD_M", .03, .25),
        ("SIMVLA_GOAL_NAV_YAW_STD_RAD", .1, math.pi),
        ("SIMVLA_GOAL_ARM_XYZ_STD_M", .01, .25),
    ):
        try:
            value = float(environ.get(name, default))
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be finite and in [0, {maximum}]") from None
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be finite and in [0, {maximum}]")
        result.append(value)
    return tuple(result)


def track_loaded_home_reset(robot: str, skill: str, holding: bool, value: str | None = None) -> bool:
    """Keep a loaded AI Worker reset on its planned path, not ahead of the real wrist."""
    value = os.environ.get("SIMVLA_TRACK_LOADED_HOME", "1") if value is None else value
    if value not in ("0", "1"):
        raise ValueError("SIMVLA_TRACK_LOADED_HOME must be 0 or 1")
    return value == "1" and robot == "aiworker" and skill == "arm.reset" and bool(holding)


def loaded_home_arm_indices(plan_names, robot_names):
    """Map only AI Worker's seven left-arm joints; the full cuRobo plan includes locked lift."""
    names = [f"arm_l_joint{i}" for i in range(1, 8)]
    for available in (plan_names, robot_names):
        if any(available.count(name) != 1 for name in names):
            raise ValueError("Loaded-home plan and robot must name each left-arm joint exactly once")
    return ([robot_names.index(name) for name in names],
            [plan_names.index(name) for name in names])


def loaded_home_hold_allowed(holding, episode_step, *, is_home_reset, is_navigation):
    """Keep joint ownership across a home-plan pause and subsequent loaded navigation."""
    return bool(holding) and episode_step > 1 and (is_home_reset or is_navigation)


def postrelease_final_home_step(step, skill_ids, payloads, *, reset_id, gripper_id) -> bool:
    """Recognize the second reset after opening, never a loaded carry reset.

    A two-stage release first retreats the open hand, then sends it home.  The
    collector must not mistake any earlier reset (while still holding an object)
    for that final home motion.
    """
    if step < 2 or step >= len(skill_ids):
        return False
    return (int(skill_ids[step]) == reset_id
            and int(skill_ids[step - 1]) == reset_id
            and int(skill_ids[step - 2]) == gripper_id
            and float(payloads[step - 2][0]) <= 0)


def cartesian_fallback_allowed(value: str | None = None) -> bool:
    """Strict profiles must not jog after cuRobo rejects an arm motion."""
    value = os.environ.get("SIMVLA_REQUIRE_CUROBO_PLAN", "0") if value is None else value
    if value not in ("0", "1"):
        raise ValueError("SIMVLA_REQUIRE_CUROBO_PLAN must be 0 or 1")
    return value == "0"


def configure_physics_substeps(cfg) -> None:
    """Optional finer contact/control integration without changing recorded FPS."""
    raw = os.environ.get("SIMVLA_PHYSICS_SUBSTEPS")
    if raw is None:
        return
    if not raw.isdecimal() or not 1 <= int(raw) <= 24:
        raise ValueError("SIMVLA_PHYSICS_SUBSTEPS must be an integer in [1, 24]")
    control_dt = cfg.sim.dt * cfg.decimation
    cfg.decimation = int(raw)
    cfg.sim.dt = control_dt / cfg.decimation
    cfg.sim.render_interval = cfg.decimation


def planar_hold_command(anchor: Sequence[float], current: Sequence[float]) -> tuple[float, float, float]:
    """Small position-hold velocity in the collector's forward/right/yaw convention.

    Pose arguments are world x/y/yaw. The base actuator consumes velocity, so a zero
    command alone allows reaction torque from the arm to rotate the parked base.
    """
    if len(anchor) != 3 or len(current) != 3 or not all(
            math.isfinite(float(v)) for v in (*anchor, *current)):
        raise ValueError("base hold needs finite x/y/yaw poses")
    vx = max(-0.05, min(0.05, 4.0 * (anchor[0] - current[0])))
    vy = max(-0.05, min(0.05, 4.0 * (anchor[1] - current[1])))
    yaw_error = (anchor[2] - current[2] + math.pi) % (2 * math.pi) - math.pi
    omega = max(-0.15, min(0.15, 4.0 * yaw_error))
    c, s = math.cos(current[2]), math.sin(current[2])
    return c * vx + s * vy, s * vx - c * vy, omega


def augmentation_seeds(base_seed: int) -> tuple[int, int]:
    """Return stable, independent seeds for fixed camera and lighting perturbations."""
    rng = random.Random(int(base_seed))
    return rng.randrange(0, 2**32), rng.randrange(0, 2**32)


def pinch_axis_xy_correction(
    goal_position: Sequence[float],
    jaw_offset_world: Sequence[float],
    object_position: Sequence[float],
) -> tuple[float, float]:
    """Translate a wrist goal so its physical jaw midpoint lies over an object's XY axis.

    ``jaw_offset_world`` is the wrist-to-jaw-midpoint vector rotated by the desired wrist
    orientation. Z is intentionally unchanged: grasp height and orientation remain authored.
    """
    if len(goal_position) < 3 or len(jaw_offset_world) < 2 or len(object_position) < 2:
        raise ValueError("goal, jaw offset, and object positions must have at least 2/3 coordinates")
    return (
        float(object_position[0]) - float(goal_position[0]) - float(jaw_offset_world[0]),
        float(object_position[1]) - float(goal_position[1]) - float(jaw_offset_world[1]),
    )


def pinch_center_z_correction(
    goal_position: Sequence[float],
    jaw_offset_world: Sequence[float],
    object_position: Sequence[float],
    target_height_above_object_m: float = 0.0,
) -> float:
    """Translate a wrist goal so its jaw midpoint reaches a height above the object root."""
    if len(goal_position) < 3 or len(jaw_offset_world) < 3 or len(object_position) < 3:
        raise ValueError("goal, jaw offset, and object positions must have three coordinates")
    target_height = float(target_height_above_object_m)
    if not math.isfinite(target_height):
        raise ValueError("target height above object must be finite")
    return (float(object_position[2]) + target_height - float(goal_position[2])
            - float(jaw_offset_world[2]))


def anchor_relative_ik_target(action_term, env_index: int, position, quaternion) -> None:
    """Initialize a relative-IK target at the measured EEF pose before a jog.

    Call once at the start of a corrective jog. Repeating this every control frame discards
    reference motion whenever the physical arm lags its commanded pose.
    """
    controller = action_term._ik_controller
    controller.ee_pos_des[env_index] = position
    controller.ee_quat_des[env_index] = quaternion


def advance_pregrasp_stage(
    stage: int,
    orient_position_error_m: float,
    rotation_error_deg: float,
    grasp_position_error_m: float,
    position_tolerance_m: float = 0.03,
    rotation_tolerance_deg: float = 20.0,
    final_rotation_tolerance_deg: float = 5.0,
) -> int:
    """Reach far, orient, move to near standoff, finish orientation, then close in.

    The far waypoint can be a kinematic singularity for an extended arm. Permit a small,
    bounded residual there so the arm can change posture at the near standoff, but do not begin
    the final object-axis approach until the stricter final orientation tolerance is met.
    """
    values = (
        orient_position_error_m, rotation_error_deg, grasp_position_error_m,
        position_tolerance_m, rotation_tolerance_deg, final_rotation_tolerance_deg,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("pregrasp errors and tolerances must be finite")
    if not 0 <= int(stage) <= 3:
        raise ValueError("pregrasp stage must be 0 (far), 1 (orient), 2 (standoff), or 3 (approach)")
    if orient_position_error_m < 0 or rotation_error_deg < 0 or grasp_position_error_m < 0:
        raise ValueError("pregrasp errors must be nonnegative")
    if (position_tolerance_m <= 0 or rotation_tolerance_deg <= 0
            or final_rotation_tolerance_deg <= 0):
        raise ValueError("pregrasp tolerances must be positive")
    stage = int(stage)
    if stage == 0 and orient_position_error_m <= position_tolerance_m:
        return 1
    if stage == 1 and rotation_error_deg <= rotation_tolerance_deg:
        return 2
    if (stage == 2 and grasp_position_error_m <= position_tolerance_m
            and rotation_error_deg <= final_rotation_tolerance_deg):
        return 3
    return stage


def mug_grasp_position_tolerance(object_name: str, nominal: float, closing_next: bool) -> float:
    """Allow bounded wrist-frame residual for a mug; the physical jaws validate the close."""
    if closing_next and str(object_name).startswith("mug"):
        return max(float(nominal), 0.040)
    return float(nominal)


def measured_locked_joints(locked, joint_names, positions, tolerance=1e-4):
    """Refresh real locked joints; preserve synthetic planning-only jaw joints."""
    if len(joint_names) != len(positions) or len(set(joint_names)) != len(joint_names):
        raise ValueError("joint names and positions must have matching unique entries")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("locked-joint tolerance must be finite and nonnegative")
    measured = dict(zip(joint_names, positions))
    updated = dict(locked)
    for name, old in locked.items():
        if name in measured:
            value = float(measured[name])
            if not math.isfinite(value):
                raise ValueError(f"non-finite locked joint {name}")
            if abs(value - old) > tolerance:
                updated[name] = value
    return updated


def jaw_pad_offset(body_name: str) -> tuple[float, float, float]:
    """Contact-surface centre in a distal finger's own frame.

    FFW-SG2's body origin is a linkage frame, 19.64 mm behind the contact
    surface centre. Measured from the public distal mesh's inner-pad bounds;
    confusing these points drives the palm too far into a side grasp.
    RB-Y1's inner pad is instead 30.5 mm along local -Z from its finger origin.
    These are embodiment-specific mesh measurements, not shared wrist offsets.
    """
    if body_name.startswith("gripper_") and "_rh_p12_rn_" in body_name:
        if body_name.endswith("_r2"):
            return (0.0, -0.00365865, 0.019641065)
        if body_name.endswith("_l2"):
            return (0.0, 0.00365865, 0.019641065)
    if body_name in {"ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2"}:
        return (-0.003, 0.0, -0.0305)
    if body_name in {"gripper1L", "gripper1R", "gripper2L", "gripper2R"}:
        return (0.0, 0.0, 0.1088)
    return (0.0, 0.0, 0.0)


def jaw_contact_midpoint(robot, env_ids, first, second):
    """Tensor midpoint of physical pads; does not change body-origin gap checks.

    Accepts either one environment index or a batch. Uses tensor methods only,
    so importing the profile module still does not import Torch or Isaac.
    """
    points = []
    for index in (first, second):
        position = robot.data.body_pos_w[env_ids, index]
        offset = jaw_pad_offset(robot.body_names[index])
        if any(offset):
            local = position.new_tensor(offset).expand_as(position)
            quat = robot.data.body_quat_w[env_ids, index]
            xyz = quat[..., 1:]
            twice_cross = 2 * xyz.cross(local, dim=-1)
            rotated = local + quat[..., :1] * twice_cross + xyz.cross(twice_cross, dim=-1)
            position = position + rotated
        points.append(position)
    return 0.5 * (points[0] + points[1])


def mug_jaw_contact_ready(
    jaw_axis_error_m: float,
    jaw_height_above_object_m: float,
    lateral_tolerance_m: float = 0.020,
    min_height_m: float = 0.040,
    max_height_m: float = 0.075,
) -> bool:
    """Whether the closed-loop reach has physically placed both jaws around a mug.

    Wrist position alone is a poor proxy across robot tool frames. This bounded contact window
    checks that the jaw midpoint is close to the mug axis and within the cylindrical body.
    The pinned public mug mesh extends about 0.086 m above its root; a 0.09-0.15 m
    window let the planner stop at the rim or even above it, where pads briefly loaded
    but slid off during lift.
    """
    values = (jaw_axis_error_m, jaw_height_above_object_m, lateral_tolerance_m,
              min_height_m, max_height_m)
    if any(not math.isfinite(float(value)) for value in values):
        return False
    if lateral_tolerance_m <= 0 or min_height_m < 0 or max_height_m <= min_height_m:
        raise ValueError("mug jaw contact limits must define a positive finite window")
    return (0.0 <= float(jaw_axis_error_m) <= lateral_tolerance_m
            and min_height_m < float(jaw_height_above_object_m) < max_height_m)


def mug_grasp_target_height(mug_height_m: float = 0.08, requested_m: float | None = None) -> float:
    """Aim the jaw midpoint inside, never on the boundary of, the mug contact gate."""
    if not math.isfinite(mug_height_m) or mug_height_m <= 0.040:
        raise ValueError("SIMVLA_MUG_HEIGHT must exceed the mug contact minimum (0.040 m)")
    upper = min(0.075, mug_height_m)
    target = (0.040 + upper) / 2 if requested_m is None else float(requested_m)
    if not math.isfinite(target) or not 0.040 < target < upper:
        raise ValueError(f"SIMVLA_MUG_GRASP_HEIGHT must be strictly between 0.040 and {upper:.3f} m")
    return target


def gripper_open_target_reached(joint_positions, open_targets, tolerance_m: float):
    """Per-environment open gate; a slow-moving jaw is not a fully open jaw.

    The inputs are tensors, but this module deliberately does not import Torch so it
    remains safe to load before the simulator application starts.
    """
    if not math.isfinite(tolerance_m) or not 0 < tolerance_m <= 0.01:
        raise ValueError("SIMVLA_GRIPPER_OPEN_TARGET_TOL_M must be in (0, 0.01] m")
    if joint_positions.shape[-1] != open_targets.shape[-1]:
        raise ValueError("gripper joint and open-target widths must match")
    return (joint_positions - open_targets).abs().amax(dim=-1) <= tolerance_m


def postrelease_retreat_target(eef_xyz, object_xyz, base_xyz, distance_m: float,
                               lift_m: float) -> tuple[float, float, float]:
    """Withdraw an open gripper horizontally toward its base before lifting away.

    Vertical-only reset can hook a mug's rim even with fully open jaws. The target
    is derived from live positions so a mobile base's final parking pose is respected.
    """
    values = (*eef_xyz, *object_xyz, *base_xyz, distance_m, lift_m)
    if (len(eef_xyz) != 3 or len(object_xyz) != 3 or len(base_xyz) != 3
            or any(not math.isfinite(float(value)) for value in values)
            or not 0 < distance_m <= 0.3 or not -0.05 <= lift_m <= 0.15):
        raise ValueError("postrelease retreat needs finite 3D poses and bounded distance/lift")
    dx = float(base_xyz[0]) - float(object_xyz[0])
    dy = float(base_xyz[1]) - float(object_xyz[1])
    length = math.hypot(dx, dy)
    if length < 1e-4:
        raise ValueError("cannot retreat toward a base coincident with the object")
    return (float(eef_xyz[0]) + distance_m * dx / length,
            float(eef_xyz[1]) + distance_m * dy / length,
            float(eef_xyz[2]) + lift_m)


def jaw_gap_shows_object_contact(jaw_gap_m: float, minimum_gap_m: float = 0.015) -> bool:
    """Reject a close that has collapsed to near-zero separation with nothing between the jaws."""
    if not math.isfinite(float(jaw_gap_m)) or not math.isfinite(float(minimum_gap_m)):
        return False
    if minimum_gap_m < 0:
        raise ValueError("minimum jaw gap must be nonnegative")
    return float(jaw_gap_m) >= float(minimum_gap_m)


def mug_pad_contacts_loaded(left_force_n: float, right_force_n: float,
                            minimum_force_n: float = 1.0) -> bool:
    """Require both instrumented mug pads to carry a measurable normal load."""
    values = (left_force_n, right_force_n, minimum_force_n)
    if any(not math.isfinite(float(value)) for value in values):
        return False
    if minimum_force_n < 0:
        raise ValueError("minimum pad force must be nonnegative")
    return float(left_force_n) >= float(minimum_force_n) and \
        float(right_force_n) >= float(minimum_force_n)


def mug_pre_lift_grasp_ready(
    object_to_eef_m: float,
    jaw_gap_m: float,
    jaw_axis_error_m: float,
    jaw_height_above_object_m: float,
    pad_forces_n: tuple[float, float] | None = None,
) -> bool:
    """Use measured two-pad contact when available, otherwise fall back to geometry.

    Body origins and a mug's rigid-body root are not a universal contact frame across robots.
    When sensors are configured, bilateral force plus a non-collapsed gap is stronger evidence
    than those root-relative geometric proxies; object proximity remains a coarse sanity bound.
    """
    values = (object_to_eef_m, jaw_gap_m)
    if any(not math.isfinite(float(value)) for value in values):
        return False
    if float(object_to_eef_m) > 0.15 or not jaw_gap_shows_object_contact(float(jaw_gap_m)):
        return False
    if pad_forces_n is not None:
        return mug_pad_contacts_loaded(*pad_forces_n)
    if any(not math.isfinite(float(value))
           for value in (jaw_axis_error_m, jaw_height_above_object_m)):
        return False
    return mug_jaw_contact_ready(float(jaw_axis_error_m), float(jaw_height_above_object_m),
                                 lateral_tolerance_m=0.025)


def post_lift_object_retained(
    lifted_by_m: float,
    jaw_gap_m: float,
    minimum_lift_m: float = 0.05,
    minimum_jaw_gap_m: float = 0.015,
    pad_forces_n: tuple[float, float] | None = None,
) -> bool:
    """Require observed object motion and a non-collapsed, loaded grip after a commanded lift."""
    values = (lifted_by_m, jaw_gap_m, minimum_lift_m, minimum_jaw_gap_m)
    if any(not math.isfinite(float(value)) for value in values):
        return False
    if minimum_lift_m < 0 or minimum_jaw_gap_m < 0:
        raise ValueError("post-lift verification limits must be nonnegative")
    retained = (float(lifted_by_m) >= float(minimum_lift_m)
                and jaw_gap_shows_object_contact(float(jaw_gap_m), float(minimum_jaw_gap_m)))
    return retained and (pad_forces_n is None or mug_pad_contacts_loaded(*pad_forces_n))


def choose_pregrasp_back_sign(
    goal_position: Sequence[float],
    tool_axis: Sequence[float],
    reference_position: Sequence[float],
    standoff_m: float,
) -> int:
    """Choose the standoff side nearest the measured wrist, not the robot base."""
    if len(goal_position) != 3 or len(tool_axis) != 3 or len(reference_position) != 3:
        raise ValueError("pregrasp positions and tool axis must each have three values")
    values = [float(v) for v in (*goal_position, *tool_axis, *reference_position, standoff_m)]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("pregrasp geometry must be finite")
    if standoff_m < 0:
        raise ValueError("pregrasp standoff must be nonnegative")
    axis_norm = math.sqrt(sum(float(v) ** 2 for v in tool_axis))
    if axis_norm <= 1e-9:
        raise ValueError("pregrasp tool axis must be nonzero")
    axis = [float(v) / axis_norm for v in tool_axis]
    minus = [float(goal_position[i]) - axis[i] * standoff_m for i in range(3)]
    plus = [float(goal_position[i]) + axis[i] * standoff_m for i in range(3)]
    minus_distance = sum((minus[i] - float(reference_position[i])) ** 2 for i in range(3))
    plus_distance = sum((plus[i] - float(reference_position[i])) ** 2 for i in range(3))
    return -1 if minus_distance <= plus_distance else 1


def grasp_geometry(robot: str, object_name: str, override: str | None = None) -> str:
    """Choose whether a grasp may use the calibrated AI Worker cylindrical-mug corrections.

    Pinch-frame offsets, axis snapping, and bottle-sized acceptance windows are not generic
    grasp rules. In particular they must not rewrite an authored bowl or handle grasp.
    """
    if override is None:
        return "cylindrical_mug" if robot == "aiworker" and object_name.startswith("mug") else "authored"
    if override not in {"authored", "cylindrical_mug"}:
        raise ValueError("SIMVLA_GRASP_GEOMETRY must be 'authored' or 'cylindrical_mug'")
    return override


def arm_grasp_geometries(
    robot: str,
    right_object: str,
    left_object: str,
    override: str | None = None,
) -> tuple[str, str]:
    """Select geometry independently for each arm; left-only tasks need not name a right object."""
    return (
        grasp_geometry(robot, right_object, override),
        grasp_geometry(robot, left_object, override),
    )


def left_postgrasp_lift_enabled(value: str | None) -> bool:
    """Parse the opt-in left-hand lift without treating ``'0'`` as true."""
    if value in (None, "", "0"):
        return False
    if value == "1":
        return True
    raise ValueError("SIMVLA_POSTGRASP_LIFT_LEFT must be '0' or '1'")


def recorded_episode_env_ids(episode_count: int, reset_ids: list[int], success_ids: list[int]) -> list[int | None]:
    """Associate exported episodes with reset envs when Isaac Lab leaves ``env_id`` unset."""
    if episode_count == len(reset_ids):
        return reset_ids
    if episode_count == len(success_ids):
        return success_ids
    return [None] * episode_count


def apply_ik_joint_deadzone(env, value: str | None = None) -> float:
    """Set both SimVLA arm DLS deadzones, defaulting to preserving small corrections."""
    raw = os.environ.get("SIMVLA_IK_JOINT_DEADZONE", "0") if value is None else value
    try:
        deadzone = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("SIMVLA_IK_JOINT_DEADZONE must be a finite nonnegative number") from exc
    if not math.isfinite(deadzone) or deadzone < 0:
        raise ValueError("SIMVLA_IK_JOINT_DEADZONE must be a finite nonnegative number")

    for term_name in ("armL_action", "armR_action"):
        term = env.action_manager.get_term(term_name)
        term._ik_controller.cfg.delta_joint_deadzone = deadzone
    return deadzone
