"""Which robot's base-frame EEF home positions eef_home compares against.

WHY. composed.py carried HOME_R/HOME_L as module constants -- ANUBIS's home, measured once and
hardcoded (same values as terminations.py:16). Every composed condition containing eef_home
therefore compared RB-Y1's end effectors against where an ANUBIS parks its hands, which never
matches, so an RB-Y1 success carrying eef_home could not fire. The mug-to-sink task is the first
RB-Y1 task whose success includes eef_home, which is why this surfaces now.

Detection is by JOINT NAMES, not an env var: the articulation is the ground truth for which robot
is loaded, an env var is a claim about it. "right_arm_0" exists only on RB-Y1;
"arm1_base_link_joint" only on Anubis (the same discriminators rby1_kitchen_cfg's nine blocks
substitute between).

RBY1's numbers are MEASURED, not derived: SIMVLA_PRINT_EEF_HOME=1 makes composed.py print
eef_pos_base at first evaluation (frame ~0, arms still at home), and the measured values are baked
here (measured during the RB-Y1 sink-campaign work).

Stdlib only, importable without torch/Omniverse, so the mapping is unit-testable.
"""
from __future__ import annotations

#: (right, left) base-frame EEF positions at the home pose, metres.
ANUBIS = ((0.2257, -0.0988, 1.0351), (0.2203, 0.1080, 1.0342))

#: Measured on Isaac-Kitchen-v1202r-00 under init pose F01, by scripts/simvla/campose_probe.py
#: (job 2081832, 2026-08-24), which is the ONLY trustworthy ruler for this number:
#:     [campose] POSE F01 R=(0.354,-0.244,1.075) L=(0.354,0.242,1.079) height=1.077 reach=0.354 OK
#:
#: WHY NOT composed.py's SIMVLA_PRINT_EEF_HOME. That path prints eef_pos_base at EVERY termination
#: evaluation and the measurement procedure was "take the first line", on the assumption that the
#: arms are still at home then. They are not: they are in flight and sagging. The value baked here
#: on 2026-08-23 by that method was
#:     right=(0.1403, -0.5084, 1.2464)  left=(0.2642, 0.1084, 1.1857)
#: which fails the free self-check -- the init pose is MIRROR SYMMETRIC, so the base-frame hands
#: must be too, and |right_y|=0.508 against |left_y|=0.108 is a 0.40 m asymmetry that no amount of
#: gravity sag produces. It sat 0.380 m from the true home against eef_home's own 0.12 m radius,
#: so every RB-Y1 success carrying eef_home was unfireable for as long as it was in place.
#:
#: campose_probe measures the SAME quantity by the SAME math (quat_rotate_inverse against
#: base_link, composed.py:129) but writes the pose to state AND targets, settles SETTLE_STEPS
#: frames, and ASSERTS the mirror symmetry -- it prints OK / UNSETTLED rather than leaving it to
#: be eyeballed. Re-measure with it, not with PRINT_EEF_HOME, whenever the init pose changes.
#:
#: z = 1.077 clears the 0.95 m counter by 0.127 m, so a mug parked here by arm.reset survives the
#: drive to the sink. Values are the mean of the two mirrored readings, so left is the exact
#: mirror of right rather than carrying the 2 mm measurement noise into the constant.
RBY1 = ((0.354, -0.243, 1.077), (0.354, 0.243, 1.077))

#: AI WORKER (FFW_SG2), by forward kinematics over the USD's joint graph at the default -0.30 m
#: lift home. The right arm is the exact kinematic mirror of the left, so both end effectors have
#: the same forward reach and height and equal lateral distance. The default task now requires
#: both wrists within 0.05 m of this actual home. Re-measure whenever the init pose changes.
# The previous (0.2873, +/-0.2444, 1.0420) was the old top-lift C2 LEFT
# pose mirrored onto the right, not the selected symmetric -0.30 m home.
# These values are FK over the public USD joint graph at AIWORKER_LIFT_HOMES.
AIWORKER = ((0.3249344, -0.3930621, 1.0229979), (0.3249344, 0.3930621, 1.0229979))
AIWORKER_BY_LIFT = {
    "-0.30": AIWORKER,
    "-0.35": ((0.3414013, -0.4875063, 1.0052757), (0.3414013, 0.4875063, 1.0052757)),
    # The optional top-lift fallback retains its original asymmetric C2 joints.
    "0.00": ((0.2615405, -0.3954619, 1.0299626), (0.2872952, 0.2444529, 1.0420349)),
}


def home_for(joint_names):
    names = set(joint_names)
    if "right_arm_0" in names:
        if RBY1 is None:
            raise KeyError(
                "RB-Y1 home poses not measured yet: run once with SIMVLA_PRINT_EEF_HOME=1 and "
                "bake the printed values into homes.RBY1 (plan Task 3)."
            )
        return RBY1
    if "arm1_base_link_joint" in names:
        return ANUBIS
    # arm_r_joint1 is unique to FFW_SG2 among the robots this repo carries; Anubis spells its
    # shoulder arm1_base_link_joint and RB-Y1 spells it right_arm_0.
    if "arm_r_joint1" in names:
        import os
        key = f"{float(os.environ.get('SIMVLA_AIWORKER_LIFT_M', '-0.30')):.2f}"
        if key not in AIWORKER_BY_LIFT:
            raise ValueError(f"AI Worker home has no measured pose for lift {key}")
        return AIWORKER_BY_LIFT[key]
    raise KeyError(
        f"cannot tell which robot this is from its joints ({sorted(names)[:6]}...): "
        f"none of 'right_arm_0' (RB-Y1), 'arm1_base_link_joint' (Anubis) or "
        f"'arm_r_joint1' (AI Worker) present"
    )
