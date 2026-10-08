"""How fast the mobile base approaches its nav goal, per robot.

Stdlib only, so it is unit-testable; simvla_video.py cannot be imported at all (it calls
AppLauncher at module scope), which is why this is its own module rather than a dict in there.

WHY PER ROBOT. These five numbers were inline constants in simvla_video (`goal_reached_distance`
... `slowdown_radius`), shared by every robot. Both nav phases ramp the commanded base velocity
linearly inside `slowdown_radius`:

    speed = default_speed                                        , dist  > slowdown_radius
            min_speed + (default_speed - min_speed) * dist/radius , dist <= slowdown_radius

With the shared radius of 0.75 m, the handoff from nav phase 1 to phase 2 at 0.2 m happens deep
inside the ramp, so the ENTIRE final approach to the object ran between 0.101 and 0.113 m/s -- a
12% spread over the last 20 cm, which reads as the base crawling the moment it gets near the
object. That was acceptable on Anubis and too slow on RB-Y1.

The action these feed is `mdp.JointVelocityActionCfg` on base_prismatic_x/y_joint, at unit scale,
so THE NUMBERS ARE LITERAL m/s -- not normalised action units. That is what makes the overshoot
bound below a real check rather than a guess.

The default profile values retain the legacy behavior; robot-specific object standoff overrides
are explicit and need public-asset collection evidence before they can be treated as validated.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

#: The env's control rate, Hz. Only used for the overshoot bound in the docstring/tests.
CONTROL_HZ = 20.0


@dataclass(frozen=True)
class NavProfile:
    """One robot's base-approach tuning.

    goal_reached_distance -- phase 2 ends, and the base stops translating, inside this radius (m).
                             It is where the base parks, so the ARM then plans from there: loosening
                             it trades grasp reachability for arrival time. Not a speed knob.
    goal_reached_yaw      -- phase 0/3 alignment tolerance (rad).
    default_speed         -- commanded base speed outside slowdown_radius (m/s).
    min_speed             -- the floor the ramp decays to at zero distance (m/s). Never 0, or the
                             base would asymptotically stop short of goal_reached_distance.
    slowdown_radius       -- where the ramp begins (m).
    """

    goal_reached_distance: float
    goal_reached_yaw: float
    default_speed: float
    min_speed: float
    slowdown_radius: float
    #: Tolerance for the FINAL in-place turn (nav phase 3) only. Separate from goal_reached_yaw
    #: because the two gates do opposite jobs and one number cannot serve both: phase 0 aligns the
    #: bearing BEFORE driving straight, so a loose value there sends the robot off course (measured:
    #: at 0.35 rad the base drifted up to 20 deg off and closed distance at 0.3 mm/frame, timeouts
    #: went 0 -> 114); phase 3 only sets the parked orientation, which arm.place does not depend on
    #: because it authors a WORLD-frame target. Defaults to goal_reached_yaw, so a profile that
    #: does not set it behaves exactly as before.
    goal_final_yaw: float = None
    #: Does the far-approach phase STEER AT THE GOAL, or drive blind along the current heading?
    #:
    #: simvla_video's nav phase 1 is `vx = speed, vy = 0`: it consults the goal once, in phase 0,
    #: to aim the robot, and then never again. Phase 2, which takes over inside 0.2 m, steers at
    #: the goal every frame. The two use different laws for no reason anyone recorded.
    #:
    #: THE FRAME IS NOT THE PROBLEM, and an earlier reading of this file said it was. RB-Y1's dummy
    #: base does slide along world X and Y (model.urdf parents both prismatic joints ahead of
    #: base_revolute_z_joint), but simvla_video does NOT feed these numbers to the joints directly:
    #: pre_process_actions rotates them out of the body frame by the base yaw first, and negates
    #: column 1 while doing it. Body-frame in, world-frame out -- the pipeline is coherent, and
    #: phase 2's trailing `*= -1` is what cancels that negation rather than a stray sign patch.
    #: Anything written here must therefore be BODY frame, matching phase 2 term for term.
    #:
    #: WHAT IS actually WRONG is that driving blind spends any heading error as lateral miss. Over
    #: kitchen 1202's ~1.2 m approach, phase 0's own 0.1 rad gate is worth up to 0.12 m of it,
    #: against the 0.2 m radius at which phase 2 takes over. An episode that misses that radius
    #: does not retry -- it drives on until it wedges, which is the sink pilot's recorded signature:
    #: base frozen at dist 0.686-0.688 m, yaw frozen with it, commanded 0.2 m/s and moving 1 mm per
    #: 100 frames, until the 3000-frame timeout. Every episode, which is why goal_step never left 0.
    #:
    #: DEFAULT FALSE, so Anubis is byte-identical: its tasks arrive, and a robot whose pipeline
    #: works is not the place to test this.
    closed_loop_approach: bool = False
    #: How far LEFT of the target the base parks, in metres, so the named arm does not have to
    #: reach across the body. Positive is the base's own +Y (left), which is what a RIGHT arm
    #: needs; the goal generator negates it for a left arm.
    #:
    #: WHY IT IS PER ROBOT. plan_nav_to_prim hardcodes 0.05 m for either arm. On the AI Worker the
    #: right EEF rests about 0.144 m right of the base centreline, so 0.05 leaves the target
    #: 0.094 m across the body and the arm falls short -- every failed-reach delta in the campaign
    #: is negative in Y, by 0.08 to 0.45 m, with the goal sitting near base Y=0 while the hand sat
    #: near Y=-0.14. Parking 0.144 m left puts the bottle exactly at the arm's natural Y.
    #:
    #: YAW CONTROL SHAPE. The two nav rotation phases command `sign(err) * 0.463 rad/s` -- full
    #: speed or nothing, with no ramp at either end. The command therefore STEPS from 0 to
    #: 0.463 when a turn starts and back to 0 when it ends, and those acceleration steps ring
    #: the AI Worker's lift column: the head camera sits ~1.43 m above the base on a chain whose
    #: solver_position_iteration_count has to stay at 4 (8 kills the prismatic base joints), so
    #: it is compliant and the ringing shows up as visible camera shake while rotating.
    #:
    #: yaw_kp     -- proportional gain, rad/s per rad of error. The command becomes
    #:               clamp(kp * err, +-0.463), so a turn larger than 0.463/kp radians still runs
    #:               at full speed and only the final approach ramps down.
    #: yaw_slew   -- max change in the commanded rate per control step, rad/s per step. This is
    #:               what removes the START impulse; the proportional term only fixes the stop.
    #:
    #: THE DEFAULTS REPRODUCE BANG-BANG EXACTLY -- a huge gain saturates the clamp at every
    #: error, and a huge slew never binds -- so Anubis and RB-Y1 are byte-identical.
    yaw_kp: float = 1e9
    yaw_slew: float = 1e9
    yaw_rate_max: float = 0.46293550729751587
    #: EXTRA gap, metres, added to plan_nav_to_prim's furniture park (bbox edge - safety -
    #: wheel_r) for PLAIN furniture only -- tables, counters. Handle parks (fridge) are sited
    #: by fridge_siting and the object-on-furniture grasp approach keeps its proven distance;
    #: neither reads this. Exists because wheel_r = 0.23 is a hand-typed stand-in for the whole
    #: base (robot_footprint.py) that is known to undershoot real footprints (Anubis measures
    #: 0.340), and RB-Y1 visibly clips furniture on the drive to the dining table.
    #: Defaults to 0.0, so no robot moves unless its profile opts in.
    furniture_extra_standoff_m: float = 0.0
    #: EXTRA gap, metres, for the OTHER CASE-2 park: an object ON furniture (the grasp
    #: approach), which furniture_extra_standoff_m deliberately leaves alone. Same undershoot,
    #: different reach budget: the grasp target is on the near edge of the counter, the table
    #: place is a full inset further in, so this can be smaller than the fixture gap and must be
    #: tuned separately. MEASURED on kitchen 1550 (2026-08-31): the mug park sat the base front
    #: 0.033 m inside the base cabinet's envelope (cabinet body + handles protrude 0.038 m past
    #: the countertop the park is measured from) -- the bump in every success video was
    #: geometry, not overshoot. Defaults to 0.0, so no robot moves unless its profile opts in.
    object_extra_standoff_m: float = 0.0
    #: Max change in the commanded LINEAR speed per control step, m/s per step -- the
    #: translation twin of yaw_slew. The drive used to step 0 -> default_speed in one frame at
    #: the rotate->translate hand-off. The huge default never binds, so it is bang-bang exact.
    lin_slew: float = 1e9
    #: MEASURED, AND THE GEOMETRIC ARGUMENT DID NOT SURVIVE IT. Parking the AI Worker 0.144 m
    #: left -- which puts bottle0 at base-frame Y=-0.1440, exactly the arm's rest offset,
    #: verified arithmetically -- did not help and looks worse: 1 grasp in 107 closes against
    #: 9 in 386 at 0.05, roughly 2.5x fewer per close. Both counts are small, so this is a
    #: direction rather than a proof; what it does rule out is that the cross-body reach was
    #: the dominant term, which is what the failed-reach deltas (every one negative in Y, by
    #: 0.08 to 0.45 m) had suggested.
    #:
    #: The likely error in the reasoning: where the hand HANGS at rest is not where the arm
    #: REACHES best. Someone should measure the reachable workspace centre before trying again.
    #:
    #: Defaults to 0.05, which is the existing hardcoded value, so no robot moves because of this.
    arm_lateral_bias: float = 0.05

    def __post_init__(self):
        if self.goal_final_yaw is None:
            object.__setattr__(self, "goal_final_yaw", self.goal_reached_yaw)

    def speed_at(self, dist: float) -> float:
        """The commanded speed at `dist` metres from the goal -- the same ramp simvla_video runs.

        Duplicated from the run loop rather than shared with it because the loop's version is
        vectorised over envs in torch; this one exists so the profile can be reasoned about, and
        tested, without a GPU. Any change must move both.
        """
        if dist > self.slowdown_radius:
            return self.default_speed
        return self.min_speed + (self.default_speed - self.min_speed) * (
            dist / self.slowdown_radius
        )


#: Anubis: the previous inline constants, unchanged.
#:
#: RB-Y1: the ramp starts at 0.30 m instead of 0.75 m, so most of it is spent near the object
#: rather than 55 cm out, and both speeds are raised. Terminal accuracy is deliberately NOT touched
#: -- goal_reached_distance and goal_reached_yaw decide where the base parks, and the arm plans its
#: grasp from there.
#:
#: These are the SECOND set of RB-Y1 numbers. The first (0.25 / 0.15 / 0.25) ran a full 32-env
#: campaign successfully -- 6/32 demos, up from 5 -- but read as too fast by eye, so this backs off
#: about 20% while keeping the shape of the change. Recorded because "we already tried faster and
#: it worked, it just looked wrong" is the sort of thing that gets re-litigated otherwise.
#:
#: THIRD set (2026-08-31): the STOP was the complaint, not the cruise. Phase 2 zeroes the command
#: in one step the moment the base enters the 0.02 m band, and it bypasses lin_slew (slewing the
#: stop would coast ~2 cm past the goal, straight into the counter gap). So the only lever is
#: the floor the ramp decays to: min_speed 0.13 -> 0.05 and the ramp starts at 0.35 m instead
#: of 0.30, so the base ARRIVES at ~0.06 m/s and the final step is 2.3x smaller. Cost: the last
#: 0.35 m takes ~3.2 s instead of ~1.9 s. Cruise (0.20 m/s) is untouched.
#:
#: Resulting commanded speed over the final approach, RB-Y1 against Anubis:
#:     0.35 m  0.200 vs 0.120            0.20 m  0.136 vs 0.113
#:     0.10 m  0.093 vs 0.107            0.02 m  0.059 vs 0.101 (the stop step)
#: At the worst case (0.20 m/s, 20 Hz) that is 10 mm per step against a 20 mm stop RADIUS, i.e. a
#: 40 mm diameter: the base cannot step over its own goal band, which is the failure that would
#: turn "faster" into "never arrives".
PROFILES: dict[str, NavProfile] = {
    "anubis": NavProfile(
        goal_reached_distance=0.02,
        goal_reached_yaw=0.1,
        default_speed=0.15,
        min_speed=0.1,
        slowdown_radius=0.75,
        # On public kitchen 813 the wrist target was 0.633 m ahead of base, beyond the measured
        # 0.555 m grasp reach; the jaw missed the mug axis by 0.13-0.19 m across reset samples.
        # Close the object-on-counter park by 0.10 m. The measured forward footprint is 0.2325 m,
        # leaving about 0.178 m from the cabinet edge with the template's 0.28 m safety.
        object_extra_standoff_m=-0.10,
    ),
    "rby1": NavProfile(
        goal_reached_distance=0.02,
        # Bearing gate stays TIGHT: phase 1 drives along the body x axis, so any slack here is
        # driven straight into the floor as off-course travel. Measured at 0.35: distance closed
        # at 0.3 mm/frame and the robot never arrived.
        goal_reached_yaw=0.1,
        default_speed=0.20,
        min_speed=0.05,
        slowdown_radius=0.35,
        # ...but the FINAL turn gate is loose. RB-Y1 parks exactly on the sink goal
        # (dist 0.001-0.003 m) and then has to turn 90 degrees in place with its arms extended and
        # an effort-limited base (100 against damping 1745). At 0.1 rad that turn never closed and
        # the step after it was never reached in three full runs.
        goal_final_yaw=0.35,
        # ON. Turned off while the effort-limit fix was under test, on the strength of an offline
        # simulation showing the blind law converges from every legal spawn -- but that simulation
        # has NO WALLS and NO DISTURBANCES, and the kitchen does. MEASURED on kitchen 926
        # (job 2094569): env2 drove to base (-0.913, +0.557), which is outside BOTH legal spawn
        # bands, and sat wedged in a wall corner at dist_to_goal 2.245 for the rest of the episode
        # while other envs reached the same goal at 0.169. Phase 1 consults the goal once, in
        # phase 0, and then drives its heading forever; nothing recovers a robot pushed off that
        # line. The clean-plant result was true and irrelevant.
        closed_loop_approach=True,
        # RB-Y1 rotation used to stop in ONE control step at the phase gate and translate the
        # next -- visible as a jerk in every campaign video. Same shaped law and values the AI
        # Worker proved: turns over 26.5 deg still run at full 0.463 rad/s, the last 26.5 deg
        # decays proportionally, and the rate may change by at most 0.046 rad/s per 20 Hz step.
        yaw_kp=1.0,
        yaw_slew=0.046,
        # See the field note. 0.14 is a MEASURED compromise, not a guess. RB-Y1's base
        # extends 0.345 m forward of its origin (rby1m model.urdf base_link), so the first
        # 0.10 left a 0.105 m gap the place lean closed on video (job 2122894). But 0.18
        # starved pose-bank collection to 0 successes in two independent kitchen-1550 runs
        # (jobs 2123567, 2123944; banks of 5 and 2 poses, all high-z): the table place sits
        # at the edge of the arm's ~0.555 m reach, and every centimetre of park clearance is
        # a centimetre of reach spent. 0.14 splits the window.
        furniture_extra_standoff_m=0.14,
        # Public kitchen-813 candidate-0 still missed by 0.216 m with a -0.05 m adjustment:
        # the measured wrist target remained about 0.574 m ahead and 0.086 m lateral in the
        # parked base frame, beyond the ~0.555 m grasp reach. Move the object-on-counter park
        # another 0.05 m inward; the authored target should then be ~0.49 m from the object-facing
        # base frame, while nominal front clearance remains positive at about 0.065 m (measured
        # footprint 0.345 m, wheel radius 0.23 m, safety 0.28 m). This is a reachability
        # experiment requiring a fresh clearance check and GPU collection; the 0.14 m furniture
        # margin remains independent and preserves the field measurement for kitchen-1550.
        object_extra_standoff_m=-0.10,
        # 0.02 m/s per 20 Hz step: 0 -> 0.20 m/s in 10 steps (0.5 s), the same ramp-in feel
        # as the yaw slew, ending the one-frame lurch when translation starts.
        lin_slew=0.02,
    ),
    #: AI WORKER (FFW_SG2). Started from RB-Y1's shape rather than Anubis's, because the two
    #: share the thing that drove RB-Y1's numbers: a planar dummy base that has to turn in
    #: place with the arms extended, against a drive that is a velocity PD with stiffness ~0.
    #: Anubis's tight 0.1 rad final gate never closed that turn in three full runs.
    #:
    #: THESE ARE INHERITED, NOT MEASURED ON THIS ROBOT. AI Worker is 1.61 m to RB-Y1's 1.47
    #: and carries its mass differently, so retune from a [navtr] trace rather than trusting
    #: them. If the base creeps at a fraction of commanded speed, read the effort-limit note
    #: on RB-Y1's base actuator first -- that failure looked exactly like a steering bug and
    #: was not one. AIWORKER_CFG ships effort_limit_sim=1e5 on the base, well above the ~349 N
    #: the PD term can demand at these speeds, so it should not bind here.
    "aiworker": NavProfile(
        goal_reached_distance=0.02,
        goal_reached_yaw=0.1,
        default_speed=0.20,
        min_speed=0.13,
        slowdown_radius=0.30,
        goal_final_yaw=0.35,
        closed_loop_approach=True,
        arm_lateral_bias=0.05,   # see the note on the field: 0.144 MEASURED WORSE
        # 1.0 rad/s per rad: turns over 26.5 deg still run at the full 0.463 rad/s and
        # only the last 26.5 deg ramps. 0.046 rad/s per step reaches full speed in ~10
        # steps (0.5 s) instead of one, which is what stops the column ringing.
        yaw_kp=1.0,
        yaw_slew=0.046,
        # The public kitchen-813 mug grasp parks at 0.76 m in the AI Worker's base frame; a 0.10 m
        # closer park moved the best left-arm wrist to 0.075 m short with 0.020 m jaw lateral error
        # before contact disturbed the mug. Tightening further to -0.15 worsened the sampled reach
        # and is not retained. The -0.10 profile keeps 0.185 m nominal counter clearance. Regenerate
        # the goal JSON for this author-time adjustment to take effect.
        object_extra_standoff_m=-0.10,
    ),
}

#: Per-field environment overrides, for tuning a run without editing this file.
_OVERRIDES = {
    "SIMVLA_NAV_GOAL_DISTANCE": "goal_reached_distance",
    "SIMVLA_NAV_GOAL_YAW": "goal_reached_yaw",
    "SIMVLA_NAV_FINAL_YAW": "goal_final_yaw",
    "SIMVLA_NAV_DEFAULT_SPEED": "default_speed",
    "SIMVLA_NAV_MIN_SPEED": "min_speed",
    "SIMVLA_NAV_SLOWDOWN_RADIUS": "slowdown_radius",
    "SIMVLA_NAV_FURNITURE_EXTRA": "furniture_extra_standoff_m",
    "SIMVLA_NAV_OBJECT_EXTRA": "object_extra_standoff_m",
    "SIMVLA_NAV_LIN_SLEW": "lin_slew",
    "SIMVLA_NAV_YAW_SLEW": "yaw_slew",
    "SIMVLA_NAV_YAW_RATE_MAX": "yaw_rate_max",
}


def profile_for(robot: str, env=None) -> NavProfile:
    """This robot's profile, with any environment overrides applied.

    An unknown robot raises rather than falling back to a default: a silent fallback would drive
    a new robot at another one's tuning and look like a control problem.

    A malformed override raises for the same reason -- an ignored override is indistinguishable in
    the log from one that was applied.
    """
    env = os.environ if env is None else env
    if robot not in PROFILES:
        raise KeyError(
            f"no nav profile for robot {robot!r}; have {', '.join(sorted(PROFILES))}"
        )
    base = PROFILES[robot]
    changes = {}
    for var, field in _OVERRIDES.items():
        raw = (env.get(var) or "").strip()
        if not raw:
            continue
        try:
            changes[field] = float(raw)
        except ValueError:
            raise ValueError(f"{var}={raw!r} is not a number") from None
        if not math.isfinite(changes[field]):
            raise ValueError(f"{var} must be finite")
        if field in {"goal_reached_yaw", "goal_final_yaw"} and not 0 < changes[field] <= math.pi:
            raise ValueError(f"{var} must be in (0, pi] radians")
        if field in {"yaw_rate_max", "yaw_slew"} and not 0 < changes[field] <= 1:
            raise ValueError(f"{var} must be in (0, 1]")
    if not changes:
        return base
    return NavProfile(**{**base.__dict__, **changes})


def describe(robot: str, p: NavProfile) -> str:
    """One log line, so a run's base tuning is recoverable from its log."""
    return (f"[nav] {robot}: default={p.default_speed} min={p.min_speed} "
            f"slowdown_radius={p.slowdown_radius} goal_dist={p.goal_reached_distance} "
            f"| speed at 0.20 m = {p.speed_at(0.20):.3f} m/s")
