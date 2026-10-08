"""SimVLA: derive the run config from the task, so nobody transcribes it again.

Stdlib only (plus skill_contract, task_template, task_bind, task_validate).

Every field here used to be a command-line flag a human copied out of a goal file by counting steps.
Two of them shipped wrong in the README, and both failed in ways that do not name themselves:

  --obj_name "bottle0" on a task targeting /world/bowl0  ->  bare KeyError, forty minutes in.
  --sub_grasp_idx_l 999 with --sub_good_goal_count_l 7   ->  999 correctly disables the left grasp
      check (the left arm grasps a drawer handle, which belongs to an articulation, not a rigid
      body, so the rigid-object check cannot apply). The run is then told to collect seven left-arm
      successes that the disabled check is the only thing that could produce. It reaches
      stage=collect_left and spins there until its time limit, printing nothing.

Here both come from the same bound prims. They cannot disagree, because they are not stated. The
coupling is enforced on BOTH arms: --sub_grasp_idx_r 999 with --sub_good_goal_count_r 7 is the
identical spin, and it is reachable the moment a template opens the drawer with the RIGHT arm.

Two facts do all the work, and they are the two a human counting steps gets wrong:

  WHAT each arm grasps. Not "the step whose skill is arm.grasp" — an arm reaches an object with
  any of task_validate.GRASP_SKILLS (arm.grasp, arm.handle_grasp, arm.fridge_handle_grasp), and the
  reference task's LEFT arm uses arm.handle_grasp. Matching only arm.grasp made the left target
  look ABSENT, so the rigid-body guard below never ran and the right answer came out for the wrong
  reason. Both arms now bind through the same set — and through the first grasp in it whose target
  is a RIGID BODY, not simply the first. An arm may grasp a fridge handle to open the fridge and
  then grasp the bowl; the handle is a means, the bowl is the task. Taking the first grasp step
  gave obj_name="none" for a task whose object is right there in the second one.

  WHEN a grasp can be checked. The check fires at the step AFTER the arm returns home — but only
  while that arm is still holding THE MANIPULATION OBJECT. It is not enough to know THAT the
  gripper closed; the scan has to know WHAT it closed on. The order matters four ways. An arm is
  free to reset home before it goes to grasp (`arm.reset, nav, arm.grasp, close, arm.reset`), and a
  scan that takes an arm's FIRST reset while merely knowing a close happens somewhere would point
  the check at a nav step, gripper empty. An arm is free to close, place, and OPEN before it resets
  (`grasp, close, place, open, arm.reset`), and a scan that never forgets a close once seen would
  point the check at a reset with an already-empty gripper — the object is on the counter, every
  candidate fails the 0.15 m distance check, and the run spins to its time limit printing nothing.
  An arm is free to grasp, close on and reset while holding something that is NOT the object (a
  fridge handle), and a scan that starts at the top of the script would verify the bowl against a
  hand that is holding a door. And an arm is free to do that AFTER it has released the object — one
  arm doing all of v813 (`grasp bowl, close, place, open, handle_grasp, close, reset`) closes its
  gripper twice, and a scan that only latches "a close happened since the bowl's grasp" points the
  check at a hand holding the drawer handle with the bowl already in the drawer.

  So the scan starts at the grasp step that bound the manipulation object, carries the prim of the
  most recent grasp skill on that arm, and counts a close only when the hand is AT the manipulation
  object. `gripper.set(grasp=False)` clears the state again. The check therefore only ever lands on
  a reset that happens WHILE the arm holds THAT object. An arm that never resets while holding it
  has no valid check point, and the answer is GRASP_CHECK_DISABLED.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .task_bind import BindingError, BoundPrim
from .task_template import TaskTemplate, role_ref
from .task_validate import GRASP_SKILLS, SQUEEZE_SKILLS, SequenceError, arms_of, gripper_closes

#: No step ever has this index, which is how a grasp check is switched off — and derive_run_config
#: REFUSES a script long enough to contain step 999, because in one the sentinel names a real step
#: and the "safe by construction" pairing below (999 alongside --obj_name "none") stops being safe.
GRASP_CHECK_DISABLED = 999

#: What --target_idx is when NEITHER arm has a manipulation object, and the task therefore randomizes
#: no object at all.
#:
#: --target_idx is a STEP index, not an object index. At reset exactly ONE object is displaced by a
#: ±5 cm pose sample (isaaclab/envs/mdp/events.py:903, stashed in variable.rand_samples_simvla), and
#: simvla_gen.py:1522 adds that same sample to the authored pose of the step where
#: `step_i == target_idx` — `step_i` enumerating the GOAL SCRIPT's steps. So target_idx names the one
#: step whose authored pose must follow the object when the object moves: the manipulation grasp. It
#: reads like an object index in v813 only because the bowl is also that kitchen's object #1.
#:
#: The discipline, then, is not a value but a property: with no object to follow there is no step the
#: sample belongs on, so the sentinel must match NO STEP — `step_i == target_idx` false for every
#: step of every script. Any value a script could enumerate is disqualified, however unlikely a
#: grasp: it would add 5 cm to that step's authored pose to track an object that never moved. 0 is
#: the value this module used to emit, and 0 is step 0.
#:
#: -1 satisfies the property and nothing else does as cheaply: `step_i` comes from `enumerate(goal)`
#: and is >= 0 for every step of every script, so no step can equal it. It is safe to go negative
#: precisely because target_idx is only ever COMPARED against `step_i` (simvla_gen.py:1522) and is
#: never used to subscript the goal — if it were, -1 would silently address the LAST step.
NO_TARGET_IDX = -1

NAV_ACTIONS = ("N", "N_s")


@dataclass(frozen=True)
class RunConfig:
    obj_name: str
    obj_name_l: str
    target_idx: int
    sub_grasp_idx_r: int
    sub_grasp_idx_l: int
    sub_good_goal_count_r: int
    sub_good_goal_count_l: int
    export_groups: str
    task_language: str
    task_type: str

    def to_meta(self) -> dict:
        """The block written into the v2 goal file. simvla_gen reads it; the CLI flags override."""
        return asdict(self)


@dataclass(frozen=True)
class _Grasp:
    """Where an arm picks up its manipulation object, and which object that is."""
    step_index: int
    prim: BoundPrim


def _grasp_prims(t: TaskTemplate, bindings: dict[str, BoundPrim]) -> dict[int, BoundPrim]:
    """Step index -> bound prim, for EVERY grasp step in the script. Raises if one cannot be bound.

    This is the module's only resolution site, and it is total: after it, a grasp step's prim is a
    BoundPrim, never a None that the rigid-body guard below would read as "not a rigid body".

    UNBOUND and NOT-A-RIGID-BODY are different facts, and sharing a code path between them is how a
    whole run went quietly to waste. `arm.grasp(A_r, "/world/bowl0")` — a literal path, which
    validate_template explicitly permits (task_template.py:117) — and `arm.grasp(A_r, @target)` with
    `target` missing from the bindings both used to resolve to None, and None was read as "this arm
    grasps nothing rigid": obj_name="none", target_idx=-1, sub_grasp_idx_r=999,
    sub_good_goal_count_r=0. An EMPTY bindings dict gave byte-identical output. Both pass
    validate_template AND validate_sequence, neither crashes and neither spins — the run records a
    full dataset with the grasp check switched off and the object never randomized. Silently useless
    data is the one failure mode this module exists to abolish, so an unresolvable grasp is loud.

    A literal path is refused rather than used because a BoundPrim is not decoration: is_rigid_body
    decides whether a grasp check is even possible, and `name` is the key env.scene.rigid_objects is
    dereferenced by. A raw string carries neither.
    """
    prims: dict[int, BoundPrim] = {}
    for i, step in enumerate(t.steps):
        if step.skill not in GRASP_SKILLS:
            continue
        value = step.params.get("prim_path", "")
        ref = role_ref(value)
        if ref is None:
            raise BindingError(
                f"task {t.name!r} step {i} ({step.skill}): prim_path {value!r} is a literal prim "
                f"path, not a @role. It carries no BoundPrim, so this module cannot tell whether it "
                f"is a rigid body (checkable) or part of an articulation (not), and cannot name the "
                f"key env.scene.rigid_objects is dereferenced by. A template may write a literal; a "
                f"run config cannot be derived from one. Declare a role and grasp that."
            )
        prim = bindings.get(ref)
        if prim is None:
            bound = ", ".join(sorted(bindings)) or "nothing is bound"
            raise BindingError(
                f"task {t.name!r} step {i} ({step.skill}): prim_path {value!r} names no bound role "
                f"({bound}). An UNBOUND grasp is not the same fact as a grasp on something that is "
                f"not a rigid body — the second disables the grasp check honestly, the first would "
                f"blank obj_name, target_idx and the check for a task whose object is right there in "
                f"the script, and record a whole dataset without ever saying so."
            )
        prims[i] = prim
    return prims


def _grasp_target(t: TaskTemplate, prims: dict[int, BoundPrim], arm: str) -> _Grasp | None:
    """This arm's manipulation object: its first grasp step whose bound prim IS a rigid body.

    GRASP_SKILLS and arms_of are imported from task_validate rather than redeclared, because a
    second copy of either is a copy that drifts — and the drift is silent. `arm.grasp` alone would
    miss the reference task's left arm (`arm.handle_grasp`) and report "this arm grasps nothing";
    an arms_of blind to `A_b` would say the same of `arm.grasp(A_b, @target)`, which every arm
    skill declares and the validator accepts.

    FIRST RIGID grasp, not first grasp. A handle is a means, not the object of the task:

        A_r arm.fridge_handle_grasp(@fridge_handle), G_r close, N open, G_r open,
        A_r arm.grasp(@target), G_r close, A_r arm.reset, N close

    is one arm opening a fridge and then picking up the bowl inside it. Its first grasp binds a
    fridge handle, which belongs to an articulation; the guard below then blanks obj_name, target_idx
    and the grasp check — for a task whose object is right there in the second grasp. The run does
    not crash (`--obj_name none` is only ever dereferenced under the guard of a live grasp check,
    and "none" is emitted only alongside GRASP_CHECK_DISABLED, which matches no step: see
    simvla_gen.py:1959-1963 — the pairing is safe by construction). It quietly stops checking the
    grasp, and quietly randomizes the wrong step. Both are lies told in silence, which is the class
    of failure this module exists to remove.

    The guard is on the TARGET, not the skill name, and it applies to both arms, because either arm
    can be told to pull a drawer open. A grasp check compares the end-effector against
    env.scene.rigid_objects[obj_name]; a drawer handle is part of an articulation and simply is not
    in that dict, so checking it is not merely unhelpful — it is impossible (THAT is where the bare
    KeyError lives: a real handle name handed to the check, not the sentinel "none"). An arm whose
    every grasp is non-rigid therefore HAS no manipulation object (that is v813's left arm): no
    obj_name, no target_idx, no check.

    `prims` is _grasp_prims' output, so every grasp step here HAS a bound prim: "unbound" was refused
    before this function ran, and cannot arrive disguised as "not a rigid body".
    """
    for i, step in enumerate(t.steps):
        if step.skill in GRASP_SKILLS and arm in arms_of(step.action):
            if prims[i].is_rigid_body:
                return _Grasp(step_index=i, prim=prims[i])
    return None


def _grasp_check_index(
    t: TaskTemplate, prims: dict[int, BoundPrim], arm: str, grasp: _Grasp
) -> int:
    """The index at which the grasp is verified: the step after the first `arm.reset` that follows
    this arm closing its gripper ON THE MANIPULATION OBJECT, with no intervening release. By then the
    arm has returned home still holding that object — which is exactly what the check asks: is
    env.scene.rigid_objects[obj_name] still within 0.15 m of the end-effector.

    The scan therefore tracks TWO things, and the second is the one every earlier version was missing:
    `held`, the prim of the most recent grasp skill on this arm — WHAT the hand is on — and `closed`,
    whether the gripper shut while the hand was on the manipulation object. Knowing only THAT a
    gripper closed is not knowing what it closed on, and a hand can be redirected between the grasp
    and the close.

    The ordering cuts four ways, and each one is a real authoring:

    `arm.reset, N_s, arm.grasp, close, arm.reset, N` — home first, THEN go get the thing. The reset
    that verifies the grasp is the SECOND one. A scan gated only on "a close occurs somewhere in the
    episode" returns the first reset: a nav step, gripper empty.

    `arm.grasp, close, arm.place, open, arm.reset` — close on the object, put it down, let go, and
    only then go home. There is no valid check point at all. A release clears `closed`, so the reset
    that follows it does not count as holding anything.

    `arm.fridge_handle_grasp, close, nav.open_articulation, arm.reset` before the bowl is ever
    touched — the arm pulls the fridge open and goes home holding the DOOR. What rules that reset out
    is `held`: the hand is on the fridge handle there, not on the bowl, so the close does not count.
    The `from_step` start offset does NOT carry this — it is redundant, and provably so: start the
    scan at 0 and all of this module's tests stay green, because `held` cannot be the manipulation
    prim before that prim's own first grasp. Keep the offset if you like; do not credit it, and do
    not delete `held` on the strength of it. `held` is the thing carrying the correctness.

    And the one no start offset could catch, because it happens after the grasp: ONE arm doing the
    whole of v813 —

        N_s, arm.grasp(@target), close, arm.place(@container), open,
        arm.handle_grasp(@handle), close, arm.reset, arm.pause

    The arm picks up the bowl, puts it in the drawer, lets go, grasps the DRAWER HANDLE, closes on
    it, and resets home. A scan that latches any close after the bowl's grasp returns 8 — a step at
    which the bowl is lying in the drawer and the hand is holding a door. `held` is the handle there,
    not the bowl, so the close does not count, and the honest answer is that this arm never returns
    home holding the bowl: GRASP_CHECK_DISABLED. (The same reading covers a hand redirected BEFORE
    the close, `arm.grasp(@target), arm.handle_grasp(@handle), close, arm.reset`, and one redirected
    to another rigid object, `arm.grasp(@target), arm.grasp(@mug), close, arm.reset`. All three
    validate; all three used to derive a check index pointing at the wrong thing in the hand.)

    Every one of them points the check at a step where the gripper does not hold the object. Every
    candidate env then fails the 0.15 m distance check, every env is reset, and the run never leaves
    its collect stage: it spins to its time limit printing nothing.

    An arm that never closes on the object — or has released it before its next reset — has no grasp
    to check, and the scan falls through to GRASP_CHECK_DISABLED. That is a different disablement
    from "the arm has no rigid target at all", which is _grasp_target's to decide.
    """
    held: BoundPrim | None = None
    closed = False
    for i, step in enumerate(t.steps[grasp.step_index:], start=grasp.step_index):
        if arm not in arms_of(step.action):
            continue
        if step.skill in GRASP_SKILLS:
            # The hand is now on THIS prim, whatever it was on before.
            held = prims[i]
        elif step.skill == "gripper.set":
            # A close counts only if the hand is on the manipulation object. A release always clears.
            closed = gripper_closes(step) and held == grasp.prim
        elif closed and step.skill == "arm.reset":
            nxt = i + 1
            return nxt if nxt < len(t.steps) else GRASP_CHECK_DISABLED
    return GRASP_CHECK_DISABLED


def derive_run_config(
    t: TaskTemplate,
    bindings: dict[str, BoundPrim],
    sub_good_goal_count: int = 7,
) -> RunConfig:
    """`sub_good_goal_count` is per-arm run tuning: how many verified grasps to bank before the run
    samples only from them. It applies to whichever arm has a LIVE check; an arm whose check is
    disabled always gets 0, on both sides, because a disabled check is the only thing that could
    have produced the successes it would then be waiting for.

    PRECONDITION: `t` is VALIDATED — validate_template(t) and validate_sequence(t) have both passed.
    Nothing here re-checks them, and nothing here is meaningful without them: this function derives
    facts FROM a coherent script, it does not decide whether the script is coherent. What it does
    check is what those two cannot see — the bindings, and the things a run config simply cannot
    express (below). Each of those is a REFUSAL, because every one of them otherwise produces a
    config that runs to EXIT=0 and records a dataset that is quietly worthless.
    """
    if not t.steps:
        # Every field below is a fact about a script. There is no script, so there is nothing to say
        # — and "Manipulation", which the action scan used to fall through to, says that an empty
        # task manipulates something. It is the sort of confident-sounding wrong answer that this
        # module exists to stop shipping to a GPU.
        raise ValueError(f"task {t.name!r} has no steps: an empty script has no run config")

    if len(t.steps) > GRASP_CHECK_DISABLED:
        # "No step ever has this index" is an assertion about scripts, and it holds only below 1000
        # steps. In a longer one, step 999 EXISTS: a task with no rigid grasp derives obj_name="none"
        # and sub_grasp_idx_r=999, the mask fires at that real step, and the check dereferences
        # env.scene.rigid_objects["none"] — the bare KeyError this module exists to prevent, arriving
        # by way of the sentinel that was supposed to prevent it. One guard makes the docstring's
        # "safe by construction" true instead of merely likely.
        raise SequenceError(
            f"task {t.name!r} has {len(t.steps)} steps, so step {GRASP_CHECK_DISABLED} exists. "
            f"GRASP_CHECK_DISABLED = {GRASP_CHECK_DISABLED} switches a grasp check off by naming NO "
            f"step, and it is emitted alongside --obj_name 'none'; in a script this long it names a "
            f"REAL step, the check fires there, and env.scene.rigid_objects['none'] is a KeyError. "
            f"The sentinel is only safe below {GRASP_CHECK_DISABLED} steps."
        )

    # Resolve every grasp step's prim FIRST, and refuse the unresolvable. An unbound grasp must not
    # reach the rigid-body guard, which would read it as "this arm grasps nothing rigid".
    prims = _grasp_prims(t, bindings)

    # _grasp_target applies the rigid-body guard: an arm that grasps only handles has no manipulation
    # object, and None here means exactly that — never "the prim did not resolve".
    right = _grasp_target(t, prims, "right")
    left = _grasp_target(t, prims, "left")

    # THE manipulation object — singular, and it must be ONE arm's. The reset event randomizes
    # exactly one asset (the one whose x-range is (-0.05, 0.05): events.py:903), and target_idx names
    # the ONE step whose authored pose follows it. Two refusals below; both are cases where a config
    # CAN be emitted and is wrong in silence, which is worse than no config at all.
    #
    # A_b first, because a two-handed grasp binds both arms to the same object and would otherwise
    # trip the two-object refusal, whose message would then name one prim twice.
    for grasp in (right, left):
        if grasp is None:
            continue
        action = t.steps[grasp.step_index].action
        if not action.endswith("_b"):
            continue
        if t.steps[grasp.step_index].skill in SQUEEZE_SKILLS:
            # A squeeze pins the object (--obj_fix_init), so the reset event applies no ±5 cm noise to
            # any half — the left-half-only patch this refusal guards against has nothing to patch.
            # A_b is in fact the CORRECT action for a squeeze: both palms close on the object at once.
            continue
        # The executor corrects only the LEFT half of an A_b payload: an A_b payload is 14 floats,
        # [:7] LEFT and [7:14] RIGHT (goal_format.py:28), and the object-tracking noise at
        # simvla_gen.py:1523 is `pose_exec[:6] += noise` — the left half alone. (Six lines above, the
        # AUTHORING noise patches both halves explicitly, `pose_save[0:3]` and `pose_save[7:10]`, so
        # this is an oversight in the executor, not a convention.) On such a step the bowl moves up to
        # 5 cm at reset, the left arm's pose follows it and the RIGHT arm's chases the stale authored
        # pose — while the right arm is the one being checked, and 5 cm is well inside the 0.15 m
        # threshold, so the check PASSES and nothing prints. Widening the executor's patch is a
        # runtime change needing its own GPU verification; refusing what this module cannot express
        # correctly is this module's job.
        raise SequenceError(
            f"task {t.name!r}: the manipulation grasp of {grasp.prim.prim_path!r} is step "
            f"{grasp.step_index}, whose action is {action!r} — it drives BOTH arms. That step is the "
            f"one --target_idx names, and the executor patches only the LEFT half of an A_b payload "
            f"(14 floats: [:7] left, [7:14] right; simvla_gen.py:1523 adds the reset event's ±5 cm "
            f"sample with `pose_exec[:6] += noise`). The right arm would go on reaching for where the "
            f"object no longer is — by under 5 cm, inside the 0.15 m grasp check, so the run would "
            f"record the whole dataset and never say so. Author the manipulation grasp on ONE arm "
            f"(A_r or A_l); the other arm may still grasp it on its own step."
        )

    # EXPERIMENTAL bimanual escape hatch. The two refusals below both exist for ONE reason: the
    # reset event randomizes an asset by ±5 cm and only target_idx follows it, so a second arm on
    # the same object chases a stale pose. Disable the randomization (run with
    # simvla_video --obj_fix_init) and that hazard is gone: both arms grasp the fixed authored pose,
    # nothing moves, no step needs to track a displacement. This gate is env-var-scoped so the
    # DEFAULT stays a refusal — no other caller's invariant changes. Pending a proper
    # TaskTemplate.fix_object field, it lets a wide-object bimanual grasp be authored and driven.
    _bimanual_fixed = bool(os.environ.get("SIMVLA_BIMANUAL_FIXED"))

    # A squeeze binds BOTH arms to ONE object BY DESIGN and pins the object (a fixed-pose heuristic,
    # not a randomized pick), so the ±5 cm/target_idx hazard the two-arm refusal guards against
    # cannot arise. When both manipulation grasps are squeeze skills this is that intended pattern —
    # the proper, template-recorded form of what SIMVLA_BIMANUAL_FIXED did out of band.
    _both_squeeze = (
        right is not None and left is not None
        and t.steps[right.step_index].skill in SQUEEZE_SKILLS
        and t.steps[left.step_index].skill in SQUEEZE_SKILLS
    )
    _fixed_object = _bimanual_fixed or _both_squeeze

    if right is not None and left is not None and not _fixed_object:
        # Two rigid manipulation objects, ONE randomized asset — and WHICH one the environment
        # randomizes is a fact of the env config (events.py: whichever asset has x-range
        # (-0.05, 0.05)), recorded nowhere in the TaskTemplate and nowhere in a BoundPrim. The
        # information is genuinely absent, so no derivation can infer it. Pinning target_idx to the
        # right arm — which this module used to do — is a coin flip: when it loses, the right hand
        # collects a displacement its object never underwent and the left arm's object moves with no
        # step tracking it. Both errors are under 5 cm, both sit inside the 0.15 m check, and neither
        # prints. Refuse instead. (Which is why there is no right-arm precedence left below: with
        # both arms bound we do not get there.)
        raise SequenceError(
            f"task {t.name!r} grasps a rigid manipulation object with BOTH arms: right -> "
            f"{right.prim.prim_path!r} (step {right.step_index}), left -> {left.prim.prim_path!r} "
            f"(step {left.step_index}). The reset event randomizes exactly ONE asset and --target_idx "
            f"names the ONE step whose authored pose follows it. WHICH asset is randomized lives in "
            f"the env config (events.py: the asset whose x-range is (-0.05, 0.05)) and is recorded "
            f"nowhere in the template or its bindings, so no derivation can tell which of these two "
            f"the run will move — and guessing wrong is silent: the miss is under 5 cm, well inside "
            f"the 0.15 m grasp check. Give the manipulation object to ONE arm."
        )

    idx_r = _grasp_check_index(t, prims, "right", right) if right else GRASP_CHECK_DISABLED
    idx_l = _grasp_check_index(t, prims, "left", left) if left else GRASP_CHECK_DISABLED

    # At most one arm has a manipulation object — the refusal above is what makes this true — so this
    # is not a precedence rule but a selection of the only candidate there is. It is not always the
    # right arm's: in the mirror of v813 the right arm opens the drawer and the LEFT arm carries the
    # bowl, and reading target_idx off the right arm unconditionally emitted the no-target sentinel
    # there. The bowl then moved 5 cm at reset while the left arm's authored grasp pose stayed where
    # the bowl no longer was: the grasp misses, the 0.15 m check can still pass, and the failure never
    # announces itself.
    manipulation = right or left

    actions = {s.action for s in t.steps}
    has_nav = any(a in NAV_ACTIONS for a in actions)
    has_manip = any(a not in NAV_ACTIONS for a in actions)
    task_type = (
        "NavManipulation" if has_nav and has_manip
        else "Navigation" if has_nav
        else "Manipulation"
    )

    return RunConfig(
        obj_name=right.prim.name if right else "none",
        obj_name_l=left.prim.name if left else "none",
        # A STEP index: the manipulation grasp's own step, whichever arm makes it — the step whose
        # authored pose has to follow the object when the reset event displaces it. With no such
        # grasp on either arm, nothing is randomized and the sentinel matches no step. See
        # NO_TARGET_IDX.
        # NO_TARGET_IDX under the bimanual-fixed hatch: the object is pinned (--obj_fix_init), so no
        # step's authored pose has to follow a displacement — naming one would patch it with a noise
        # that never occurs.
        target_idx=NO_TARGET_IDX if _fixed_object
        else (manipulation.step_index if manipulation else NO_TARGET_IDX),
        sub_grasp_idx_r=idx_r,
        sub_grasp_idx_l=idx_l,
        # 0 if and only if the check is off. Asking a disabled check for successes is the bug — and
        # it is the same bug on either arm, so the coupling is stated twice, identically.
        sub_good_goal_count_r=0 if idx_r == GRASP_CHECK_DISABLED else sub_good_goal_count,
        sub_good_goal_count_l=0 if idx_l == GRASP_CHECK_DISABLED else sub_good_goal_count,
        export_groups=";".join(",".join(str(i) for i in g) for g in t.subtask_groups),
        task_language=t.language,
        task_type=task_type,
    )
