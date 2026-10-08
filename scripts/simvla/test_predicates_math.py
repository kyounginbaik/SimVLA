"""The condition primitives' geometry, as pure functions of tensors.

This is the point of the predicates_math / mdp.composed split: the geometry inlined nine times in
kitchen/mdp/terminations.py can only run on a GPU against a live stage, which is why no version of
it has ever been tested. Here it is a function of its arguments.

Run: python -m pytest scripts/simvla/test_predicates_math.py -q
torch only: no Omniverse, no isaaclab.
"""

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from predicate_contract import REGISTRY, omittable_params, resolve_roles, validate_spec
from predicates_math import IMPL, EvalContext, compile_spec
from skill_contract import Bool, Choice


def ctx(**kw):
    kw.setdefault("num_envs", 2)
    kw.setdefault("device", "cpu")
    return EvalContext(**kw)


def test_every_declared_primitive_has_an_implementation():
    """The half-wired check. A primitive the composer can offer but nothing can evaluate is the
    G_b bug in another costume."""
    assert sorted(IMPL) == sorted(REGISTRY)


def test_every_param_a_spec_may_omit_has_a_python_default():
    """validate_spec lets a param be omitted only when it "takes its declared default" — but
    compile_spec passes ONLY the params the spec carries, so the default has to be on the Python
    side or nothing supplies it. `{"eef_home": {"radius": 0.18}}` validated clean and then raised
    "eef_home() missing 1 required keyword-only argument: 'arm'" at Isaac Lab env load: unreachable
    from the composer (which always writes Choice defaults), reachable from a hand-written
    template, and it dies in the one place no test can reach.

    Driven off predicate_contract.omittable_params — the very set validate_spec uses — so a
    primitive declared LATER with an omittable param and no Python default fails here, not on a GPU.
    """
    for pid, declared in REGISTRY.items():
        sig = inspect.signature(IMPL[pid])
        for name in sorted(omittable_params(declared)):
            assert name in sig.parameters, f"{pid}() has no {name!r} param at all"
            assert sig.parameters[name].default is not inspect.Parameter.empty, (
                f"{pid}.{name} may be omitted from a spec, so {pid}() must default it"
            )


def test_the_python_defaults_are_the_declared_ones():
    """Same value, not merely some value. predicates_math cannot import predicate_contract (it is
    torch-only, and the registry must stay importable without torch), so the two declarations are
    held together here: an arm Choice re-defaulted to 'both' in the registry and left at 'right' in
    the implementation would silently score a two-arm condition as one-arm."""
    for pid, declared in REGISTRY.items():
        sig = inspect.signature(IMPL[pid])
        for p in declared.params:
            if isinstance(p, (Choice, Bool)) and p.default is not None:
                assert sig.parameters[p.name].default == p.default, (
                    f"{pid}.{p.name}: registry says {p.default!r}, "
                    f"{pid}() defaults to {sig.parameters[p.name].default!r}"
                )


def test_an_omitted_choice_validates_and_compiles_and_evaluates():
    """The three steps the bug fell between: validate_spec accepted it, compile_spec raised, and
    nothing ever evaluated it. All three, on the exact spec from the finding."""
    spec = {"all": [{"eef_home": {"radius": 0.18}}]}
    assert validate_spec(spec, ["target"], "success") == []
    fn = compile_spec(spec)
    c = ctx(
        eef_pos_base={"right": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                      "left": torch.zeros(2, 3)},
        home={"right": torch.zeros(3), "left": torch.zeros(3)},
    )
    assert fn(c).tolist() == [True, False], "and it means the declared default arm: right alone"


@pytest.mark.parametrize("spec,extra", [
    ({"obj_near_eef": {"role": "@b", "radius": 0.2}},
     dict(obj_pos_w={"b": torch.zeros(2, 3)},
          eef_pos_w={"right": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                     "left": torch.zeros(2, 3)})),
    ({"gripper_open": {"gap": 0.079}},
     dict(gripper_gap={"right": torch.tensor([0.09, 0.05]), "left": torch.tensor([0.09, 0.09])})),
])
def test_the_other_two_arm_primitives_compile_without_an_arm(spec, extra):
    """obj_near_eef and gripper_open declare the same shared ARM Choice, so they had the same hole.
    Each is given a LEFT arm that would change the answer if the default silently became 'both'.

    Validated in the AUTHORED form ('@b', which is what a template carries) and compiled in the
    RESOLVED form emit writes — the same two forms the real pipeline has."""
    from predicate_contract import resolve_roles
    assert validate_spec(spec, ["b"], "retry") == []
    resolved = resolve_roles(spec, {"b": "b"})
    assert compile_spec(resolved)(ctx(**extra)).tolist() == [True, False]


def test_obj_z_with_both_bounds_is_a_band():
    c = ctx(obj_pos_w={"bowl0": torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.9]])})
    out = IMPL["obj_z"](c, role="bowl0", lo=0.6, hi=0.8)
    assert out.tolist() == [True, False]


def test_obj_z_with_only_hi_is_below():
    c = ctx(obj_pos_w={"bowl0": torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.5]])})
    out = IMPL["obj_z"](c, role="bowl0", hi=0.3)
    assert out.tolist() == [True, False]


def test_obj_z_with_only_lo_is_above():
    c = ctx(obj_pos_w={"bowl0": torch.tensor([[0.0, 0.0, 0.95], [0.0, 0.0, 0.85]])})
    out = IMPL["obj_z"](c, role="bowl0", lo=0.90)
    assert out.tolist() == [True, False]


def test_obj_z_is_strict_at_both_bounds():
    c = ctx(obj_pos_w={"bowl0": torch.tensor([[0.0, 0.0, 0.6], [0.0, 0.0, 0.8]])})
    out = IMPL["obj_z"](c, role="bowl0", lo=0.6, hi=0.8)
    assert out.tolist() == [False, False]


def test_obj_near_eef_is_strict_at_the_radius():
    c = ctx(
        obj_pos_w={"b": torch.tensor([[0.10, 0.0, 0.0], [0.11, 0.0, 0.0]])},
        eef_pos_w={"right": torch.zeros(2, 3)},
    )
    out = IMPL["obj_near_eef"](c, role="b", arm="right", radius=0.10)
    assert out.tolist() == [False, False]


# ---------------------------------------------------------------- obj_between_eefs
#
# obj_near_eef alone was the certified-far-less-than-it-claims bug this predicate replaces for
# bimanual holds: a 0.2 m sphere around EACH palm never required the two to be on OPPOSITE sides
# of the object, in contact, at the SAME instant. Every case below is a scene obj_near_eef(arm=
# "both") would have accepted and a real two-handed hold cannot produce.


def test_obj_between_eefs_accepts_a_genuine_opposed_squeeze():
    c = ctx(
        obj_pos_w={"b": torch.zeros(1, 3)},
        eef_pos_w={"right": torch.tensor([[0.08, 0.0, 0.0]]),
                   "left": torch.tensor([[-0.08, 0.0, 0.0]])},
    )
    out = IMPL["obj_between_eefs"](c, role="b", near=0.10)
    assert out.tolist() == [True]


def test_obj_between_eefs_rejects_both_palms_on_the_same_side():
    """Both palms close to the object but never OPPOSING it -- e.g. resting on the same
    forearm -- is exactly what a radius-only check cannot tell apart from a hold. Both distances
    here are also within `near`, isolating the missing-opposition defect specifically."""
    c = ctx(
        obj_pos_w={"b": torch.zeros(1, 3)},
        eef_pos_w={"right": torch.tensor([[0.08, 0.0, 0.0]]),
                   "left": torch.tensor([[0.09, 0.0, 0.0]])},
    )
    out = IMPL["obj_between_eefs"](c, role="b", near=0.20)
    assert out.tolist() == [False]


def test_obj_between_eefs_rejects_a_palm_outside_near_even_when_genuinely_opposed():
    """Perfectly antiparallel (cos == -1) is not enough on its own -- each palm must also be
    within contact range. Isolates the missing-proximity defect specifically."""
    c = ctx(
        obj_pos_w={"b": torch.zeros(1, 3)},
        eef_pos_w={"right": torch.tensor([[0.08, 0.0, 0.0]]),
                   "left": torch.tensor([[-0.30, 0.0, 0.0]])},
    )
    out = IMPL["obj_between_eefs"](c, role="b", near=0.10)
    assert out.tolist() == [False]


def test_obj_between_eefs_rejects_sequential_single_hand_contact_not_simultaneous_squeeze():
    """The concrete failure mode a video review of kitchen 1511's milkcarton episode found: the
    LEFT hand contacts the carton in early frames then WITHDRAWS to empty counter for the rest of the
    episode, while the RIGHT hand is absent until later and then stays close -- at frame 13 the
    left wrist camera shows empty counter while the right shows the carton filling the frame, at
    the SAME instant. That is sequential single-hand contact, not a simultaneous two-handed
    squeeze -- and the old obj_near_eef(arm="both") spec passed it, because it only ever asked
    "was each hand near at some point in its own trace", never "are both near AND opposed right
    now".

    Reproduced as one evaluation instant: right in contact, left withdrawn far off but on the
    SAME axis (so direction alone would look opposed). Must fail -- and specifically because each
    palm's distance is checked INDEPENDENTLY (an AND over both), not because the two are averaged
    or the nearer one alone is allowed to carry the check. A min()-style aggregate would wrongly
    pass on the right palm alone, exactly reproducing this episode's false positive; the assertion
    below confirms this scene really does exercise that distinction, not a case a min-aggregate
    would also have refused for some unrelated reason.
    """
    r = torch.tensor([[0.08, 0.0, 0.0]])   # in contact
    l = torch.tensor([[-1.20, 0.0, 0.0]])  # withdrawn -- "empty counter"
    c = ctx(obj_pos_w={"b": torch.zeros(1, 3)}, eef_pos_w={"right": r, "left": l})

    near = 0.20
    out = IMPL["obj_between_eefs"](c, role="b", near=near)
    assert out.tolist() == [False]

    r_norm = torch.linalg.norm(r, dim=-1)
    l_norm = torch.linalg.norm(l, dim=-1)
    cos = (r * l).sum(dim=-1) / (r_norm * l_norm)
    assert cos.item() < -0.99, "sanity: the two vectors must be genuinely opposed in this scene"
    assert min(r_norm.item(), l_norm.item()) < near, (
        "sanity: a min()-aggregate WOULD wrongly pass this scene on the right palm alone -- "
        "confirming the real predicate's AND-per-palm check is what this test actually exercises"
    )


def test_obj_between_eefs_rejects_a_palm_at_the_object_centre_without_nan():
    """A palm coincident with the object's own centre is a degenerate direction, not evidence of
    a hold -- must fail explicitly, and the division inside the predicate must never leak a NaN
    out to the caller."""
    c = ctx(
        obj_pos_w={"b": torch.zeros(1, 3)},
        eef_pos_w={"right": torch.tensor([[0.08, 0.0, 0.0]]),
                   "left": torch.tensor([[0.0, 0.0, 0.0]])},
    )
    out = IMPL["obj_between_eefs"](c, role="b", near=0.20)
    assert out.tolist() == [False]
    assert torch.isfinite(out.float()).all()


def test_obj_between_eefs_opposition_threshold_is_strict_at_minus_one_half():
    """cos == -0.5 is exactly 120 degrees apart -- deliberately excluded (`< -0.5`, not `<=`),
    the same strict-boundary discipline obj_z/obj_near_eef/joint_pos already document."""
    theta = math.acos(-0.5)
    r = torch.tensor([[0.10, 0.0, 0.0]])
    l = torch.tensor([[0.10 * math.cos(theta), 0.10 * math.sin(theta), 0.0]])
    c = ctx(obj_pos_w={"b": torch.zeros(1, 3)}, eef_pos_w={"right": r, "left": l})
    out = IMPL["obj_between_eefs"](c, role="b", near=0.20)
    assert out.tolist() == [False]


def test_obj_between_eefs_validates_resolves_and_evaluates_through_the_real_pipeline():
    """The three steps the composer/emit/eval pipeline actually takes a spec through, on THIS
    predicate specifically -- same discipline as test_an_omitted_choice_validates_and_compiles_
    and_evaluates and test_the_other_two_arm_primitives_compile_without_an_arm."""
    from predicate_contract import resolve_roles
    spec = {"all": [{"obj_between_eefs": {"role": "@b", "near": 0.12}}]}
    assert validate_spec(spec, ["b"], "success") == []
    resolved = resolve_roles(spec, {"b": "b"})
    fn = compile_spec(resolved)
    c = ctx(
        obj_pos_w={"b": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
        eef_pos_w={
            "right": torch.tensor([[0.08, 0.0, 0.0], [0.08, 0.0, 0.0]]),
            "left": torch.tensor([[-0.08, 0.0, 0.0], [-0.30, 0.0, 0.0]]),
        },
    )
    assert fn(c).tolist() == [True, False]


def test_arm_both_is_the_conjunction():
    """task1's close_home is (distance_r < r) AND (distance_l < r)."""
    c = ctx(
        eef_pos_base={"right": torch.zeros(2, 3),
                      "left": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])},
        home={"right": torch.zeros(3), "left": torch.zeros(3)},
    )
    out = IMPL["eef_home"](c, arm="both", radius=0.1)
    assert out.tolist() == [True, False]


def test_eef_home_measures_in_the_base_frame():
    c = ctx(
        eef_pos_base={"right": torch.tensor([[0.2257, -0.0988, 1.0351], [0.0, 0.0, 0.0]])},
        home={"right": torch.tensor([0.2257, -0.0988, 1.0351])},
    )
    out = IMPL["eef_home"](c, arm="right", radius=0.02)
    assert out.tolist() == [True, False]


def test_obj_near_prim_xy_only_ignores_height():
    c = ctx(
        obj_pos_w={"bowl0": torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 0.0]])},
        art_body_pos_w={"cab": torch.zeros(2, 1, 3)},
        art_body_names={"cab": ["corpus"]},
    )
    out = IMPL["obj_near_prim"](c, role="bowl0", target_role="cab", radius=0.28, xy_only=True)
    assert out.tolist() == [True, False]


def test_experimental_mug_table_requires_released_mug_away_from_gripper():
    repo = Path(__file__).resolve().parents[2]
    template = json.loads((repo / "examples/experimental/mug_counter_to_table_right.json").read_text())
    spec = resolve_roles(template["success"], {"target": "mug0"})
    assert validate_spec(template["success"], ["target"], "success") == []
    mug = torch.tensor([[1.10, -3.30, 0.90], [1.10, -3.30, 0.90]])
    c = ctx(
        obj_pos_w={"mug0": mug, "table": torch.tensor([[1.10, -3.58, 0.41]] * 2)},
        eef_pos_w={"right": torch.tensor([[1.12, -3.30, 0.90], [1.40, -3.30, 0.90]])},
        eef_pos_base={"right": torch.zeros(2, 3)}, home={"right": torch.zeros(3)},
        gripper_gap={"right": torch.tensor([0.08, 0.08])},
        goal_index=torch.tensor([8, 8]), max_steps=9,
    )
    assert compile_spec(spec)(c).tolist() == [False, True]


def test_obj_near_prim_selects_a_named_body():
    pos = torch.zeros(2, 2, 3)
    pos[:, 1, 0] = 3.0                       # drawer body is 3 m along x
    c = ctx(
        obj_pos_w={"bowl0": torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])},
        art_body_pos_w={"cab": pos},
        art_body_names={"cab": ["corpus", "drawer_0_0"]},
    )
    out = IMPL["obj_near_prim"](c, role="bowl0", target_role="cab", body="drawer_0_0", radius=0.1)
    assert out.tolist() == [True, False]


def test_obj_near_prim_z_override_replaces_the_target_height():
    c = ctx(
        obj_pos_w={"m": torch.tensor([[0.0, 0.0, 0.88], [0.0, 0.0, 0.0]])},
        art_body_pos_w={"sink": torch.zeros(2, 1, 3)},
        art_body_names={"sink": ["corpus"]},
    )
    out = IMPL["obj_near_prim"](c, role="m", target_role="sink", radius=0.05, z_override=0.88)
    assert out.tolist() == [True, False]


def test_door_midpoint_shifts_toward_the_door_along_the_shared_axis():
    """The basin-centre heuristic sink/task1/task1_deploy/task1_molmospace each re-implement:
    bodies [1] and [2] are the doors; whichever axis they share is the one to average along."""
    pos = torch.zeros(1, 3, 3)
    pos[:, 0, 0] = 0.0        # corpus at x=0
    pos[:, 1, 0] = 2.0        # door 1 at x=2
    pos[:, 2, 0] = 2.0        # door 2 at x=2 -> x is shared, so midpoint x = 1.0
    c = ctx(
        num_envs=1,
        obj_pos_w={"m": torch.tensor([[1.0, 0.0, 0.0]])},
        art_body_pos_w={"sink": pos},
        art_body_names={"sink": ["corpus", "door_a", "door_b"]},
    )
    out = IMPL["obj_near_prim"](c, role="m", target_role="sink", anchor="door_midpoint",
                                radius=0.01)
    assert out.tolist() == [True]


def test_joint_pos_by_name():
    c = ctx(
        art_joint_pos={"cab": torch.tensor([[0.0, 0.5], [0.2, 0.5]])},
        art_joint_names={"cab": ["corpus_to_door_0_1", "corpus_to_drawer_0_0"]},
    )
    out = IMPL["joint_pos"](c, role="cab", joint="corpus_to_door_0_1", hi=0.03)
    assert out.tolist() == [True, False]


def test_joint_pos_by_index_matches_task2s_raw_indexing():
    c = ctx(art_joint_pos={"cab": torch.tensor([[0.0], [0.2]])}, art_joint_names={"cab": ["j"]})
    out = IMPL["joint_pos"](c, role="cab", joint=0, hi=0.03)
    assert out.tolist() == [True, False]


def test_joint_pos_is_strict_at_the_bound():
    c = ctx(art_joint_pos={"cab": torch.tensor([[0.03], [0.02]])}, art_joint_names={"cab": ["j"]})
    out = IMPL["joint_pos"](c, role="cab", joint="j", hi=0.03)
    assert out.tolist() == [False, True]


def test_joint_pos_names_the_joints_it_has_when_the_name_is_unknown():
    c = ctx(art_joint_pos={"cab": torch.zeros(2, 1)}, art_joint_names={"cab": ["only_joint"]})
    with pytest.raises(KeyError, match="only_joint"):
        IMPL["joint_pos"](c, role="cab", joint="typo", hi=0.03)


def test_gripper_open():
    c = ctx(gripper_gap={"right": torch.tensor([0.09, 0.05])})
    assert IMPL["gripper_open"](c, arm="right", gap=0.079).tolist() == [True, False]


def test_robot_fell():
    c = ctx(base_pos_w=torch.tensor([[0.0, 0.0, -0.5], [0.0, 0.0, 0.1]]))
    assert IMPL["robot_fell"](c, z=-0.1).tolist() == [True, False]


def test_last_subtask():
    c = ctx(goal_index=torch.tensor([3, 1]), max_steps=4)
    assert IMPL["last_subtask"](c).tolist() == [True, False]


def test_last_subtask_says_why_when_the_script_state_is_absent():
    with pytest.raises(ValueError, match="script"):
        IMPL["last_subtask"](ctx())


def test_compile_all_is_a_conjunction():
    c = ctx(obj_pos_w={"b": torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.0, 0.7]])},
            base_pos_w=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, -0.5]]))
    fn = compile_spec({"all": [
        {"obj_z": {"role": "b", "lo": 0.6, "hi": 0.8}},
        {"not": {"robot_fell": {"z": -0.1}}},
    ]})
    assert fn(c).tolist() == [True, False]


def test_compile_any_is_a_disjunction():
    c = ctx(obj_pos_w={"b": torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 1.0]])},
            base_pos_w=torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]]))
    fn = compile_spec({"any": [
        {"obj_z": {"role": "b", "hi": 0.3}},
        {"robot_fell": {"z": -0.1}},
    ]})
    assert fn(c).tolist() == [True, False]


def test_compile_refuses_an_unknown_primitive_by_name():
    with pytest.raises(KeyError, match="obj_zz"):
        compile_spec({"all": [{"obj_zz": {}}]})
