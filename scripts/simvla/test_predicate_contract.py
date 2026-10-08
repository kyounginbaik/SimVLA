"""The predicate registry: the palette of condition primitives a task composes.

Run: python -m pytest scripts/simvla/test_predicate_contract.py -q
Stdlib only: no torch, no Omniverse.
"""

import json

import pytest

from predicate_contract import (
    REGISTRY,
    SpecError,
    leaves,
    physical_only,
    resolve_roles,
    spec_warnings,
    unregistered_predicates,
    validate_spec,
)

ROLES = {"target", "container"}


def test_the_nine_primitives_are_declared():
    assert sorted(REGISTRY) == [
        "eef_home", "gripper_open", "joint_pos", "last_subtask", "obj_between_eefs",
        "obj_near_eef", "obj_near_prim", "obj_z", "robot_fell",
    ]


def test_every_primitive_declares_a_known_phase():
    assert {s.phase for s in REGISTRY.values()} == {"physical", "script"}
    assert REGISTRY["last_subtask"].phase == "script", (
        "last_subtask reads variable.env_goal_indices, which does not exist during eval"
    )


def test_leaves_walks_the_whole_tree_in_order():
    spec = {"all": [
        {"obj_z": {"role": "@target", "lo": 0.6}},
        {"not": {"robot_fell": {"z": -0.1}}},
        {"any": [{"last_subtask": {}}]},
    ]}
    assert [pid for pid, _ in leaves(spec)] == ["obj_z", "robot_fell", "last_subtask"]


def test_a_good_spec_has_no_problems():
    spec = {"all": [
        {"obj_z": {"role": "@target", "lo": 0.6, "hi": 0.8}},
        {"eef_home": {"arm": "right", "radius": 0.18}},
        {"joint_pos": {"role": "@container", "joint": "corpus_to_drawer_0_0", "hi": 0.03}},
        {"last_subtask": {}},
    ]}
    assert validate_spec(spec, ROLES, kind="success") == []


def test_every_problem_is_reported_at_once():
    """Fixing a composed condition one error per run is whack-a-mole — same contract as
    validate_template."""
    spec = {"all": [
        {"obj_zz": {"role": "@target"}},                      # unknown primitive
        {"eef_home": {"arm": "right", "radius": 0.1, "nope": 1}},  # unknown param
        {"obj_z": {"role": "@missing", "lo": 1.0}},           # undeclared role
    ]}
    problems = validate_spec(spec, ROLES, kind="success")
    assert len(problems) == 3
    assert any("obj_zz" in p for p in problems)
    assert any("nope" in p for p in problems)
    assert any("@missing" in p for p in problems)


def test_an_empty_all_is_refused():
    """An empty conjunction is vacuously true — a success term that always fires."""
    problems = validate_spec({"all": []}, ROLES, kind="success")
    assert any("empty" in p for p in problems)


def test_a_success_made_only_of_script_phase_is_refused():
    """This is task4: success = 'reached the last step', which marks a demo successful for
    merely finishing the script."""
    problems = validate_spec({"all": [{"last_subtask": {}}]}, ROLES, kind="success")
    assert any("last_subtask" in p and "physical" in p for p in problems)


def test_a_retry_made_only_of_script_phase_is_allowed():
    assert validate_spec({"any": [{"last_subtask": {}}]}, ROLES, kind="retry") == []


def test_a_missing_required_param_is_reported():
    problems = validate_spec({"all": [{"eef_home": {"arm": "right"}}]}, ROLES, kind="success")
    assert any("radius" in p for p in problems)


def test_a_required_text_param_is_reported_but_an_optional_one_is_not():
    """joint_pos.joint is required — a joint range with no joint means nothing. obj_near_prim.body
    is optional and defaults to body 0, the way sink()/task1() measure to the corpus."""
    missing_joint = validate_spec(
        {"all": [{"joint_pos": {"role": "@container", "hi": 0.03}}]}, ROLES, kind="success")
    assert any("joint" in p for p in missing_joint)

    no_body = validate_spec(
        {"all": [{"obj_near_prim": {"role": "@target", "target_role": "@container",
                                    "radius": 0.28}}]}, ROLES, kind="success")
    assert no_body == []


def test_an_omitted_optional_bound_is_fine():
    assert validate_spec({"any": [{"obj_z": {"role": "@target", "hi": 0.3}}]},
                         ROLES, kind="retry") == []


def test_obj_z_with_neither_bound_is_refused():
    problems = validate_spec({"all": [{"obj_z": {"role": "@target"}}]}, ROLES, kind="success")
    assert any("lo" in p and "hi" in p for p in problems)


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_a_non_finite_bound_is_refused(literal):
    """json.loads ACCEPTS JSON's Infinity/-Infinity/NaN, and json.dumps writes them back — so a
    hand-written template can carry one all the way to the env config, where they are three
    undefined Python NAMEs and the config NameErrors at import (the same class of failure the
    true/false/null rewrite exists for, on a value the type check waves through: they are real
    floats). They also mean nothing as a bound: an infinite one can never be crossed, and every
    comparison against NaN is False."""
    spec = json.loads('{"all": [{"obj_z": {"role": "@target", "lo": %s}}]}' % literal)
    problems = validate_spec(spec, ROLES, kind="success")
    assert any("FINITE" in p for p in problems), problems


def test_a_non_finite_number_is_refused_wherever_a_number_may_go():
    """Not just `lo`: every Float/OptFloat param goes through the same check."""
    spec = {"all": [{"obj_near_eef": {"role": "@target", "arm": "right",
                                      "radius": float("inf")}}]}
    assert any("FINITE" in p for p in validate_spec(spec, ROLES, kind="success"))


def test_an_ordinary_bound_is_still_accepted():
    """The finiteness check must not refuse the numbers every real condition is made of."""
    spec = {"all": [{"obj_z": {"role": "@target", "lo": 0.6, "hi": 0.8}},
                    {"robot_fell": {"z": -0.1}}]}
    assert validate_spec(spec, ROLES, kind="success") == []


def test_proximity_without_a_lift_warns_but_does_not_refuse():
    """simvla_eval.py:695 records that proximity alone scored a non-grasping policy 4/20."""
    spec = {"all": [{"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}}]}
    assert validate_spec(spec, ROLES, kind="success") == []
    assert any("4/20" in w for w in spec_warnings(spec, kind="success"))


def test_proximity_with_a_lift_does_not_warn():
    spec = {"all": [
        {"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}},
        {"obj_z": {"role": "@target", "lo": 0.90}},
    ]}
    assert spec_warnings(spec, kind="success") == []


def test_resolve_roles_substitutes_bound_prim_names():
    spec = {"all": [
        {"obj_z": {"role": "@target", "lo": 0.6, "hi": 0.8}},
        {"joint_pos": {"role": "@container", "joint": "corpus_to_drawer_0_0", "hi": 0.03}},
        {"eef_home": {"arm": "right", "radius": 0.18}},
    ]}
    out = resolve_roles(spec, {"target": "bottle0", "container": "base_cabinet"})
    assert out["all"][0]["obj_z"]["role"] == "bottle0"
    assert out["all"][1]["joint_pos"]["role"] == "base_cabinet"
    assert out["all"][2]["eef_home"]["arm"] == "right", "non-role params are untouched"
    assert spec["all"][0]["obj_z"]["role"] == "@target", "the input spec is not mutated"


def test_resolve_roles_refuses_an_unbound_role():
    with pytest.raises(SpecError, match="target"):
        resolve_roles({"all": [{"obj_z": {"role": "@target", "lo": 1.0}}]}, {})


def test_physical_only_drops_script_phase_leaves():
    spec = {"all": [
        {"obj_z": {"role": "bottle0", "lo": 0.9}},
        {"last_subtask": {}},
    ]}
    assert physical_only(spec) == {"all": [{"obj_z": {"role": "bottle0", "lo": 0.9}}]}


def test_physical_only_returns_none_when_nothing_physical_remains():
    assert physical_only({"all": [{"last_subtask": {}}]}) is None


def test_physical_only_drops_an_unregistered_predicate_indistinguishably():
    """The behaviour unregistered_predicates exists to make visible, pinned so it cannot be
    mistaken for an accident: a pid the registry does not declare vanishes exactly like a
    script-phase leaf, with nothing in the result to say it was ever there."""
    spec = {"all": [{"obj_z": {"role": "b", "lo": 0.9}}, {"was_renamed": {}}]}
    assert physical_only(spec) == {"all": [{"obj_z": {"role": "b", "lo": 0.9}}]}


def test_unregistered_predicates_names_what_the_registry_does_not_declare():
    spec = {"all": [
        {"obj_z": {"role": "b", "lo": 0.9}},
        {"was_renamed": {}},
        {"not": {"also_gone": {"x": 1}}},
        {"last_subtask": {}},          # declared, merely script-phase — NOT unregistered
    ]}
    assert unregistered_predicates(spec) == ["also_gone", "was_renamed"]
    assert unregistered_predicates({"all": [{"obj_z": {"role": "b", "lo": 0.9}}]}) == []


def test_a_role_ref_param_accepts_a_literal_entity_name():
    """target_role="sink_cabinet" (no '@') is a LITERAL: it names a fixture that exists on every
    kitchen under one name. resolve_roles passes literals through and composed.py raises loudly
    when one matches nothing, so the validator permitting them closes no silent hole. A non-string
    or empty value is still refused."""
    from predicate_contract import validate_spec
    spec = {"obj_near_prim": {"role": "@target", "target_role": "sink_cabinet", "radius": 0.12}}
    assert validate_spec(spec, {"target"}, kind="success") == []
    bad = {"obj_near_prim": {"role": "@target", "target_role": 7, "radius": 0.12}}
    assert any("literal entity name" in p for p in validate_spec(bad, {"target"}, kind="success"))
