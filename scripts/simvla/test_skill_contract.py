"""Tests for the skill registry.

Run with: pytest scripts/simvla/test_skill_contract.py -v
Stdlib only — no Omniverse, no torch.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

from skill_contract import (
    ACTIONS,
    REGISTRY,
    SKILL_ID,
    Choice,
    Float,
    PrimPath,
    SkillContractError,
    skill,
    skills_for_action,
    validate_registry,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test gets an empty registry; the real skills live in another module."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


def test_a_skill_registers_itself_with_its_declared_metadata():
    @skill(id="arm.grasp", actions=("A_r", "A_l"), label="Move arm to grasp",
           params=[PrimPath("prim_path")])
    class ArmGrasp:
        def plan(self, ctx, action, params):
            return [0.0] * 7

    spec = REGISTRY["arm.grasp"]
    assert spec.actions == ("A_r", "A_l")
    assert spec.label == "Move arm to grasp"
    assert [p.name for p in spec.params] == ["prim_path"]
    assert spec.is_runtime is False, "a skill with plan() is authored, not runtime-resolved"


def test_a_skill_with_resolve_is_runtime_resolved():
    @skill(id="arm.pause", actions=("A_r",), label="Pause", params=[])
    class Pause:
        def resolve(self, ctx, envs, params):
            return [0.0] * 7

    assert REGISTRY["arm.pause"].is_runtime is True


def test_a_skill_must_define_exactly_one_of_plan_or_resolve():
    """This is the refrigerator rule. The old code let a skill return a raw float and hope the
    executor recognised it; -0.4 matched nothing, so the robot drove to a fixed corner. A skill
    now says which kind it is, or it cannot be registered."""
    @skill(id="bad.neither", actions=("N",), label="Neither", params=[])
    class Neither:
        pass

    with pytest.raises(SkillContractError, match="bad.neither"):
        validate_registry(executor_actions={"N"})

    REGISTRY.clear()

    @skill(id="bad.both", actions=("N",), label="Both", params=[])
    class Both:
        def plan(self, ctx, action, params):
            return [0.0]

        def resolve(self, ctx, envs, params):
            return [0.0]

    with pytest.raises(SkillContractError, match="bad.both"):
        validate_registry(executor_actions={"N"})


def test_validate_rejects_an_action_the_executor_cannot_run():
    """This is the G_b bug. SubtaskDialog offered G_b; simvla_gen's TASK_IDS had no G_b entry, and
    the lookup was unguarded, so any goal using it died with KeyError at load. Now the mismatch is
    an import-time error naming the skill, and the GUI list IS the registry, so it cannot arise."""
    @skill(id="gripper.both", actions=("G_b",), label="Both grippers", params=[])
    class GripperBoth:
        def plan(self, ctx, action, params):
            return True

    with pytest.raises(SkillContractError, match="G_b"):
        validate_registry(executor_actions={"N", "A_r", "G_r"})   # no G_b channel


def test_duplicate_ids_are_rejected():
    @skill(id="dupe", actions=("N",), label="One", params=[])
    class One:
        def plan(self, ctx, action, params):
            return [0.0]

    with pytest.raises(SkillContractError, match="dupe"):
        @skill(id="dupe", actions=("N",), label="Two", params=[])
        class Two:
            def plan(self, ctx, action, params):
                return [0.0]


def test_skill_ids_are_stable_regardless_of_import_order():
    """The int indexes the executor's dispatch, so both sides must agree on it regardless of which
    module imported the skills first. (Goal files persist the skill NAME — see SKILL_ID's
    docstring for why the int must never be written to disk.)"""
    @skill(id="z.last", actions=("N",), label="Z", params=[])
    class Z:
        def plan(self, ctx, action, params):
            return [0.0]

    @skill(id="a.first", actions=("N",), label="A", params=[])
    class A:
        def plan(self, ctx, action, params):
            return [0.0]

    assert SKILL_ID()["a.first"] < SKILL_ID()["z.last"], "ids must be assigned in sorted-id order"


def test_skill_ids_are_sorted_stable_even_when_registered_out_of_order():
    """Register skills in a deliberately scrambled order in a fresh registry and confirm the ids
    still come out in sorted-id order, not insertion order."""
    ids = ["m.mid", "a.first", "z.last", "b.second"]
    for sid in ids:
        @skill(id=sid, actions=("N",), label=sid, params=[])
        class Anon:
            def plan(self, ctx, action, params):
                return [0.0]

    assigned = SKILL_ID()
    assert [sid for sid, _ in sorted(assigned.items(), key=lambda kv: kv[1])] == sorted(ids)


def test_actions_and_skills_for_action_drive_the_gui():
    @skill(id="nav.a", actions=("N",), label="Alpha", params=[])
    class NavA:
        def resolve(self, ctx, envs, params):
            return [0.0]

    @skill(id="arm.b", actions=("A_r", "A_l"), label="Beta", params=[Choice("side", ["L", "R"])])
    class ArmB:
        def plan(self, ctx, action, params):
            return [0.0]

    assert set(ACTIONS()) == {"N", "A_r", "A_l"}
    assert [s.label for s in skills_for_action("A_r")] == ["Beta"]
    assert [s.label for s in skills_for_action("N")] == ["Alpha"]


def test_params_carry_enough_for_the_gui_to_render_them():
    p = Choice("which_arm", ["Left", "Right", "Both"])
    assert p.name == "which_arm" and p.options == ["Left", "Right", "Both"]
    f = Float("back_off_m", default=0.25)
    assert f.name == "back_off_m" and f.default == 0.25


def test_module_is_stdlib_only():
    """skill_contract.py must be importable with no GPU and no Omniverse: simvla_data_generator.py
    and simvla_gen.py both boot Omniverse at import (~60s), and this module has to live somewhere
    that does not, or it cannot be unit tested. Parse the AST rather than just importing, so a
    lazily-imported (function-local) third-party dependency cannot slip past a naive check."""
    stdlib_names = set(sys.stdlib_module_names)

    # The scripts-side module is now a compatibility import. Inspect the canonical implementation
    # that both CPU and simulator entry points load.
    module_path = Path(__file__).resolve().parents[2] / "src/simvla/skill_contract.py"
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])

    non_stdlib = imported_roots - stdlib_names
    assert not non_stdlib, f"skill_contract.py imports non-stdlib module(s): {non_stdlib}"

    # Belt and suspenders: actually importing it must not pull in any Omniverse/torch/etc module.
    # Diff sys.modules across the import — an absolute check would blame this module for torch,
    # which a sibling test (test_timeout_logic.py, test_kitchen_*.py) already imported into the
    # shared process. That fails only under `pytest scripts/simvla/`, i.e. exactly how CI runs it.
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("skill_contract_purity_check", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    banned_prefixes = ("omni", "isaaclab", "pxr", "torch", "scene_synthesizer", "numpy")
    pulled_in = set(sys.modules) - before
    banned = sorted(m for m in pulled_in if m.split(".")[0] in banned_prefixes)
    assert not banned, f"importing skill_contract.py pulled in {banned}"


def test_a_non_callable_plan_does_not_pass_for_a_plan():
    """`plan = None` (a stub someone meant to come back to) and `plan = -0.4` (a literal — the
    exact shape the refrigerator skill shipped in) both satisfy hasattr. The promise is that a
    half-wired skill is caught at import, not with a TypeError an hour into a GPU job."""
    @skill(id="bad.stub", actions=("N",), label="Stub", params=[])
    class Stub:
        plan = None

    @skill(id="bad.literal", actions=("N",), label="A raw float, again", params=[])
    class Literal:
        plan = -0.4

    with pytest.raises(SkillContractError) as exc:
        validate_registry({"N"})
    assert "bad.stub" in str(exc.value)
    assert "bad.literal" in str(exc.value)
    assert REGISTRY["bad.stub"].is_runtime is False


def test_every_offender_is_reported_at_once():
    """Porting 16 skills one error per run is whack-a-mole."""
    @skill(id="bad.both", actions=("N",), label="Both", params=[])
    class Both:
        def plan(self, ctx, action, params):
            return []

        def resolve(self, ctx, envs, params):
            return []

    @skill(id="bad.no_channel", actions=("G_b",), label="The G_b bug", params=[])
    class NoChannel:
        def plan(self, ctx, action, params):
            return []

    with pytest.raises(SkillContractError) as exc:
        validate_registry({"N"})
    message = str(exc.value)
    assert "bad.both" in message and "bad.no_channel" in message


def test_a_skill_with_no_actions_is_rejected():
    """It would register cleanly and pass validate_registry (set() - executor is empty), then sit
    there unreachable: in no GUI dropdown, on no executor channel. Silently dead is the disease."""
    with pytest.raises(SkillContractError, match="no actions"):
        @skill(id="bad.orphan", actions=(), label="Unreachable", params=[])
        class Orphan:
            def plan(self, ctx, action, params):
                return []


def test_actions_given_as_a_bare_string_is_rejected():
    """The missing comma: ("N") is the str "N", and tuple("A_r") is ("A", "_", "r")."""
    with pytest.raises(SkillContractError, match="not a tuple"):
        @skill(id="bad.comma", actions="A_r", label="Missing comma", params=[])
        class MissingComma:
            def plan(self, ctx, action, params):
                return []


def test_re_registering_the_same_class_is_not_a_duplicate():
    """A skills module executed twice (a flat import and a package import, or importlib.reload in
    a Qt dev loop) must not raise 'Duplicate skill id' from code that is working fine."""
    def define():
        @skill(id="arm.grasp", actions=("A_r",), label="Grasp", params=[])
        class ArmGrasp:
            def plan(self, ctx, action, params):
                return []

        return ArmGrasp

    first = define()
    # Re-registering the same class object is a no-op; a different class under that id is not.
    skill(id="arm.grasp", actions=("A_r",), label="Grasp", params=[])(first)
    assert REGISTRY["arm.grasp"].cls is first

    with pytest.raises(SkillContractError, match="Duplicate"):
        define()


def test_a_choices_default_is_declared_not_positional():
    """It used to be options[0]. That makes the dropdown's ORDER load-bearing: nav.to_prim reads
    which_arm as an arm_bias of +/-5 cm, so reordering the options for readability would silently
    move every base pose authored afterwards. Order and meaning must be independent."""
    a = Choice("which_arm", ["Both", "Left", "Right"], default="Both")
    b = Choice("which_arm", ["Right", "Left", "Both"], default="Both")
    assert a.default == b.default == "Both", "reordering the options changed the default"

    # Undeclared still falls back to the first option, so existing declarations keep working.
    assert Choice("which_arm", ["Both", "Left"]).default == "Both"


def test_a_choice_cannot_default_to_a_non_option():
    """A typo'd default is a hard error, not a dropdown that quietly opens on the wrong entry."""
    with pytest.raises(SkillContractError, match="not one of"):
        Choice("which_arm", ["Both", "Left", "Right"], default="Rihgt")

    with pytest.raises(SkillContractError, match="no options"):
        Choice("which_arm", [])
