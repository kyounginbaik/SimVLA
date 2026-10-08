"""SimVLA: one place to declare a skill.

Stdlib only — no torch, isaaclab, omni or scene_synthesizer. Both simvla_data_generator.py and
simvla_gen.py boot Omniverse at import (~60s), so the contract has to live somewhere that does
not, or it cannot be unit tested.

A skill's identity used to be spelled out in three places that could silently disagree:
SubtaskDialog's dropdowns, SKILL_REGISTRY's (action, usage) key, and simvla_gen's TASK_IDS plus a
hand-written dispatch branch per magic float. Two live bugs came straight out of that: G_b was
offered by the GUI but missing from TASK_IDS (KeyError at load), and the refrigerator skill
returned a raw [-0.4, -0.4, -0.4] that matched no sentinel, so it fell through and became a
literal coordinate — 73 of the 7,888 goal files on disk carry it.

Now: one declaration. The GUI, the goal file and the executor all derive from it.
"""

from __future__ import annotations

from dataclasses import dataclass


class SkillContractError(Exception):
    """A skill is half-wired. Raised at import, never at run time."""


# ---- param declarations: what the GUI renders, and what lands in the goal file ----

@dataclass(frozen=True)
class PrimPath:
    name: str


@dataclass(frozen=True)
class Choice:
    name: str
    options: list[str]
    default: str | None = None

    def __post_init__(self):
        """The default is declared, never positional.

        It used to be options[0]. That makes the dropdown's *order* load-bearing: reordering
        ["Both", "Left", "Right"] for readability silently moves every base pose authored
        afterwards by 5 cm, because nav.to_prim reads which_arm as an arm_bias. Meaning encoded in
        a position, recovered by convention, is the disease this contract exists to cure — so a
        Choice states its default, and a default that is not one of the options is a hard error
        rather than a dropdown that quietly opens on the wrong entry.
        """
        if not self.options:
            raise SkillContractError(f"Choice {self.name!r} declares no options")
        if self.default is None:
            object.__setattr__(self, "default", self.options[0])
        elif self.default not in self.options:
            raise SkillContractError(
                f"Choice {self.name!r} defaults to {self.default!r}, which is not one of "
                f"{self.options}"
            )


@dataclass(frozen=True)
class Float:
    name: str
    default: float = 0.0


@dataclass(frozen=True)
class Bool:
    name: str
    default: bool = False


Param = PrimPath | Choice | Float | Bool


@dataclass
class SkillSpec:
    id: str
    actions: tuple[str, ...]
    label: str
    params: list[Param]
    cls: type

    @property
    def is_runtime(self) -> bool:
        """True iff the goal can only be computed once the robot is running."""
        return callable(getattr(self.cls, "resolve", None))


REGISTRY: dict[str, SkillSpec] = {}


def skill(*, id: str, actions: tuple[str, ...], label: str, params: list[Param]):
    """Declare a skill. The class must define exactly one of plan() or resolve().

    plan(ctx, action, params)     -> the goal, computed while authoring
    resolve(ctx, envs, params)    -> the goal, computed from live robot state at run time
    """
    def decorator(cls: type) -> type:
        if isinstance(actions, str):
            raise SkillContractError(
                f"Skill {id!r} declares actions={actions!r}, a string, not a tuple. A missing "
                f'comma: ("N") is "N", not ("N",). tuple("A_r") is ("A", "_", "r").'
            )
        if not actions:
            raise SkillContractError(
                f"Skill {id!r} declares no actions, so nothing can ever invoke it: it appears in "
                f"no GUI dropdown and has no executor channel."
            )
        if REGISTRY.get(id) is not None and REGISTRY[id].cls is not cls:
            raise SkillContractError(f"Duplicate skill id {id!r}")
        REGISTRY[id] = SkillSpec(id=id, actions=tuple(actions), label=label,
                                  params=list(params), cls=cls)
        return cls

    return decorator


def SKILL_ID() -> dict[str, int]:
    """skill id -> a small int, assigned in sorted-id order.

    These ints are an in-memory, per-run detail: they index the executor's dispatch (see
    simvla_gen's skill_ids_tensor). Goal files persist the skill *name*, never this int — because
    sorting makes the int independent of import order but NOT of the registry's contents. Adding
    one skill renumbers every id after it alphabetically, which would silently reinterpret every
    goal file ever written. Persist the name; derive the int at load.
    """
    return {sid: i for i, sid in enumerate(sorted(REGISTRY))}


def ACTIONS() -> tuple[str, ...]:
    """Every action code any registered skill declares. The GUI and the executor both read this."""
    seen: list[str] = []
    for spec in REGISTRY.values():
        for action in spec.actions:
            if action not in seen:
                seen.append(action)
    return tuple(sorted(seen))


def skills_for_action(action: str) -> list[SkillSpec]:
    """What the GUI offers once an action is picked."""
    return [s for s in REGISTRY.values() if action in s.actions]


def validate_registry(executor_actions: set[str]) -> None:
    """Fail loudly, at import, if any skill is half-wired.

    `executor_actions` is the set of action channels simvla_gen actually implements. Passing it in
    (rather than importing simvla_gen, which would boot Omniverse) is what lets this run in a test.

    Every offender is reported at once. Porting 16 skills one error-per-run would be whack-a-mole.
    """
    problems: list[str] = []

    for spec in REGISTRY.values():
        # callable(), not hasattr(): `plan = None` (a stub) or `plan = -0.4` (a literal — the exact
        # shape of the refrigerator bug) passes hasattr, then dies with TypeError mid-run, an hour
        # into a GPU job. The promise is that a half-wired skill cannot reach a run.
        has_plan = callable(getattr(spec.cls, "plan", None))
        has_resolve = callable(getattr(spec.cls, "resolve", None))

        if has_plan and has_resolve:
            problems.append(
                f"{spec.id!r} defines both plan() and resolve(). A goal is either computed while "
                f"authoring or resolved at run time — pick one."
            )
        elif not has_plan and not has_resolve:
            problems.append(
                f"{spec.id!r} defines neither plan() nor resolve() (as callables). This is how the "
                f"refrigerator skill shipped: it returned a raw float and no branch recognised it."
            )

        missing = set(spec.actions) - executor_actions
        if missing:
            problems.append(
                f"{spec.id!r} declares action(s) {sorted(missing)} that the executor has no channel "
                f"for. This is the G_b bug: the GUI offered it, the executor had never heard of it, "
                f"and the goal died with KeyError at load."
            )

    if problems:
        raise SkillContractError(
            f"{len(problems)} half-wired skill(s):\n  " + "\n  ".join(problems)
        )
