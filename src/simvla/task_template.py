"""SimVLA: a task is a composition of skills over roles.

Stdlib only (plus skill_contract), so it can be unit-tested with no GPU and no Omniverse.

Adding a task used to mean authoring a goal file and then re-deriving, by hand, a set of CLI
arguments the goal file already implied: --obj_name is the prim step 0 points at, --sub_grasp_idx_r
is an index INTO the script, --export_groups is more indices. Nothing checked they agreed. The
README shipped --obj_name "bottle0" for a task targeting /world/bowl0, and it died with a bare
KeyError minutes into a GPU run.

A template states the sequence once. The skill ids, action channels and parameter schemas all come
from skill_contract.REGISTRY, so a template cannot name a skill the executor does not have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .predicate_contract import spec_warnings, validate_spec
from .scene_spec import SceneObject, scene_from_dicts, scene_to_dicts, validate_scene
from .skill_contract import REGISTRY, PrimPath

ROLE_PREFIX = "@"


class TaskTemplateError(Exception):
    """A template is incoherent. Raised at load, never at run time."""


@dataclass(frozen=True)
class Role:
    """A slot a prim fills, matched per kitchen.

    Exactly one matcher:
      object_type       -- a rigid object the kitchen generator places ("bowl", "mug", ...)
      articulation_with -- a feature the articulation must have ("drawer", "door", or the
                           furniture name like "base_cabinet"), OR a LIST of features it must have
                           ALL of, e.g. ["door", "base_cabinet"]
      handle_of         -- the handle sub-prim of another role's articulation

    WHY A LIST. One feature cannot name a cabinet door. Measured on a generated l_shaped kitchen,
    eleven articulations carry "door" -- both sink-cabinet doors, five wall-cabinet doors, the
    fridge, the freezer, the microwave and the oven -- so `articulation_with="door"` is always
    ambiguous. Narrowing to the furniture instead ("base_cabinet") is worse than ambiguous: whether
    that cabinet's handled segment is a hinged DOOR or a prismatic DRAWER depends on the seed, so
    the same template silently binds a drawer on half of them, and an arc-about-a-hinge skill then
    runs against something that travels in a straight line. The intersection names it exactly.

    Stored as a tuple either way, so consumers have one shape to handle.
    """

    name: str
    object_type: str | None = None
    articulation_with: str | tuple[str, ...] | list[str] | None = None
    handle_of: str | None = None

    def __post_init__(self):
        matchers = [self.object_type, self.articulation_with, self.handle_of]
        if sum(m is not None for m in matchers) != 1:
            raise TaskTemplateError(
                f"Role {self.name!r} must declare exactly one of object_type / "
                f"articulation_with / handle_of; got {matchers}"
            )
        if self.articulation_with is not None:
            features = ((self.articulation_with,)
                        if isinstance(self.articulation_with, str)
                        else tuple(self.articulation_with))
            if not features or not all(isinstance(f, str) and f for f in features):
                raise TaskTemplateError(
                    f"Role {self.name!r}: articulation_with must be a non-empty feature string or "
                    f"a list of them, got {self.articulation_with!r}"
                )
            object.__setattr__(self, "articulation_with", features)


@dataclass(frozen=True)
class TemplateStep:
    skill: str
    action: str
    params: dict = field(default_factory=dict)
    language: str = ""


@dataclass
class TaskTemplate:
    name: str
    language: str
    roles: list[Role]
    steps: list[TemplateStep]
    subtask_groups: list[list[int]] = field(default_factory=list)
    scene: list[SceneObject] = field(default_factory=list)
    #: The composed condition that ends an episode successfully. REQUIRED: without it the emitted
    #: env config falls back to mdp.task2 — a put-a-bowl-in-a-drawer check — for every task, and
    #: simvla_gen gates demo export on that term firing (simvla_gen.py:3181).
    success: dict | None = None
    #: The composed condition that abandons an episode and retries. Optional: emit writes a default
    #: invariant set when it is absent (see task_emit.default_retry).
    retry: dict | None = None


def role_ref(value) -> str | None:
    """'@target' -> 'target'. A literal prim path -> None."""
    if isinstance(value, str) and value.startswith(ROLE_PREFIX):
        return value[len(ROLE_PREFIX):]
    return None


def language_problems(language) -> list[str]:
    """A task's `language` is also a DIRECTORY NAME. Everything wrong with this one.

    simvla_gen.py:3217 builds the LeRobot output path as

        f"{os.environ['SIMVLA_LEROBOT_ROOT']}/{args_cli.task_language}/{args_cli.task}"

    and task_runconfig.derive_run_config sets `task_language=t.language`, verbatim, into the v2 goal
    file's run_config — from where run_config_merge assigns it to args_cli.task_language. So the
    string a composer types is pasted into a filesystem path, and until now the only thing standing
    between the two was that a human typed it on a command line and would have noticed.

    Nobody types it any more. "Put the bowl in the drawer / fridge" — an entirely reasonable sentence
    to write in a GUI — silently makes a NESTED directory and splits one task's episodes across two.
    A leading "/" makes the path ABSOLUTE and throws SIMVLA_LEROBOT_ROOT away entirely. ".." climbs
    out of it. A trailing space makes a directory whose name nothing typed at a shell will ever match.
    None of these fails: they all record a dataset, somewhere else.

    Refused at AUTHORING, not at use. By the time simvla_gen reads task_language the run is minutes
    from writing frames, and the goal file — the artefact, the thing that gets copied around — is
    already wrong. This function is what validate_template asks, and emit() calls validate_template
    before it opens a single file.

    A TRAILING PERIOD IS LEGAL and must stay so: the reference task's language is
    "Put bowl inside drawer." — the demonstrated dataset's own label, on the only task known to have
    recorded one end to end. It is a sentence. Only "." twice in a row is a path.
    """
    if not isinstance(language, str) or not language:
        return [
            f"language must be a non-empty string — it is the label every frame of the dataset is "
            f"trained against, and a path component — got {language!r}"
        ]

    bad: list[str] = []
    if "/" in language or "\\" in language:
        bad.append("a path separator ('/' or '\\')")
    if ".." in language:
        bad.append("'..'")
    if "\x00" in language:
        bad.append("a NUL")
    if language != language.strip():
        bad.append("leading or trailing whitespace")

    if not bad:
        return []
    return [
        f"language {language!r} contains " + ", ".join(bad) + " — and the language is a DIRECTORY "
        f"NAME: simvla_gen.py:3217 writes the LeRobot dataset to "
        f"$SIMVLA_LEROBOT_ROOT/<task_language>/<task>, and a v2 goal file carries task_language "
        f"straight from this template. It would not fail; it would record the dataset somewhere "
        f"else. (A trailing period is fine — the reference task is 'Put bowl inside drawer.')"
    ]


def validate_template(t: TaskTemplate) -> None:
    """Fail loudly, at load, listing every problem at once.

    Fixing a twelve-step template one error per run is whack-a-mole.
    """
    problems: list[str] = []
    role_names = {r.name for r in t.roles}

    problems.extend(language_problems(t.language))

    for r in t.roles:
        if r.handle_of is not None and r.handle_of not in role_names:
            problems.append(
                f"role {r.name!r} is the handle of {r.handle_of!r}, which is not a declared role"
            )

    for i, step in enumerate(t.steps):
        spec = REGISTRY.get(step.skill)
        if spec is None:
            problems.append(f"step {i}: no skill {step.skill!r} in the registry")
            continue
        if step.action not in spec.actions:
            problems.append(
                f"step {i}: {step.skill!r} does not declare the action {step.action!r} "
                f"(it declares {list(spec.actions)})"
            )
        declared = {p.name: p for p in spec.params}
        for key, value in step.params.items():
            if key not in declared:
                problems.append(f"step {i}: {step.skill!r} declares no param {key!r}")
                continue
            if isinstance(declared[key], PrimPath):
                if not isinstance(value, str) or not value:
                    problems.append(
                        f"step {i}: param {key!r} must be a non-empty string prim path, "
                        f"got {value!r}"
                    )
                    continue
                ref = role_ref(value)
                if ref is not None and ref not in role_names:
                    problems.append(f"step {i}: {value!r} names no declared role")

    for g, group in enumerate(t.subtask_groups):
        for idx in group:
            if not (0 <= idx < len(t.steps)):
                problems.append(
                    f"subtask_groups[{g}]: step index {idx} is out of range "
                    f"(the task has {len(t.steps)} steps)"
                )

    if t.success is None:
        problems.append(
            "the template declares no success condition. Without one the emitted env config gets "
            "mdp.task2 — object at drawer height, right EEF home, drawer joint closed — whatever "
            "the task is; and simvla_gen.py:3181 gates demo export on that term firing, so a wrong "
            "condition yields no data at all, not a wrong metric."
        )
    else:
        problems.extend(validate_spec(t.success, role_names, kind="success"))
    if t.retry is not None:
        problems.extend(validate_spec(t.retry, role_names, kind="retry"))

    validate_scene(t.scene)
    if t.scene:                                    # empty scene = fallback, deferred to generator
        types_present = {o.object_type for o in t.scene}
        for r in t.roles:
            if r.object_type is not None and r.object_type not in types_present:
                problems.append(
                    f"role {r.name!r} manipulates a {r.object_type!r}, but the scene has no "
                    f"{r.object_type!r} object (types present: {sorted(types_present)})"
                )

    if problems:
        raise TaskTemplateError(
            f"{len(problems)} problem(s) in template {t.name!r}:\n  " + "\n  ".join(problems)
        )


def to_json(t: TaskTemplate) -> str:
    return json.dumps(
        {
            "name": t.name,
            "language": t.language,
            "roles": [
                {
                    "name": r.name,
                    "object_type": r.object_type,
                    "articulation_with": r.articulation_with,
                    "handle_of": r.handle_of,
                }
                for r in t.roles
            ],
            "steps": [
                {"skill": s.skill, "action": s.action, "params": s.params, "language": s.language}
                for s in t.steps
            ],
            "subtask_groups": t.subtask_groups,
            "scene": scene_to_dicts(t.scene),
            "success": t.success,
            "retry": t.retry,
        },
        indent=4,
        allow_nan=False,
    )


def from_json(text: str) -> TaskTemplate:
    raw = json.loads(text)
    return TaskTemplate(
        name=raw["name"],
        language=raw["language"],
        roles=[Role(**r) for r in raw["roles"]],
        steps=[TemplateStep(**s) for s in raw["steps"]],
        subtask_groups=[list(g) for g in raw.get("subtask_groups", [])],
        scene=scene_from_dicts(raw.get("scene", [])),
        success=raw.get("success"),
        retry=raw.get("retry"),
    )


def template_warnings(t: TaskTemplate) -> list[str]:
    """Non-fatal traps worth showing the author. Never refuses."""
    return spec_warnings(t.success, kind="success") if t.success else []
