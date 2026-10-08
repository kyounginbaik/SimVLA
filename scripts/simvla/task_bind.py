"""SimVLA: bind a task's roles to a kitchen's prims.

Stdlib only (plus task_template).

A role binds by object TYPE, which is what the kitchen generator already places
(kitchen_build.OBJECT_TYPES: bottle, bowl, apple, sodacan, nutella, mug). So a template written once
binds to every kitchen that has the pieces — and the twelve rotations of a kitchen share prim names,
so a rotation family costs nothing extra.

Ambiguity and absence are REPORTED, never guessed. A fan-out that silently skips kitchens reads as
"it covered everything" when it did not.
"""

from __future__ import annotations

from dataclasses import dataclass

from task_template import TaskTemplate


class BindingError(Exception):
    """A role could not be bound in this kitchen. Names the role and the reason."""


@dataclass(frozen=True)
class BoundPrim:
    prim_path: str
    #: True only for prims in env.scene.rigid_objects. A drawer handle belongs to an articulation
    #: and is NOT one — which is the whole reason its grasp check has to be disabled.
    is_rigid_body: bool
    #: Placement order in the kitchen's object list. NOT --target_idx, which is a STEP index into the
    #: goal script (task_runconfig.NO_TARGET_IDX says why). The two coincide in the reference task —
    #: v813 grasps at step 1 and its bowl is object 1 — and that coincidence cost a review.
    object_index: int

    @property
    def name(self) -> str:
        return self.prim_path.rstrip("/").rsplit("/", 1)[-1]

    @property
    def scene_entity(self) -> str:
        """The name of the SCENE ENTITY this prim belongs to — the ONLY name a composed condition
        may name, because that is what env.scene.rigid_objects / env.scene.articulations are keyed
        by.

        For a rigid role the prim IS its own scene entity: /world/bowl0 -> 'bowl0', and
        simvla_data_generator emits `bowl0 = RigidObjectCfg(...)`.

        For an articulation role it is NOT. bind_roles binds such a role to the DESCENDANT segment
        it matched (/world/base_cabinet/drawer_0_0), and a handle role to a segment below that
        (/world/base_cabinet/drawer_0_0/door_handle) — while the env config enumerates only
        FIRST-LEVEL prims (`len(path.split("/")) == 3`) into scene entities, so the entity is
        `base_cabinet` and nothing named `drawer_0_0` exists. Resolving @container to 'drawer_0_0'
        is what made every kitchen emitted from templates/bowl_to_drawer.json raise
        "the composed condition references 'drawer_0_0'" on its first termination evaluation.

        The drawer WITHIN the articulation is selected by a predicate's own `body` param
        (obj_near_prim(..., body="drawer_0_0")) and its joint by `joint`
        (joint_pos(..., joint="corpus_to_drawer_0_0")) — both matched against the ARTICULATION's
        body_names/joint_names. So the entity is exactly what those params need to be resolved
        against.

        NOT used by _target_prim: that one is rigid-only, where `name` and `scene_entity` agree.
        """
        if self.is_rigid_body:
            return self.name
        parts = self.prim_path.rstrip("/").split("/")
        # ['', 'world', 'base_cabinet', ...] -> 'base_cabinet'. A path too short to have a
        # first-level prim (nothing bind_roles produces) falls back to the last segment rather
        # than IndexError-ing on a caller that built a BoundPrim by hand.
        return parts[2] if len(parts) > 2 else self.name


@dataclass(frozen=True)
class KitchenScene:
    #: [(object_type, prim_path), ...] in placement order.
    objects: list[tuple[str, str]]
    #: prim_path -> the features it has, e.g. {"/world/base_cabinet/drawer_0_0": ["drawer","handle"]}
    articulations: dict[str, list[str]]


def bind_roles(t: TaskTemplate, scene: KitchenScene) -> dict[str, BoundPrim]:
    bound: dict[str, BoundPrim] = {}

    # Objects and articulations first; handle_of roles depend on them.
    for role in t.roles:
        if role.object_type is not None:
            matches = [
                (i, path) for i, (otype, path) in enumerate(scene.objects)
                if otype == role.object_type
            ]
            if not matches:
                raise BindingError(
                    f"role {role.name!r}: this kitchen has no object of type "
                    f"{role.object_type!r}"
                )
            if len(matches) > 1:
                raise BindingError(
                    f"role {role.name!r} is ambiguous: {len(matches)} objects of type "
                    f"{role.object_type!r} ({', '.join(p for _, p in matches)}). "
                    f"Pick one by clicking it in the composer."
                )
            i, path = matches[0]
            bound[role.name] = BoundPrim(prim_path=path, is_rigid_body=True, object_index=i)

        elif role.articulation_with is not None:
            # If another role is the handle OF this one, the task opens this articulation by its
            # handle — so only articulations that actually HAVE a handle can satisfy it. This
            # disambiguates the common case (several drawers, only some handled: e.g. a base_cabinet
            # drawer has a door_handle, a range drawer does not) and correctly SKIPS a kitchen whose
            # only matching drawer is handle-less, instead of binding one the handle step then fails
            # on with 'Prim not found: <drawer>/door_handle'. 'handle' is in an articulation's feature
            # list exactly when it has a door_handle child (see _scene_from_stage / _label_supports).
            needs_handle = any(r.handle_of == role.name for r in t.roles)
            # role.articulation_with is normalised to a TUPLE by Role.__post_init__, and every
            # feature in it must be present: a conjunction, not an alternation. See Role's
            # docstring for why one feature cannot name a cabinet door.
            wanted = role.articulation_with
            matches = [
                path for path, features in scene.articulations.items()
                if all(w in features for w in wanted) and (not needs_handle or "handle" in features)
            ]
            qual = " and ".join(repr(w) for w in wanted) + (" and a handle" if needs_handle else "")
            if not matches:
                raise BindingError(
                    f"role {role.name!r}: this kitchen has no articulation with a {qual}"
                )
            if len(matches) > 1:
                raise BindingError(
                    f"role {role.name!r} is ambiguous: {len(matches)} articulations have a "
                    f"{qual} ({', '.join(matches)})"
                )
            bound[role.name] = BoundPrim(
                prim_path=matches[0], is_rigid_body=False, object_index=-1
            )

    for role in t.roles:
        if role.handle_of is not None:
            parent = bound.get(role.handle_of)
            if parent is None:
                raise BindingError(
                    f"role {role.name!r} is the handle of {role.handle_of!r}, which did not bind"
                )
            bound[role.name] = BoundPrim(
                prim_path=f"{parent.prim_path}/door_handle",
                is_rigid_body=False,
                object_index=-1,
            )

    return bound


def bind_many(
    t: TaskTemplate, scenes: dict[str, KitchenScene]
) -> tuple[dict[str, dict[str, BoundPrim]], list[str]]:
    """Bind across kitchens. Returns (bound_by_kitchen, skipped) — never drops a kitchen silently."""
    bound: dict[str, dict[str, BoundPrim]] = {}
    skipped: list[str] = []
    for task_name, scene in scenes.items():
        try:
            bound[task_name] = bind_roles(t, scene)
        except BindingError as exc:
            skipped.append(f"{task_name}: {exc}")
    return bound, skipped
