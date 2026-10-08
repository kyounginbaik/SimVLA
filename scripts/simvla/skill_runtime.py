"""What a runtime-resolved skill is allowed to know about the running simulation.

A handful of SimVLA skills cannot compute their target while authoring: it depends on where the
robot actually IS when the step runs — the live eef pose, the base yaw, whether there is ramen in
the scene. Those skills define ``resolve(ctx, envs, params, eef_idx=None)`` instead of ``plan()``,
and ``ctx`` is the object defined here. ``simvla_gen.py`` builds one per goal-eval pass; the skill
reads the sim through it rather than reaching into the executor's locals.

That is now the whole of this module. It used to also hold the *dispatch*:

    class SkillSentinel:
        PAUSE = 0.3            # ... eight more magic floats
    RUNTIME_RESOLVERS: Dict[float, Callable] = {}

    @register_runtime_resolver(SkillSentinel.PAUSE)
    def resolve_pause(ctx, env_idx, eef_idx): ...

A goal file said "pause" by putting 0.3 in all four quaternion slots, and the executor recovered
that meaning with ``torch.isclose(skill_r, full_like(skill_r, 0.3)).all(dim=1)`` — a float, in a
coordinate slot, standing for a verb. Every failure mode of that encoding actually happened or was
one edit away from happening: a payload that matched no sentinel silently became a coordinate and
drove the robot to the corner of the kitchen (the refrigerator, in 73 goal files); the values were
not "well outside the valid joint-angle range" as the docstring claimed but ordinary numbers
(0.3, -0.12, -0.25) that a real pose could hit; and one line of rotation noise in the quaternion
slots would have erased every arm sentinel at once.

A step now says which skill it is by NAME, in the goal file (``goal_format`` v2), and the executor
dispatches on ``skill_ids_tensor``, an integer tensor. There is nothing to sniff.

**The v1 values still exist, in exactly one place: ``goal_format.py``.** 7,723 goal files on disk
are still v1 and their steps are *defined* by those floats, so the quarantined legacy reader keeps
its own model of the old dispatch (``CHANNEL_SENTINELS``, ``FLAG_THRESHOLD``, ``_isclose_all``) and
uses it to recover a skill name at load. That is the values as *archaeology* — read once, at the
edge, and converted to a name. Nothing downstream ever sees them. Do not bring them back here.

This module is intentionally Isaac/USD-free — no ``omni.usd``, no ``isaaclab.app``, no ``pxr`` — so
both ``simvla_data_generator.py`` (authoring) and ``simvla_gen.py`` (executing) can import it
without double-initializing AppLauncher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeContext:
    """Snapshot of runtime simulation state needed by skill resolvers.

    Build once per goal-eval pass in ``simvla_gen.py``; resolvers read from it rather than reaching
    into the surrounding ``simvla_gen`` scope directly. Keeping this small + explicit makes the
    skill↔runtime contract obvious.
    """

    robot: Any                # env.scene.articulations["robot"]
    env_origins: Any          # env.scene.env_origins; shape (N_env, 3)
    r_eef_idx: int
    l_eef_idx: int
    base_link_idx: int
    has_ramen: bool = False
    has_sweet_potato: bool = False
    # Scene rigid objects — needed by resolvers that read object poses at runtime (e.g.
    # skills.resolve_grasp_target → env.scene.rigid_objects[...]). Pass `env.scene.rigid_objects`
    # here; defaults to None for resolvers that don't need it.
    rigid_objects: Any = None
