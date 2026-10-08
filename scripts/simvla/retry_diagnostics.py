"""Explain simple top-level retry unions without re-running stateful predicates.

The collector reads the termination manager's already-computed retry mask. Re-evaluating a
stateful predicate such as ``base_floor_collision`` would advance its consecutive-frame counter a
second time. Instead, evaluate the stateless branches from captured simulator state and infer the
single remaining branch from the true union.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def describe_retry_union(
    spec: Mapping[str, Any],
    retry_envs: Sequence[int],
    *,
    object_z: Mapping[str, Sequence[float]],
    base_z: Sequence[float],
) -> list[str]:
    """Return one diagnostic line per retrying environment.

    ``obj_z`` and ``robot_fell`` are measured directly. If a top-level ``any`` has exactly one
    other leaf and every measured leaf is false, that remaining leaf is true by the definition of
    the already-computed union. Unknown leaves are never called, so diagnostics cannot mutate
    predicate history.
    """
    leaves = spec.get("any")
    if not isinstance(leaves, list):
        return [f"env={env_i} retry=true branches=unavailable(non-any spec)" for env_i in retry_envs]

    lines = []
    for env_i in retry_envs:
        known: list[str] = []
        known_true = False
        unknown: list[str] = []
        for leaf in leaves:
            if not isinstance(leaf, Mapping) or len(leaf) != 1:
                unknown.append("invalid_leaf")
                continue
            name, raw_params = next(iter(leaf.items()))
            params = raw_params or {}
            if name == "obj_z":
                role = params["role"]
                value = float(object_z[role][env_i])
                lo, hi = params.get("lo"), params.get("hi")
                hit = True
                if lo is not None:
                    hit = hit and value > float(lo)
                if hi is not None:
                    hit = hit and value < float(hi)
                known_true |= hit
                known.append(f"obj_z={'true' if hit else 'false'}(z={value:.4f})")
            elif name == "robot_fell":
                value = float(base_z[env_i])
                threshold = float(params["z"])
                hit = value < threshold
                known_true |= hit
                known.append(f"robot_fell={'true' if hit else 'false'}(z={value:.4f})")
            else:
                unknown.append(str(name))

        if len(unknown) == 1 and not known_true:
            known.append(f"{unknown[0]}=true(inferred_from_union)")
        else:
            known.extend(f"{name}=unknown" for name in unknown)
        lines.append(f"env={env_i} retry=true branches=" + " ".join(known))
    return lines
