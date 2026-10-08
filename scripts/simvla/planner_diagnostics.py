"""Read-only, JSON-safe planner failure evidence (never changes collision checks)."""
from __future__ import annotations

import json
import math


def overlapping_link_pairs(payload, ignored):
    """Report positive sphere penetration, excluding configured pair ignores."""
    spheres, links = payload["spheres_base_frame"], payload["sphere_links"]
    if len(spheres) != len(links):
        raise ValueError("sphere/link counts differ")
    excluded = {frozenset((a, b)) for a, others in ignored.items() for b in others}
    overlaps = {}
    for i, (first, a) in enumerate(zip(spheres, links)):
        if len(first) != 4 or not all(math.isfinite(v) for v in first):
            raise ValueError("invalid sphere")
        if first[3] <= 0:
            continue
        for second, b in zip(spheres[i + 1:], links[i + 1:]):
            if a == b or frozenset((a, b)) in excluded or second[3] <= 0:
                continue
            penetration = first[3] + second[3] - math.dist(first[:3], second[:3])
            if penetration > 0:
                pair = tuple(sorted((a, b)))
                overlaps[pair] = max(overlaps.get(pair, 0.), penetration)
    return [{"links": list(pair), "penetration_m": depth}
            for pair, depth in sorted(overlaps.items(), key=lambda item: -item[1])]


def failure_snapshot(kinematics, joint_state, *, arm, step, status):
    """Capture exactly the ordered start state and its collision spheres."""
    q = joint_state.position.reshape(1, -1)
    config = kinematics.kinematics_config
    spheres = kinematics.get_state(q).get_link_spheres().reshape(-1, 4)
    index_to_name = {int(index): name for name, index in config.link_name_to_idx_map.items()}
    payload = {
        "arm": arm, "step": int(step), "status": str(status),
        "joint_names": list(joint_state.joint_names),
        "joint_position": q.detach().cpu().reshape(-1).tolist(),
        "spheres_base_frame": spheres.detach().cpu().tolist(),
        "sphere_links": [index_to_name[int(index)]
                         for index in config.link_sphere_idx_map.detach().cpu().reshape(-1).tolist()],
    }
    # Refuse nonfinite diagnostics instead of producing invalid JSON.
    return json.dumps(payload, allow_nan=False)
