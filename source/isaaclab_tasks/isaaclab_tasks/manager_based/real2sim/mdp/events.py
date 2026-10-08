"""Gain randomisation parameterised by the DAMPING RATIO rather than raw damping."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_gains_by_damping_ratio(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_inertia: dict[str, float],
    stiffness_range: tuple[float, float],
    zeta_range: tuple[float, float],
):
    """Draw stiffness log-uniformly, then damping as D = zeta * 2*sqrt(K*I).

    Sampling K and D independently -- as `randomize_actuator_gains` does -- puts most of the
    probability mass on combinations no actuator could have: measured over the 512 gain sets
    that survived the tightest acceptance test on 2026-09-01, 31% of joints sat below 2/10 of
    critical damping and only 32% inside a sane 0.3..2.0 band, with single arms spanning
    zeta 0.02 to 20. The free-space replay objective cannot separate those (its damping
    marginal stays at ~2.5 of its 3.0-decade prior at every threshold), so the fit is free to
    return an undamped shoulder -- which tracks a smooth trajectory perfectly and drops the
    mug the moment it touches anything.

    `zeta` is defined against a FIXED reference inertia per joint (scripts/simvla/joint_inertia.py,
    locked-chain inertia about the joint axis at zero configuration). The true inertia is
    configuration-dependent, so this is a change of coordinates that keeps every draw physically
    realisable -- not a claim about the instantaneous dynamics.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    lo_k, hi_k = math.log(stiffness_range[0]), math.log(stiffness_range[1])
    lo_z, hi_z = math.log(zeta_range[0]), math.log(zeta_range[1])

    for actuator in asset.actuators.values():
        idx = actuator.joint_indices
        names = [asset.data.joint_names[i] for i in
                 (range(asset.num_joints) if isinstance(idx, slice) else idx)]
        local = [k for k, n in enumerate(names) if n in joint_inertia]
        if not local:
            continue
        inertia = torch.tensor([joint_inertia[names[k]] for k in local],
                               device=asset.device, dtype=torch.float32)
        cols = torch.tensor(local, device=asset.device, dtype=torch.long)
        shape = (len(env_ids), len(local))

        k = torch.exp(torch.rand(shape, device=asset.device) * (hi_k - lo_k) + lo_k)
        z = torch.exp(torch.rand(shape, device=asset.device) * (hi_z - lo_z) + lo_z)
        d = z * 2.0 * torch.sqrt(k * inertia.unsqueeze(0))

        stiffness = actuator.stiffness[env_ids].clone()
        damping = actuator.damping[env_ids].clone()
        stiffness[:, cols] = k
        damping[:, cols] = d
        actuator.stiffness[env_ids] = stiffness
        actuator.damping[env_ids] = damping
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
            asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)
