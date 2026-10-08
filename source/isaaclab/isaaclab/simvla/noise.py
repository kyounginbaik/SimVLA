"""Scene-cfg perturbation helpers used by SimVLA replay / data-gen.

Two top-level entry points:

- :func:`add_camera_noise_to_scene_cfg_once`: jitter each camera's
  position and orientation in the env config.
- :func:`add_light_noise_to_scene_cfg_once`: jitter the scene light's
  intensity, color temperature, and (subtle) RGB tint.

Both return a deepcopied env_cfg so the caller can keep the original
template intact. The randomization is "once" — the returned config
holds a single fixed jitter; call again with a new seed to resample.
"""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from isaaclab.simvla.rot import axis_angle_to_quat_wxyz, quat_mul_wxyz_np


def _noisy_pose_once(base_pos, base_q, rng, pos_sigma, rot_max_deg, yaw_only):
    """Sample one perturbed pose around (base_pos, base_q)."""
    if pos_sigma <= 0:
        dpos = np.zeros(3, dtype=np.float64)
    else:
        v = rng.normal(0.0, 1.0, size=(3,))
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n = 1.0
        v = v / n
        r = float(pos_sigma) * (rng.random() ** (1.0 / 3.0))
        dpos = v * r

    new_pos = base_pos + dpos

    ang = rng.uniform(-math.radians(rot_max_deg), math.radians(rot_max_deg))
    if yaw_only:
        dq = axis_angle_to_quat_wxyz([0, 1, 0], ang)
    else:
        axis = rng.normal(size=(3,))
        axis /= (np.linalg.norm(axis) + 1e-12)
        dq = axis_angle_to_quat_wxyz(axis, ang)

    new_q = quat_mul_wxyz_np(dq, base_q)
    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)
    return new_pos, new_q


def add_camera_noise_to_scene_cfg_once(
    env_cfg,
    cams=("front", "wrist_left", "wrist_right"),
    pos_sigma=0.05,
    rot_max_deg=2.0,
    yaw_only=False,
    seed=None,
):
    """Return a deepcopy of `env_cfg` with each named camera's offset jittered."""
    rng = np.random.default_rng(seed)
    cfg = deepcopy(env_cfg)

    if isinstance(cams, str):
        cams = (cams,)

    for name in cams:
        if not hasattr(cfg.scene, name):
            continue
        cam = getattr(cfg.scene, name)
        if not hasattr(cam, "offset"):
            continue

        base_pos = np.array(cam.offset.pos, dtype=np.float64)
        base_q = np.array(cam.offset.rot, dtype=np.float64)

        new_pos, new_q = _noisy_pose_once(
            base_pos, base_q, rng, pos_sigma, rot_max_deg, yaw_only
        )
        cam.offset.pos = tuple(new_pos.tolist())
        cam.offset.rot = tuple(new_q.tolist())

    return cfg


def add_light_noise_to_scene_cfg_once(
    env_cfg,
    intensity_base=6000.0,
    intensity_stop_range=1.5,
    kelvin_choices=(2700, 3200, 4000, 5000, 6500, 8000, 9000),
    kelvin_uniform=None,
    rgb_jitter_sigma=0.03,
    seed=None,
):
    """Return a deepcopy of `env_cfg` with the scene light randomly perturbed.

    ``intensity_stop_range`` is in *log2 stops* — the intensity multiplier
    is ``2 ** U(-r, +r)``. Color temperature is picked from
    ``kelvin_choices`` (discrete) unless ``kelvin_uniform=(lo, hi)`` is
    provided, in which case it's sampled uniformly from that range. A
    small RGB tint around white is also applied.
    """
    rng = np.random.default_rng(seed)
    cfg = deepcopy(env_cfg)

    if not hasattr(cfg.scene, "light") or not hasattr(cfg.scene.light, "spawn"):
        return cfg

    light = cfg.scene.light.spawn

    m = 2.0 ** rng.uniform(-intensity_stop_range, intensity_stop_range)
    light.intensity = float(intensity_base * m)

    if kelvin_uniform is not None:
        k_lo, k_hi = kelvin_uniform
        light.enable_color_temperature = True
        light.color_temperature = float(rng.uniform(k_lo, k_hi))
    else:
        light.enable_color_temperature = True
        light.color_temperature = float(rng.choice(kelvin_choices))

    jitter = rng.normal(0.0, rgb_jitter_sigma, size=(3,))
    col = np.clip(1.0 + jitter, 0.7, 1.3)
    light.color = (float(col[0]), float(col[1]), float(col[2]))

    return cfg
