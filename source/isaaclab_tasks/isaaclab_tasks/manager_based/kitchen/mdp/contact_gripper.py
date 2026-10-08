"""Binary gripper command that holds the jaw pose reached at bilateral mug contact."""

from __future__ import annotations

from dataclasses import MISSING
import math
import os

import torch
from simvla.gripper import resolve_pad_joint_indices

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass


class ContactHoldingBinaryJointPositionAction(BinaryJointPositionAction):
    """Stop driving the fingers through a mug once both pads are loaded.

    The ordinary binary action continues toward a fully closed joint target during
    lift. On these two grippers, that collapses the jaw gap and unloads both pads.
    Preserve the measured contact configuration with a small closing preload until
    an open action or an environment reset clears the latch.
    """

    cfg: ContactHoldingBinaryJointPositionActionCfg

    def __init__(self, cfg, env):
        if not all(math.isfinite(value) for value in (
            cfg.minimum_pad_force_n, cfg.preload_fraction, cfg.target_pad_force_n,
            cfg.maximum_pad_force_n, cfg.balance_step_fraction,
        )):
            raise ValueError("gripper control settings must be finite")
        super().__init__(cfg, env)
        if cfg.minimum_pad_force_n < 0 or not 0 <= cfg.preload_fraction <= 1:
            raise ValueError("gripper contact force and preload settings must be nonnegative and bounded")
        if (cfg.target_pad_force_n < cfg.minimum_pad_force_n
                or cfg.maximum_pad_force_n < cfg.target_pad_force_n
                or not 0 < cfg.balance_step_fraction <= 1):
            raise ValueError("gripper force-balance thresholds and step must be ordered and positive")
        self._pad_joint_indices = (resolve_pad_joint_indices(self._joint_names, cfg.pad_joint_names)
                                   if cfg.force_balance_enabled else ())
        if len(cfg.contact_sensor_names) != 2:
            raise ValueError("contact holding requires exactly two pad sensors")
        self._contact_sensors = tuple(env.scene.sensors.get(name) for name in cfg.contact_sensor_names)
        if any(sensor is None for sensor in self._contact_sensors):
            raise ValueError(f"missing gripper contact sensors: {cfg.contact_sensor_names}")
        if len(cfg.additional_contact_sensor_names) != 2:
            raise ValueError("additional contact sensors must be grouped by the two fingers")
        names = list(cfg.contact_sensor_names) + [
            name for group in cfg.additional_contact_sensor_names for name in group]
        if len(set(names)) != len(names):
            raise ValueError("a contact sensor cannot contribute to multiple finger measurements")
        self._additional_contact_sensors = tuple(
            tuple(env.scene.sensors.get(name) for name in group)
            for group in cfg.additional_contact_sensor_names)
        if any(sensor is None for group in self._additional_contact_sensors for sensor in group):
            raise ValueError(f"missing finger-segment contact sensors: {cfg.additional_contact_sensor_names}")
        self._closing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._latched = torch.zeros_like(self._closing)
        self._balance_frozen = torch.zeros_like(self._closing)
        self._hold_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._contact_pos = torch.zeros_like(self._hold_target)
        self._preload_by_pad = torch.full(
            (self.num_envs, 2), cfg.preload_fraction, device=self.device)
        self._balance_ticks = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _pad_load(self, sensor) -> torch.Tensor:
        forces = getattr(sensor.data, "force_matrix_w", None)
        # These sensors are object-filtered. Net force also includes the counter
        # and other objects, so it cannot substitute for missing mug measurements.
        if forces is None:
            return torch.zeros(self.num_envs, device=self.device)
        vectors = forces.reshape(self.num_envs, -1, 3)
        if vectors.shape[1] == 0:
            return torch.zeros(self.num_envs, device=self.device)
        loads = torch.linalg.vector_norm(vectors, dim=-1).amax(dim=-1)
        return torch.where(torch.isfinite(loads), loads, torch.zeros_like(loads))

    def _finger_load(self, index: int) -> torch.Tensor:
        """Total object-filtered normal load over distinct links of one finger."""
        load = self._pad_load(self._contact_sensors[index])
        for sensor in self._additional_contact_sensors[index]:
            load = load + self._pad_load(sensor)
        return load

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        values = actions.reshape(self.num_envs)
        self._closing = values == 0 if actions.dtype == torch.bool else values < 0
        self._latched[~self._closing] = False
        self._balance_frozen[~self._closing] = False
        holding = self._closing & self._latched
        self._processed_actions[holding] = self._hold_target[holding]

    def apply_actions(self):
        ready = self._closing & ~self._latched
        if torch.any(ready):
            loaded = (self._finger_load(0) >= self.cfg.minimum_pad_force_n) & (
                self._finger_load(1) >= self.cfg.minimum_pad_force_n
            )
            new_latches = ready & loaded
            if torch.any(new_latches):
                joint_pos = self._asset.data.joint_pos[new_latches][:, self._joint_ids]
                self._contact_pos[new_latches] = joint_pos
                self._preload_by_pad[new_latches] = self.cfg.preload_fraction
                self._hold_target[new_latches] = joint_pos + self.cfg.preload_fraction * (
                    self._close_command - joint_pos
                )
                self._processed_actions[new_latches] = self._hold_target[new_latches]
                self._latched[new_latches] = True
                for env_id in new_latches.nonzero(as_tuple=False).flatten().tolist():
                    print(f"[gripper-hold] env{env_id} {self.cfg.contact_sensor_names}: "
                          f"holding contact joint targets {self._hold_target[env_id].tolist()}", flush=True)
        if self.cfg.force_balance_enabled:
            holding = self._closing & self._latched
            if self.cfg.freeze_at_target_force:
                loads = torch.stack([self._finger_load(index) for index in (0, 1)], dim=1)
                reached = holding & ~self._balance_frozen & torch.all(
                    (loads >= self.cfg.target_pad_force_n)
                    & (loads <= self.cfg.maximum_pad_force_n), dim=1)
                self._balance_frozen[reached] = True
                for env_id in reached.nonzero(as_tuple=False).flatten().tolist():
                    print(f"[gripper-hold] env{env_id} balanced targets frozen at "
                          f"{loads[env_id].tolist()} N", flush=True)
                holding &= ~self._balance_frozen
            if torch.any(holding):
                loads = torch.stack([self._finger_load(index) for index in (0, 1)], dim=1)
                fractions = self._preload_by_pad[holding]
                measured = loads[holding]
                step = self.cfg.balance_step_fraction
                fractions = torch.where(
                    measured < self.cfg.target_pad_force_n,
                    torch.clamp(fractions + step, max=1.0), fractions)
                fractions = torch.where(
                    measured > self.cfg.maximum_pad_force_n,
                    torch.clamp(fractions - step, min=0.0), fractions)
                self._preload_by_pad[holding] = fractions
                # Articulation order may be proximal1, proximal2, distal1, distal2.
                # Sensor-to-jaw association must use names, not contiguous halves.
                updated = self._hold_target[holding].clone()
                initial_all = self._contact_pos[holding]
                for pad, joint_ids in enumerate(self._pad_joint_indices):
                    indices = list(joint_ids)
                    initial = initial_all[:, indices]
                    updated[:, indices] = initial + fractions[:, pad:pad + 1] * (
                        self._close_command[indices] - initial)
                self._hold_target[holding] = updated
                self._processed_actions[holding] = self._hold_target[holding]
                self._balance_ticks[holding] += 1
                if (bool(holding[0]) and int(self._balance_ticks[0]) % 60 == 0
                        and bool(torch.any(loads[0] >= self.cfg.minimum_pad_force_n))):
                    print(f"[gripper-balance] env0 finger_load_n={loads[0].tolist()} "
                          f"preload={self._preload_by_pad[0].tolist()}", flush=True)
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._latched[env_ids] = False
        self._closing[env_ids] = False
        self._balance_frozen[env_ids] = False
        self._balance_ticks[env_ids] = 0


@configclass
class ContactHoldingBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Contact-aware close parameters for an instrumented parallel gripper."""

    class_type: type = ContactHoldingBinaryJointPositionAction
    contact_sensor_names: tuple[str, str] = MISSING
    additional_contact_sensor_names: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    pad_joint_names: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    minimum_pad_force_n: float = 1.0
    preload_fraction: float = float(os.environ.get("SIMVLA_GRIPPER_PRELOAD_FRACTION", "0.40"))
    force_balance_enabled: bool = os.environ.get("SIMVLA_GRIPPER_FORCE_BALANCE", "0") == "1"
    freeze_at_target_force: bool = os.environ.get("SIMVLA_GRIPPER_FREEZE_AT_FORCE", "0") == "1"
    target_pad_force_n: float = float(os.environ.get("SIMVLA_GRIPPER_TARGET_FORCE_N", "6.0"))
    maximum_pad_force_n: float = float(os.environ.get("SIMVLA_GRIPPER_MAX_FORCE_N", "12.0"))
    # Per physics substep, not per exported control frame. When comparing finer
    # integration, scale this inversely with substeps to preserve preload rate.
    balance_step_fraction: float = float(os.environ.get("SIMVLA_GRIPPER_BALANCE_STEP_FRACTION", "0.005"))
