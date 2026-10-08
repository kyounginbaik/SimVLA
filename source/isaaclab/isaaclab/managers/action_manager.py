# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations
import ipdb
import inspect
import torch
import torch.nn.functional as F
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.assets import AssetBase

from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

import numpy as np
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.math import euler_xyz_from_quat


def reorder_left_gripper_in_between(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 20)
    return torch.cat(
        [
            x[..., 0:9],     # L pos + L rot6d
            x[..., 18:19],   # L gripper
            x[..., 9:18],    # R pos + R rot6d
            x[..., 19:20],   # R gripper
        ],
        dim=-1,
    )


@torch.no_grad()
def quat_wxyz_to_R_scipy(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """
    quat_wxyz: (B,4) or (4,) torch tensor in (w,x,y,z) on any device
    returns:   (B,3,3) torch tensor on same device/dtype (NOTE: computed on CPU via SciPy)
    """
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz.unsqueeze(0)  # (1,4)

    # normalize like your code
    q = F.normalize(quat_wxyz, dim=-1)

    # SciPy runs on CPU, and expects xyzw
    q_np = q.detach().cpu().numpy().astype(np.float64)          # (B,4) wxyz
    q_xyzw = np.concatenate([q_np[:, 1:4], q_np[:, 0:1]], axis=-1)  # (B,4) xyzw

    Rm = R.from_quat(q_xyzw).as_matrix()  # (B,3,3) float64 numpy
    return torch.from_numpy(Rm).to(device=quat_wxyz.device, dtype=quat_wxyz.dtype)

class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        # call the base class constructor
        super().__init__(cfg, env)
        # parse config to obtain asset to which the term is applied
        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False

        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    @abstractmethod
    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """
        # check if config is None
        if cfg is None:
            raise ValueError("Action manager configuration is None. Please provide a valid configuration.")

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # create buffers to store actions
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)
        self._joint_position_replay_targets = None
        self._joint_position_replay_joint_ids = None
        self._joint_position_replay_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # check if any term has debug visualization implemented
        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Dimension"] = "r"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action
    
    @property
    def action_absolute(self) -> torch.Tensor:
        """The absolute actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        action_abs = []
        for name, term in self._terms.items():
            # Arm
            if name in ['armR_action', 'armL_action']:
                quat = term._ik_controller.ee_quat_des
                if quat.ndim == 1:
                    quat = quat.unsqueeze(0)  # shape (1, 4)

                # Normalize quaternion
                quat = F.normalize(quat, dim=-1)
                R = quat_wxyz_to_R_scipy(quat)  # (B,3,3)

                delta_local = torch.tensor([0.0, 0.0, -0.10956],
                                           device=R.device, dtype=R.dtype).view(1, 3, 1)

                delta_world = (R @ delta_local).squeeze(-1)  # (B,3)
                pos = term._ik_controller.ee_pos_des.clone() + delta_world
                rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1) 
                pos[:,0] += 0.095
                pos[:,2] += -0.823356
                action_abs.append(pos)    # 3
                action_abs.append(rot6d)  # 6

            # Gripper
            elif name in ['gripperR_action', 'gripperL_action']:
                # Export the requested binary command, not the contact controller's
                # intermediate motor target. The latter depends on joint polarity
                # and force feedback and cannot be decoded as open/close reliably.
                gripper_real = torch.where(term.raw_actions[:, :1] < 0, 0.1, -1.6)
                action_abs.append(gripper_real) # 1
            # Mobile base
            elif name in ['base_action']:
                action_abs.append(term.processed_actions[:, :3])

        action_abs = torch.cat(action_abs, dim=-1)  # or dim=0 if they're 1D

        base_link_idx = self._env.scene.articulations["robot"].find_bodies("base_link")[0][0]
        r, p, yaw = euler_xyz_from_quat(
            self._env.scene.articulations["robot"].data.body_quat_w[:, base_link_idx]
        ) 

        base_world = action_abs[:, -3:]           # (num_envs, 3) -> [vx_world, vy_world, omega]
        vx_world = base_world[:, 0]
        vy_world = base_world[:, 1]
        omega    = base_world[:, 2]        

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx_local =  cos_yaw * vx_world + sin_yaw * vy_world
        vy_local = -sin_yaw * vx_world + cos_yaw * vy_world

        base_local = torch.stack(
            [
                vx_local,  # so vy_local = -base_local[:, 0]
                vy_local,  # so vx_local = -base_local[:, 1]
                omega,      # unchanged
            ],
            dim=1,
        )  # shape: (num_envs, 3)

        action_abs = reorder_left_gripper_in_between(action_abs)

        action_dict = {
                        "ee_6D_pos": action_abs,
                        "base": base_local,
                        "joint_pos": self._env.scene.articulations["robot"]._joint_pos_target_sim,
                    }
        return action_dict
    
    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        self._joint_position_replay_mask[env_ids] = False
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)
        self._joint_position_replay_substep = 0

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            # ipdb.set_trace()
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def set_joint_position_replay_targets(
        self, targets: torch.Tensor | None, *, joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Optionally replay recorded motor targets through the physical actuators.

        Ordinary action terms still provide velocity/effort commands. Position
        targets are overridden after them at each physics substep, never written
        into the joint state. Passing None disables the override; reset disables
        it for the reset environments. Callers must validate dataset joint names.
        A subset may also execute a planned arm trajectory without overriding
        the adaptive gripper or other arm. Targets retain the full robot shape.
        A (num_envs, decimation, num_joints) sequence additionally preserves each
        physics-substep target; process_action rewinds it at each control step.
        """
        if targets is None:
            self._joint_position_replay_targets = None
            self._joint_position_replay_joint_ids = None
            self._joint_position_replay_mask[:] = False
            return
        robot = self._env.scene.articulations["robot"]
        shape = robot.data.joint_pos.shape
        valid_shape = targets.shape == shape or (
            targets.ndim == 3 and targets.shape[0] == shape[0]
            and targets.shape[1] == self._env.cfg.decimation and targets.shape[2] == shape[1])
        if not valid_shape or not bool(torch.isfinite(targets).all()):
            raise ValueError("recorded joint targets must be finite and match the robot joint-state shape")
        for label, indices, size in (("joint_ids", joint_ids, targets.shape[-1]),
                                     ("env_ids", env_ids, targets.shape[0])):
            if indices is not None and (not len(indices) or
                    any(type(i) is not int or not 0 <= i < size for i in indices) or
                    len(set(indices)) != len(indices)):
                raise ValueError(f"{label} must contain unique in-range integer indices")
        self._joint_position_replay_targets = targets.to(self.device).detach().clone()
        self._joint_position_replay_substep = 0
        self._joint_position_replay_joint_ids = None if joint_ids is None else list(joint_ids)
        self._joint_position_replay_mask[:] = env_ids is None
        if env_ids is not None:
            self._joint_position_replay_mask[list(env_ids)] = True

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()
        if self._joint_position_replay_targets is not None:
            ids = torch.where(self._joint_position_replay_mask)[0]
            if ids.numel():
                joint_ids = self._joint_position_replay_joint_ids
                targets = self._joint_position_replay_targets[ids]
                if targets.ndim == 3:
                    index = self._joint_position_replay_substep
                    if index >= targets.shape[1]:
                        raise RuntimeError("Recorded motor substeps exhausted before the next control step")
                    targets = targets[:, index]
                    self._joint_position_replay_substep += 1
                if joint_ids is not None:
                    targets = targets[:, joint_ids]
                self._env.scene.articulations["robot"].set_joint_position_target(
                    targets, joint_ids=joint_ids, env_ids=ids)

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # parse action terms from the config
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
