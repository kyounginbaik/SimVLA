"""RB-Y1 variants of the two kitchen MDP terms that are hardcoded for Anubis.

`molmospace_rby1.py` was authored by copying the Anubis Molmospace env and swapping in
RBY1_CFG, but it kept referencing the SHARED `observations.ee_6d_pos` and
`terminations.task1_molmospace`. Both are Anubis-specific in ways that do not transfer:

  * terminations.task1_molmospace looks up bodies `gripper1R` / `gripper1L`. RB-Y1 has no
    such bodies (its jaws are ee_finger_r1/r2, ee_finger_l1/l2), so the term raises
    ValueError on the first env.step() and the run dies.
  * the same function compares each EE against module-level `home_r` / `home_l`, which are
    ANUBIS base-frame home positions. On RB-Y1 the true home is ~0.45 m lower in z, so the
    `distance < 0.1` gate could never be satisfied. This one fails SILENTLY: no error, just
    a success condition that is always False.
  * observations.ee_6d_pos reads the grippers by hardcoded index, joint_pos[:, -1] and
    [:, -3]. RB-Y1's 29 joints end l1, l2, r1, r2, so -1 is gripper_finger_r2 and -3 is
    gripper_finger_l2 -- the wrong finger of each pair AND left/right transposed.

Rather than make the shared functions robot-aware (they are imported by ~9,310 Anubis
kitchen envs), the RB-Y1 forms live here and are wired up only by molmospace_rby1.py.
observations.py and terminations.py are left byte-identical.

Every RB-Y1 number below is MEASURED, not inferred, by
`scripts/simvla/rby1_probe.py --task Isaac-Deploy-Molmospace-RBY1 --robot rby1`
(job 2048623, 2026-08-18); see the log at mspace_logs/probe.log.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse

from .observations import ee_6d_pos

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


#: Measured home EE positions in base_link frame, at the env's reset pose.
#: Anubis, for contrast, is (0.2257,-0.0988,1.0351) / (0.2203,0.1080,1.0342).
_HOME_R_RBY1 = (0.0074, -0.2199, 0.5859)   # ee_link1, the RIGHT arm
_HOME_L_RBY1 = (0.0101, 0.2198, 0.5867)    # ee_link2, the LEFT arm

#: Measured jaw separation with the jaws commanded fully open: 0.08600 m.
#: Anubis tests `> 0.079` against a 0.080 m travel, i.e. 1 mm below full open. The same 1 mm
#: margin is kept here so the two robots' success criteria are equally strict.
#: NOTE this is a TIGHT test by construction -- it asks for a fully open gripper, not merely a
#: released one -- and it is inherited from Anubis's semantics deliberately, not chosen.
_JAW_OPEN_RBY1 = 0.085


def ee_6d_pos_rby1(env: ManagerBasedRLEnv) -> torch.Tensor:
	"""ee_6d_pos with the two gripper dims read by NAME instead of by Anubis's fixed indices.

	The first 21 dims (both arms' xyz + 6-D rotation) are produced by the shared function
	unchanged: its frame convention (R @ [0,0,-0.10956], x += 0.095, z += -0.823356) is a fixed
	dataset convention with no robot term, and calibration/rby1.json inverts exactly it.
	Only the trailing two gripper dims are Anubis-indexed, so only those are recomputed.
	"""
	obs = ee_6d_pos(env).clone()

	robot = env.scene.articulations["robot"]
	names = robot.joint_names
	# The *2 fingers, NOT the *1 fingers. The scaling below (0.1 - 1.7*q/0.04) encodes
	# closed -> +0.1 and open -> -1.6, which assumes the joint opens POSITIVE. That holds for
	# Anubis, whose open_command_expr is {"gripper2.*": 0.04} -- every gripper joint positive.
	# RB-Y1's is ANTISYMMETRIC: l1 -> -0.04 while l2 -> +0.04. Reading l1/r1 therefore flips the
	# sign and yields ~+2.2 against a [-1.6, 0.1] envelope. Measured, not guessed: an earlier
	# build of this override used l1/r1 and the checkpoint reported
	# l_gripper=2.2250, r_gripper=2.2250 out of range. l2/r2 open positive, matching the formula.
	l_idx = names.index("gripper_finger_l2")
	r_idx = names.index("gripper_finger_r2")

	obs[:, -2] = 0.1 - 1.7 * robot.data.joint_pos[:, l_idx] / 0.04
	obs[:, -1] = 0.1 - 1.7 * robot.data.joint_pos[:, r_idx] / 0.04
	return obs


def task1_molmospace_rby1(
	env: ManagerBasedRLEnv,
	sink_pos: torch.Tensor,
	obj_cfg: SceneEntityCfg = SceneEntityCfg("mug0"),
) -> torch.Tensor:
	"""task1_molmospace with RB-Y1's measured home poses and jaw bodies.

	Success = object near the sink AND both EEs returned home AND the right jaw open.
	Structurally identical to the Anubis form; only the three Anubis literals are replaced.
	"""
	obj: RigidObject = env.scene[obj_cfg.name]
	obj_pos = obj.data.body_pos_w.squeeze(1) - env.scene.env_origins

	# 1. Object close to the sink
	radius = 0.2
	sinks_pos = sink_pos.unsqueeze(0).repeat(env.scene.num_envs, 1)
	distance = torch.linalg.norm(sinks_pos[:, :2] - obj_pos[:, :2], dim=-1)
	close_sink = distance <= radius

	# 2. Both EEs back at their home pose, in base frame
	robot = env.scene.articulations["robot"]
	eef_r_idx = robot.find_bodies("ee_link1")[0][0]
	eef_l_idx = robot.find_bodies("ee_link2")[0][0]
	base_idx = robot.find_bodies("base_link")[0][0]

	base_pos_w = robot.data.body_pos_w[:, base_idx]
	base_quat_w = robot.data.body_quat_w[:, base_idx]

	eef_r_pos_base = quat_rotate_inverse(base_quat_w, robot.data.body_pos_w[:, eef_r_idx] - base_pos_w)
	eef_l_pos_base = quat_rotate_inverse(base_quat_w, robot.data.body_pos_w[:, eef_l_idx] - base_pos_w)

	# Built on the robot's own device rather than a module-level cuda:0 literal, so this
	# does not break on cpu or a second GPU.
	dev = eef_r_pos_base.device
	home_r = torch.tensor(_HOME_R_RBY1, device=dev, dtype=eef_r_pos_base.dtype)
	home_l = torch.tensor(_HOME_L_RBY1, device=dev, dtype=eef_l_pos_base.dtype)

	radius_home = 0.1
	close_home = (torch.linalg.norm(eef_r_pos_base - home_r, dim=-1) < radius_home) & (
		torch.linalg.norm(eef_l_pos_base - home_l, dim=-1) < radius_home
	)

	# 3. Right jaw open -- RB-Y1's jaws are ee_finger_r1 / ee_finger_r2.
	jaw_a = robot.find_bodies("ee_finger_r1")[0][0]
	jaw_b = robot.find_bodies("ee_finger_r2")[0][0]
	jaw_sep = torch.linalg.norm(
		robot.data.body_pos_w[:, jaw_a] - robot.data.body_pos_w[:, jaw_b], dim=1
	)
	is_open = jaw_sep > _JAW_OPEN_RBY1

	return close_home & close_sink & is_open
