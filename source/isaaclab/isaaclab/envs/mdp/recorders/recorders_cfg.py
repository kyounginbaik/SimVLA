# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
	"""Configuration for the initial state recorder term."""

	class_type: type[RecorderTerm] = recorders.InitialStateRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
	"""Configuration for the step state recorder term."""

	class_type: type[RecorderTerm] = recorders.PostStepStatesRecorder


@configclass
class TerminalStateRecorderCfg(RecorderTermCfg):
	"""One physical snapshot before an episode is reset."""

	class_type: type[RecorderTerm] = recorders.TerminalStateRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
	"""Configuration for the step action recorder term."""

	class_type: type[RecorderTerm] = recorders.PreStepActionsRecorder

@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
	"""Configuration for the step policy observation recorder term."""

	class_type: type[RecorderTerm] = recorders.PreStepFlatPolicyObservationsRecorder

@configclass
class LeRobotPostStepActionsRecorderCfg(RecorderTermCfg):
	"""Configuration for the step action recorder term."""

	class_type: type[RecorderTerm] = recorders.LeRobotPostStepActionsRecorder

@configclass
class LeRobotObservationsRecorderCfg(RecorderTermCfg):
	"""Configuration for the camera recorder term."""

	class_type: type[RecorderTerm] = recorders.LeRobotObservationsRecorder

@configclass
class PreLeRobotObservationsRecorderCfg(RecorderTermCfg):
	"""Configuration for the camera recorder term."""

	class_type: type[RecorderTerm] = recorders.PreLeRobotObservationsRecorder

##
# Recorder manager configurations.
##

@configclass
class PreActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
	"""Recorder configurations for recording actions and states."""

	# record_initial_state = InitialStateRecorderCfg()
	record_pre_step_actions = LeRobotPostStepActionsRecorderCfg()
	record_post_step_states = PreLeRobotObservationsRecorderCfg()


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
	"""Recorder configurations for recording actions and states."""

	record_initial_state = InitialStateRecorderCfg()
	record_terminal_state = TerminalStateRecorderCfg()
	record_pre_step_actions = LeRobotPostStepActionsRecorderCfg()
	record_post_step_states = LeRobotObservationsRecorderCfg()
	# record_post_step_states = LeRobotPostStepStatesRecorderCfg()
	# record_post_step_states = PostStepStatesRecorderCfg()
	# record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
