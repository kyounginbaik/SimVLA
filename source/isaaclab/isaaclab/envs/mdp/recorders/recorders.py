# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm

import torch
import os
import json
import time
import ipdb
from isaaclab.utils import convert_dict_to_backend

class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class TerminalStateRecorder(RecorderTerm):
    """Keep physical terminal state before automatic reset hides the outcome."""

    def record_pre_reset(self, env_ids):
        indices = slice(None) if env_ids is None else env_ids
        lengths = getattr(self._env, "episode_length_buf", None)
        if lengths is not None and not bool(torch.any(lengths[indices] > 0)):
            return None, None

        def select(value):
            if isinstance(value, dict):
                return {key: select(item) for key, item in value.items()}
            return value[indices]

        return "terminal_state", select(self._env.scene.get_state(is_relative=True))

class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["policy"]

class LeRobotPostStepActionsRecorder(RecorderTerm):
    """Record final motor targets and optionally every physics-substep target."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        option = os.environ.get("SIMVLA_RECORD_JOINT_SUBSTEPS", "0")
        if option not in ("0", "1"):
            raise ValueError("SIMVLA_RECORD_JOINT_SUBSTEPS must be 0 or 1")
        env._record_joint_substeps = option == "1"

    def record_post_step(self):
        actions = self._env.action_manager.action_absolute
        if self._env._record_joint_substeps:
            samples = self._env._recorded_joint_substeps
            if len(samples) != self._env.cfg.decimation:
                raise RuntimeError("Incomplete physics-substep motor recording")
            actions["joint_substeps"] = torch.stack(samples, dim=1)
        return "actions", actions
#1. Single process Single thread: 0.505

#class LeRobotObservationsRecorder(RecorderTerm):
#   def __init__(self, *args, **kwargs):
#       super().__init__(*args, **kwargs)
#       self.cameras = ["front", "wrist_left", "wrist_right", "back"]
#       self.writers = {
#           cam: AsyncVideoWriter(f"videos/{cam}.mp4", fps=30)
#           for cam in self.cameras
#       }
#
#   def record_pre_step(self):
#       for cam in self.cameras:
#           rgb = self._env.scene.sensors[cam].data.output['rgb']  # (num_env, H, W, 3)
#           self.writers[cam].write(rgb)
#   
#       return "observations", self._env.obs_buf["policy"]
#
#   def close(self):
#       for w in self.writers.values():
#           w.close()

# 2. Single process Multi thread: 0.47


class PreLeRobotObservationsRecorder(RecorderTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def record_pre_step(self):
        return "observations", self._env.obs_buf["policy"]





#import concurrent.futures
#class LeRobotObservationsRecorder(RecorderTerm):
#   def __init__(self, *args, **kwargs):
#       super().__init__(*args, **kwargs)
#
#   def camera_to_hdf5(self, camera_name):
##      start = time.time()
#       camera = self._env.scene._sensors[camera_name]
#       output = convert_dict_to_backend({'rgb': camera.data.output['rgb'][0]}, backend="numpy")
##      print(f"[record_pre_step] Unit duration: {time.time() - start:.3f}s")
#       return camera_name, output
#
#   def record_pre_step(self):
#       r_dict = self._env.obs_buf["policy"]
#       cam_dict = {}
##      start_time = time.time()
#       camera = self._env.scene._sensors['front'].data.output['rgb']
##      print("just reading: ", time.time()-start_time)
#       with concurrent.futures.ThreadPoolExecutor() as executor:
#           cameras = ["front", "wrist_left", "wrist_right", "back"]
#           futures = [executor.submit(self.camera_to_hdf5, cam) for cam in cameras]
#
#           for f in concurrent.futures.as_completed(futures):
#               cam_name, rgb = f.result()
#               cam_dict[cam_name] = rgb
#   
#       r_dict["images"] = cam_dict
##print(f"[record_pre_step] Total duration: {time.time() - start_time:.3f}s")
#       return "observations", r_dict

# Just having the camer_data_list takes 0.5 sec
#import concurrent.futures
#import time
#import numpy as np
#
## Function outside the class
#def camera_to_hdf5_worker(camera_data, camera_name):
#   start_time = time.time()
#   # convert_dict_to_backend only needs the rgb data
#   output = convert_dict_to_backend({'rgb': camera_data['rgb']}, backend="numpy")
#   print(f"[camera_to_hdf5_worker] Unit duration: {time.time() - start_time:.3f}s")
#   return camera_name, output
#
#class LeRobotObservationsRecorder(RecorderTerm):
#   def __init__(self, *args, **kwargs):
#       super().__init__(*args, **kwargs)
#
#   def record_pre_step(self):
#       r_dict = self._env.obs_buf["policy"]
#       cam_dict = {}
#       start_time = time.time()
#
#       # Prepare minimal picklable data
#       cameras = ["front", "wrist_left", "wrist_right", "back"]
#       camera_data_list = [
#           (self._env.scene._sensors[cam].data.output, cam) for cam in cameras
#       ]
#       print(f"[record_pre_step] Total duration: {time.time() - start_time:.3f}s")
#       # Use ProcessPoolExecutor safely
#       with concurrent.futures.ProcessPoolExecutor() as executor:
#           futures = [
#               executor.submit(camera_to_hdf5_worker, data, name)
#               for data, name in camera_data_list
#           ]
#
#           for f in concurrent.futures.as_completed(futures):
#               cam_name, rgb = f.result()
#               cam_dict[cam_name] = rgb
#
#       r_dict["images"] = cam_dict
#       print(f"[record_pre_step] Total duration: {time.time() - start_time:.3f}s")
#       return "observations", r_dict
#
#
import copy
class LeRobotObservationsRecorder(RecorderTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def record_pre_step(self):
        r_dict = self._env.obs_buf["policy"]   # this is a reused dict/buffer

        cameras = ["front", "wrist_left", "wrist_right"]

        cam_dict = {}
        for cam in cameras:
            t = self._env.scene.sensors[cam].data.output["rgb"]
            cam_dict[cam] = t.detach().clone().cpu().numpy().copy()

        out = dict(r_dict)
        out["images"] = cam_dict

        return "observations", out

#   def record_pre_step(self):
#       """
#       Record observations from multiple cameras in parallel across bins.
#       """
#       r_dict = self._env.obs_buf["policy"]
#       cam_dict = {}
#       cameras = ["front", "wrist_left", "wrist_right"]
#       for camera in cameras:
##cam_dict[camera] = convert_dict_to_backend({'rgb': self._env.scene.sensors[camera].data.output['rgb']}, backend="torch")
#           cam_dict = convert_dict_to_backend({camera: self._env.scene.sensors[camera].data.output['rgb'].clone() for camera in cameras}, backend="numpy")
##cam_dict[camera] = {'rgb': self._env.scene.sensors[camera].data.output['rgb']}
#       r_dict["images"] = cam_dict
#
#       return "observations", r_dict
