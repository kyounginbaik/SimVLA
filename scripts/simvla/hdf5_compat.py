"""Compatibility helpers for IsaacLab recorder-manager variants."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _stack_leaf(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, list):
        if not value:
            return torch.empty(0)
        if all(isinstance(item, torch.Tensor) for item in value):
            return torch.stack([item.detach() for item in value])
        if all(isinstance(item, np.ndarray) for item in value):
            return torch.from_numpy(np.stack(value))
        return torch.as_tensor(value)
    return torch.as_tensor(value)


def _normalise_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_tree(child) for key, child in value.items()}
    return _stack_leaf(value)


def hdf5_episode(episode: Any) -> Any:
    """Return a shallow episode copy with stackable HDF5 leaves and replay actions.

    SimVLA's LeRobot recorder keeps frames in Python lists and stores absolute policy features in
    ``actions``. The HDF5 handler expects tensor leaves, while IsaacLab's HDF5 replay expects the
    actual action vector passed to ``env.step``. A dedicated recorder supplies that vector as
    ``raw_actions``; keep the original episode untouched for the LeRobot writer.
    """
    out = copy.copy(episode)
    data = _normalise_tree(episode.data)
    if "raw_actions" in data:
        data["actions"] = data.pop("raw_actions")
    out.data = data
    return out


def write_success_if_missing(
    recorder_manager: Any,
    episode: Any,
    *,
    succeeded: bool,
    expected_success_count: int,
) -> bool:
    """Write one successful episode when the manager has only returned it.

    Stock IsaacLab writes an episode before returning it from ``record_pre_reset``. Some SimVLA
    research checkouts return the same ``EpisodeData`` for the LeRobot writer but omit that HDF5
    write. Compare the handler's public episode count with the manager's public success count so
    the collector fills the missing record without duplicating stock behavior.

    Returns whether this call wrote the episode. Failed episodes are never demonstration data.
    """
    if not succeeded:
        return False
    handler = getattr(recorder_manager, "_dataset_file_handler", None)
    if handler is None:
        return False
    if handler.get_num_episodes() >= int(expected_success_count):
        return False
    handler.write_episode(hdf5_episode(episode))
    handler.flush()
    return True
