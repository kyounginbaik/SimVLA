"""Reusable simulator buffers must become immutable episode snapshots."""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("backend", [torch, np])
def test_recorded_joint_actions_do_not_alias_live_targets(backend):
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/utils/datasets/episode_data.py"
    spec = importlib.util.spec_from_file_location("episode_snapshot_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episode = module.EpisodeData()
    live = backend.asarray([[1., 2.], [3., 4.]])
    episode.add("actions", {"joint_pos": live[0]})
    live += 10
    episode.add("actions/joint_pos", live[0])
    live[:] = 0
    first, second = episode.data["actions"]["joint_pos"]
    assert first.tolist() == [1., 2.]
    assert second.tolist() == [11., 12.]
    if backend is torch:
        assert first.data_ptr() != second.data_ptr()
    else:
        assert not np.shares_memory(first, second)
