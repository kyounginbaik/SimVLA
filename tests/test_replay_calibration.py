"""Every public collection robot must decode the shared simulated action frame."""
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("robot", ["anubis", "rby1", "aiworker"])
def test_simulated_dataset_frame_round_trip(robot):
    calibration = json.loads((ROOT / "scripts/simvla/calibration" / f"{robot}.json").read_text())
    # Shared producer contract, regardless of the physical wrist's link geometry.
    offset = np.array([.095, 0., -.823356])
    local = np.array([0., 0., -.10956])
    position = np.array([.46, .12, 1.14])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    encoded = position + rotation @ local + offset
    decoded = (encoded + np.asarray(calibration["real_to_sim_frame_offset_xyz"])
               + rotation @ np.asarray(calibration["eef_to_grasp_offset_local_xyz"]))
    np.testing.assert_allclose(decoded, position, atol=1e-12)


def test_frame_contract_matches_shared_producer():
    source = (ROOT / "source/isaaclab/isaaclab/managers/action_manager.py").read_text()
    assert "torch.tensor([0.0, 0.0, -0.10956]" in source
    assert "pos[:,0] += 0.095" in source
    assert "pos[:,2] += -0.823356" in source
