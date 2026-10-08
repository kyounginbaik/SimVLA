from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/simvla"))
from collector_profile import configure_physics_substeps


@pytest.mark.parametrize("initial", [1, 6])
def test_finer_integration_preserves_twenty_fps_and_is_idempotent(monkeypatch, initial):
    cfg = SimpleNamespace(decimation=initial, sim=SimpleNamespace(dt=.05/initial, render_interval=initial))
    monkeypatch.setenv("SIMVLA_PHYSICS_SUBSTEPS", "6")
    configure_physics_substeps(cfg)
    configure_physics_substeps(cfg)
    assert cfg.sim.dt == pytest.approx(1/120)
    assert cfg.sim.dt * cfg.decimation == pytest.approx(.05)
    assert cfg.sim.render_interval == 6


@pytest.mark.parametrize("invalid", ["0", "25", "1.5", "nan", "-1"])
def test_invalid_integration_override_is_rejected(monkeypatch, invalid):
    monkeypatch.setenv("SIMVLA_PHYSICS_SUBSTEPS", invalid)
    with pytest.raises(ValueError, match="PHYSICS_SUBSTEPS"):
        configure_physics_substeps(SimpleNamespace())
