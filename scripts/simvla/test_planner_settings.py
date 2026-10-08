"""Tests for bounded cuRobo search-budget overrides."""

import pytest

from planner_settings import bounded_float_env, bounded_int_env, curobo_search_budget


def test_search_budget_defaults_preserve_tuned_values(monkeypatch):
    for name in ("SIMVLA_PLANNER_ATTEMPTS", "SIMVLA_TRAJOPT_SEEDS", "SIMVLA_GRAPH_SEEDS"):
        monkeypatch.delenv(name, raising=False)
    assert curobo_search_budget() == (10, 12, 12, True)


def test_search_budget_accepts_smoke_overrides(monkeypatch):
    monkeypatch.setenv("SIMVLA_PLANNER_ATTEMPTS", "2")
    monkeypatch.setenv("SIMVLA_TRAJOPT_SEEDS", "2")
    monkeypatch.setenv("SIMVLA_GRAPH_SEEDS", "1")
    monkeypatch.setenv("SIMVLA_FINETUNE_TRAJOPT", "false")
    assert curobo_search_budget() == (2, 2, 1, False)


@pytest.mark.parametrize("raw", ["0", "21", "bad"])
def test_search_budget_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("SIMVLA_PLANNER_ATTEMPTS", raw)
    with pytest.raises(SystemExit, match="SIMVLA_PLANNER_ATTEMPTS"):
        curobo_search_budget()


def test_bounded_int_env_accepts_boundaries(monkeypatch):
    monkeypatch.setenv("BUDGET", "1")
    assert bounded_int_env("BUDGET", 3, 1, 5) == 1
    monkeypatch.setenv("BUDGET", "5")
    assert bounded_int_env("BUDGET", 3, 1, 5) == 5


def test_reset_rotation_tolerance_can_be_tuned_without_changing_default(monkeypatch):
    monkeypatch.delenv("SIMVLA_RESET_ROT_TOL_DEG", raising=False)
    assert bounded_float_env("SIMVLA_RESET_ROT_TOL_DEG", 10.0, 0.0, 180.0) == 10.0
    monkeypatch.setenv("SIMVLA_RESET_ROT_TOL_DEG", "25")
    assert bounded_float_env("SIMVLA_RESET_ROT_TOL_DEG", 10.0, 0.0, 180.0) == 25.0


@pytest.mark.parametrize("raw", ["nan", "181", "not-a-number"])
def test_reset_rotation_tolerance_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv("SIMVLA_RESET_ROT_TOL_DEG", raw)
    with pytest.raises(SystemExit, match="SIMVLA_RESET_ROT_TOL_DEG"):
        bounded_float_env("SIMVLA_RESET_ROT_TOL_DEG", 10.0, 0.0, 180.0)


def test_search_budget_rejects_invalid_finetune_value(monkeypatch):
    monkeypatch.setenv("SIMVLA_FINETUNE_TRAJOPT", "sometimes")
    with pytest.raises(SystemExit, match="SIMVLA_FINETUNE_TRAJOPT"):
        curobo_search_budget()
