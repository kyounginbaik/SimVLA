"""Validated runtime settings for cuRobo's search budget."""

from __future__ import annotations

import os


def bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read an integer environment override and fail early on malformed/out-of-range values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer (got {raw!r})") from exc
    if not minimum <= value <= maximum:
        raise SystemExit(f"{name} must be between {minimum} and {maximum} (got {value})")
    return value


def bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    """Read a finite float environment override within the inclusive bounds."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number (got {raw!r})") from exc
    if not minimum <= value <= maximum:
        raise SystemExit(f"{name} must be between {minimum} and {maximum} (got {value})")
    return value


def curobo_search_budget() -> tuple[int, int, int, bool]:
    """Return attempts, trajectory seeds, graph seeds, and finetune toggle."""
    finetune_raw = os.environ.get("SIMVLA_FINETUNE_TRAJOPT", "1").strip().lower()
    if finetune_raw not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise SystemExit(
            "SIMVLA_FINETUNE_TRAJOPT must be a boolean (0/1, true/false, yes/no, on/off)"
        )
    finetune = finetune_raw in {"1", "true", "yes", "on"}
    return (
        bounded_int_env("SIMVLA_PLANNER_ATTEMPTS", 10, 1, 20),
        bounded_int_env("SIMVLA_TRAJOPT_SEEDS", 12, 1, 32),
        bounded_int_env("SIMVLA_GRAPH_SEEDS", 12, 1, 32),
        finetune,
    )
