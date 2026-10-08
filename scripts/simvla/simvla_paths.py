"""SimVLA: where things live on disk.

Standard library only — no isaaclab, omni, pxr, scene_synthesizer or torch. Every other
module in this pipeline boots Omniverse at import time, which makes them impossible to unit
test; keeping path resolution here means it can be tested in milliseconds.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Goal JSONs are written and read here. It is an env var because the directory can be huge
#: (the author's is 356 MB / 15,810 files) and does not belong inside a git checkout. Matches
#: the convention already used by SIMVLA_OUTPUT_DIR, SIMVLA_HOUSE_DIR, SIMVLA_ROOM_DIR,
#: SIMVLA_DEPLOY_ENV_DIR, SIMVLA_EXTRA_ASSETS_DIR and SIMVLA_LEROBOT_ROOT.
GOALS_DIR_ENV = "SIMVLA_GOALS_DIR"
REPO_ROOT_ENV = "SIMVLA_REPO_ROOT"


def assets_dir() -> Path:
    """External asset tree, or the legacy checkout's asset directory."""
    value = os.environ.get("SIMVLA_ASSETS_DIR")
    return Path(value).expanduser().resolve() if value else repo_root() / "source/isaaclab_assets/data"


def robot_asset_path(value: str) -> Path:
    """Resolve cuRobo's legacy paths without requiring assets inside the checkout."""
    path = Path(value).expanduser()
    if path.name == "ffw_sg2_follower.urdf" and os.environ.get("SIMVLA_AIWORKER_URDF_PATH"):
        return Path(os.environ["SIMVLA_AIWORKER_URDF_PATH"]).expanduser().resolve()
    if path.is_absolute():
        return path
    prefix = Path("source/isaaclab_assets/data")
    if path.is_relative_to(prefix):
        return assets_dir() / path.relative_to(prefix)
    root = os.environ.get("SIMVLA_ROBOT_MODELS_DIR")
    return (Path(root).expanduser().resolve() if root else repo_root()) / path


def validate_robot_asset(robot: str, usd_path: str | Path) -> None:
    """Reject a smoke/collector robot flag that disagrees with the task's articulation USD."""
    expected = {
        "anubis": "anubis_simvla.usd",
        "aiworker": "ffw_sg2.usd",
        "rby1": "model_simvla_black_gripper.usd",
    }
    try:
        marker = expected[robot]
    except KeyError:
        raise ValueError(
            f"Unsupported robot {robot!r}; expected one of {', '.join(expected)}"
        ) from None
    if robot == "aiworker" and os.environ.get("SIMVLA_AIWORKER_USD_PATH"):
        override = Path(os.environ["SIMVLA_AIWORKER_USD_PATH"]).expanduser().resolve()
        configured_path = Path(usd_path).expanduser().resolve()
        if configured_path == override:
            return
    configured = Path(usd_path).name
    if configured != marker:
        raise ValueError(
            f"Robot/task mismatch: --robot {robot!r} expects an articulation USD named "
            f"{marker!r}, but this task config uses {configured!r}. Adapt the kitchen task "
            f"for {robot!r} before running this workflow."
        )


def validate_robot_model_file(robot: str) -> Path:
    """Fail before Kit starts when a selected robot's external USD is missing."""
    if robot == "rby1":
        root = os.environ.get("SIMVLA_RBY1M_DIR")
        if not root:
            raise FileNotFoundError(
                "RB-Y1 requires SIMVLA_RBY1M_DIR pointing to the extracted rby1m/ directory"
            )
        path = Path(root).expanduser() / "models/rby1m/urdf/model/model_simvla_black_gripper.usd"
    elif robot == "anubis":
        path = assets_dir() / "Robots/anubis_simvla.usd"
    elif robot == "aiworker":
        path = assets_dir() / "Robots/MM/aiworker/ffw_sg2.usd"
    else:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of anubis, aiworker, rby1")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"{robot} articulation USD is missing: {path}. Check SIMVLA_ASSETS_DIR and "
            "SIMVLA_RBY1M_DIR against the extracted asset bundle."
        )
    return path


def repo_root() -> Path:
    """The checkout root: $SIMVLA_REPO_ROOT, else the nearest ancestor with a pyproject.toml."""
    env = os.environ.get(REPO_ROOT_ENV)
    if env:
        return Path(os.path.expanduser(env))

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    raise RuntimeError(
        f"Cannot find the repo root. Set {REPO_ROOT_ENV}, or run from inside the checkout."
    )


def goals_dir() -> Path:
    """Resolve an explicit goal directory or the bundled examples.

    Older research checkouts without examples retain scripts/simvla/goals as fallback.
    """
    env = os.environ.get(GOALS_DIR_ENV)
    bundled = repo_root() / "examples/goals"
    default = bundled if bundled.is_dir() else repo_root() / "scripts/simvla/goals"
    path = Path(os.path.expanduser(env)) if env else default
    path.mkdir(parents=True, exist_ok=True)
    return path


def lerobot_root() -> Path:
    """LeRobot export root, defaulting to a stable checkout-local output directory.

    Collection must still finalize successfully when users have not sourced ``.env``. Keeping
    this default separate from the raw run's ``--output_root`` avoids mixing staged episodes
    with the consolidated dataset while preserving an explicit external destination.
    """
    value = os.environ.get("SIMVLA_LEROBOT_ROOT")
    return Path(value).expanduser().resolve() if value else repo_root() / "outputs/lerobot"


def goal_files(stem: str) -> tuple[Path, Path]:
    """The (<stem>.json, <stem>.reloadable.json) pair for a goal.

    `stem`, not `task`: simvla_deploy.py names its goals '<task>-<task_type>', so this cannot
    assume the caller is handing over a bare task id.
    """
    directory = goals_dir()
    return directory / f"{stem}.json", directory / f"{stem}.reloadable.json"
