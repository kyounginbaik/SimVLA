"""Preserve workflow results across Isaac Sim's process shutdown."""
import json
import os
from pathlib import Path


def record_status(exit_code: int) -> None:
    """Write the launcher-owned result file before Kit can terminate the process."""
    path = os.environ.get("SIMVLA_STATUS_FILE")
    if path:
        Path(path).write_text(json.dumps({"exit_code": int(exit_code)}))


def resolve_exit_code(process_exit: int, status_path: str | Path) -> int:
    """Kit may return zero after failure; require its pre-shutdown status too."""
    if process_exit:
        return process_exit
    try:
        code = json.loads(Path(status_path).read_text())["exit_code"]
    except (OSError, ValueError, KeyError, TypeError):
        return 1
    return code if type(code) is int and 0 <= code <= 255 else 1


def close_with_status(exit_code, clear_context, close_app):
    """A full filesystem must not prevent simulator cleanup after a failed run."""
    try:
        record_status(exit_code)
    except OSError as exc:
        print(f"[status] could not preserve workflow result: {exc}", flush=True)
        exit_code = exit_code or 1
    finally:
        try:
            clear_context()
        finally:
            close_app()
    return exit_code
