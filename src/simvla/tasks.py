"""Load and validate task templates without starting a simulator."""
from importlib.resources import files
from pathlib import Path

from . import skills as _skills  # Registers schemas without importing torch or Isaac Sim.
from .task_template import TaskTemplate, from_json, template_warnings, validate_template
from .task_validate import validate_sequence


def list_templates() -> list[str]:
    return sorted(p.name[:-5] for p in files("simvla").joinpath("templates").iterdir()
                  if p.name.endswith(".json"))


def load_template(name_or_path: str | Path) -> TaskTemplate:
    """Read a bundled template name or a JSON path, then validate it."""
    name = str(name_or_path)
    if name in list_templates():
        text = files("simvla").joinpath("templates", name + ".json").read_text()
    else:
        text = Path(name).read_text()
    try:
        task = from_json(text)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed task template {name!r}: {exc}") from exc
    validate_template(task)
    validate_sequence(task)
    return task


__all__ = ["TaskTemplate", "list_templates", "load_template", "template_warnings"]
