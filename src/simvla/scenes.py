"""Generate procedural kitchens on CPU; export geometry for inspection."""
from pathlib import Path

from .placements import KITCHEN_TYPES


def generate_kitchen(layout: str = "single_wall", *, seed: int = 0):
    """Return a scene-synthesizer Scene with SimVLA's kitchen fixtures.

    This creates furniture geometry, not an Isaac Sim physics environment.
    Object placement with external meshes is available through kitchen_build.
    """
    if layout not in KITCHEN_TYPES:
        raise ValueError(f"Unknown layout {layout!r}; choose from {KITCHEN_TYPES}")
    try:
        from .kitchen_build import build_kitchen
    except ModuleNotFoundError as exc:
        raise RuntimeError("Scene generation requires: pip install 'simvla[scenes]' "
                           "(from a checkout: pip install '.[scenes]')") from exc
    scene, _, _, _ = build_kitchen(layout, [], [], seed=seed)
    return scene


def export_kitchen(output: str | Path, *, layout: str = "single_wall", seed: int = 0) -> Path:
    """Write a GLB containing generated kitchen geometry. Refuse to overwrite."""
    output = Path(output)
    if output.suffix.lower() != ".glb":
        raise ValueError("Use a .glb output path")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    scene = generate_kitchen(layout, seed=seed)
    data = scene.scene.export(file_type="glb")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
    return output

