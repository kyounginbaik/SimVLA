"""Prepare a local FFW-SG2 USD overlay with collision on its visible distal jaw meshes.

Run with the simulator's Python so `pxr` is available:
    ./isaaclab.sh -p scripts/tools/patch_aiworker_gripper_collisions.py

The downloaded USD is never modified. The output is a sibling asset selected at runtime with
SIMVLA_AIWORKER_USD_PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


SCHEMA = "simvla.aiworker.gripper-collision-overlay.v2"
DEFAULT_STATIC_FRICTION = 1.5
DEFAULT_DYNAMIC_FRICTION = 1.2
FINGER_LINKS = (
    ("gripper_l_rh_p12_rn_r2", "r2"),
    ("gripper_l_rh_p12_rn_l2", "l2"),
    ("gripper_r_rh_p12_rn_r2", "r2"),
    ("gripper_r_rh_p12_rn_l2", "l2"),
)
ROOT_PRIM = "/ffw_sg2_follower"
GEOMETRY_ROOT_PRIM = "/Root/ffw_sg2_follower"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_for_link(stage, link_name: str, visual_name: str, Usd, UsdGeom):
    link = stage.GetPrimAtPath(f"{ROOT_PRIM}/{link_name}")
    if not link or not link.IsValid():
        raise ValueError(f"AI Worker USD is missing jaw link {link_name}")
    visuals = stage.GetPrimAtPath(f"{ROOT_PRIM}/{link_name}/visuals/{visual_name}")
    meshes = [prim for prim in Usd.PrimRange(visuals) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError(
            f"expected one visible jaw mesh below {link_name}/visuals/{visual_name}; "
            f"found {len(meshes)}"
        )
    mesh = UsdGeom.Mesh(meshes[0])
    points = mesh.GetPointsAttr().Get()
    if not points or len(points) < 4:
        raise ValueError(f"jaw mesh {meshes[0].GetPath()} has no useful surface geometry")
    return mesh


def _validate_overlay(
    stage, Usd, UsdGeom, UsdPhysics, UsdShade,
    static_friction: float, dynamic_friction: float,
) -> list[str]:
    changes = []
    metadata = stage.GetRootLayer().customLayerData
    material_path = metadata.get("simvla_gripper_material_path")
    material_prim = stage.GetPrimAtPath(material_path) if material_path else None
    if not material_prim or not material_prim.IsValid():
        raise ValueError("overlay has no authored AI Worker jaw physics material")
    material = UsdPhysics.MaterialAPI(material_prim)
    for key, attr, expected in (
        ("simvla_gripper_static_friction", material.GetStaticFrictionAttr(),
         static_friction),
        ("simvla_gripper_dynamic_friction", material.GetDynamicFrictionAttr(),
         dynamic_friction),
    ):
        value = metadata.get(key)
        if value is None or abs(float(value) - float(expected)) > 1e-6:
            raise ValueError(f"overlay material metadata {key} is invalid: {value!r}")
        authored = attr.Get()
        if authored is None or abs(float(authored) - float(value)) > 1e-6:
            raise ValueError(f"overlay jaw physics material {key} does not match its metadata")
    for link_name, visual_name in FINGER_LINKS:
        mesh = _mesh_for_link(stage, link_name, visual_name, Usd, UsdGeom)
        prim = mesh.GetPrim()
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(f"jaw mesh {prim.GetPath()} has no CollisionAPI")
        collision = UsdPhysics.MeshCollisionAPI(prim)
        if not collision:
            raise ValueError(f"jaw mesh {prim.GetPath()} has no MeshCollisionAPI")
        approximation = collision.GetApproximationAttr().Get()
        if approximation != "convexHull":
            raise ValueError(
                f"jaw mesh {prim.GetPath()} approximation is {approximation!r}, not convexHull"
            )
        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        bound_material = binding.GetMaterial() if binding else None
        if not bound_material or bound_material.GetPath() != material_prim.GetPath():
            raise ValueError(f"jaw mesh {prim.GetPath()} is not bound to the authored pad material")
        changes.append(str(prim.GetPath()))
    return changes


def patch_asset(
    source: Path, geometry_source: Path, output: Path,
    *, static_friction: float = DEFAULT_STATIC_FRICTION,
    dynamic_friction: float = DEFAULT_DYNAMIC_FRICTION,
) -> dict:
    """Copy and patch `source`; refuse to overwrite an unrelated existing `output`."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    source = source.expanduser().resolve(strict=True)
    geometry_source_asset_path = geometry_source.expanduser().absolute()
    geometry_source = geometry_source_asset_path.resolve(strict=True)
    output = output.expanduser().absolute()
    if source == output.resolve(strict=False):
        raise ValueError("output must differ from the downloaded/source USD")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output.parent}")
    source_hash = sha256(source)
    geometry_hash = sha256(geometry_source)
    if (not math.isfinite(static_friction) or not math.isfinite(dynamic_friction)
            or not 0.0 < dynamic_friction <= static_friction <= 10.0):
        raise ValueError("friction must satisfy 0 < dynamic <= static <= 10")

    if output.exists():
        try:
            stage = Usd.Stage.Open(str(output))
        except Exception as exc:  # noqa: BLE001 - USD parse failures are unrelated-file collisions.
            raise FileExistsError(f"refusing to overwrite unrelated file: {output}") from exc
        if stage is None:
            raise ValueError(f"could not open existing output USD: {output}")
        metadata = stage.GetRootLayer().customLayerData
        if metadata.get("simvla_patch_schema") != SCHEMA:
            raise FileExistsError(f"refusing to overwrite unrelated file: {output}")
        if (metadata.get("simvla_source_sha256") != source_hash
                or metadata.get("simvla_geometry_sha256") != geometry_hash
                or float(metadata.get("simvla_gripper_static_friction", -1)) != static_friction
                or float(metadata.get("simvla_gripper_dynamic_friction", -1)) != dynamic_friction):
            raise FileExistsError(
                f"existing overlay was made from different source/geometry hashes: {output}"
            )
        meshes = _validate_overlay(
            stage, Usd, UsdGeom, UsdPhysics, UsdShade, static_friction, dynamic_friction)
        return {"output": str(output), "source_sha256": source_hash,
                "geometry_sha256": geometry_hash, "output_sha256": sha256(output),
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
                "collision_meshes": meshes,
                "already_prepared": True}

    shutil.copyfile(source, output)
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        output.unlink(missing_ok=True)
        raise ValueError(f"could not open copied USD: {output}")

    try:
        material_path = f"{ROOT_PRIM}/simvla_gripper_pad_material"
        pad_material = UsdShade.Material.Define(stage, material_path)
        pad_physics = UsdPhysics.MaterialAPI.Apply(pad_material.GetPrim())
        pad_physics.CreateStaticFrictionAttr().Set(static_friction)
        pad_physics.CreateDynamicFrictionAttr().Set(dynamic_friction)
        pad_physics.CreateRestitutionAttr().Set(0.0)
        for link_name, visual_name in FINGER_LINKS:
            visuals_root = stage.GetPrimAtPath(f"{ROOT_PRIM}/{link_name}/visuals")
            if not visuals_root or not visuals_root.IsValid():
                raise ValueError(f"jaw link {link_name} has no visuals scope")
            # The configured USD authors these empty scopes as instanceable, which prevents the
            # local overlay from adding the matching mesh reference beneath them.
            visuals_root.SetInstanceable(False)
            visual_path = f"{ROOT_PRIM}/{link_name}/visuals/{visual_name}"
            visual_prim = stage.GetPrimAtPath(visual_path)
            if not visual_prim or not visual_prim.IsValid():
                visual_prim = UsdGeom.Xform.Define(stage, visual_path).GetPrim()
            # API schemas must be authored on the composed mesh prims, not instance proxies.
            visual_prim.SetInstanceable(False)
            if not any(child.IsA(UsdGeom.Mesh) for child in Usd.PrimRange(visual_prim)):
                source_visual = (
                    f"{GEOMETRY_ROOT_PRIM}/{link_name}/visuals/{visual_name}"
                )
                reference = os.path.relpath(geometry_source_asset_path, output.parent)
                visual_prim.GetReferences().AddReference(reference, source_visual)

            mesh = _mesh_for_link(stage, link_name, visual_name, Usd, UsdGeom)
            prim = mesh.GetPrim()
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(
                    f"jaw mesh already has collision geometry; inspect before patching: "
                    f"{prim.GetPath()}"
                )
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("convexHull")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(pad_material)

        stage.GetRootLayer().customLayerData = {
            **stage.GetRootLayer().customLayerData,
            "simvla_patch_schema": SCHEMA,
            "simvla_source_sha256": source_hash,
            "simvla_geometry_sha256": geometry_hash,
            "simvla_gripper_material_path": material_path,
            "simvla_gripper_static_friction": static_friction,
            "simvla_gripper_dynamic_friction": dynamic_friction,
        }
        stage.GetRootLayer().Save()
        meshes = _validate_overlay(
            stage, Usd, UsdGeom, UsdPhysics, UsdShade, static_friction, dynamic_friction)
    except Exception:
        output.unlink(missing_ok=True)
        raise

    return {"output": str(output), "source_sha256": source_hash,
            "geometry_sha256": geometry_hash, "output_sha256": sha256(output),
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "collision_meshes": meshes,
            "already_prepared": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    assets_dir = Path(os.environ.get("SIMVLA_ASSETS_DIR", "external/simvla-assets/assets"))
    parser.add_argument(
        "--source", type=Path,
        default=assets_dir / "Robots/MM/aiworker/ffw_sg2.usd",
        help="downloaded FFW-SG2 USD to copy (default: SIMVLA_ASSETS_DIR/Robots/MM/aiworker/ffw_sg2.usd)",
    )
    parser.add_argument(
        "--geometry-source", type=Path,
        default=assets_dir / "Robots/FFW_SG2.usd",
        help="public USD containing the matching SG2 jaw visuals (default: SIMVLA_ASSETS_DIR/Robots/FFW_SG2.usd)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="new overlay path (default: sibling ffw_sg2_simvla_grip.usd)",
    )
    parser.add_argument("--static-friction", type=float, default=DEFAULT_STATIC_FRICTION)
    parser.add_argument("--dynamic-friction", type=float, default=DEFAULT_DYNAMIC_FRICTION)
    args = parser.parse_args(argv)
    source = args.source.expanduser().absolute()
    if (not 0.0 < args.dynamic_friction <= args.static_friction <= 10.0):
        parser.error("friction must satisfy 0 < dynamic <= static <= 10")
    output = args.output or source.with_name("ffw_sg2_simvla_grip.usd")
    result = patch_asset(
        source, args.geometry_source, output,
        static_friction=args.static_friction, dynamic_friction=args.dynamic_friction)
    print(json.dumps(result, indent=2))
    print("Select it for AI Worker collection with:")
    print(f"export SIMVLA_AIWORKER_USD_PATH={result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
