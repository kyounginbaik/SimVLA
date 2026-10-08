"""Tests for the external AI Worker jaw-collider overlay builder."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

pxr = pytest.importorskip("pxr", reason="run through isaaclab.sh -p for USD schema tests")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
from patch_aiworker_gripper_collisions import (  # noqa: E402
    FINGER_LINKS,
    GEOMETRY_ROOT_PRIM,
    ROOT_PRIM,
    patch_asset,
)


def _write_usd(path: Path, root_path: str, *, with_finger_meshes: bool) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    for link_name, visual_name in FINGER_LINKS:
        link_path = f"{root_path}/{link_name}"
        UsdGeom.Xform.Define(stage, link_path)
        visual_path = f"{link_path}/visuals"
        UsdGeom.Xform.Define(stage, visual_path)
        if not with_finger_meshes:
            continue
        visual_path = f"{visual_path}/{visual_name}"
        UsdGeom.Xform.Define(stage, visual_path)
        mesh = UsdGeom.Mesh.Define(stage, f"{visual_path}/mesh")
        mesh.CreatePointsAttr([
            (-0.01, -0.01, -0.02), (0.01, -0.01, -0.02),
            (0.01, 0.01, -0.02), (-0.01, 0.01, -0.02),
            (-0.01, -0.01, 0.02), (0.01, -0.01, 0.02),
            (0.01, 0.01, 0.02), (-0.01, 0.01, 0.02),
        ])
        mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])
        mesh.CreateFaceVertexIndicesAttr([
            0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
            1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7,
        ])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def test_overlay_references_visible_jaws_and_adds_convex_collision(tmp_path):
    source = tmp_path / "configured.usd"
    geometry = tmp_path / "geometry.usd"
    output = tmp_path / "prepared.usd"
    _write_usd(source, ROOT_PRIM, with_finger_meshes=False)
    _write_usd(geometry, GEOMETRY_ROOT_PRIM, with_finger_meshes=True)
    original_source_bytes = source.read_bytes()
    original_geometry_bytes = geometry.read_bytes()

    result = patch_asset(source, geometry, output)
    assert result["already_prepared"] is False
    assert len(result["collision_meshes"]) == 4
    assert result["static_friction"] == 1.5
    assert result["dynamic_friction"] == 1.2
    stage = Usd.Stage.Open(str(output))
    assert len(_validate(stage)) == 4

    second = patch_asset(source, geometry, output)
    assert second["already_prepared"] is True
    assert source.read_bytes() == original_source_bytes
    assert geometry.read_bytes() == original_geometry_bytes


def _validate(stage):
    result = []
    for link_name, visual_name in FINGER_LINKS:
        mesh_path = f"{ROOT_PRIM}/{link_name}/visuals/{visual_name}/mesh"
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
        assert mesh and len(mesh.GetPointsAttr().Get()) == 8
        prim = mesh.GetPrim()
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        collision = UsdPhysics.MeshCollisionAPI(prim)
        assert collision.GetApproximationAttr().Get() == "convexHull"
        material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        assert physics.GetStaticFrictionAttr().Get() == pytest.approx(1.5)
        assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(1.2)
        result.append(mesh_path)
    return result


def test_overlay_refuses_to_replace_an_unrelated_file(tmp_path):
    source = tmp_path / "configured.usd"
    geometry = tmp_path / "geometry.usd"
    output = tmp_path / "prepared.usd"
    _write_usd(source, ROOT_PRIM, with_finger_meshes=False)
    _write_usd(geometry, GEOMETRY_ROOT_PRIM, with_finger_meshes=True)
    output.write_text("not a USD overlay")

    with pytest.raises(FileExistsError, match="unrelated file"):
        patch_asset(source, geometry, output)
    assert output.read_text() == "not a USD overlay"
