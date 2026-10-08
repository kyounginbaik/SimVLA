"""USD-only checks; no Kit/GPU needed when usd-core is installed."""
import importlib.util
from pathlib import Path

import pytest


def test_sparse_omnipbr_input_is_added_without_rebinding_or_overwriting():
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    path = Path(__file__).parents[1] / "source/isaaclab/isaaclab/simvla/materials.py"
    spec = importlib.util.spec_from_file_location("material_color_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mat = UsdShade.Material.Define(stage, "/material")
    shader = UsdShade.Shader.Define(stage, "/material/Shader")
    shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier",
                                    Sdf.ValueTypeNames.Token).Set("OmniPBR")
    color = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
    color.Set((.1, .2, .3))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    assert module.prepare_omnipbr_color_inputs([mesh.GetPrim()] * 2) == ["/material/Shader"]
    assert shader.GetInput("project_uvw").Get() is False
    assert tuple(color.Get()) == pytest.approx((.1, .2, .3))
    shader.GetInput("project_uvw").Set(True)
    assert module.prepare_omnipbr_color_inputs([mesh.GetPrim()]) == []
    assert shader.GetInput("project_uvw").Get() is True
    assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath() == mat.GetPath()
    other = UsdShade.Shader.Define(stage, "/material/Other")
    other.CreateIdAttr("UsdPreviewSurface")
    module.prepare_omnipbr_color_inputs([mesh.GetPrim()])
    assert not other.GetInput("project_uvw")
