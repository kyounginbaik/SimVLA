"""The staged RB-Y1 USD must not depend on the maintainer's absolute home path."""

import importlib.util
import sys
from pathlib import Path

import pytest


def test_rby1_reference_is_rewritten_to_relative_usd(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Sdf

    script = Path(__file__).resolve().parents[1] / "scripts/tools/relativize_rby1_usd.py"
    spec = importlib.util.spec_from_file_location("relativize_rby1_usd", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    source = tmp_path / "maintainer home" / "robot.usda"
    source.parent.mkdir()
    layer = Sdf.Layer.CreateNew(str(source))
    prim = Sdf.CreatePrimInLayer(layer, "/Robot")
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = "Xform"
    prim.referenceList.prependedItems = [Sdf.Reference(str(tmp_path / "old" / "model.usd"))]
    assert layer.Save()
    original = source.read_bytes()

    output = tmp_path / "downloaded" / "robot.usd"
    module.relativize(source, output)

    assert source.read_bytes() == original
    assert list(Sdf.Layer.FindOrOpen(str(output)).GetExternalReferences()) == ["model.usd"]
    assert b"maintainer home" not in output.read_bytes()
