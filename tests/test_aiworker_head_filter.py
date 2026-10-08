"""The head fix must not remove other collision pairs or mutate asset files."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("count", [1, 3])
def test_only_head_and_torso_are_filtered(monkeypatch, count):
    path = Path(__file__).parents[1] / "source/isaaclab_assets/isaaclab_assets/robots/aiworker_collisions.py"
    spec = importlib.util.spec_from_file_location("head_filter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    edits = []

    class Prim:
        def __init__(self, path): self.path = path
        def GetChild(self, name): return Prim(f"{self.path}/{name}")
        def GetPath(self): return self.path
        def IsValid(self): return True
        def CreateFilteredPairsRel(self): return self
        def AddTarget(self, target): edits.append((self.path, target))

    roots = [Prim(f"/World/envs/env_{i}/Robot") for i in range(count)]
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        UsdPhysics=SimpleNamespace(FilteredPairsAPI=SimpleNamespace(Apply=lambda prim: prim))))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.spawners.from_files.from_files",
                        SimpleNamespace(spawn_from_usd=lambda *a, **kw: roots[0]))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils",
                        SimpleNamespace(find_matching_prims=lambda path: roots))
    assert module.spawn_aiworker_from_usd("/World/envs/.*/Robot", object()) is roots[0]
    assert edits == [(f"{root.path}/head_link2", f"{root.path}/arm_base_link") for root in roots]
