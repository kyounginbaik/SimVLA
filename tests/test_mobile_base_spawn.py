"""The yaw override must not alter any real arm joint or downloaded asset."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("robots", [1, 3])
def test_only_virtual_yaw_is_unbounded(monkeypatch, robots):
    path = Path(__file__).parents[1] / "source/isaaclab_assets/isaaclab_assets/robots/mobile_base.py"
    spec = importlib.util.spec_from_file_location("mobile_base_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    edits = []

    class Prim:
        def __init__(self, name):
            self.name = name
        def GetName(self):
            return self.name
        def CreateLowerLimitAttr(self, value):
            edits.append((self.name, "lo", value))
        def CreateUpperLimitAttr(self, value):
            edits.append((self.name, "hi", value))

    roots = [[Prim("arm_joint"), Prim("base_revolute_z_joint")] for _ in range(robots)]
    monkeypatch.setitem(sys.modules, "pxr", SimpleNamespace(
        Usd=SimpleNamespace(PrimRange=lambda root: root),
        UsdPhysics=SimpleNamespace(RevoluteJoint=lambda prim: prim)))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.spawners.from_files.from_files",
                        SimpleNamespace(spawn_from_usd=lambda *a, **kw: roots[0]))
    monkeypatch.setitem(sys.modules, "isaaclab.sim.utils",
                        SimpleNamespace(find_matching_prims=lambda path: roots))
    assert module.spawn_mobile_base_from_usd("/World/envs/.*/Robot", object()) is roots[0]
    assert edits == [("base_revolute_z_joint", side, value)
                     for _ in range(robots)
                     for side, value in (("lo", float("-inf")), ("hi", float("inf")))]
