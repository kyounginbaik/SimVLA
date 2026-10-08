from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/simvla"))
from sink_collision import configure_sink_collisions, repair_sink_collisions


def legacy_spec():
    return {"all": [{"obj_near_prim": {"role": "mug0", "target_role": "sink_cabinet",
                                      "radius": .12, "anchor": "door_midpoint", "z_override": .88}},
                    {"eef_home": {"arm": "both", "radius": .05}}, {"last_subtask": {}}]}


def test_interior_gate_preserves_home_and_script_checks():
    from sink_collision import kitchen813_interior_spec
    original = legacy_spec()
    fixed = kitchen813_interior_spec(original)
    assert original == legacy_spec()
    assert fixed["all"][:2] == original["all"][1:]
    assert kitchen813_interior_spec(fixed) == fixed
    with pytest.raises(ValueError):
        kitchen813_interior_spec({"all": [{"eef_home": {"radius": .05}}]})


def test_interior_gate_rejects_countertop_floor_and_outside_region():
    torch = pytest.importorskip("torch")
    from sink_collision import kitchen813_interior_spec
    from predicates_math import EvalContext, compile_spec
    positions = torch.tensor([[1.795, -2.262, .747], [1.795, -2.262, .9011],
                              [1.795, -2.262, .01], [1.6, -2.262, .747]])
    ctx = EvalContext(num_envs=4, obj_pos_w={"mug0": positions},
                      art_body_pos_w={"sink_cabinet": torch.tensor([[[1.8686, -2.2621, 0.]]]).expand(4, 1, 3)},
                      eef_pos_base={"left": torch.zeros(4, 3), "right": torch.zeros(4, 3)},
                      home={"left": torch.zeros(3), "right": torch.zeros(3)},
                      goal_index=torch.full((4,), 7), max_steps=8)
    assert compile_spec(kitchen813_interior_spec(legacy_spec()))(ctx).tolist() == [True, False, False, False]


def test_cavity_goal_requires_both_recorded_settings(monkeypatch):
    from sink_collision import validate_sink_goal_environment
    metadata = {"diagnostic_sink_cavity": {}}
    for name in ("SIMVLA_SINK_CAVITY_COLLISIONS", "SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError):
        validate_sink_goal_environment(metadata)
    monkeypatch.setenv("SIMVLA_SINK_CAVITY_COLLISIONS", "1")
    with pytest.raises(ValueError):
        validate_sink_goal_environment(metadata)
    monkeypatch.setenv("SIMVLA_KITCHEN813_SINK_INTERIOR_GATE", "1")
    validate_sink_goal_environment(metadata)


def test_interior_configuration_requires_exact_scene_and_collision_repair():
    cfg = SimpleNamespace(scene=SimpleNamespace(kitchen=SimpleNamespace(
        spawn=SimpleNamespace(func=lambda: None, usd_path="file:/assets/Kitchen/kitchen_813_00.usd"))),
        terminations=SimpleNamespace(success=SimpleNamespace(params={"spec": legacy_spec()})))
    with pytest.raises(ValueError, match="repaired kitchen"):
        configure_sink_collisions(cfg, "0", "1")
    cfg.scene.kitchen.spawn.usd_path = "file:/assets/Kitchen/kitchen_813_01.usd"
    with pytest.raises(ValueError, match="repaired kitchen"):
        configure_sink_collisions(cfg, "1", "1")
    cfg.scene.kitchen.spawn.usd_path = "file:/assets/Kitchen/kitchen_813_00.usd"
    configure_sink_collisions(cfg, "1", "1")
    assert cfg.terminations.success.params["spec"]["all"][-1] == {"obj_z": {"role": "mug0", "lo": .73, "hi": .83}}


def test_cavity_goal_generator_preserves_grasp_and_discloses_gate(tmp_path):
    import importlib.util
    import json
    import copy
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("cavity_diagnostic", root / "scripts/tools/make_sink_cavity_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    before = copy.deepcopy(source)
    result = module.make_goal(source, [1.79, -2.262, 1.08])
    assert source == before
    assert result["goals"][0][:5] == before["goals"][0][:5]
    assert result["goals"][0][6:] == before["goals"][0][6:]
    assert result["goals"][0][5]["skill"] == "arm.place"
    assert result["diagnostic_sink_cavity"]["accepted_episode"] is False
    assert result["diagnostic_sink_cavity"]["requires_environment"]["SIMVLA_KITCHEN813_SINK_INTERIOR_GATE"] == "1"
    import goal_format
    path = tmp_path / "goal.json"
    path.write_text(json.dumps(result))
    _, metadata = goal_format.read(path)
    assert metadata["diagnostic_sink_cavity"] == result["diagnostic_sink_cavity"]
    source["kitchen_sub_num"] = 1
    with pytest.raises(ValueError, match="813 rotation 00"):
        module.make_goal(source, [1.79, -2.262, 1.08])


def test_sink_repair_is_explicit_and_idempotently_configured():
    original = lambda: None
    cfg = SimpleNamespace(scene=SimpleNamespace(kitchen=SimpleNamespace(
        spawn=SimpleNamespace(func=original))))
    configure_sink_collisions(cfg, "0")
    assert cfg.scene.kitchen.spawn.func is original
    with pytest.raises(ValueError):
        configure_sink_collisions(cfg, "yes")
    configure_sink_collisions(cfg, "1")
    wrapped = cfg.scene.kitchen.spawn.func
    assert wrapped is not original
    configure_sink_collisions(cfg, "1")
    assert cfg.scene.kitchen.spawn.func is wrapped


def test_sink_repair_preserves_shape_collisions_and_body_apis():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Kitchen").GetPrim()
    cabinet = UsdGeom.Xform.Define(stage, "/Kitchen/sink_cabinet").GetPrim()
    corpus = UsdGeom.Xform.Define(stage, "/Kitchen/sink_cabinet/corpus").GetPrim()
    for prim in (cabinet, corpus):
        UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.ArticulationRootAPI.Apply(cabinet)
    UsdPhysics.RigidBodyAPI.Apply(corpus)
    for name in ("sink", "sink_countertop"):
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Mesh.Define(stage, str(corpus.GetPath()) + "/" + name).GetPrim())
    cube = UsdGeom.Cube.Define(stage, str(corpus.GetPath()) + "/back").GetPrim()
    UsdPhysics.CollisionAPI.Apply(cube)
    other = UsdGeom.Xform.Define(stage, "/Kitchen/not_sink_cabinet").GetPrim()
    UsdPhysics.CollisionAPI.Apply(other)
    report = repair_sink_collisions(stage, root.GetPath())
    assert len(report["decomposed_meshes"]) == 2
    assert len(report["removed_xform_collision_apis"]) == 2
    assert cabinet.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert corpus.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not corpus.HasAPI(UsdPhysics.CollisionAPI)
    assert cube.HasAPI(UsdPhysics.CollisionAPI)
    assert other.HasAPI(UsdPhysics.CollisionAPI)
    for path in report["decomposed_meshes"]:
        mesh = stage.GetPrimAtPath(path)
        assert mesh.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexDecomposition"
    assert repair_sink_collisions(stage, root.GetPath())["removed_xform_collision_apis"] == []


def test_sink_repair_refuses_unrecognized_geometry_without_mutation():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    cabinet = UsdGeom.Xform.Define(stage, "/Kitchen/sink_cabinet").GetPrim()
    UsdGeom.Xform.Define(stage, "/Kitchen/sink_cabinet/corpus")
    UsdPhysics.CollisionAPI.Apply(cabinet)
    with pytest.raises(ValueError, match="collidable sink mesh"):
        repair_sink_collisions(stage, "/Kitchen")
    assert cabinet.HasAPI(UsdPhysics.CollisionAPI)


@pytest.mark.parametrize("script", ["simvla_gen.py", "simvla_replay.py", "smoke_test.py"])
def test_same_sink_configuration_hook_in_all_runtime_entrypoints(script):
    source = (Path(__file__).parents[1] / "scripts/simvla" / script).read_text()
    assert source.index("configure_sink_collisions(") < source.index("gym.make(")
