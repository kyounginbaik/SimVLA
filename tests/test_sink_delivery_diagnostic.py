import copy
import importlib.util
import json
from pathlib import Path
import pytest


def test_sink_diagnostic_preserves_grasp_and_does_not_weaken_success():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "sink_diagnostic", root / "scripts/tools/make_rby1_sink_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    before = copy.deepcopy(source)
    result = module.make_goal(source)
    assert source == before
    assert result["goals"][0][:4] == before["goals"][0][:4]
    assert result["goals"][0][6:] == before["goals"][0][6:]
    assert result["goals"][0][4]["goal"][1] == -2.04
    assert result["goals"][0][5]["params"]["down_m"] == 0
    assert result["run_config"] == before["run_config"]


@pytest.mark.parametrize("advance,reach", [(-.01,.3),(.16,.3),(0,.14),(0,.41),
                                         (float("nan"),.3),(0,float("inf"))])
def test_sink_diagnostic_rejects_unsafe_offsets(advance, reach):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "sink_diagnostic", root / "scripts/tools/make_rby1_sink_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="parking advance"):
        module.make_goal({}, parking_advance_m=advance, forward_m=reach)


def test_short_reach_moves_parking_without_modifying_source():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "sink_diagnostic", root / "scripts/tools/make_rby1_sink_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    original = copy.deepcopy(source)
    result = module.make_goal(source, parking_advance_m=.1, forward_m=.3)
    assert source == original
    assert result["goals"][0][4]["goal"][0] == pytest.approx(source["goals"][0][4]["goal"][0] + .1)
    assert result["goals"][0][5]["params"]["forward_m"] == .3


def test_absolute_sink_target_preserves_other_steps_and_predicates():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "sink_diagnostic", root / "scripts/tools/make_rby1_sink_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    relative = module.make_goal(source, parking_advance_m=.1, forward_m=.3)
    absolute = module.make_goal(source, parking_advance_m=.1, forward_m=.3,
                                place_position=[1.67, -2.26, 1.12])
    expected = copy.deepcopy(relative)
    expected["goals"][0][5].update(skill="arm.place",
                                  params={"prim_path": "/world/sink_cabinet"},
                                  goal=[1.67, -2.26, 1.12, 1., 0., 0., 0.])
    assert absolute == expected
    for value in ([1., 2.], [1., float("nan"), 3.]):
        with pytest.raises(ValueError, match="three finite"):
            module.make_goal(source, place_position=value)


def test_final_straight_approach_preserves_turn_grasp_and_release_order():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "sink_diagnostic", root / "scripts/tools/make_rby1_sink_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads((root / "examples/goals/Isaac-Kitchen-v813r-00.json").read_text())
    original = copy.deepcopy(source)
    old = module.make_goal(source, parking_advance_m=.15, forward_m=.3,
                           place_position=[1.88, -2.20, 1.08])
    new = module.make_goal(source, parking_advance_m=.15, forward_m=.3,
                           place_position=[1.88, -2.20, 1.08], final_approach_m=.05)
    assert source == original
    a, b = old["goals"][0], new["goals"][0]
    assert len(b) == 9 and b[:5] == a[:5] and b[6:] == a[5:]
    assert b[5]["goal"][0] - b[4]["goal"][0] == pytest.approx(.05)
    assert b[5]["goal"][1:] == b[4]["goal"][1:]
    assert b[5]["skill"] == "nav.to_prim"
    assert new["run_config"]["export_groups"] == "3;6,7,8"
    assert new["diagnostic_final_approach"]["accepted_episode"] is False
    for value in (-.01, .081, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="final approach"):
            module.make_goal(source, final_approach_m=value)
    with pytest.raises(ValueError, match="absolute placement"):
        module.make_goal(source, final_approach_m=.05)
