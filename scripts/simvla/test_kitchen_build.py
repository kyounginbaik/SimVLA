"""Tests for the pure kitchen build layer.

Run with: pytest scripts/simvla/test_kitchen_build.py -v
Needs scene_synthesizer but NOT Omniverse and NOT the BODex assets.
"""

import pytest
import trimesh

from kitchen_build import (
    GEOMETRY2MATERIAL,
    OBJECT_TYPES,
    collect_joints,
    LOCATION_LABELS,
    MATERIALS,
    build_kitchen,
    material_display_names,
    object_scale,
    pick_materials,
    resolve_mesh,
)


def _bodex_mesh(tmp_path, obj_dir, extents):
    """A stand-in for a BODex mesh, laid out the way the real dataset is.

    The directory name is what matters: matching_meshes reads it via path.split('/')[-3] and tests
    it against the type's CATEGORIES prefixes. A flat 'bowl_simplified.obj' passed the old rule --
    a bare substring search over the whole path -- and would silently match nothing now, which is
    the right answer for a file that looks nothing like the data.
    """
    path = tmp_path / "use_data" / obj_dir / "mesh"
    path.mkdir(parents=True, exist_ok=True)
    mesh = path / "simplified.obj"
    trimesh.creation.box(extents=extents).export(mesh)
    return str(mesh)


@pytest.fixture
def bowl_mesh(tmp_path):
    return _bodex_mesh(tmp_path, "core_bowl_0123456789abcdef", (0.08, 0.08, 0.08))


@pytest.fixture
def mug_mesh(tmp_path):
    return _bodex_mesh(tmp_path, "core_mug_0123456789abcdef", (0.06, 0.06, 0.08))


from kitchen_build import (
    KITCHEN_TYPES,
    PLACEMENTS,
    PLACEMENT_BY_LOC,
    match_support_nodes,
)


def test_placement_registry_covers_the_four_legacy_locations():
    """loc 1-4 must keep their exact meaning; the registry is their new home."""
    assert PLACEMENT_BY_LOC[1].label == "Above cabinet"
    assert PLACEMENT_BY_LOC[2].label == "Above dishwasher"
    assert PLACEMENT_BY_LOC[3].label == "Fridge shelf 1"
    assert PLACEMENT_BY_LOC[4].label == "Above island"


def test_wall_cabinet_locations_track_the_dynamic_flag():
    """The menu entries and the physics must flip together. A menu that offers
    'Inside wall cabinet' while the export strips the door's joint would produce
    tasks that can never succeed."""
    from kitchen_build import WALL_CABINET_DYNAMIC
    labels = {p.label for p in PLACEMENTS}
    if WALL_CABINET_DYNAMIC:
        assert "Inside wall cabinet" in labels
        assert "On wall cabinet top" in labels
    else:
        assert "Inside wall cabinet" not in labels
        assert "On wall cabinet top" not in labels


def test_placement_locs_are_unique_and_labels_are_unique():
    locs = [p.loc for p in PLACEMENTS]
    labels = [p.label for p in PLACEMENTS]
    assert len(locs) == len(set(locs)), f"duplicate loc values: {locs}"
    assert len(labels) == len(set(labels)), f"duplicate menu labels: {labels}"


def test_location_labels_is_derived_from_the_registry():
    from kitchen_build import LOCATION_LABELS
    assert LOCATION_LABELS == {p.loc: p.label for p in PLACEMENTS}


def test_persisted_placement_vocabulary_is_frozen():
    """Templates store the support label as each object's "placement". Measured across all 20
    templates in scripts/simvla/templates/, exactly two values are ever used. Renaming either
    silently breaks every template that says it, so they are frozen."""
    from kitchen_build import PLACEMENT_BY_KEY
    assert "dishwasher" in PLACEMENT_BY_KEY, "20 templates say placement=dishwasher"
    assert "island" in PLACEMENT_BY_KEY, "20 templates say placement=island"
    assert PLACEMENT_BY_KEY["dishwasher"].patterns == ("countertop_dishwasher",)
    assert PLACEMENT_BY_KEY["island"].patterns == ("kitchen_island/countertop",)


def test_every_placement_has_a_unique_key():
    from kitchen_build import PLACEMENTS, PLACEMENT_BY_KEY
    keys = [p.key for p in PLACEMENTS]
    assert len(keys) == len(set(keys)), f"duplicate placement keys: {keys}"
    assert all(k and k.strip() == k for k in keys), "keys must be non-empty and unpadded"
    assert len(PLACEMENT_BY_KEY) == len(PLACEMENTS)


def test_templates_on_disk_only_use_known_placement_keys():
    """The real compatibility check: parse every shipped template and assert each placement it
    names still resolves. This fails the moment a rename orphans a template."""
    import json
    from pathlib import Path
    from kitchen_build import PLACEMENT_BY_KEY

    # Anchored to this file, not the CWD: the previous glob("scripts/simvla/templates/*.json")
    # matched nothing when pytest ran from scripts/simvla/ (and several tests here chdir to a
    # tmp_path), so the test passed by validating zero templates.
    templates = sorted((Path(__file__).parent / "templates").glob("*.json"))
    assert templates, "no templates found — the path anchor is wrong, not the templates"

    unknown = {}
    for path in templates:
        for obj in json.loads(path.read_text()).get("scene") or []:
            placement = obj.get("placement")
            if placement and placement not in PLACEMENT_BY_KEY:
                unknown.setdefault(placement, []).append(path.name)
    assert not unknown, f"templates reference placements the registry no longer knows: {unknown}"


def test_every_labeled_support_is_reachable_from_some_location(bowl_mesh, monkeypatch, tmp_path):
    """The defect this locks down: refrigerator_3nd/_4nd were labeled as supports but no
    menu entry could ever select them. Labeling and selection now come from one table, so
    the two sets must agree exactly."""
    monkeypatch.chdir(tmp_path)
    for kitchen_name in ("island", "l_shaped", "peninsula", "u_shaped", "single_wall"):
        kitchen, _data, _objects, supports = build_kitchen(
            kitchen_name, [], [bowl_mesh], seed=0
        )
        reachable = set()
        for placement in PLACEMENTS:
            reachable.update(match_support_nodes(kitchen, placement))
        assert set(supports) == reachable, (
            f"{kitchen_name}: labeled-but-unreachable="
            f"{sorted(set(supports) - reachable)}, "
            f"reachable-but-unlabeled={sorted(reachable - set(supports))}"
        )


def test_available_locations_matches_what_the_kitchen_actually_has(bowl_mesh, monkeypatch, tmp_path):
    """LOCATION_KITCHENS is a hand-maintained table because probing costs 5-7s per
    kitchen and the GUI needs the answer instantly. This test is what keeps it honest."""
    monkeypatch.chdir(tmp_path)
    from kitchen_build import KITCHEN_TYPES, available_locations

    for kitchen_name in KITCHEN_TYPES:
        kitchen, _data, _objects, _supports = build_kitchen(
            kitchen_name, [], [bowl_mesh], seed=0
        )
        actually_present = {
            p.loc for p in PLACEMENTS if match_support_nodes(kitchen, p)
        }
        declared = {p.loc for p in available_locations(kitchen_name)}
        assert declared == actually_present, (
            f"{kitchen_name}: LOCATION_KITCHENS claims {sorted(declared)} "
            f"but the built kitchen has {sorted(actually_present)}"
        )


def test_support_geom_ids_label_every_offered_location(bowl_mesh, monkeypatch, tmp_path):
    """The batch path's half of the registry, exercised against real kitchens.

    goal_generator._label_supports() feeds support_geom_ids()[key] straight into
    kitchen.label_support(geom_ids=...), which re.search()es it over the scene's geometry
    node names. Nothing else in this suite ever ran that string through label_support, which
    is how two bugs survived: only patterns[0] was emitted, and a raw fnmatch glob was being
    reinterpreted as a regex. Both are invisible until a real kitchen is on the other end —
    on single_wall the wall cabinets are 'wall_cabinet_1/...'/'wall_cabinet_2/...', so the
    literal 'wall_cabinet/top' matched nothing, label_support silently registered no label,
    and generate_kitchen_scene then died with a KeyError inside support_generator because
    scene_spec.kitchen_surfaces() (derived from the registry, not from what got labeled)
    still advertised the location.

    So: for every registry key, on every kitchen type available_locations() says offers it,
    the label must actually register and support_generator must be able to iterate it.
    """
    monkeypatch.chdir(tmp_path)
    import re
    import numpy as np
    from kitchen_build import (
        KITCHEN_BUILDERS,
        add_fixtures,
        available_locations,
        support_geom_ids,
        support_geom_regex,
    )

    geom_ids = support_geom_ids()
    assert set(geom_ids) == {p.key for p in PLACEMENTS}

    for kitchen_name in KITCHEN_TYPES:
        # Built the way goal_generator.generate_kitchen_scene builds it: raw builder, the same
        # unwrap_geometries() call, then add_fixtures() -- and NOT build_kitchen (which labels
        # by node name -- that is the GUI path, and the path already known to work). The batch
        # path now calls add_fixtures() too (both paths must agree on what a kitchen contains,
        # since support_geom_ids()/kitchen_surfaces() advertise fixture placements to both), so
        # this mirror has to call it as well or it tests a kitchen shape generate_kitchen_scene
        # no longer actually produces.
        kitchen = KITCHEN_BUILDERS[kitchen_name](seed=0, counter_height=0.95)
        kitchen.unwrap_geometries("(sink_cabinet/sink_countertop|countertop_.*|.*countertop)")
        add_fixtures(kitchen, kitchen_name, np.random.default_rng(0), 0.95)
        geometry_nodes = list(kitchen.scene.graph.nodes_geometry)

        for placement in available_locations(kitchen_name):
            # The emitted regex must select exactly the geometry the GUI path reaches
            # through match_support_nodes() -- same rows, same nodes, plus the geometry
            # children unwrap_geometries() splits off (e.g. countertop_base_cabinet is a
            # group node; countertop_base_cabinet/geometry_0 is what carries the mesh).
            expected = set()
            for node in match_support_nodes(kitchen, placement):
                expected |= {
                    g for g in geometry_nodes if g == node or g.startswith(node + "/")
                }
            selected = set(
                filter(re.compile(support_geom_regex(placement)).search, geometry_nodes)
            )
            assert selected == expected, (
                f"{kitchen_name}/{placement.key}: regex selects "
                f"{sorted(selected - expected)} extra, misses {sorted(expected - selected)}"
            )

            support_data = kitchen.label_support(
                label=placement.key, geom_ids=geom_ids[placement.key]
            )
            assert support_data, (
                f"{kitchen_name}: label_support found no supports for {placement.key!r} "
                f"(geom_ids={geom_ids[placement.key]!r}) — the label is never registered, so "
                f"support_generator(support_ids={placement.key!r}) will KeyError at scene "
                f"build time even though kitchen_surfaces({kitchen_name!r}) offers it"
            )
            # The failure the missing label actually produces downstream.
            kitchen.support_generator(support_ids=placement.key)


def test_available_locations_rejects_unknown_kitchen():
    from kitchen_build import available_locations
    with pytest.raises(ValueError, match="nonexistent"):
        available_locations("nonexistent")


def test_every_material_group_used_by_geometry2material_exists():
    missing = sorted(set(GEOMETRY2MATERIAL.values()) - set(MATERIALS))
    assert missing == [], f"geometry2material references unknown material groups: {missing}"


def test_no_material_group_is_empty():
    empty = sorted(g for g, choices in MATERIALS.items() if not choices)
    assert empty == [], f"material groups with no choices: {empty}"


def test_pick_materials_covers_every_group():
    picks = pick_materials()
    assert set(picks) == set(MATERIALS)
    for group, choice in picks.items():
        assert choice in MATERIALS[group]


def test_material_display_names_falls_back_to_mdl_stem():
    picks = {
        "countertop": ("Base/Stone/Granite_Dark.mdl", "Granite_Dark", 0.25),
        "floor": ("Base/Wood/Oak.mdl", None, 0.25),
    }
    assert material_display_names(picks) == {"countertop": "Granite_Dark", "floor": "Oak"}


def test_resolve_mesh_picks_a_matching_file():
    files = ["/a/use_data/core_bowl_aa/mesh/simplified.obj",
             "/a/use_data/core_mug_bb/mesh/simplified.obj"]
    assert resolve_mesh("bowl", files) == files[0]


def test_resolve_mesh_raises_when_nothing_matches():
    with pytest.raises(ValueError, match="bottle"):
        resolve_mesh("bottle", ["/a/use_data/core_bowl_aa/mesh/simplified.obj"])


def test_resolve_mesh_will_not_hand_a_can_row_a_candle():
    """'can' is a substring of both 'candle' and 'sodacan'. Under the old rule a can row could
    spawn either; matching on the category prefix is what stops it."""
    files = ["/a/use_data/core_can_aa/mesh/simplified.obj",
             "/a/use_data/sem_Candle_bb/mesh/simplified.obj",
             "/a/use_data/sem_SodaCan_cc/mesh/simplified.obj"]
    assert resolve_mesh("can", files) == files[0]
    assert resolve_mesh("candle", files) == files[1]
    assert resolve_mesh("sodacan", files) == files[2]


def test_a_withdrawn_mesh_is_never_spawned():
    """MESH_EXCLUDE entries stay in use_data with their grasp data intact, but nothing places
    them."""
    from scene_spec import MESH_EXCLUDE
    withdrawn = sorted(MESH_EXCLUDE)[0]
    files = [f"/a/use_data/{withdrawn}/mesh/simplified.obj"]
    from kitchen_build import matching_meshes
    obj_type = "vase" if withdrawn.startswith("sem_Vase") else "cap"
    assert matching_meshes(obj_type, files) == []


def test_build_kitchen_rejects_unknown_type(bowl_mesh):
    with pytest.raises(ValueError, match="nonexistent"):
        build_kitchen("nonexistent", [], [bowl_mesh])


def test_build_kitchen_places_object_at_expected_node_id(bowl_mesh, monkeypatch, tmp_path):
    """The whole preview design rests on this: a placed object owns graph node '<obj_n>0'."""
    monkeypatch.chdir(tmp_path)
    specs = [{"obj_n": "bowl0", "type": "bowl", "loc": 1}]
    kitchen, kitchen_data, objects, supports = build_kitchen("island", specs, [bowl_mesh], seed=0)

    assert kitchen_data["kitchen_type"] == "island"
    assert kitchen_data["bowl0"] == bowl_mesh

    assert len(objects) == 1
    assert objects[0]["label"] == "bowl0"
    assert objects[0]["node_id"] == "bowl00"
    assert objects[0]["location"] == LOCATION_LABELS[1]

    # The node id must actually exist in the trimesh scene graph, and own geometry.
    graph_nodes = set(kitchen.scene.graph.nodes)
    assert "bowl00" in graph_nodes
    assert any(n.startswith("bowl00/") for n in graph_nodes)

    # The object was placed "Above cabinet", which labels countertop_base_cabinet as a
    # support; an island kitchen also labels the island top and the fridge shelves.
    assert "countertop_base_cabinet" in supports
    assert "kitchen_island/countertop" in supports
    assert all(n in graph_nodes for n in supports), "every reported support must be real"


def test_build_kitchen_skips_island_location_on_non_island_kitchen(bowl_mesh, monkeypatch, tmp_path):
    """loc=4 ("Above island") has no valid support on a non-island kitchen: the object is
    neither placed nor recorded in kitchen_data (contrast with the fit-failure skip path,
    which still records a kitchen_data entry)."""
    monkeypatch.chdir(tmp_path)
    specs = [{"obj_n": "bowl0", "type": "bowl", "loc": 4}]
    _kitchen, kitchen_data, objects, supports = build_kitchen("l_shaped", specs, [bowl_mesh], seed=0)
    assert objects == []
    assert "bowl0" not in kitchen_data
    assert kitchen_data == {"kitchen_type": "l_shaped"}
    # loc=4 ("Above island") is impossible on this kitchen type, but _label_supports still
    # unconditionally labels the supports that DO exist for l_shaped (no island top).
    assert "kitchen_island/countertop" not in supports
    assert "countertop_base_cabinet" in supports


def test_collect_joints_returns_only_drivable_joints(monkeypatch, tmp_path, bowl_mesh):
    """Joint sliders need revolute/prismatic joints with a real range and a real child node.

    'fixed' joints cannot move and 'floating' is what holds a placed object on its support
    (see apply_placements) — neither belongs in the joint panel.
    """
    monkeypatch.chdir(tmp_path)
    specs = [{"obj_n": "bowl0", "type": "bowl", "loc": 1}]
    kitchen, _kitchen_data, _objects, _supports = build_kitchen(
        "l_shaped", specs, [bowl_mesh], seed=0
    )

    joints = collect_joints(kitchen)
    assert joints, "an l_shaped kitchen has doors and drawers; none were found"

    graph_nodes = set(kitchen.scene.graph.nodes)
    for joint in joints:
        assert joint["type"] in ("revolute", "prismatic")
        assert joint["upper"] > joint["lower"], f"{joint['name']} has no range to drive"
        assert joint["child"] in graph_nodes, f"{joint['child']} is not a scene node"
        assert len(joint["axis"]) == 3
        assert len(joint["origin"]) == 4 and len(joint["origin"][0]) == 4
        # the group is what the preview folds the sliders under
        assert joint["group"] == joint["child"].split("/")[0]

    # the fridge doors are the ones users actually reach for; make sure they survived
    assert any("refrigerator" in j["child"] for j in joints)


def test_collect_joints_excludes_the_floating_joint_holding_a_placed_object(
    monkeypatch, tmp_path, bowl_mesh
):
    """A placed object hangs off its support by a 'floating' joint. It is not a door."""
    monkeypatch.chdir(tmp_path)
    specs = [{"obj_n": "bowl0", "type": "bowl", "loc": 1}]
    kitchen, _d, _o, _s = build_kitchen("island", specs, [bowl_mesh], seed=0)

    joints = collect_joints(kitchen)
    assert not any(j["child"] == "bowl00" for j in joints)
    assert not any("floating" in j["name"] for j in joints)


def test_object_scale_covers_every_allowed_object_type():
    """The bug this locks down: goal_generator's scale code was an if/elif over bottle/mug/bowl
    while the allowed types also include apple, sodacan and nutella — placing one raised
    UnboundLocalError. Adding a type to OBJECT_TYPES without a dimension must fail HERE, loudly,
    not at scene-generation time."""
    for obj_type in OBJECT_TYPES:
        dims = object_scale(obj_type, size=1.0, mug_height=0.08)
        assert len(dims) == 3, f"{obj_type} did not produce a [w, d, h]"
        assert all(d > 0 for d in dims), f"{obj_type} produced a non-positive dimension: {dims}"


def test_object_scale_applies_the_size_jitter():
    base = object_scale("bottle", size=1.0, mug_height=0.08)
    bigger = object_scale("bottle", size=1.1, mug_height=0.08)
    assert bigger == pytest.approx([b * 1.1 for b in base])


def test_object_scale_takes_the_mug_height_from_the_caller():
    """A mug's height is the one dimension the task profile sets (TASK_PROFILES['mug_height'])."""
    short = object_scale("mug", size=1.0, mug_height=0.08)
    tall = object_scale("mug", size=1.0, mug_height=0.09)
    assert short[2] == pytest.approx(0.08)
    assert tall[2] == pytest.approx(0.09)
    assert short[0] == tall[0], "only the height should depend on mug_height"


def test_object_scale_rejects_an_unknown_type_by_name():
    with pytest.raises(ValueError, match="banana"):
        object_scale("banana", size=1.0, mug_height=0.08)


NEW_SURFACE_LOCS = {
    5: "Sink counter",
    6: "Corner counter",
    7: "On stovetop",
    8: "On fridge top",
    9: "Inside base cabinet",
    10: "Inside sink cabinet",
    11: "Inside dishwasher",
    12: "In oven",
}


@pytest.mark.parametrize("loc,label", sorted(NEW_SURFACE_LOCS.items()))
def test_new_location_is_registered(loc, label):
    assert PLACEMENT_BY_LOC[loc].label == label


@pytest.mark.parametrize("loc", sorted(NEW_SURFACE_LOCS))
def test_new_location_actually_places_an_object(loc, bowl_mesh, monkeypatch, tmp_path):
    """A location that resolves to geometry but is too small or too cluttered for
    place_object to find a pose is worse than no location at all: it fails silently
    at scene-build time. Every registered location must place on every kitchen type
    that claims it."""
    monkeypatch.chdir(tmp_path)
    from kitchen_build import available_locations

    for kitchen_name in KITCHEN_TYPES:
        if loc not in {p.loc for p in available_locations(kitchen_name)}:
            continue
        specs = [{"obj_n": "bowl0", "type": "bowl", "loc": loc}]
        _k, kitchen_data, objects, _s = build_kitchen(
            kitchen_name, specs, [bowl_mesh], seed=0
        )
        assert objects, f"loc {loc} claims {kitchen_name} but nothing was placed"
        assert objects[0]["node_id"] == "bowl00"
        assert kitchen_data["bowl0"] == bowl_mesh


# loc 13/14 (wall cabinet top / interior) are not in NEW_SURFACE_LOCS above -- they are
# gated behind WALL_CABINET_DYNAMIC (see test_wall_cabinet_locations_track_the_dynamic_flag)
# rather than being unconditional like loc 5-12, so they get their own parametrized pair
# that skips cleanly when the flag is off instead of a bare KeyError.
WALL_CABINET_SURFACE_LOCS = {
    13: "On wall cabinet top",
    14: "Inside wall cabinet",
}


@pytest.mark.parametrize("loc,label", sorted(WALL_CABINET_SURFACE_LOCS.items()))
def test_wall_cabinet_location_is_registered(loc, label):
    """Mirrors test_new_location_is_registered for loc 13/14."""
    from kitchen_build import WALL_CABINET_DYNAMIC
    if not WALL_CABINET_DYNAMIC:
        pytest.skip("wall cabinet placements disabled by WALL_CABINET_DYNAMIC")
    assert PLACEMENT_BY_LOC[loc].label == label


@pytest.mark.parametrize("loc", sorted(WALL_CABINET_SURFACE_LOCS))
def test_wall_cabinet_location_actually_places_an_object(loc, bowl_mesh, monkeypatch, tmp_path):
    """Mirrors test_new_location_actually_places_an_object (loc 5-12) for the wall
    cabinet locations loc 13/14, which NEW_SURFACE_LOCS does not cover."""
    from kitchen_build import WALL_CABINET_DYNAMIC, available_locations
    if not WALL_CABINET_DYNAMIC:
        pytest.skip("wall cabinet placements disabled by WALL_CABINET_DYNAMIC")
    monkeypatch.chdir(tmp_path)

    for kitchen_name in KITCHEN_TYPES:
        if loc not in {p.loc for p in available_locations(kitchen_name)}:
            continue
        specs = [{"obj_n": "bowl0", "type": "bowl", "loc": loc}]
        _k, kitchen_data, objects, _s = build_kitchen(
            kitchen_name, specs, [bowl_mesh], seed=0
        )
        assert objects, f"loc {loc} claims {kitchen_name} but nothing was placed"
        assert objects[0]["node_id"] == "bowl00"
        assert kitchen_data["bowl0"] == bowl_mesh


def test_fridge_shelves_are_ordered_top_to_bottom(bowl_mesh, monkeypatch, tmp_path):
    """Shelf 1 is the highest. Name order is NOT height order at the bottom: shelf_4
    (z=0.080) sits below shelf_0 (z=0.246)."""
    monkeypatch.chdir(tmp_path)
    from kitchen_build import fridge_shelf_order, _node_top_z

    kitchen, _d, _o, _s = build_kitchen("island", [], [bowl_mesh], seed=0)
    order = fridge_shelf_order(kitchen)
    assert len(order) == 5, f"expected 5 fridge shelves, got {order}"
    assert order[0] == "refrigerator/shelf_3", f"top shelf should be shelf_3, got {order[0]}"
    assert order[-1] == "refrigerator/shelf_4", f"bottom shelf should be shelf_4, got {order[-1]}"

    heights = [_node_top_z(kitchen, n) for n in order]
    assert heights == sorted(heights, reverse=True), (
        f"shelves not ordered top-to-bottom: {list(zip(order, heights))}"
    )


def test_all_five_fridge_shelves_and_three_door_shelves_are_registered():
    labels = {p.label for p in PLACEMENTS}
    for i in range(1, 6):
        assert f"Fridge shelf {i}" in labels
    for i in range(1, 4):
        assert f"Fridge door shelf {i}" in labels


def test_every_fridge_shelf_places_an_object(bowl_mesh, monkeypatch, tmp_path):
    """Every registered fridge location must place on every kitchen type that claims
    it, not just island (mirrors test_new_location_actually_places_an_object for loc
    5-12, which already covers this shape for the non-fridge locations)."""
    monkeypatch.chdir(tmp_path)
    from kitchen_build import available_locations

    fridge_locs = [p.loc for p in PLACEMENTS if p.group == "Refrigerator"]
    assert len(fridge_locs) == 8
    for loc in fridge_locs:
        for kitchen_name in KITCHEN_TYPES:
            if loc not in {p.loc for p in available_locations(kitchen_name)}:
                continue
            specs = [{"obj_n": "bowl0", "type": "bowl", "loc": loc}]
            _k, _data, objects, _s = build_kitchen(
                kitchen_name, specs, [bowl_mesh], seed=0
            )
            assert objects, f"fridge loc {loc} placed nothing on {kitchen_name}"


def test_fridge_shelf_locations_match_the_measured_order(bowl_mesh, monkeypatch, tmp_path):
    """Close the gap between the hand-written PLACEMENTS rows and fridge_shelf_order():
    loc 3, 20, 21, 22, 23 are menu positions 1-5 and must resolve, via
    match_support_nodes, to exactly the shelf fridge_shelf_order() measures at that
    position. Without this, a transposed pattern (e.g. loc=21/22 swapped) would pass
    every other test in this file, because test_fridge_shelves_are_ordered_top_to_bottom
    never looks at PLACEMENTS and the other fridge tests don't check WHICH shelf a loc
    lands on, only that it lands on something.

    The expectation is derived from fridge_shelf_order(kitchen) itself, not a second
    hardcoded list of node names -- a test that restates the constant it is checking
    would not catch the drift it exists to prevent.
    """
    monkeypatch.chdir(tmp_path)
    from kitchen_build import fridge_shelf_order

    kitchen, _d, _o, _s = build_kitchen("island", [], [bowl_mesh], seed=0)
    measured = fridge_shelf_order(kitchen)

    shelf_locs_in_menu_order = [3, 20, 21, 22, 23]  # Fridge shelf 1..5
    resolved = []
    for loc in shelf_locs_in_menu_order:
        nodes = match_support_nodes(kitchen, PLACEMENT_BY_LOC[loc])
        assert len(nodes) == 1, f"loc {loc} should resolve to exactly one shelf, got {nodes}"
        resolved.append(nodes[0])

    assert resolved == measured, (
        f"menu order (loc {shelf_locs_in_menu_order}) resolves to {resolved}, "
        f"but the measured top-to-bottom order is {measured}"
    )


def test_build_kitchen_uses_an_explicitly_named_mesh(mug_mesh, monkeypatch, tmp_path):
    """An object row may name its exact mesh. resolve_mesh() picks a RANDOM match, so with
    only one candidate in mesh_files a test that merely asserts "the chosen mesh came back"
    would pass even if the named mesh were silently ignored and resolve_mesh() were still
    doing the picking. Monkeypatching resolve_mesh to raise proves the stronger property:
    build_kitchen never calls it at all when spec["mesh"] is set.
    """
    monkeypatch.chdir(tmp_path)
    from kitchen_build import available_locations

    def _must_not_be_called(*_args, **_kwargs):
        raise AssertionError("resolve_mesh must not be called when spec['mesh'] is set")

    monkeypatch.setattr("kitchen_build.resolve_mesh", _must_not_be_called)

    specs = [{"obj_n": "mug0", "type": "mug", "loc": available_locations("island")[0].loc,
              "mesh": mug_mesh}]
    _kitchen, kitchen_data, _objects, _supports = build_kitchen(
        "island", specs, [mug_mesh], seed=0
    )
    assert kitchen_data["mug0"] == mug_mesh


def test_build_kitchen_still_picks_at_random_when_no_mesh_is_named(mug_mesh, monkeypatch, tmp_path):
    """Absent/None "mesh" must fall back to resolve_mesh(), exactly as every caller behaved
    before the field existed."""
    monkeypatch.chdir(tmp_path)
    from kitchen_build import available_locations

    specs = [{"obj_n": "mug0", "type": "mug", "loc": available_locations("island")[0].loc,
              "mesh": None}]
    _kitchen, kitchen_data, _objects, _supports = build_kitchen(
        "island", specs, [mug_mesh], seed=0
    )
    assert "mug" in kitchen_data["mug0"].lower()


def test_matching_meshes_is_what_resolve_mesh_chooses_from(bowl_mesh):
    """One definition of what "bowl" means. When the gallery and the random pick disagreed about
    that, a user could choose a mesh the generator would never have offered."""
    from kitchen_build import matching_meshes, resolve_mesh

    others = ["/data/use_data/core_mug_aaa/mesh/simplified.obj"]
    files = [bowl_mesh] + others

    assert matching_meshes("bowl", files) == [bowl_mesh]
    assert resolve_mesh("bowl", files) in matching_meshes("bowl", files)
    assert matching_meshes("teapot", files) == []


def _kitchen_type_index_maps():
    """Every str->int kitchen-type encoding in the repo, read as source.

    The public exporter lives in hdf5_dataset_file_handler and imports the Isaac stack.
    The separate research-only stream_finalize_lerobot is not shipped or called here.
    Read the actual exporter's dict literal from the AST; this guards the boundary
    between a newly registered layout and a KeyError at the very end of a collection run,
    after the episodes are on disk.
    """
    import ast
    from pathlib import Path

    here = Path(__file__).resolve()
    sources = {
        "hdf5_dataset_file_handler.kmap":
            here.parents[2] / "source" / "isaaclab" / "isaaclab" / "utils" / "datasets"
            / "hdf5_dataset_file_handler.py",
    }
    maps = {}
    for label, path in sources.items():
        assert path.exists(), f"{label}: {path} has moved"
        var = label.split(".")[-1]
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == var for t in node.targets)
                and isinstance(node.value, ast.Dict)
            ):
                maps[label] = {
                    k.value: v.value for k, v in zip(node.value.keys, node.value.values)
                }
        assert label in maps, f"{label} is gone; the encoding it defined is unaccounted for"
    return maps


def test_every_registered_kitchen_type_has_a_dataset_encoding():
    """A layout the wizard can build but the finaliser cannot encode is a run that dies at
    the finish line: KITCHEN_TYPE_MAP[ktype_str] raises KeyError after every episode is
    already staged. galley was exactly that until it was added here."""
    from kitchen_build import KITCHEN_TYPES

    for label, kmap in _kitchen_type_index_maps().items():
        missing = sorted(set(KITCHEN_TYPES) - set(kmap))
        assert not missing, (
            f"{label} has no index for {missing}; a collection run on that layout raises "
            "KeyError during LeRobot finalization, after the episodes are on disk"
        )


def test_the_kitchen_type_encodings_are_unique_and_stable():
    """The index is written into every episode of every dataset ever collected. Reusing one
    silently relabels history, and the two maps disagreeing splits the same layout across two
    encodings depending on which writer ran."""
    maps = _kitchen_type_index_maps()
    for label, kmap in maps.items():
        assert len(set(kmap.values())) == len(kmap), f"{label} reuses an index"

    # The five encodings the existing corpus was written with. Pinned, not derived: this is
    # the assertion that a new layout must take a FRESH index rather than renumber the rest.
    legacy = {"island": 0, "l_shaped": 1, "peninsula": 2, "u_shaped": 3, "single_wall": 4}
    for label, kmap in maps.items():
        for name, idx in legacy.items():
            assert kmap.get(name) == idx, (
                f"{label} re-encodes {name} as {kmap.get(name)}, not {idx}; every dataset "
                "collected so far is mislabelled"
            )

    first = next(iter(maps.values()))
    assert all(value == first for value in maps.values()), "kitchen-type encodings have drifted apart"
