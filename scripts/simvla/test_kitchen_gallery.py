"""The gallery's tile scenes and paging.

Tests the DATA the page is built from — the same discipline as test_kitchen_preview.py, which
never asserts on the DOM. A gallery scene is checked through trimesh's own graph, not by eye.

Run: ~/miniconda3/envs/env_isaaclab/bin/python -m pytest scripts/simvla/test_kitchen_gallery.py -v
"""

import trimesh

from kitchen_gallery import kitchen_tiles, mesh_tiles, paginate, tile_label


def _obj(tmp_path, name, extents):
    """A stand-in for a BODex mesh, the way test_kitchen_build.py makes them."""
    path = tmp_path / name
    trimesh.creation.box(extents=extents).export(path)
    return str(path)


def test_mesh_tiles_makes_one_tile_per_path_carrying_the_path_as_its_id(tmp_path):
    """The id is what comes back when the user clicks, and it must be the untouched path: it goes
    straight into an object row's `mesh` field, which build_kitchen loads."""
    paths = [_obj(tmp_path, f"mug_{i}.obj", (0.1, 0.1, 0.1)) for i in range(3)]
    scene, tiles = mesh_tiles(paths)

    assert [t["id"] for t in tiles] == paths
    assert len(scene.geometry) == 3


def test_mesh_tiles_normalises_wildly_different_sizes_to_one_cell(tmp_path):
    """A cereal box beside a soap bar is otherwise one enormous slab and one speck."""
    big = _obj(tmp_path, "cerealbox_0.obj", (0.30, 0.08, 0.40))
    small = _obj(tmp_path, "soapbar_0.obj", (0.02, 0.01, 0.03))
    scene, tiles = mesh_tiles([big, small], cell=1.0)

    sizes = [max(geom.extents) for geom in scene.geometry.values()]
    assert max(sizes) - min(sizes) < 1e-6, f"tiles are not the same size: {sizes}"
    assert max(sizes) <= 1.0, "a tile must fit inside its cell"


def test_mesh_tiles_lays_them_out_on_a_grid(tmp_path):
    """Row-major: per_row across, then wrap. Overlapping tiles would be unclickable."""
    paths = [_obj(tmp_path, f"mug_{i}.obj", (0.1, 0.1, 0.1)) for i in range(5)]
    scene, tiles = mesh_tiles(paths, per_row=2, cell=1.0)

    centres = [
        scene.graph.get(t["node_raw"])[0][:3, 3] for t in tiles
    ]
    xs = [round(float(c[0]), 6) for c in centres]
    ys = [round(float(c[1]), 6) for c in centres]
    assert xs == [0.0, 1.0, 0.0, 1.0, 0.0], f"columns wrong: {xs}"
    assert ys == [0.0, 0.0, -1.0, -1.0, -2.0], f"rows wrong: {ys}"


def test_mesh_tiles_node_names_are_what_threejs_will_see(tmp_path):
    """GLTFLoader sanitizes every node name; a panel matching raw names would match nothing."""
    scene, tiles = mesh_tiles([_obj(tmp_path, "mug_0.obj", (0.1, 0.1, 0.1))])
    from kitchen_preview import sanitize_three_name

    assert tiles[0]["node"] == sanitize_three_name(tiles[0]["node_raw"])


def test_mesh_tiles_skips_a_mesh_that_will_not_load(tmp_path):
    """One corrupt file out of 148 must not cost the whole gallery."""
    good = _obj(tmp_path, "mug_0.obj", (0.1, 0.1, 0.1))
    bad = tmp_path / "mug_1.obj"
    bad.write_text("this is not an obj")

    scene, tiles = mesh_tiles([good, str(bad)])

    assert [t["id"] for t in tiles] == [good]


def test_mesh_tiles_stand_a_mesh_up_the_way_the_placer_will(tmp_path):
    """The picker's whole job is to show the mesh you are about to place. It used to draw the raw
    file frame, so every core_* mesh — which stores +Y up — appeared lying on its side while the
    scene stood it upright, and every unlabelled sem_* one appeared upright while the scene laid
    it down. Same up vector on both sides, so the tile cannot disagree with the counter."""
    tall_in_y = _obj(tmp_path, "bottle_0.obj", (0.06, 0.18, 0.06))
    scene, tiles = mesh_tiles([tall_in_y])

    drawn = next(iter(scene.geometry.values())).extents
    assert drawn.argmax() == 2, f"the long axis should be standing up in Z, got extents {drawn}"


def test_mesh_tiles_follow_a_per_mesh_override_over_the_type_fallback(tmp_path):
    """A sem_* mesh in MESH_UP is +Z up already; applying the core_* fallback to it would lay it
    down in the picker — the exact defect the table exists to prevent in the scene."""
    from mesh_orientation import MESH_UP

    listed = next(n for n, up in MESH_UP.items() if tuple(up) == (0, 0, 1))
    mesh_dir = tmp_path / listed / "mesh"
    mesh_dir.mkdir(parents=True)
    trimesh.creation.box(extents=(0.06, 0.06, 0.18)).export(mesh_dir / "simplified.obj")

    scene, tiles = mesh_tiles([str(mesh_dir / "simplified.obj")])

    drawn = next(iter(scene.geometry.values())).extents
    assert drawn.argmax() == 2, f"an entry of (0,0,1) must leave it standing, got {drawn}"


def test_mesh_tiles_take_the_up_the_type_gives_when_the_mesh_has_no_entry(tmp_path):
    """apple and sodacan are the two types the pipeline has always stood on +Z, and the picker has
    to make the same exception the placer does."""
    tall_in_z = _obj(tmp_path, "apple_0.obj", (0.06, 0.06, 0.18))
    scene, tiles = mesh_tiles([tall_in_z], obj_type="apple")

    drawn = next(iter(scene.geometry.values())).extents
    assert drawn.argmax() == 2, f"an apple keeps its +Z, got {drawn}"


def test_kitchen_tiles_are_left_in_the_frame_they_were_built_in():
    """Rooms come from build_kitchen already Z-up. Running them through the object correction
    would tip every layout in the gallery onto its wall."""
    room = trimesh.Scene(trimesh.creation.box(extents=(4.2, 3.1, 2.4)))
    scene, tiles = kitchen_tiles([("island", 0, room)])

    drawn = next(iter(scene.geometry.values())).extents
    assert drawn.argmin() == 2, f"the 2.4 m ceiling should still be the short axis, got {drawn}"


def test_mesh_tiles_of_nothing_is_an_empty_scene(tmp_path):
    scene, tiles = mesh_tiles([])
    assert tiles == []
    assert len(scene.geometry) == 0


def test_tile_label_names_the_object_not_the_path(tmp_path):
    assert tile_label("/data/use_data/core_mug_1038e4ea/mesh/simplified.obj") == "core_mug_1038e4ea"


def test_tile_label_falls_back_to_the_whole_path_when_too_shallow():
    """'/data/mug.obj' normalises (os.sep-split) to ['', 'data', 'mug.obj'] -- length 3 -- so a
    naive parts[-3] lands on the leading '' from the absolute path's leading slash, a blank button
    in the gallery list. Empty segments must be dropped before counting, and fewer than three real
    segments must fall back to the whole path rather than ever return blank. Same fix, same
    reasoning, as shortMesh() in kitchen_wizard.py's form page -- the two must agree."""
    assert tile_label("/data/mug.obj") == "/data/mug.obj"


def test_kitchen_tiles_carry_type_and_seed_in_the_id():
    """The id is what the form stores: picking a tile must reproduce that exact room."""
    scenes = [
        ("island", 0, trimesh.Scene(trimesh.creation.box(extents=(3, 3, 2)))),
        ("l_shaped", 1, trimesh.Scene(trimesh.creation.box(extents=(4, 2, 2)))),
    ]
    scene, tiles = kitchen_tiles(scenes)

    assert [t["id"] for t in tiles] == ["island:0", "l_shaped:1"]
    assert [t["label"] for t in tiles] == ["island", "l shaped"]
    assert len(scene.geometry) == 2


def test_paginate_reports_the_page_and_the_total():
    items = list(range(50))
    page_items, page, pages = paginate(items, 2, per_page=24)
    assert page_items == list(range(24, 48))
    assert (page, pages) == (2, 3)


def test_paginate_clamps_a_page_out_of_range():
    """A stale Next from a page that no longer exists must land somewhere real, not raise."""
    assert paginate(list(range(10)), 99, per_page=24)[1] == 1
    assert paginate(list(range(10)), 0, per_page=24)[1] == 1
    assert paginate(list(range(50)), 99, per_page=24)[1] == 3


def test_paginate_of_nothing_is_one_empty_page():
    assert paginate([], 1, per_page=24) == ([], 1, 1)


import json

from kitchen_gallery import gallery_options, render_gallery_page


def _page(tmp_path, count=3, page=1, pages=1):
    paths = [_obj(tmp_path, f"mug_{i}.obj", (0.1, 0.1, 0.1)) for i in range(count)]
    scene, tiles = mesh_tiles(paths)
    return render_gallery_page(scene, tiles, title="Pick a mug", page=page, pages=pages), tiles


def test_gallery_options_declare_every_id_the_page_can_send():
    """The step publishes these, and the server refuses any id it did not publish — so a button on
    screen whose id is missing here is a button that answers 400 and looks dead."""
    tiles = [{"id": "/a.obj", "node": "t0", "label": "a"}, {"id": "/b.obj", "node": "t1", "label": "b"}]
    options = gallery_options(tiles, page=2, pages=3)

    assert "/a.obj" in options and "/b.obj" in options
    assert "" in options, "Cancel answers with the empty id"
    assert "page:1" in options and "page:3" in options, "both page turns are reachable from page 2"


def test_gallery_options_omit_the_page_turn_that_does_not_exist():
    tiles = [{"id": "/a.obj", "node": "t0", "label": "a"}]
    first = gallery_options(tiles, page=1, pages=3)
    last = gallery_options(tiles, page=3, pages=3)

    assert "page:0" not in first and "page:2" in first
    assert "page:4" not in last and "page:2" in last


def test_the_page_carries_every_tile_and_posts_to_answer(tmp_path):
    page, tiles = _page(tmp_path, count=3)
    assert "__GALLERY__" not in page, "the payload placeholder must be substituted"
    for tile in tiles:
        assert tile["id"] in page
        assert tile["node"] in page
    assert "/answer" in page
    assert "simvlaPost" in page


def test_the_page_offers_a_button_per_tile_as_well_as_the_3d(tmp_path):
    """Raycasting a small tile is fiddly, and a click that misses looks like a broken page. The
    panel lists the same tiles as buttons, so every tile is reachable without aiming."""
    page, tiles = _page(tmp_path, count=3)
    payload = json.loads(page.split("var DATA = ")[1].split(";\n")[0])
    assert [t["id"] for t in payload["tiles"]] == [t["id"] for t in tiles]


def test_the_page_knows_where_it_is_in_the_paging(tmp_path):
    page, _ = _page(tmp_path, count=3, page=2, pages=7)
    payload = json.loads(page.split("var DATA = ")[1].split(";\n")[0])
    assert (payload["page"], payload["pages"]) == (2, 7)


def test_a_tile_label_containing_a_script_tag_cannot_break_out(tmp_path):
    """Tile labels and ids are filesystem paths. A path containing </script> would otherwise end
    the script block early and the rest of the page would be parsed as live markup.

    Built from one real tile rather than mesh_tiles([]): trimesh 4.10.1's exporter refuses to
    export an empty scene at all ("Can't export empty scenes!"), which is a pre-existing trimesh
    constraint this test is not about -- the fake hostile tile below stands in regardless of what
    the scene actually contains.
    """
    scene, tiles = mesh_tiles([_obj(tmp_path, "mug_0.obj", (0.1, 0.1, 0.1))])
    tiles = [{"id": "/x</script><script>window.pwned=1</script>.obj", "node": "t0", "label": "x"}]
    page = render_gallery_page(scene, tiles, title="t")

    assert "</script><script>window.pwned" not in page


# ---------------------------------------------------------------------------------------------
# Fix round 1 (coordinator review): the panel had no double-submission guard, so two rapid clicks
# (two tiles, or a tile then Cancel) could both reach the server inside the ~1s window before the
# director wakes from wait_answer() and republishes -- the single answer slot is last-write-wins.
# Same no-DOM discipline as kitchen_preview's _panel_js_function tests: these pin the guard's
# structural shape in the rendered _PANEL source, not by executing it in a browser.
# ---------------------------------------------------------------------------------------------

import re

import kitchen_gallery


def _gallery_js_function(name):
    """The source of one top-level helper inside kitchen_gallery._PANEL's <script>.

    Same convention as kitchen_preview's _panel_js_function: the script is indented two spaces
    inside the IIFE, so a helper runs from `function name(` to the first line that is exactly
    `  }`.
    """
    src = kitchen_gallery._PANEL
    start = src.index("function " + name + "(")
    return src[start:src.index("\n  }", start)]


def test_answer_refuses_a_second_call_while_one_is_in_flight():
    """FINDING 1: the answer slot is single and last-write-wins -- a second call while the first
    is still in flight must be a no-op, not a second POST. The guard must run before setBusy(true)
    and the POST, or a re-entrant call would already be past the door it is meant to close."""
    body = _gallery_js_function("answer")
    guard_idx = body.index("if (answering) { return; }")
    assert guard_idx < body.index("setBusy(true)")
    assert guard_idx < body.index("simvlaPost")


def test_answer_only_re_enables_on_a_non_200_response():
    """The 200 path leaves the row disabled -- the step poller is about to reload the tab anyway --
    and only a rejected answer, which leaves the user on the same page, re-enables it. Same shape
    as _CHOICE_BODY in kitchen_wizard.py."""
    body = _gallery_js_function("answer")
    assert body.count("setBusy(false)") == 1
    non_200_branch = body[body.index("if (r.status !== 200)"):]
    assert "setBusy(false)" in non_200_branch


def test_set_busy_disables_every_button_and_restores_the_paging_boundary():
    """setBusy(true) must reach every clickable button in the panel -- tiles, Prev, Next, Cancel --
    since they are all built from the same '#gal button' query. Restoring on re-enable must not
    just flip every button back on: that would also re-enable a page turn that does not exist."""
    body = _gallery_js_function("setBusy")
    assert "answering = busy;" in body, (
        "setBusy() must actually arm the `answering` flag the guard above reads -- delete this "
        "line and `if (answering)` is always false, and the double-submission guard the tests "
        "above pin the SHAPE of is defeated end to end"
    )
    assert "document.querySelectorAll('#gal button')" in body
    assert "b.disabled = busy" in body
    assert "DATA.page <= 1" in body and "DATA.page >= DATA.pages" in body


def test_raycast_click_handler_is_bound_by_the_same_guard():
    """Not a button, so setBusy()'s disabling would not stop a click on the 3D view by itself --
    the raycast handler needs its own check. It selects rather than answering now, but the guard
    still matters: selecting repaints materials, and doing that while an answer is in flight would
    fight the reload that is about to happen."""
    handler = _gallery_js_function("onViewClick")
    assert "if (answering) { return; }" in handler
    assert "renderer.domElement.addEventListener('click', onViewClick);" in kitchen_gallery._PANEL


def test_the_gallery_page_only_fetches_from_its_own_origin(tmp_path):
    """FINDING 2: one of this task's own constraints is that pages fetch nothing from the network.
    test_kitchen_preview.py checks this for kitchen_preview.render_page; the gallery panel needs
    the same guarantee and had no test of its own."""
    page, _ = _page(tmp_path, count=3)
    assert not re.search(r"\bfetch\s*\(\s*[\"']https?:", page)


def test_gallery_options_of_a_single_page_offers_no_page_turns():
    """FINDING 3: a gallery with everything on one page must not offer a Prev or a Next id."""
    tiles = [{"id": "/a.obj", "node": "t0", "label": "a"}]
    assert gallery_options(tiles, page=1, pages=1) == ["/a.obj", ""]


def test_gallery_options_of_no_tiles_is_just_cancel():
    """FINDING 3: an empty gallery (e.g. a filter that matched nothing) must still answer something
    -- Cancel -- rather than publishing an empty options list."""
    assert gallery_options([], page=1, pages=1) == [""]


def test_gallery_options_matches_every_id_the_rendered_panel_can_send(tmp_path):
    """FINDING 3: gallery_options() and the panel's own answer() calls are built from the same
    (tiles, page, pages) in two different places -- Python for the server's allow-list, JS for what
    the buttons and the raycast handler actually send. The server refuses anything not published,
    so drift between the two is exactly the failure this task must not ship. Pull the tile ids back
    out of the rendered page's own payload (not the `tiles` list handed in) so this checks what the
    page actually carries, not just what was passed to both functions."""
    page, tiles = _page(tmp_path, count=3, page=2, pages=3)
    payload = json.loads(page.split("var DATA = ")[1].split(";\n")[0])

    sendable = {t["id"] for t in payload["tiles"]}
    if payload["page"] > 1:
        sendable.add("page:" + str(payload["page"] - 1))
    if payload["page"] < payload["pages"]:
        sendable.add("page:" + str(payload["page"] + 1))
    sendable.add("")  # Cancel

    assert set(gallery_options(tiles, page=2, pages=3)) == sendable


# ---- what the user asked for after seeing the first galleries ----


def test_every_tile_gets_a_material_so_it_is_not_rendered_black(tmp_path):
    """A mesh with no material exports to glTF with none, and three.js falls back to a dark
    default -- the first galleries came out as black silhouettes. Every tile carries its own
    light PBR material instead."""
    paths = [_obj(tmp_path, f"mug_{i}.obj", (0.1, 0.1, 0.1)) for i in range(2)]
    scene, tiles = mesh_tiles(paths)

    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, "material", None)
        assert material is not None, "a tile with no material renders black"
        assert max(material.baseColorFactor[:3]) > 120, (
            f"tile material is too dark to read as geometry: {material.baseColorFactor}"
        )


def test_kitchen_tiles_get_the_same_material_treatment():
    scenes = [("island", 0, trimesh.Scene(trimesh.creation.box(extents=(3, 3, 2))))]
    scene, tiles = kitchen_tiles(scenes)

    material = getattr(next(iter(scene.geometry.values())).visual, "material", None)
    assert material is not None and max(material.baseColorFactor[:3]) > 120


def test_each_tile_owns_its_material_through_the_glb_export(tmp_path):
    """Asserted on the EXPORTED glTF, not on the Python objects: trimesh deduplicates materials
    that compare equal, so distinct instances with the same name collapse into one on the way out
    and every mesh ends up pointing at it — after which recolouring the selected tile would
    recolour the whole gallery. Measured before the fix: 6 meshes, 1 material."""
    import json
    import struct

    paths = [_obj(tmp_path, f"mug_{i}.obj", (0.1, 0.1, 0.1)) for i in range(3)]
    scene, tiles = mesh_tiles(paths)

    glb = scene.export(file_type="glb")          # exactly what scene_to_html embeds
    length = struct.unpack("<I", glb[12:16])[0]
    doc = json.loads(glb[20:20 + length].decode("utf-8"))

    used = [p.get("material") for mesh in doc["meshes"] for p in mesh["primitives"]]
    assert len(doc.get("materials", [])) == 3, f"materials collapsed on export: {doc.get('materials')}"
    assert len(set(used)) == 3, f"tiles share one exported material: {used}"


def test_labels_in_the_3d_view_are_off_unless_asked_for(tmp_path):
    """Five kitchen layouts want their names in the view; 24 mesh tiles would just collide."""
    scene, tiles = mesh_tiles([_obj(tmp_path, "mug_0.obj", (0.1, 0.1, 0.1))])

    plain = json.loads(render_gallery_page(scene, tiles, title="t").split("var DATA = ")[1]
                       .split(";\n")[0])
    labelled = json.loads(render_gallery_page(scene, tiles, title="t", labels_in_view=True)
                          .split("var DATA = ")[1].split(";\n")[0])

    assert plain["labels"] is False
    assert labelled["labels"] is True


def test_a_click_in_the_3d_view_selects_and_does_not_answer():
    """The complaint that started this: orbiting the view meant clicking a tile by accident, and
    a click committed immediately. A 3D click now only moves the highlight."""
    handler = _gallery_js_function("onViewClick")

    assert "select(" in handler, "a 3D click must select"
    assert "answer(" not in handler, "a 3D click must not commit"


def test_the_panel_commits_the_selection_with_its_own_button(tmp_path):
    """Selecting in the view is only half of it -- there has to be something to press once the
    highlight is on the tile you want."""
    page, _ = _page(tmp_path, count=3)

    assert "gal-use" in page, "no button to commit the selected tile"
    use = _gallery_js_function("commitSelection")
    assert "answer(" in use, "the commit button must answer"


def test_tiles_carry_the_object_s_real_size(tmp_path):
    """Every tile is scaled to the same cell, so the gallery destroys the one property that
    decides whether a gripper can close on the thing. The real extents ride along instead —
    in the PLACED frame, so the third number is the height the object will actually have on the
    counter and the panel agrees with the picture beside it."""
    path = _obj(tmp_path, "mug_0.obj", (0.09, 0.08, 0.11))
    scene, tiles = mesh_tiles([path])

    # No override for this name and no type given, so the placer's core_* fallback applies:
    # mesh +Y becomes world up, and the 0.08 deep axis becomes the 0.08 tall one.
    assert [round(v, 3) for v in tiles[0]["size"]] == [0.09, 0.11, 0.08]


def test_kitchen_tiles_carry_the_room_s_real_size():
    scenes = [("island", 0, trimesh.Scene(trimesh.creation.box(extents=(4.2, 3.1, 2.4))))]
    scene, tiles = kitchen_tiles(scenes)

    assert [round(v, 2) for v in tiles[0]["size"]] == [4.2, 3.1, 2.4]


def test_the_page_carries_sizes_units_and_the_grid_width(tmp_path):
    """The panel needs the units to write "9 x 8 x 11 cm", and per_row to move the selection up
    and down a row with the arrow keys rather than only along it."""
    scene, tiles = mesh_tiles([_obj(tmp_path, "mug_0.obj", (0.09, 0.08, 0.11))], per_row=6)
    page = render_gallery_page(scene, tiles, title="t", per_row=6, units="cm")
    data = json.loads(page.split("var DATA = ")[1].split(";\n")[0])

    assert data["units"] == "cm"
    assert data["per_row"] == 6
    # In the placed frame, like the tile beside it: mesh +Y is up, so 0.08 is the height.
    assert [round(v, 3) for v in data["tiles"][0]["size"]] == [0.09, 0.11, 0.08]


def test_hovering_the_view_previews_a_tile_without_selecting_it(tmp_path):
    """Sweeping the grid to find the right shape should not commit you to anything, and should
    not lose the selection you already made."""
    handler = _gallery_js_function("onViewHover")

    assert "answer(" not in handler, "hovering must never answer"
    assert "select(" not in handler, "hovering must not change the selection either"


def test_the_keyboard_drives_the_whole_gallery(tmp_path):
    """Arrow keys move, Enter commits, Escape cancels, PageUp/PageDown turn the page."""
    handler = _gallery_js_function("onKeyDown")

    for key in ("ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown",
                "Enter", "Escape", "PageUp", "PageDown"):
        assert key in handler, f"{key} does nothing"
    page, _ = _page(tmp_path, count=3)
    assert "keydown" in page, "the handler is never bound"


def test_mesh_tiles_mark_which_meshes_have_grasps(tmp_path):
    """Choose… is where a mesh is picked, so it is where 'this one will not grasp' has to appear."""
    backed = _obj(tmp_path, "backed.obj", (0.1, 0.1, 0.1))
    bare = _obj(tmp_path, "bare.obj", (0.1, 0.1, 0.1))
    _scene, tiles = mesh_tiles([backed, bare], grasp_backed={backed})
    by_id = {t["id"]: t for t in tiles}
    assert by_id[backed]["grasps"] is True
    assert by_id[bare]["grasps"] is False


def test_mesh_tiles_leave_grasps_unknown_when_not_told(tmp_path):
    """No manifest: say nothing rather than claim every mesh is ungraspable."""
    _scene, tiles = mesh_tiles([_obj(tmp_path, "a.obj", (0.1, 0.1, 0.1))])
    assert tiles[0]["grasps"] is None


def test_the_page_carries_the_grasp_marker_through_serialization(tmp_path):
    """render_gallery_page's payload dict names its fields explicitly (see id/node/label/size
    above) -- grasps is easy to drop back out of that list, and every other test in this file would
    still pass. True, False, and a kitchen_tiles-shaped tile that carries no "grasps" key at all
    (kitchen_tiles() never sets one, and render_gallery_page is shared by both galleries) all have
    to survive to DATA.tiles -- the last one as JSON null, not a KeyError and not a vanished key."""
    backed = _obj(tmp_path, "backed.obj", (0.1, 0.1, 0.1))
    bare = _obj(tmp_path, "bare.obj", (0.1, 0.1, 0.1))
    scene, tiles = mesh_tiles([backed, bare], grasp_backed={backed})
    tiles.append({"id": "island:0", "node": tiles[0]["node"], "label": "island"})  # no "grasps" key

    page = render_gallery_page(scene, tiles, title="t")
    payload = json.loads(page.split("var DATA = ")[1].split(";\n")[0])

    by_id = {t["id"]: t["grasps"] for t in payload["tiles"]}
    assert by_id[backed] is True
    assert by_id[bare] is False
    assert by_id["island:0"] is None
