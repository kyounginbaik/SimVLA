"""Tests for the self-contained kitchen preview page.

Run with: pytest scripts/simvla/test_kitchen_preview.py -v
Needs only trimesh — no Omniverse, no scene_synthesizer, no BODex — EXCEPT for
test_placed_object_is_not_a_direct_child_of_world, which imports scene_synthesizer
(lazily, inside the test) to check a structural premise of the drag-math fix in
kitchen_preview.py, and test_the_anubis_reference_is_posed_by_its_own_config, which reads the
robot's own USD joints in a pxr subprocess (see _run_with_pxr — USD libraries only, no Isaac boot
and no GPU). Both skip where their dependency is missing. kitchen_preview.py itself still imports
only trimesh + stdlib.
"""

import base64
import fnmatch
import itertools
import json
import re
import tempfile
import urllib.request

import numpy as np
import pytest
import trimesh

import kitchen_preview

from kitchen_preview import (
    PreviewServer,
    _column_major_to_matrix,
    auto_open_verdict,
    build_preview_html,
    object_node_names,
    render_page,
    sanitize_three_name,
)

MATERIALS = {"countertop": "Granite_Dark", "cabinet": "Paint_Eggshell_Denim", "floor": "Oak"}


def _scene():
    """A toy stand-in for a generated kitchen: one shell part, two placed objects."""
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(2.0, 0.6, 0.1)),
        node_name="base_cabinet/countertop",
        geom_name="base_cabinet/countertop",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        node_name="bowl00/geometry_0",
        geom_name="bowl00/geometry_0",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(0.05, 0.05, 0.2)),
        node_name="bottle00/geometry_0",
        geom_name="bottle00/geometry_0",
    )
    return scene


def _objects():
    return [
        {"label": "bowl0", "node_id": "bowl00", "location": "Above cabinet"},
        {"label": "bottle0", "node_id": "bottle00", "location": "In refrigerator"},
    ]


def _supports():
    return ["base_cabinet/countertop"]


def test_sanitize_three_name_mirrors_threejs():
    # three.js: name.replace(/\s/g, "_").replace(/[\[\]\.:\/]/g, "")
    assert sanitize_three_name("bowl00/geometry_0") == "bowl00geometry_0"
    assert sanitize_three_name("a b") == "a_b"
    assert sanitize_three_name("x[0].y:z/w") == "x0yzw"
    assert sanitize_three_name("bowl00") == "bowl00"


def test_object_node_names_respects_the_slash_boundary():
    """node 'mug100' belongs to object mug10, NOT to object mug1 — a prefix test would confuse them."""
    scene = trimesh.Scene()
    for name in ("mug10/geometry_0", "mug100/geometry_0"):
        scene.add_geometry(trimesh.creation.box(), node_name=name, geom_name=name)

    assert object_node_names(scene, "mug10") == ["mug10geometry_0"]
    assert object_node_names(scene, "mug100") == ["mug100geometry_0"]


def test_page_is_self_contained(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    assert not re.search(r"<script[^>]+\bsrc\s*=", page), "page pulls in an external script"
    assert not re.search(r"<link[^>]+href=[\"']https?:", page), "page pulls in an external stylesheet"
    assert not re.search(r"\bfetch\s*\(\s*[\"']https?:", page), "page fetches from a remote host"


def test_panel_is_injected_exactly_once(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()
    assert page.count("</body>") == 1
    assert page.count("simvla-panel") >= 1


def test_panel_width_is_a_responsive_clamp_not_a_fixed_pixel(tmp_path):
    """Guards against reverting scripts/simvla/kitchen_preview.py's #simvla-panel back to a bare
    `width:280px`.

    A CSS width has no behaviour a test can exercise -- this only checks the rule text is present
    in the rendered payload, not that the panel actually reads wider on a real browser (that needs
    a human looking at it, at a few viewport sizes). The panel was widened because its content --
    object labels with locations, a material name per group, a joint list -- truncates at a fixed
    280px on anything with a long name; a fixed WIDER number was rejected too, because on a narrow
    laptop screen it would eat into the 3D view the preview exists to show. clamp() is what makes
    it grow on a wide display while never shrinking below the old 280px floor.
    """
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    match = re.search(r"#simvla-panel\{[^}]*\bwidth:([^;}]+)", page)
    assert match, "no width rule found for #simvla-panel"
    width = match.group(1).strip()
    assert width == "clamp(280px, 26vw, 460px)", (
        f"#simvla-panel's width is {width!r}, not the responsive clamp -- a bare fixed pixel "
        "width (e.g. 280px) is the truncation bug this test was added to catch: long object "
        "labels, material names and joint names get cut off with no room to grow on a wide "
        "screen. See the comment above #simvla-panel's width rule in kitchen_preview.py."
    )


def test_object_labels_locations_and_materials_appear(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    for label in ("bowl0", "bottle0"):
        assert label in page
    for location in ("Above cabinet", "In refrigerator"):
        assert location in page
    for group, name in MATERIALS.items():
        assert group in page
        assert name in page


def test_object_nodes_are_embedded_sanitized(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()
    # what three.js will actually see, not the raw trimesh names
    assert "bowl00geometry_0" in page
    assert "bottle00geometry_0" in page


def test_supports_are_embedded_sanitized_in_the_payload(tmp_path):
    """The JS matches mesh ancestor names against DATA.supports (see kitchen_preview.py's
    _PANEL script), so supports must go through the same sanitize_three_name treatment as
    objects — 'base_cabinet/countertop' must show up as 'base_cabinetcountertop', not with
    the slash three.js would have stripped anyway."""
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    match = re.search(r'var DATA = (\{.*?\});\s*\n', page, re.DOTALL)
    assert match, "could not find the DATA payload in the page"
    data = json.loads(match.group(1))

    assert data["supports"] == ["base_cabinetcountertop"]


def test_embedded_glb_decodes_back_to_the_scene(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    match = re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page)
    assert match, "no base64 GLB payload found in the page"

    glb = base64.b64decode(match.group(1))
    reloaded = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")

    names = set(reloaded.geometry)
    assert "bowl00/geometry_0" in names
    assert "base_cabinet/countertop" in names


def test_page_only_fetches_from_its_own_origin(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()
    assert 'fetch("/placements"' in page or "fetch('/placements'" in page
    assert not re.search(r"\bfetch\s*\(\s*[\"']https?:", page)


def test_column_major_to_matrix_transposes_threejs_layout():
    # THREE.Matrix4.toArray() is column-major; translation lands in elements 12,13,14.
    flat = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0.5, 1.5, 0.95, 1]
    assert _column_major_to_matrix(flat) == [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 1.5],
        [0.0, 0.0, 1.0, 0.95],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_column_major_to_matrix_rejects_wrong_length():
    try:
        _column_major_to_matrix([1, 2, 3])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-16-element matrix")


def test_server_serves_the_page():
    server = PreviewServer("<html><body>kitchen</body></html>")
    try:
        body = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "kitchen" in body
    finally:
        server.shutdown()


def test_server_records_posted_placements():
    server = PreviewServer("<html><body>kitchen</body></html>")
    try:
        assert server.placements == {}

        payload = json.dumps(
            {"bowl00": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 1.5, 0.95, 1]}
        ).encode()
        request = urllib.request.Request(
            server.url + "placements",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 204

        matrix = server.placements["bowl00"]
        assert matrix[0][3] == 0.5    # translation is the last COLUMN of a row-major matrix
        assert matrix[1][3] == 1.5
        assert matrix[2][3] == 0.95
        assert matrix[3] == [0.0, 0.0, 0.0, 1.0]
    finally:
        server.shutdown()


def test_placed_object_is_not_a_direct_child_of_world():
    """Structural regression guard for the drag-math fix in kitchen_preview.py (findRoots()).

    What this proves: kitchen_preview.py's JS assumed every draggable object's Object3D
    parent was the gltf 'world' node (identity transform), so Object3D.position could be
    treated as world-space. That assumption was false — a placed object is nested under
    whatever furniture piece it landed on (e.g. a countertop, which itself carries a real
    rotation+translation relative to 'world'). This test builds a real scene_synthesizer
    kitchen_island scene, places one object on a countertop exactly as kitchen_build.py
    does, and asserts (a) the object's trimesh graph node is NOT a direct child of 'world',
    and (b) the furniture it landed on has a non-identity transform relative to 'world' —
    i.e. the parent-frame problem findRoots() now works around (by reparenting draggable
    roots onto 'world' with THREE.Object3D.attach()) is real, not hypothetical.

    What this does NOT prove: it says nothing about kitchen_preview.py's actual behavior.
    The drag/yaw math lives entirely in JS that runs in a browser; there is no headless GL
    stack in this environment to execute it, so the JS fix itself is unverified by any
    test here. This only pins down the trimesh/scene_synthesizer scene-graph shape the
    fix relies on. kitchen_preview.py itself imports only trimesh + stdlib — this test is
    the one exception in this file, and it imports scene_synthesizer lazily so the rest of
    the suite stays usable without it.

    If a future scene_synthesizer/trimesh change flattens placed-object parenting so this
    test fails, that's a green light to reconsider (not blindly remove) the attach()-based
    reparenting in kitchen_preview.py's findRoots() — re-verify against a real scene before
    touching it, the way this test's numbers were verified.
    """
    import scene_synthesizer as synth
    from scene_synthesizer import procedural_scenes as ps
    from scene_synthesizer import utils

    kitchen = ps.kitchen_island(seed=0, counter_height=0.95)
    kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")

    mesh_path = tempfile.mktemp(suffix=".obj")
    trimesh.creation.box(extents=(0.08, 0.08, 0.08)).export(mesh_path)

    kitchen.place_objects(
        obj_id_iterator=utils.object_id_generator("bowl0"),
        obj_asset_iterator=synth.assets.asset_generator(
            itertools.repeat(mesh_path, 1),
            scale=0.1,
            up=(0, 1, 0),
            front=(0, 0, -1),
            origin=("com", "bottom", "com"),
            align=True,
        ),
        obj_support_id_iterator=kitchen.support_generator(support_ids="base_cabinet"),
        obj_position_iterator=utils.PositionIteratorGrid(
            step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04
        ),
        obj_orientation_iterator=utils.orientation_generator_uniform_around_z(
            lower=0.0, upper=0.0
        ),
    )

    graph = kitchen.scene.graph
    assert "bowl00" in graph.nodes, "setup did not actually place bowl00 — test is broken"

    parent = graph.transforms.parents.get("bowl00")
    assert parent is not None, "bowl00 has no parent in the graph — test is broken"
    assert parent != "world", (
        "bowl00 is now a direct child of 'world': the parent-frame problem this test "
        "guards against no longer exists in this scene_synthesizer/trimesh version. "
        "kitchen_preview.py's findRoots() attach()-based reparenting may be unnecessary "
        "now — re-verify against a real scene before simplifying it away."
    )

    # The premise only matters in practice if that parent chain carries a non-identity
    # transform. Confirm the countertop this object landed on actually rotates/translates
    # relative to 'world' (this is the ~90 degree rotation the code review verified).
    countertop_matrix, _ = graph.get("countertop_base_cabinet", "world")
    assert not np.allclose(countertop_matrix, np.eye(4)), (
        "countertop_base_cabinet has an identity transform relative to 'world', so "
        "Object3D.position on its children would already be world-space — the parent-"
        "frame bug this test guards against would not apply here. Re-verify kitchen_"
        "preview.py's findRoots() fix is still needed before relying on this test."
    )


def test_auto_open_trusts_BROWSER_even_over_ssh(monkeypatch):
    """Under VS Code Remote, $BROWSER opens the page on the USER's machine and forwards the
    port — so auto-opening is right even though we are on the far end of an SSH connection."""
    monkeypatch.setenv("BROWSER", "/vscode-server/bin/helpers/browser.sh")
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.delenv("SIMVLA_PREVIEW_OPEN", raising=False)

    opens, _reason = auto_open_verdict()
    assert opens is True


def test_auto_open_refuses_a_bare_x11_forwarded_ssh_session(monkeypatch):
    """The trap: webbrowser.open() here launches the REMOTE box's browser over X11, which has
    no GPU. WebGL fails and the user stares at a blank white page. Hand over the URL instead.

    find_vscode_helper is stubbed out because the dev box this runs on DOES have VS Code
    Remote installed — without the stub we would be testing that machine's setup, not the
    plain-SSH case this is about.
    """
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.delenv("SIMVLA_PREVIEW_OPEN", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.setattr(kitchen_preview, "find_vscode_helper", lambda: None)

    opens, reason = auto_open_verdict()
    assert opens is False
    assert "X11" in reason


def test_auto_open_on_a_local_desktop(monkeypatch):
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SIMVLA_PREVIEW_OPEN", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(kitchen_preview, "find_vscode_helper", lambda: None)

    opens, _reason = auto_open_verdict()
    assert opens is True


def test_SIMVLA_PREVIEW_OPEN_overrides_the_verdict_both_ways(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")   # would otherwise refuse
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setenv("SIMVLA_PREVIEW_OPEN", "1")
    assert auto_open_verdict()[0] is True

    monkeypatch.setenv("BROWSER", "/x")                              # would otherwise open
    monkeypatch.setenv("SIMVLA_PREVIEW_OPEN", "0")
    assert auto_open_verdict()[0] is False


def test_auto_open_finds_the_vscode_helper_without_BROWSER(monkeypatch, tmp_path):
    """$BROWSER is only exported into VS Code's OWN integrated terminal. Run the generator from
    a plain SSH shell on the same box and it is absent — at which point Python's webbrowser
    happily falls back to the remote X11 Firefox, which has no GPU and renders a blank page.
    The helper must be found on disk so the terminal you launched from stops mattering."""
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.delenv("SIMVLA_PREVIEW_OPEN", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")

    helper = tmp_path / "browser.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(kitchen_preview, "find_vscode_helper", lambda: helper)

    opens, reason = auto_open_verdict()
    assert opens is True
    assert "VS Code" in reason


def test_open_refuses_the_x11_browser_rather_than_showing_a_blank_page(monkeypatch):
    """The whole point: over SSH with no VS Code helper, there is no browser that can render.
    open() must return False so the caller prints the URL — NOT launch X11 Firefox."""
    monkeypatch.delenv("BROWSER", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
    monkeypatch.setattr(kitchen_preview, "find_vscode_helper", lambda: None)

    launched = []
    monkeypatch.setattr(kitchen_preview.webbrowser, "open", lambda u: launched.append(u))

    server = PreviewServer("<html><body>x</body></html>")
    try:
        assert server.open() is False
        assert launched == [], "it launched a browser that cannot render WebGL"
    finally:
        server.shutdown()


def test_a_stale_VSCODE_IPC_HOOK_CLI_does_not_shadow_the_live_sockets(monkeypatch):
    """The bug this locks down: a shell held open across a VS Code reconnect carries a STALE
    $VSCODE_IPC_HOOK_CLI. Trying only that one — and giving up when it fails — made the preview
    tell the user to open the URL by hand while 14 live sockets sat right next to it.
    """
    stale, live_a, live_b = "/run/user/1/dead.sock", "/run/user/1/live-a.sock", "/run/user/1/live-b.sock"

    monkeypatch.setattr(
        kitchen_preview, "_ipc_socket_candidates", lambda: [stale, live_a, live_b]
    )
    monkeypatch.setattr(kitchen_preview, "_socket_is_live", lambda p: p != stale)

    assert kitchen_preview._vscode_ipc_sockets() == [live_a, live_b]


def test_ipc_sockets_are_deduplicated():
    """$VSCODE_IPC_HOOK_CLI usually also turns up in the on-disk glob; don't try it twice."""
    import unittest.mock as mock

    same = "/run/user/1/a.sock"
    with mock.patch.object(kitchen_preview, "_ipc_socket_candidates", lambda: [same, same]), \
         mock.patch.object(kitchen_preview, "_socket_is_live", lambda p: True):
        assert kitchen_preview._vscode_ipc_sockets() == [same]


def test_open_via_vscode_gives_up_cleanly_when_no_socket_is_live(monkeypatch, tmp_path):
    helper = tmp_path / "browser.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(kitchen_preview, "find_vscode_helper", lambda: helper)
    monkeypatch.setattr(kitchen_preview, "_vscode_ipc_sockets", lambda: [])

    assert kitchen_preview._open_via_vscode("http://127.0.0.1:8777/") is False


def test_server_answers_the_health_probe():
    """The page pings this on load. Serving the page only proves the browser can READ from the
    generator; placements travel the other way, and a broken write path silently discards every
    drag. The probe is what lets the page say so up front instead of after the work is done."""
    server = PreviewServer("<html><body>kitchen</body></html>")
    try:
        with urllib.request.urlopen(server.url + "ping", timeout=5) as response:
            assert response.status == 204
    finally:
        server.shutdown()


def test_page_probes_the_connection_on_load(tmp_path):
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()
    assert 'fetch("/ping"' in page
    assert "checkConnection()" in page


def test_render_page_without_actions_has_no_action_row():
    """Every caller that is not the wizard must be unaffected."""
    page = render_page(_scene(), _objects(), {}, _supports())
    assert '"actions": []' in page


def test_the_page_marks_the_table_as_floor_dragged():
    """The preview needs to know which node uses the ground-plane branch, and it has to come
    from the payload rather than a hardcoded name in the JavaScript -- otherwise renaming the
    node in kitchen_build silently disables the drag."""
    objects = _objects() + [{"label": "table", "node_id": "table",
                             "location": "On the table", "support": None}]

    html = kitchen_preview.render_page(_scene(), objects, {}, _supports())
    assert '"floor_dragged"' in html, "the payload does not name the floor-dragged nodes"
    assert '"table"' in html


def _floor_dragged(objects, scene=None):
    """The rendered payload's floor_dragged list, read back out of the page."""
    html = kitchen_preview.render_page(scene or _scene(), objects, {}, _supports())
    return json.loads(
        re.search(r"var DATA = (\{.*?\});\s*\n", html, re.DOTALL).group(1)
    )["floor_dragged"]


def test_the_page_marks_every_chair_as_floor_dragged_too():
    """A chair stands on the floor exactly as the table does, so it drags against the ground plane
    rather than raycasting the support meshes -- and it is keyed by LABEL, not node id, because
    that is what `selected` holds. For a chair those two strings differ ("chair_0" vs "chair 0"),
    which is the case the table could never have caught: they are the same string for it."""
    objects = [
        {"label": "table", "node_id": "table", "location": "On the floor", "support": None},
        {"label": "chair 0", "node_id": "chair_0", "location": "On the floor", "support": None},
        {"label": "chair 1", "node_id": "chair_1", "location": "On the floor", "support": None},
    ]
    assert _floor_dragged(objects) == ["table", "chair 0", "chair 1"]


def test_a_fixture_that_merely_contains_a_furniture_name_does_not_drag_on_the_floor():
    """fullmatch, never a substring test -- the same rule and the same reason as
    kitchen_build._is_furniture, which has `chairlift` pinned for exactly this."""
    objects = [
        {"label": "side_table", "node_id": "side_table", "location": "", "support": None},
        {"label": "chairlift", "node_id": "chairlift", "location": "", "support": None},
        # "tabletop" is the near-miss that is not hypothetical: it is a real node in every kitchen
        # with a table, and an anchored-at-the-start match (rather than fullmatch) would sweep it
        # onto the ground plane. So would "chair_0_lamp".
        {"label": "tabletop", "node_id": "tabletop", "location": "", "support": None},
        {"label": "chair_0_lamp", "node_id": "chair_0_lamp", "location": "", "support": None},
    ]
    assert _floor_dragged(objects) == []


def test_an_object_list_built_by_hand_is_not_floor_dragged():
    """The regression this cost a decision to avoid. `support: None` is what BOTH floor-dragged
    entries carry, so "no support" reads like the same statement -- but a caller that builds its
    object list by hand omits the key entirely, and kitchen_usd_load._composer_objects does that
    for every object it loads back out of a committed USD. A `not obj.get("support")` rule would
    put all of them on the ground plane, where pickGround() ignores the surface they were resting
    on. The rule is a node-id test for that reason.
    """
    composer_style = [{"label": "bottle0", "node_id": "bottle0", "location": ""}]
    assert _floor_dragged(composer_style) == []


def _scene_with_a_table():
    """_scene() plus a table, whose top is BOTH an object node and a support node.

    That double membership is not contrived: it is what build_kitchen actually renders. A real
    "island" kitchen with table="dining_long" puts 'tabletop' in DATA.supports (loc 42 places
    objects on it) and in the table object's `nodes` (so the table can be selected and dragged
    on the floor). This toy scene reproduces exactly that overlap without a 6-second build.
    """
    scene = _scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(1.2, 0.8, 0.03)),
        node_name="table/top", geom_name="table/top",
    )
    scene.add_geometry(
        trimesh.creation.box(extents=(0.05, 0.05, 0.7)),
        node_name="table/leg_0", geom_name="table/leg_0",
    )
    objects = _objects() + [
        {"label": "table", "node_id": "table", "location": "On the floor", "support": None}
    ]
    return scene, objects, _supports() + ["table/top"]


def test_a_table_node_is_both_an_object_node_and_a_support_in_the_payload():
    """The precondition classify() has to survive, pinned on the rendered payload.

    Nothing upstream deduplicates these two lists, and neither should: the table is genuinely
    both an object (draggable) and a support (things go on it). Any classification that treats
    "has an owner" and "is a support" as mutually exclusive is therefore wrong about the table.
    """
    scene, objects, supports = _scene_with_a_table()
    data = json.loads(
        re.search(r'var DATA = (\{.*?\});\s*\n', render_page(scene, objects, {}, supports),
                  re.DOTALL).group(1)
    )

    table = next(o for o in data["objects"] if o["label"] == "table")
    assert "tabletop" in data["supports"], "the tabletop is not offered as a drop surface"
    assert "tabletop" in table["nodes"], "the tabletop is not part of the draggable table"


def test_classify_registers_a_support_mesh_even_when_it_belongs_to_an_object():
    """The fix for the payload above: pickSurface() raycasts supportMeshes only, so a support
    that also has an owner label must still land in supportMeshes.

    It did not. classify() tested ownerOf(o) FIRST and only looked at isSupportMesh(o) in the
    owner-less else branch, so once the table joined DATA.objects every table mesh -- 'tabletop'
    included -- went down the object branch and supportMeshes never saw it. A mug at loc 42 then
    had homeSupportNames == {tabletop} with no hit possible under it: the drag silently no-opped
    or dropped the mug through onto some other support. Before the table was user-placed it was
    a Fixture, absent from DATA.objects, and its top WAS a support mesh -- this is a regression
    guard, not a new feature.

    Asserted structurally because the page's JS cannot be executed here: the supportMeshes push
    must not live inside the owner-less branch. shellMeshes must stay in it, or the Solid/Ghost/
    Hidden toggle would start ghosting the objects being dragged.
    """
    body = _panel_js_function("classify")
    else_branch = body[body.index("} else {"):]
    else_branch = else_branch[:else_branch.index("\n      }")]

    assert "shellMeshes.push(o)" in else_branch, (
        "shellMeshes left the owner-less branch -- the shell toggle would now ghost objects too"
    )
    assert "supportMeshes" not in else_branch, (
        "classify() still only registers supports for owner-less meshes, so the table's top "
        "can never be a drop surface"
    )
    assert "supportMeshes.push(o)" in body, "classify() no longer collects support meshes at all"


def test_render_page_with_actions_carries_them_to_the_panel():
    """Pin the rendered DATA.actions payload itself -- the thing the `actions` parameter actually
    controls. sendDecision()'s `simvlaPost` call sits in the unconditional _PANEL script text, so
    asserting its presence would pass even for actions=None; only the payload distinguishes them."""
    page = render_page(_scene(), _objects(), {}, _supports(), actions=[
        {"id": "accept", "label": "Accept & Generate"},
        {"id": "back", "label": "Back to Setup"},
    ])
    match = re.search(r'var DATA = (\{.*?\});\s*\n', page, re.DOTALL)
    assert match, "could not find the DATA payload in the page"
    data = json.loads(match.group(1))
    assert data["actions"] == [
        {"id": "accept", "label": "Accept & Generate", "path": "/answer"},
        {"id": "back", "label": "Back to Setup", "path": "/answer"},
    ]


# ---------------------------------------------------------------------------------------------
# Fix round 1 (coordinator review): postPlacements() always fulfilled, even on failure, so
# buildActions' decision POST was unconditionally reachable after a rejected or unreachable flush
# -- exactly the "accept a scene the generator has not been told about" outcome the panel's own
# comment claims to prevent. These pin the fix's structural shape in the rendered _PANEL source, in
# the same spirit as test_task_composer.py's _js_function-based tests: no browser/DOM here, so the
# guarantee is checked by asserting on the JS text the page actually serves.
# ---------------------------------------------------------------------------------------------

def _panel_js_function(name):
    """The source of one top-level helper inside kitchen_preview._PANEL's <script>.

    Same convention as test_task_composer.py's _js_function: the script is indented two spaces
    inside the IIFE, so a helper runs from `function name(` to the first line that is exactly
    `  }` -- nested blocks close deeper and cannot end the slice. (endDrag is nested one level
    deeper inside bindDrag and is not a top-level function, so its own tests below slice it with a
    distinct delimiter instead of this helper.)
    """
    src = kitchen_preview._PANEL
    start = src.index("function " + name + "(")
    return src[start:src.index("\n  }", start)]


def test_post_placements_rejects_on_a_non_2xx_response():
    """postPlacements() must not just setStatus and quietly fulfil on a rejected flush -- a caller
    that awaits it (sendDecision, below) needs to be able to tell the flush failed so it can refuse
    to send a decision the generator was never told about."""
    body = _panel_js_function("postPlacements")
    non_2xx_branch = body[body.index("Generator rejected the placements"):body.index("NOT saved")]
    assert "throw" in non_2xx_branch, (
        "the non-2xx branch does not throw -- postPlacements() would still fulfil on a rejected flush"
    )


def test_post_placements_rejects_on_a_network_error():
    """Same reasoning as the non-2xx case above: a network error (fetch() itself rejecting) must
    also propagate, not just setStatus and swallow it."""
    body = _panel_js_function("postPlacements")
    network_branch = body[body.index("NOT saved"):]
    assert "throw" in network_branch, (
        "the network-error branch does not rethrow -- postPlacements() would still fulfil"
    )


def test_post_placements_success_path_does_not_throw():
    """Sanity check on the two tests above: the success branch (r.ok) must NOT throw, or a normal
    save would itself look like a failure to sendDecision."""
    body = _panel_js_function("postPlacements")
    success_branch = body[body.index("Placements saved"):body.index("Generator rejected the placements")]
    assert "throw" not in success_branch


def test_send_decision_does_not_post_the_answer_when_the_flush_failed():
    """The decision POST must only be reachable from postPlacements()'s SUCCESS continuation, never
    from a path that also runs when it rejected -- the bug was that the decision was chained off
    ANY settled (fulfilled OR rejected) promise, so a failed flush still sent Accept."""
    body = _panel_js_function("sendDecision")
    assert "postPlacements()" in body
    catch_idx = body.index(".catch(function (e)")
    assert "window.simvlaPost" in body[:catch_idx], "the decision POST is missing from the success path"
    assert "window.simvlaPost" not in body[catch_idx:], (
        "the decision POST is reachable from the failure handler -- a failed flush must not send it"
    )


def test_send_decision_ends_every_path_with_the_row_re_enabled():
    """The row must never leave a click disabled without saying why: both the success continuation
    and the terminal .catch must re-enable the buttons."""
    body = _panel_js_function("sendDecision")
    catch_idx = body.index(".catch(function (e)")
    success_branch = body[:catch_idx]
    failure_branch = body[catch_idx:]
    assert "x.disabled = false" in success_branch, "the success path never re-enables the row"
    assert "x.disabled = false" in failure_branch, (
        "the terminal .catch does not re-enable the row -- a decision-POST network error would "
        "leave every button disabled with no way to tell why"
    )


def test_build_actions_wires_each_button_through_send_decision():
    """buildActions itself should only wire the click -> sendDecision, so the flush/decision/catch
    logic lives in one place a reader (and a test) can reason about on its own."""
    body = _panel_js_function("buildActions")
    assert "sendDecision(" in body


def test_reset_placements_swallows_a_rejected_flush_without_letting_it_go_unhandled():
    """resetPlacements() calls postPlacements() fire-and-forget as its last statement. Now that
    postPlacements() can reject, this call must attach its own .catch or a failed flush becomes an
    unhandled promise rejection on every reset -- setStatus already told the user, there is nothing
    more for this call site to do with the failure."""
    body = _panel_js_function("resetPlacements")
    assert "postPlacements().catch(" in body


def test_end_drag_swallows_a_rejected_flush_without_letting_it_go_unhandled():
    """endDrag() (nested in bindDrag) calls postPlacements() fire-and-forget after releasing the
    drag. Same reasoning as resetPlacements above."""
    src = kitchen_preview._PANEL
    end_drag = src[
        src.index("function endDrag(event) {"):
        src.index("canvas.addEventListener('pointerup', endDrag)")
    ]
    assert "postPlacements().catch(" in end_drag


def test_yaw_change_listener_swallows_a_rejected_flush_without_letting_it_go_unhandled():
    """The yaw slider's 'change' handler used to pass postPlacements directly as the listener. Now
    that postPlacements() can reject, passing it bare would leave an unhandled promise rejection on
    every failed flush -- it must be wrapped so the rejection is caught."""
    src = kitchen_preview._PANEL
    listener = src[
        src.index("document.getElementById('simvla-yaw').addEventListener('change'"):
        src.index("document.getElementById('simvla-reset')")
    ]
    assert "'change', postPlacements)" not in listener, "still passes postPlacements bare as the listener"
    assert "postPlacements().catch(" in listener


def test_the_viewport_is_not_trimeshs_white(tmp_path):
    """The 3D ground is repainted, and the white it replaces is really gone.

    Asserting only "our colour is present" would pass while trimesh's own statement also sat in
    the page -- last writer wins in JS, so the page would still render white. The second assert
    is the one that catches that.
    """
    page = build_preview_html(_scene(), _objects(), MATERIALS, _supports(), tmp_path / "p.html").read_text()

    assert f"scene.background=new THREE.Color('{kitchen_preview.VIEWER_BG}')" in page
    assert "scene.background=new THREE.Color(0xffffff)" not in page


def test_the_background_patch_fails_loudly_if_trimesh_moves_it():
    """A trimesh upgrade must not silently restore the white ground on every page in the run.

    This is the whole reason set_viewer_background raises instead of returning the html
    unchanged: it patches a vendored, minified template nobody here controls.
    """
    with pytest.raises(RuntimeError, match="no longer sets the background"):
        kitchen_preview.set_viewer_background("<html>a template that moved on</html>")


def test_the_viewport_colour_is_overridable_without_editing_code(monkeypatch):
    """SIMVLA_VIEWER_BG is the knob; reading it at call time is what makes it usable."""
    page = kitchen_preview.set_viewer_background(
        "x" + kitchen_preview._TRIMESH_BG_ANCHOR + "y", colour="#808080"
    )
    assert "scene.background=new THREE.Color('#808080');" in page
    assert "0xffffff" not in page


def test_lighting_colours_are_not_repainted_with_the_ground():
    """0xffffff appears four times in trimesh's template; only the ground statement may change.

    Rewriting the bare colour instead of the whole statement would tint the directional lights
    and change how every mesh in the library reads -- a rendering change disguised as a theme one.
    """
    template = "L=new THREE.DirectionalLight(0xffffff,1);" + kitchen_preview._TRIMESH_BG_ANCHOR
    out = kitchen_preview.set_viewer_background(template, colour="#16211c")
    assert "DirectionalLight(0xffffff,1)" in out, "a light colour was repainted"


# --- the scale reference, and the four places it must never reach ------------------------------
#
# The reference exists to answer "is this room the size a room should be", which is a judgement the
# author can only make by looking. Everything below is about the other half: it is a RULER drawn on
# a page, and a ruler that ends up in the scene is a mannequin committed into every kitchen.
#
# It is TWO pieces of geometry, not one -- the robot and its dimension rule -- so every containment
# test below reads kitchen_preview.SCALE_NODE_PREFIX rather than naming either of them, and each one
# carries a positive control that registers BOTH the way the thing under test really reads. Two of
# the four assertions could not have failed when they named only one piece; copying the assertion
# shape onto a new piece without extending the control would reproduce exactly that.
#
# The tests that need a real built kitchen import kitchen_build inside themselves, the same
# lazy-import concession test_placed_object_is_not_a_direct_child_of_world already makes; this
# module and kitchen_preview.py both still import nothing heavier than trimesh at the top.


#: The robot every test below draws unless it is specifically about the choice. The wizard's edit
#: page has no selector and uses this one too, so it is the reference that is most drawn.
_ROBOT = kitchen_preview.DEFAULT_ROBOT


def _reference_page(scene, objects=None, supports=None, robot=_ROBOT):
    """The preview page for `scene` WITH `robot` standing in it."""
    return render_page(scene, objects if objects is not None else _objects(), MATERIALS,
                       supports if supports is not None else _supports(), scale_reference=robot)


def _scale_meshes(robot=_ROBOT):
    """Every mesh `robot`'s scale reference draws, by node name. What a containment control
    registers.

    Read off kitchen_preview._scale_pieces rather than rebuilt here, so that a piece added there is
    covered by every containment test AND by every positive control without anyone remembering to
    extend two lists -- which is the mistake the controls exist to catch.
    """
    return {node: mesh for node, mesh, _offset in kitchen_preview._scale_pieces(robot)}


def _every_scale_name():
    """Every node name EVERY robot in the registry would draw under.

    The containment claims are about the registry, not about one robot: a page draws one robot, but
    the four places a piece must never reach must be clear of all of them, and a fourth robot added
    to ROBOTS has to be covered without anyone remembering to extend a list here.
    """
    return sorted(
        node
        for robot in kitchen_preview.ROBOTS
        for node in (kitchen_preview.robot_node(robot), kitchen_preview.rule_node(robot))
    )


def _page_boxes(page):
    """(reference, kitchen): [(node, world bounds)] for the GLB the page really embeds, split by
    SCALE_NODE_PREFIX. Decoding the page back is the only check available on APPEARANCE-adjacent
    claims here -- there is no GL stack and no JS engine on this box -- but geometry does not need
    one."""
    glb = base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))
    drawn = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")

    reference, kitchen_boxes = [], []
    for node in drawn.graph.nodes_geometry:
        transform, geometry = drawn.graph[node]
        mesh = drawn.geometry[geometry].copy()
        mesh.apply_transform(transform)
        bounds = np.asarray(mesh.bounds, dtype=float)
        (reference if node.startswith(kitchen_preview.SCALE_NODE_PREFIX)
         else kitchen_boxes).append((node, bounds))
    return reference, kitchen_boxes


#: What each robot's livery has to look like once it has been through the GLB and robot_mesh: the
#: (r+g+b)/3 of its dominant colour, and the fraction of vertices that must wear it. MEASURED off
#: the assets, and the point of them is that they are DIFFERENT -- a builder that dropped the
#: materials and painted one flat tone would pass a "some colours are present" check on all three.
#:   Anubis     silver 0xC0C0C0 over 96.2%, near-black trim over 3.8%
#:   RB-Y1      mid grey 0x808080 over 85.0%, then 0x454545, 0x0A0A0A, 0xEBEBEB, 0xFEFEFF
#:   AI Worker  white 0xFFFFFF over 85.5%, dark olive-grey grippers 0x33332B over 9.8%
_LIVERY = {
    "anubis": (0xC0, 0.90),
    "rby1": (0x80, 0.80),
    "aiworker": (0xFF, 0.80),
}


@pytest.mark.parametrize("robot", sorted(kitchen_preview.ROBOTS))
def test_the_drawn_robot_is_the_cached_one_at_the_height_it_was_measured_at(robot):
    """A ruler that is 5 cm out is worse than no ruler. The GLB is the robot's own visible geometry
    normalised back to the height measured on the asset itself (see
    scripts/tools/build_robot_reference.py), so what is checked here is the mesh the page will draw.

    1e-6 m rather than an exact match: glTF stores vertex positions as float32, so the round trip
    through the cache is not bit-exact. The measured errors are 2.6e-8 to 3.7e-7 m.
    """
    mesh = kitchen_preview.robot_mesh(robot)
    if mesh is None:
        pytest.skip(f"no cached {robot} here ({kitchen_preview.robot_reference_path(robot)})")

    want = kitchen_preview.ROBOT_HEIGHTS_M[robot]
    assert abs(mesh.extents[2] - want) < 1e-6, f"the robot is {mesh.extents[2]:.7f} m, not {want}"
    assert abs(mesh.bounds[0][2]) < 1e-9, "the robot does not stand on z = 0"
    # CENTRED IN PLAN by the builder, which is what the dimension line's offset is measured from.
    # Anubis is authored 2.0 m away from its own origin, so this is not a formality.
    for axis in (0, 1):
        centre = (float(mesh.bounds[0][axis]) + float(mesh.bounds[1][axis])) / 2.0
        assert abs(centre) < 1e-6, f"the robot's plan centre is {centre:.4f} m off its own node"


@pytest.mark.parametrize("robot", sorted(kitchen_preview.ROBOTS))
def test_the_drawn_robot_wears_the_assets_own_livery(robot):
    """ONE FLAT COLOUR IS WHY THE ROBOT THIS REPLACES READ AS A MESH RATHER THAN AS A ROBOT. Each of
    these three has its own: Anubis silver, RB-Y1 mid grey, AI Worker white against dark grippers.
    build_robot_reference carries it over from the source USD's per-link materials.

    WHAT IS CHECKED IS WHAT THE PAGE WILL SHIP, not what the builder computed: the colours have to
    survive the GLB, trimesh's glTF loader AND robot_mesh's swap onto a PBR material, and it is
    exactly that last step where a form that exports correctly can still arrive blank.

    THE THREE EXPECTATIONS DIFFER (see _LIVERY), because a check loose enough to pass all three
    would also pass a robot painted one invented tone -- which is the failure this is here for.
    Anubis's is the one to read twice: its USD authors (1e-6, 1e-6, 1e-6) into a material it NAMES
    material_C0C0C0, so a builder that trusted the constant would ship a black silhouette. 0xC0
    here is that name being believed, and _diffuse_rgb records why.
    """
    mesh = kitchen_preview.robot_mesh(robot)
    if mesh is None:
        pytest.skip(f"no cached {robot} here ({kitchen_preview.robot_reference_path(robot)})")

    colours = np.asarray(mesh.visual.vertex_attributes["color"])[:, :3]
    assert len(colours) == len(mesh.vertices), "the livery does not cover every vertex"
    grey = colours.mean(axis=1)
    want, share = _LIVERY[robot]
    dominant = float((np.abs(grey - want) < 4).mean())
    assert dominant > share, (
        f"{100*dominant:.1f}% of {robot} is its own body colour 0x{want:02X}, against the "
        f"{100*share:.0f}% the asset carries -- this is not that robot's livery"
    )
    assert float((np.abs(grey - want) >= 4).mean()) > 0.01, (
        f"{robot} is one flat tone, so its trim -- the part that makes it recognisable -- is gone"
    )
    # And the material it rides on must not tint them away.
    assert list(mesh.visual.material.baseColorFactor) == [255, 255, 255, 255], (
        "a non-white baseColorFactor multiplies the livery down; see robot_mesh"
    )


@pytest.mark.parametrize("robot", sorted(kitchen_preview.ROBOTS))
def test_the_robot_is_turned_to_face_the_reader(robot):
    """A robot drawn in profile is a robot the reader has to orbit the view to recognise, and one
    drawn back-to-front is one nobody recognises at all.

    WHICH WAY EACH ONE FACES IS MEASURED, and by a different measurement per robot, because the
    three assets carry different evidence. build_robot_reference.FacingWitness names, per robot, a
    group of prims whose mean position must sit a stated distance to one side of the robot's own
    plan centre; the builder refuses to cache a robot that fails its own witness, and records the
    margin it measured in the sidecar. This asserts the margin really is a margin -- the
    asymmetry the "+x is the front" claim rests on exists in the geometry, rather than the witness
    having been satisfied by noise -- and then that +x is what ends up pointing at the camera.

    The page's camera stands on the -y side looking toward +y (the derivation is in robot_mesh), so
    the front has to end up at -y. Checked by vertex INDEX rather than by re-deriving the extreme
    point, because a rotation reorders nothing: the vertex that reaches furthest forward in the
    cache must be the one that reaches furthest toward the camera on the page.
    """
    path = kitchen_preview.robot_reference_path(robot)
    if not path.is_file():
        pytest.skip(f"no cached {robot} here ({path})")

    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["facing"] == "+x"
    part, _axis, _sign, margin = sidecar["facing_witness"]
    assert sidecar["facing_witness_m"] >= margin > 0.0, (
        f"{part} sits {sidecar['facing_witness_m']:.4f} m off {robot}'s plan centre, which is not "
        f"the {margin} m asymmetry its facing was pinned by"
    )

    raw = trimesh.load(path, force="mesh")
    drawn = kitchen_preview.robot_mesh(robot)
    assert len(raw.vertices) == len(drawn.vertices)
    assert int(np.argmin(drawn.vertices[:, 1])) == int(np.argmax(raw.vertices[:, 0])), (
        "the cache's +x -- the way it faces -- is not drawn toward -y, where the camera stands"
    )


# ============ Anubis stands in the pose its config spawns, not the one its USD authors ============

#: The height the ANUBIS USD is AUTHORED at, kept here as the thing the drawn robot must NOT be.
#: A posing step that silently did nothing would leave the reference at exactly this number, and
#: every other check in this file would still pass -- which is the whole failure this section is
#: here for. See kitchen_preview.ROBOTS for the posed number and where it comes from.
_ANUBIS_AUTHORED_HEIGHT_M = 1.7159143588633317


def test_the_shipped_anubis_cache_is_the_posed_one_not_the_authored_one():
    """THE ARTIFACT THE PAGE ACTUALLY LOADS, checked without pxr, because the cache on /lustre is
    what gets drawn and a stale one built before the posing step looks fine from every other angle.

    The builder records the pose it drew in beside the GLB, so this is a cheap read rather than a
    rebuild -- test_the_anubis_reference_is_posed_by_its_own_config below is the one that re-derives
    the kinematics from the asset.
    """
    sidecar = kitchen_preview.robot_reference_path("anubis").with_suffix(".json")
    if not sidecar.is_file():
        pytest.skip(f"no cached anubis here ({sidecar})")
    built = json.loads(sidecar.read_text())

    pose = built.get("spawn_pose")
    assert pose is not None, (
        "this Anubis was cached without a spawn pose, so it is drawn in whatever pose its USD "
        "happens to be authored in -- rebuild it"
    )
    assert pose["symbol"] == "ANUBIS_CFG"
    assert pose["config"].endswith("robots/anubis_wheels.py")
    # The fold that makes this pose visible, and the open grippers. Read off the cache rather than
    # off the config so that a cache built against an older config is caught rather than believed.
    assert pose["joint_pos"]["link12_joint"] == pytest.approx(2.356048653)
    assert pose["joint_pos"]["link22_joint"] == pytest.approx(2.356048653)
    assert all(pose["joint_pos"][f"gripper{n}_joint"] == pytest.approx(0.04) for n in (1, 2))
    assert all(pose["joint_pos"][name] == 0.0 for name in
               ("base_prismatic_x_joint", "base_prismatic_y_joint", "base_revolute_z_joint"))

    assert built["authored_height_m"] == pytest.approx(_ANUBIS_AUTHORED_HEIGHT_M, abs=5e-7)
    assert built["source_height_m"] == pytest.approx(
        kitchen_preview.ROBOT_HEIGHTS_M["anubis"], abs=5e-7
    )
    assert built["source_height_m"] < built["authored_height_m"] - 0.4, (
        "folding both elbows to 135 degrees drops this robot half a metre; a cache whose two "
        "heights are close was not posed"
    )
    # And the other two are NOT posed: this was asked for Anubis alone. .get, because a cache built
    # before posing existed has no such key and is exactly what "not posed" means.
    for robot in ("rby1", "aiworker"):
        other = kitchen_preview.robot_reference_path(robot).with_suffix(".json")
        if other.is_file():
            assert json.loads(other.read_text()).get("spawn_pose") is None, (
                f"{robot} was posed too -- only Anubis was asked for, and the other two are drawn "
                "in the pose their own USD is authored in"
            )


#: Re-derives Anubis's pose from the asset and the config, and reports what it MEASURED off the
#: drawn geometry. Runs where pxr exists; see _run_with_pxr for why that is a subprocess.
#:
#: THE TRANSFORMS ARE READ BACK OUT OF THE MESHES, by least-squares over each part's own vertices
#: before and after. That is the point of doing it this way rather than asserting on what
#: spawn_transforms returned: it measures the geometry the page will draw, so a correct FK whose
#: result was applied to the wrong prim -- or to none -- fails here.
_ANUBIS_POSE_PROBE = '''
import json, sys
import numpy as np
import build_robot_reference as brr

usd, config, out = sys.argv[1], sys.argv[2], sys.argv[3]
joint_pos = brr.spawn_pose_of(__import__("pathlib").Path(config), "ANUBIS_CFG")
before, _bbox = brr.convert_usd(usd)
after, _bbox = brr.convert_usd(usd)
brr.apply_pose(after, brr.spawn_transforms(usd, joint_pos))

def recover(link):
    """The rigid transform carrying `link`'s authored vertices onto its posed ones, plus the worst
    residual -- which is the evidence that ONE rigid motion explains the whole part."""
    key = "/" + link + "/"
    a = np.vstack([m.vertices for n, m, _c in before if key in n])
    b = np.vstack([m.vertices for n, m, _c in after if key in n])
    assert len(a) == len(b) > 3, link
    fit, *_ = np.linalg.lstsq(np.hstack([a, np.ones((len(a), 1))]), b, rcond=None)
    matrix = np.eye(4)
    matrix[:3, :] = fit.T
    residual = float(np.abs((np.hstack([a, np.ones((len(a), 1))]) @ fit) - b).max())
    return matrix, residual

def angle_of(matrix):
    return float(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))

LINKS = ["base_link", "arm1_base_link", "arm2_base_link", "link11", "link12", "link13",
         "link14", "link15", "link21", "link22", "link23", "gripper1L", "gripper1R",
         "gripper2L", "gripper2R"]
moved = {link: recover(link) for link in LINKS}

def across(parent, child):
    """The motion `child` gained over `parent`: a conjugate of the joint's own value, so its
    rotation angle is that value and nothing else in the chain can contribute to it."""
    rel = np.linalg.inv(moved[parent][0]) @ moved[child][0]
    return {"angle": angle_of(rel), "shift": float(np.linalg.norm(rel[:3, 3]))}

def gap(link_a, link_b, parts):
    ca = np.vstack([m.vertices for n, m, _c in parts if "/" + link_a + "/" in n]).mean(axis=0)
    cb = np.vstack([m.vertices for n, m, _c in parts if "/" + link_b + "/" in n]).mean(axis=0)
    return float(np.linalg.norm(ca - cb))

def height(parts):
    return brr.z_extent(parts)

def still(link):
    """How far `link`'s furthest vertex moved. The direct way to say "this did not move", with no
    fitted transform in between to carry its own arithmetic noise."""
    key = "/" + link + "/"
    a = np.vstack([m.vertices for n, m, _c in before if key in n])
    b = np.vstack([m.vertices for n, m, _c in after if key in n])
    return float(np.linalg.norm(a - b, axis=1).max())

json.dump({
    "joint_pos": joint_pos,
    "residual": {k: v[1] for k, v in moved.items()},
    "still": {link: still(link) for link in LINKS},
    "delta": {k: {"angle": angle_of(v[0]), "shift": float(np.linalg.norm(v[0][:3, 3]))}
              for k, v in moved.items()},
    "across": {name: across(p, c) for name, p, c in [
        ("arm1_base_link_joint", "arm1_base_link", "link11"),
        ("link11_joint", "link11", "link12"),
        ("link12_joint", "link12", "link13"),
        ("link13_joint", "link13", "link14"),
        ("link14_joint", "link14", "link15"),
        ("link21_joint", "link21", "link22"),
        ("link22_joint", "link22", "link23"),
        ("grippers1", "gripper1L", "gripper1R"),
        ("grippers2", "gripper2L", "gripper2R"),
    ]},
    "finger_gap": {
        "authored": [gap("gripper1L", "gripper1R", before), gap("gripper2L", "gripper2R", before)],
        "posed": [gap("gripper1L", "gripper1R", after), gap("gripper2L", "gripper2R", after)],
    },
    "height": {"authored": height(before), "posed": height(after)},
    "plan_centre": [
        [float((np.vstack([m.bounds for _n, m, _c in p]).min(axis=0)[i]
                + np.vstack([m.bounds for _n, m, _c in p]).max(axis=0)[i]) / 2.0) for i in (0, 1)]
        for p in (before, after)
    ],
}, open(out, "w"))
'''


def _run_with_pxr(script, tmp_path, args):
    """Run `script` in a subprocess that can import pxr, and return its CompletedProcess.

    THE SAME MECHANISM build_robot_reference.reexec_with_pxr uses, and for the same reason
    test_chair_placement._run_with_usd shells out: pxr's extension modules need the loader path set
    BEFORE the interpreter starts, so no amount of sys.path work inside a running pytest reaches it.

    CUDA_VISIBLE_DEVICES IS LEFT ALONE, unlike test_chair_placement's export probe. That one boots
    enough of Omniverse to want the variable gone; this one opens a stage and reads points, and
    inside a SLURM allocation the variable is how the card was handed over.
    """
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    spec = importlib.util.find_spec("isaacsim")
    libs = next(
        (c for root in (spec.submodule_search_locations if spec else [])
         for c in sorted(Path(root).glob("extscache/omni.usd.libs-*"))
         if (c / "pxr").is_dir() and (c / "bin").is_dir()),
        None,
    )
    if libs is None:
        pytest.skip("no isaacsim on this machine, so the asset's own joints cannot be read")

    path = tmp_path / "anubis_pose_probe.py"
    path.write_text(script)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(libs), str(tools), str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"), env.get("LD_LIBRARY_PATH", "")])
    return subprocess.run([sys.executable, str(path), *args], env=env,
                          capture_output=True, text=True, timeout=900)


def test_the_anubis_reference_is_posed_by_its_own_config(tmp_path):
    """THE POSE IS APPLIED, and it is the config's -- measured off the geometry, not off the code.

    The failure this exists for is the quiet one: a build that reads the config, computes a pose and
    then draws the asset exactly as authored anyway. Nothing else in this file would notice. So
    every number below is recovered from the DRAWN vertices -- each link's rigid motion is fitted to
    its own before/after point pairs -- and compared against the joint values in the config.

    WHY A RELATIVE MOTION AND NOT AN ABSOLUTE ONE. The transform a link gains over its parent is
    P . J(q) . P^-1 for some P built out of the chain above it, and a conjugation preserves rotation
    angle. So the angle between a parent's motion and its child's IS that joint's value, with no
    dependence on anything else in the arm -- which makes it a check on ONE number at a time.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[2]
    usd = repo / "source/isaaclab_assets/data/Robots/anubis_simvla.usd"
    config = repo / "source/isaaclab_assets/isaaclab_assets/robots/anubis_wheels.py"
    if not usd.is_file():
        pytest.skip(f"{usd} is not on this machine")

    out = tmp_path / "measured.json"
    done = _run_with_pxr(_ANUBIS_POSE_PROBE, tmp_path, [str(usd), str(config), str(out)])
    assert done.returncode == 0, done.stdout + done.stderr
    measured = json.loads(out.read_text())
    pose = measured["joint_pos"]

    # Read from the config's source, so this is the spawn pose and not a copy of it.
    assert len(pose) == 19, f"ANUBIS_CFG.init_state.joint_pos has {len(pose)} joints"

    # EACH PART MOVED AS ONE RIGID BODY. A part that came out as two links' motions averaged
    # together, or that lost its correspondence, cannot be fitted by a single transform.
    for link, worst in measured["residual"].items():
        assert worst < 1e-6, f"{link}'s motion is not one rigid transform: {worst:.2e} m residual"

    # THE DUMMY MOBILE BASE MOVES NOTHING. base_prismatic_x/y_joint and base_revolute_z_joint are
    # the mobile-base mechanism and the config sets all three to zero, so the robot must stand
    # exactly where its own asset stands. arm1/arm2_base_link hang off base_link by FIXED joints,
    # and link11/link21 by arm1/arm2_base_link_joint, which the config also sets to zero.
    # Asserted on the vertices rather than on a fitted transform: "did not move" is a statement
    # about the geometry, and a least-squares fit over 50k points carries ~1e-8 rad of its own.
    for link in ("base_link", "arm1_base_link", "arm2_base_link", "link11", "link21"):
        assert measured["still"][link] < 1e-9, (
            f"{link}'s geometry moved {measured['still'][link]:.6f} m -- the base mechanism is "
            "being applied as a motion instead of as the zero the config sets it to"
        )
    # And the rest of the robot did move, so the check above is a fact about the base and not
    # about a build that posed nothing at all.
    for link in ("link13", "link15", "gripper1L", "gripper2R"):
        assert measured["still"][link] > 0.05, f"{link} did not move at all"

    # EVERY ARM JOINT TURNED BY ITS OWN CONFIGURED ANGLE, including the 2.356 rad elbow fold that
    # is what makes this pose recognisable at a glance. 1e-6 rad rather than 0: these angles are
    # recovered by least squares off the meshes, and the smallest the config sets is 0.0768 rad.
    for joint, parent_child in {
        "link11_joint": "link12", "link12_joint": "link13", "link13_joint": "link14",
        "link14_joint": "link15", "link21_joint": "link22", "link22_joint": "link23",
    }.items():
        want = abs(pose[joint])
        got = measured["across"][joint]["angle"]
        assert got == pytest.approx(want, abs=1e-6), (
            f"{parent_child} turned {got:.9f} rad about {joint}, which the config sets to "
            f"{pose[joint]} -- a wrong sign, a degree/radian mix-up or an unapplied joint"
        )
    assert measured["across"]["link12_joint"]["angle"] == pytest.approx(2.356048653, abs=1e-6)
    # arm1_base_link_joint is 0.0, so link11 gains nothing over the mast it hangs on.
    assert measured["across"]["arm1_base_link_joint"]["angle"] < 1e-6

    # THE GRIPPERS ARE OPEN, at the 0.04 the config sets, and the two fingers of a hand slide
    # OPPOSITE ways -- so the pair separates by 0.08 and their relative motion carries no rotation.
    for hand, fingers in (("grippers1", ("gripper1_joint", "gripper1R_joint")),
                          ("grippers2", ("gripper2_joint", "gripper2R_joint"))):
        want = sum(abs(pose[f]) for f in fingers)
        assert measured["across"][hand]["shift"] == pytest.approx(want, abs=1e-6), (
            f"{hand}'s fingers separated by {measured['across'][hand]['shift']:.5f} m against the "
            f"{want} m the config's two 0.04 openings come to"
        )
        assert measured["across"][hand]["angle"] < 1e-6, "a finger slide is not a rotation"
    for authored, posed in zip(measured["finger_gap"]["authored"], measured["finger_gap"]["posed"]):
        assert posed > authored + 0.07, (
            f"the fingers went from {authored:.5f} m apart to {posed:.5f} m -- 0.04 has been "
            "applied as a close rather than as the open it is"
        )

    # THE HEIGHT THE DIMENSION LINE QUOTES IS THE POSED ONE.
    assert measured["height"]["authored"] == pytest.approx(_ANUBIS_AUTHORED_HEIGHT_M, abs=5e-7)
    assert measured["height"]["posed"] == pytest.approx(
        kitchen_preview.ROBOT_HEIGHTS_M["anubis"], abs=5e-7
    ), "the drawn robot is not the height kitchen_preview labels it with"

    # AND THE PLAN CENTRE WAS RE-DERIVED, not carried over: folding the arms changes the bounding
    # box, and the builder re-centres on the posed one. 2 mm is small, and it is not zero.
    (_ax, _ay), (px, py) = measured["plan_centre"]
    assert (px, py) != (_ax, _ay)
    assert abs(px - _ax) < 0.01 and abs(py - _ay) < 1e-6


def test_a_checkout_without_the_cached_robot_still_draws_the_dimension_line(tmp_path, monkeypatch):
    """THE COLD-CACHE CASE, and it is not hypothetical: the GLBs live on /lustre and are deliberately
    not in git, exactly like the chair and table libraries. A checkout that has never run
    scripts/tools/build_robot_reference.py must still render every page -- with no robot, not with a
    traceback, and not with a page that has lost its dimension line as well.
    """
    import pathlib

    monkeypatch.setenv(kitchen_preview.ROBOT_REFERENCE_ENV, str(tmp_path / "never_built"))
    for robot in kitchen_preview.ROBOTS:
        assert kitchen_preview.robot_mesh(robot) is None

        page = _reference_page(_scene(), robot=robot)
        payload = json.loads(re.search(r"var DATA = (\{.*\});", page).group(1))
        assert payload["scale_nodes"] == [
            sanitize_three_name(kitchen_preview.rule_node(robot))
        ], "the reference lost its dimension line along with the robot"
        assert len(payload["scale_labels"]) == 1

    # And the assets it would otherwise load are not something a checkout carries.
    repo = pathlib.Path(kitchen_preview.__file__).resolve().parents[2]
    default = pathlib.Path(kitchen_preview.ROBOT_REFERENCE_DIR)
    assert repo not in default.parents and repo != default, (
        f"{default} is inside {repo} -- assets are never committed"
    )


def test_the_reference_is_drawn_into_a_copy_and_never_into_the_callers_scene():
    """The root of all four containment claims. run_wizard holds ONE kitchen across the whole
    preview loop and commits that very scene, so anything added to the scene it hands render_page
    would be committed -- and no later removal can be trusted, because an exception between the add
    and the remove would skip it."""
    scene = _scene()
    before = set(scene.graph.nodes)

    page = _reference_page(scene)

    assert set(scene.graph.nodes) == before, (
        "render_page(scale_reference=<robot>) added a node to the caller's own scene"
    )
    assert not any(n.startswith(kitchen_preview.SCALE_NODE_PREFIX) for n in scene.geometry), (
        "render_page(scale_reference=<robot>) added geometry to the caller's own scene"
    )
    # And every piece really is on the page, or this test would pass on a reference never drawn.
    for node in _scale_meshes():
        assert node in page, f"{node} was never drawn, so this proves nothing about it"


def test_nothing_in_the_scale_reference_reaches_the_objects_list():
    """CONTAINMENT 1. Everything in DATA.objects is selectable, draggable and posted back as a
    placement. A robot in there would be a thing the author can pick up and move onto a
    countertop, and the position of would be sent to the director as a placement to apply. So would
    a dimension line."""
    page = _reference_page(_scene())
    payload = json.loads(re.search(r"var DATA = (\{.*\});", page).group(1))

    labels = [o["label"] for o in payload["objects"]]
    nodes = [n for o in payload["objects"] for n in o["nodes"]] + [
        o["root"] for o in payload["objects"]
    ] + list(payload["floor_dragged"])
    for name in _every_scale_name():
        assert name not in labels
        assert not any(name in n for n in nodes), (name, nodes)
    assert payload["scale_nodes"] == [sanitize_three_name(n) for n in _scale_meshes()], (
        "the payload does not name every piece of the reference, so the JS cannot keep the ones it "
        "does not know about out of shellMeshes -- they would be shell-toggled and permanently "
        "visible behind the checkbox's back"
    )


def test_nothing_in_the_scale_reference_reaches_the_placement_gate():
    """CONTAINMENT 2. fixture_overlaps measures every pair of touching objects and
    furniture_placement_problems refuses the kitchen over them. A robot standing on the floor of a
    kitchen touches the floor, and often a chair -- so a leak here does not merely add noise, it
    makes Accept & Generate refuse a kitchen that is fine.

    WITH ITS POSITIVE CONTROL, because the first version of this test could not have failed.
    scene_synthesizer's collision manager is built from the objects REGISTERED with the kitchen,
    not from a walk of kitchen.scene.graph -- measured: geometry added straight to that graph
    produces no gate pair at all. So the control registers one the way the gate really does see
    (add_object) and asserts the gate names it; only then does the absence above mean anything.

    THE CONTROL COVERS EVERY PIECE, not just the robot -- it reads the same _scale_pieces the page
    draws from. Naming one piece and copying the assertion shape onto the next is exactly how a
    vacuous test gets written a second time: a node name would be asserted absent from a list it
    could never have been in, and the test would report that as containment.

    AND EVERY ROBOT, not just the drawn one: a page carries one robot, but the gate must be clear of
    the whole registry, so the absence is asserted over _every_scale_name(). The control registers
    only the pieces this page actually drew -- a control has to register a real mesh, and loading
    all three robots would put 1.3 M faces through the collision manager to prove the same thing.
    """
    import scene_synthesizer as synth

    from kitchen_build import build_kitchen, fixture_overlaps, furniture_placement_problems

    kitchen, _data, objects, supports = build_kitchen("island", [], [], seed=0, table="dining_long")
    _reference_page(kitchen.scene, objects, supports)

    pairs = fixture_overlaps(kitchen)
    problems = furniture_placement_problems(kitchen)
    for name in _every_scale_name():
        assert not any(name in n for pair in pairs for n in pair), (name, pairs)
        assert not any(name in p for p in problems), (name, problems)

    leaked, _d, _o, _s = build_kitchen("island", [], [], seed=0, table="dining_long")
    for name, mesh in _scale_meshes().items():
        leaked.add_object(synth.assets.TrimeshAsset(mesh), name)
    control = fixture_overlaps(leaked)
    seen = {n for pair in control for n in pair}
    for name in _scale_meshes():
        assert any(name in n for n in seen), (
            f"a {name} deliberately registered with the kitchen did not reach the gate either, so "
            f"the assertion above is vacuous for it"
        )


def test_nothing_in_the_scale_reference_reaches_the_support_labels():
    """CONTAINMENT 3. _label_supports is what decides where a dragged object may be dropped and
    what the composer offers as a surface. The robot in that list would be a place to put a mug,
    and the flat tick on top of its dimension line would be a shelf at head height.

    Not vacuous, unlike the two above: match_support_nodes walks kitchen.scene.graph.nodes
    directly, so a node added to that graph really can end up here -- each piece is kept out by its
    name matching no PLACEMENTS pattern as well as by never being added.
    """
    from kitchen_build import PLACEMENTS, _label_supports, build_kitchen

    kitchen, _data, objects, supports = build_kitchen("island", [], [], seed=0)
    _reference_page(kitchen.scene, objects, supports)

    labelled = _label_supports(kitchen, "island")
    assert labelled, "this kitchen has no supports at all, so the test proves nothing"
    # The second belt: even added to the graph, these names are not something a placement can
    # select. Checked against SCALE_NODE_PREFIX + a wildcard as well as against each real name, so
    # a pattern broad enough to sweep in a future piece fails here rather than in a built kitchen.
    patterns = [p for placement in PLACEMENTS for p in placement.patterns]
    for name in _every_scale_name() + [kitchen_preview.SCALE_NODE_PREFIX + "anything_later"]:
        assert not any(name in s for s in labelled), (name, labelled)
        assert not any(fnmatch.fnmatchcase(name, p) for p in patterns), (
            f"{name} matches a placement pattern, so adding it to a kitchen's graph would make it "
            "a support surface"
        )


def test_every_id_the_preview_panel_asks_for_is_in_the_page_that_asks_for_it():
    """The failure a markup change produces and nothing else here can see: rename or drop an id and
    the control it belongs to goes dead silently. No JS engine exists on this box -- node, deno and
    bun are all absent -- so reading the rendered page back is the substitute. The wizard's own
    pages have test_every_getelementbyid_resolves_on_the_page_that_asks_for_it; this is that rule
    for the page kitchen_preview builds.

    The reference's own selector is created in JS rather than written into the markup (a page
    rendered without a reference must not carry a dead control), so it is `simvla-figure-row` -- the
    host the selector is appended to -- that has to exist.
    """
    import pathlib

    source = pathlib.Path(kitchen_preview.__file__).read_text()
    wanted = set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", source))
    assert wanted, "no getElementById calls found -- this test has lost its subject"

    for page in (render_page(_scene(), _objects(), MATERIALS, _supports()), _reference_page(_scene())):
        present = re.findall(r'\bid="([A-Za-z0-9_-]+)"', page)
        asked = {i for i in wanted if f"getElementById('{i}')" in page}
        missing = sorted(i for i in asked if i not in set(present))
        assert not missing, (
            f"the panel's script calls getElementById for {missing}, which the page does not "
            "contain -- that control is dead and no other test can see it"
        )


def test_the_reference_is_kept_out_of_the_controls_that_would_own_it():
    """The browser half of containment. The shell radio buttons set `visible = true` on everything
    they own and the drag raycaster picks from what they own, so the reference has to be separated
    from both at classify() time -- otherwise switching to Solid would own it and a drag could land
    a mug on the robot's shoulder.

    The label sprite goes on the same list for the same reason: it must not be selectable or
    shell-toggled either.
    """
    page = _reference_page(_scene())

    assert "if (isFigureMesh(o)) {" in page, "reference meshes are not separated from the shell"
    assert "figureMeshes.push(o)" in page
    assert "figureMeshes.push(sprite)" in page, "the dimension label is not separated either"


def test_no_robot_is_embedded_until_one_is_chosen():
    """THE COST IS OPT-IN, and this is the assertion that keeps it so. Each robot is 5-13 MB of GLB
    -- 8-18 MB of page -- against 0.76 MB for the same page with none. The author who never opens
    the selector must pay none of that, on every step of the wizard.

    Both no-robot states are checked: scale_reference=None (every caller outside the wizard's
    preview, which must get exactly what it got before this existed, dead control included) and
    scale_reference="" (the wizard's preview before a pick, which offers the selector and nothing
    else).
    """
    bare = render_page(_scene(), _objects(), MATERIALS, _supports())
    offered = render_page(_scene(), _objects(), MATERIALS, _supports(), scale_reference="")

    for page in (bare, offered):
        payload = json.loads(re.search(r"var DATA = (\{.*\});", page).group(1))
        assert payload["scale_nodes"] == []
        assert payload["scale_labels"] == []
        assert payload["robot"] == ""
        assert kitchen_preview.SCALE_NODE_PREFIX not in page
        assert len(page) < 2_000_000, f"{len(page)} bytes with no robot chosen"

    # The difference between the two is the selector, and nothing else.
    assert json.loads(re.search(r"var DATA = (\{.*\});", bare).group(1))["robot_choices"] == []
    choices = json.loads(re.search(r"var DATA = (\{.*\});", offered).group(1))["robot_choices"]
    assert [c["id"] for c in choices] == list(kitchen_preview.ROBOTS)


def test_choosing_a_robot_is_answered_by_the_server_and_carries_only_that_robot():
    """THE TOGGLE IS SERVER-SIDE, which is what makes one-at-a-time possible at all: three robots
    embedded together would be 26 MB of page. So the selector POSTs `robot:<id>` back through the
    same sendDecision() the button row uses, and the caller re-renders this page.

    Checked on the bytes: each robot's page carries its own two nodes and NEITHER of the other two
    robots' -- a client-side toggle would show all six here.
    """
    for robot in kitchen_preview.ROBOTS:
        page = _reference_page(_scene(), robot=robot)
        payload = json.loads(re.search(r"var DATA = (\{.*\});", page).group(1))
        assert payload["robot"] == robot
        assert payload["scale_nodes"] == [
            sanitize_three_name(n) for n in _scale_meshes(robot)
        ]
        for other in kitchen_preview.ROBOTS:
            if other != robot:
                assert kitchen_preview.robot_node(other) not in page, (
                    f"the {robot} page also carries {other} -- the choice is not server-side"
                )
        assert "sendDecision({id: 'robot:' + select.value})" in page, (
            "the selector does not post its pick back, so nothing re-renders"
        )

    with pytest.raises(ValueError):
        render_page(_scene(), _objects(), MATERIALS, _supports(), scale_reference="h2")


def test_the_reference_stands_in_the_open_and_wholly_inside_the_room():
    """Where it stands is the difference between a ruler and a bug report. The middle of a kitchen
    is where the table is, and the clearest square of the raw bounding box is a CORNER -- which
    puts the reference half outside a room whose walls are not added until commit.

    The demand is on the WHOLE assembly, not the anchor point: an inset sized for the robot alone
    would hang its dimension line out through the wall.
    """
    from kitchen_build import build_kitchen

    x_lo, x_hi, y_lo, y_hi = kitchen_preview.scale_reference_extents(_ROBOT)
    half = ((x_hi - x_lo) / 2.0, (y_hi - y_lo) / 2.0)

    for layout in ("island", "l_shaped", "u_shaped", "single_wall", "peninsula"):
        kitchen, _data, _objects_, _supports_ = build_kitchen(
            layout, [], [], seed=0, table="dining_long"
        )
        spot = kitchen_preview.clearest_floor_spot(kitchen.scene, robot=_ROBOT)
        low, high = kitchen.scene.bounds
        for axis in (0, 1):
            inside = min(spot[axis] - low[axis], high[axis] - spot[axis])
            assert inside >= half[axis], (
                f"{layout}: the reference's centre is {inside:.2f} m from the edge of the room on "
                f"axis {axis}, and the assembly reaches {half[axis]:.2f} m -- it hangs outside"
            )


# --- the dimension lines, and the numbers they claim -------------------------------------------


def test_a_dimension_rule_measures_exactly_what_it_says():
    """The whole point of the line. A rule labelled 1.26 m whose geometry is 1.27 m is worse than
    no rule at all, and that is precisely what centring a tick on each end produces -- the ticks
    are laid inside the span instead.

    Checked at both recorded robot heights, because the exactness has to be a property of
    dimension_rule and not of the one number the page happens to draw it at.
    """
    for height in kitchen_preview.ROBOT_HEIGHTS_M.values():
        rule = kitchen_preview.dimension_rule(height)
        assert abs(rule.bounds[0][2]) < 1e-12, f"the {height} m rule does not start at z = 0"
        assert abs(rule.extents[2] - height) < 1e-12, (
            f"the rule for {height} m measures {rule.extents[2]:.6f} m"
        )
        # A line, not a post: it must not read as a piece of furniture standing in the kitchen.
        assert rule.extents[0] < 0.2 and rule.extents[1] < 0.05
        assert len(rule.faces) < 100, f"{len(rule.faces)} faces for three boxes"


def test_the_robot_heights_are_the_measured_ones():
    """A wrong robot height in a scale reference is worse than no robot, so these are pinned here
    as well as documented at the constant. Each is the z extent of the VISIBLE geometry of the exact
    USD the project's own robot config spawns, measured over that stage's default prim -- see
    ROBOTS for which config chose which asset and why each qualifier is load-bearing.

    An entry appearing here without a measurement behind it is the failure this guards, so every
    height is also checked against the sidecar the builder wrote beside its cache.
    """
    assert set(kitchen_preview.ROBOT_HEIGHTS_M) == {"anubis", "rby1", "aiworker"}
    # Anubis in the pose ANUBIS_CFG spawns it in -- 1.715914 m is the same asset UNPOSED, and is
    # asserted against in test_the_shipped_anubis_cache_is_the_posed_one_not_the_authored_one.
    assert kitchen_preview.ROBOT_HEIGHTS_M["anubis"] == pytest.approx(1.1942683, abs=5e-7)
    assert kitchen_preview.ROBOT_HEIGHTS_M["rby1"] == pytest.approx(1.470000, abs=5e-7)
    assert kitchen_preview.ROBOT_HEIGHTS_M["aiworker"] == pytest.approx(1.611375, abs=5e-7)
    assert kitchen_preview.DEFAULT_ROBOT in kitchen_preview.ROBOTS

    for robot, height in kitchen_preview.ROBOT_HEIGHTS_M.items():
        sidecar = kitchen_preview.robot_reference_path(robot).with_suffix(".json")
        if not sidecar.is_file():
            continue
        measured = json.loads(sidecar.read_text())
        assert measured["source_height_m"] == pytest.approx(height, abs=5e-7), (
            f"{robot} is drawn at {height} m but its cache was built from an asset measuring "
            f"{measured['source_height_m']} m"
        )
        # RB-Y1's 1.470000 m really is exact to seven digits, so the old "no measurement is a round
        # number" check would fail on it. What is asserted instead is the thing that check was
        # after: the number came off THIS asset, not out of a datasheet.
        assert measured["height_m"] == pytest.approx(height, abs=5e-7)


def test_the_dimension_line_is_labelled_with_the_height_it_is_drawn_at():
    """A dimension line with no number on it is a stick, and a number that is not the one the
    geometry measures is worse than no line. The label is checked against ROBOT_HEIGHTS_M rather
    than against the string it was written as, and it is placed ON its rule: the sprite carries no
    leader line, so a label anywhere else annotates nothing.
    """
    for robot, spec in kitchen_preview.ROBOTS.items():
        page = _reference_page(_scene(), robot=robot)
        payload = json.loads(re.search(r"var DATA = (\{.*\});", page).group(1))
        labels = payload["scale_labels"]
        height = kitchen_preview.ROBOT_HEIGHTS_M[robot]

        assert len(labels) == 1, [item["text"] for item in labels]
        label = labels[0]
        assert spec["label"] in label["text"] and f"{height:.2f}" in label["text"], label["text"]

        reference = _page_boxes(page)[0]
        rule = next(box for node, box in reference if node == kitchen_preview.rule_node(robot))
        assert rule[0][0] - 1e-6 <= label["at"][0] <= rule[1][0] + 1e-6, (
            f"{robot}: the label sits at x = {label['at'][0]:.3f}, off its own rule at "
            f"{rule[:, 0].tolist()}"
        )
        # Above the top of the rule, and by less than the plaque is tall: any further and it reads
        # as belonging to nothing, any lower and it sits on the rule's own top tick.
        assert 0.0 < label["at"][2] - height < kitchen_preview._LABEL_WIDTH_M * 0.25 * 2

        # AND THE LINE STANDS CLEAR OF THE ROBOT, which one constant cannot do for three robots
        # 0.58, 0.70 and 0.71 m across -- so the offset is measured off the robot being drawn.
        # The G1 this replaced was 0.39 m across and its 0.32 m constant would stand the line
        # inside two of these three.
        drawn = kitchen_preview.robot_mesh(robot)
        if drawn is not None:
            body = next(box for node, box in reference
                        if node == kitchen_preview.robot_node(robot))
            assert rule[0][0] > body[1][0], (
                f"{robot}: the dimension line starts at x = {rule[0][0]:.3f}, inside a robot that "
                f"reaches {body[1][0]:.3f}"
            )


def test_the_labels_are_drawn_by_the_one_shared_sprite_helper():
    """There is no text primitive in a trimesh scene, and there is now exactly ONE answer to that in
    this run: kitchen_preview.SPRITE_LABEL_JS, a canvas texture on a camera-facing sprite. The
    gallery drew its tile labels that way first; the preview's dimension lines reuse it rather than
    being a third mechanism.

    Asserted as "the same string reaches both pages", which is the only form of this claim a test
    without a JS engine can make -- and the form that fails if somebody pastes a second copy.
    """
    import kitchen_gallery

    helper = kitchen_preview.SPRITE_LABEL_JS
    assert "CanvasTexture" in helper and "THREE.Sprite" in helper

    scene, tiles = kitchen_gallery.kitchen_tiles(
        [("island", 0, trimesh.Scene(trimesh.creation.box(extents=(1.0, 1.0, 1.0))))]
    )
    gallery = kitchen_gallery.render_gallery_page(
        scene, tiles, title="t", labels_in_view=True
    )
    preview = _reference_page(_scene())

    for name, page in (("gallery", gallery), ("preview", preview)):
        assert helper in page, f"the {name} page does not carry the shared sprite helper"
        assert page.count("new THREE.Sprite(") == 1, (
            f"the {name} page builds a sprite somewhere other than the shared helper"
        )
        assert "__SPRITE_LABEL__" not in page, f"the {name} page left the placeholder unfilled"


def test_the_drawn_reference_measures_what_it_claims_and_stands_clear_of_the_furniture():
    """The end-to-end check, on the bytes the browser will actually decode.

    Three claims: the robot is drawn at exactly the height measured on the USD IsaacLab spawns, its
    dimension line is exactly the same height, and nothing the reference draws overlaps a piece of
    the kitchen in plan. The last one is what clearest_floor_spot exists for, and it is checked
    against the assembly's real footprint rather than against the anchor point it returns.

    1e-5 m rather than an exact match: the GLB stores vertices as float32 (the robot's measured
    error is 3.7e-8 m; the rule is built here and is exact to 1e-12 before the encode).
    """
    from kitchen_build import build_kitchen

    # Every layout for the robot the edit page also draws, and every ROBOT on one layout: the five
    # layouts are what clearest_floor_spot varies over, the three robots are what the assembly's
    # footprint varies over, and fifteen real kitchen builds to cross the two would be four minutes
    # to say what these eight already say.
    cases = [(kitchen_preview.DEFAULT_ROBOT, layout) for layout in
             ("island", "l_shaped", "u_shaped", "single_wall", "peninsula")]
    cases += [(robot, "island") for robot in kitchen_preview.ROBOTS
              if robot != kitchen_preview.DEFAULT_ROBOT]

    for robot, layout in cases:
        height = kitchen_preview.ROBOT_HEIGHTS_M[robot]
        want = {kitchen_preview.robot_node(robot): height,
                kitchen_preview.rule_node(robot): height}
        if kitchen_preview.robot_mesh(robot) is None:
            want.pop(kitchen_preview.robot_node(robot))

        kitchen, _data, objects, supports = build_kitchen(
            layout, [], [], seed=0, table="dining_long"
        )
        reference, kitchen_boxes = _page_boxes(
            _reference_page(kitchen.scene, objects, supports, robot=robot)
        )

        assert {n for n, _b in reference} == set(want), sorted(n for n, _b in reference)
        assert kitchen_boxes, f"{layout}: no kitchen geometry, so the clearance proves nothing"

        for node, box in reference:
            assert abs(box[0][2]) < 1e-5, f"{layout}: {node} does not stand on the floor"
            assert abs(box[1][2] - want[node]) < 1e-5, (
                f"{layout}: {node} is drawn {box[1][2]:.6f} m tall, not {want[node]:.6f} m"
            )
            for other, box2 in kitchen_boxes:
                overlap = all(
                    box[0][axis] < box2[1][axis] and box2[0][axis] < box[1][axis]
                    for axis in (0, 1)
                )
                assert not overlap, (
                    f"{layout}: {node} stands in the footprint of {other} -- "
                    f"{box[:, :2].tolist()} against {box2[:, :2].tolist()}"
                )
