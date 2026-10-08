"""The composer's palette IS the registry. A skill the palette offers is, by construction, a skill
the executor has — that is the whole reason the skill contract exists.

Tests the DATA the page builds, never the DOM. Same discipline as test_kitchen_preview.py.

Run: pytest scripts/simvla/test_task_composer.py -v
"""

import json

import pytest

import skills  # noqa: F401
from skill_contract import REGISTRY
from task_composer import palette, param_widgets, template_from_payload
from task_template import TaskTemplateError
from task_validate import SequenceError


def test_the_palette_is_the_registry():
    """Not a hardcoded list. Before the skill contract, the GUI's skill list and the executor's
    disagreed, and G_b could be authored but not run."""
    ids = [box["id"] for box in palette()]
    assert ids == sorted(REGISTRY), "every registered skill, and nothing else"


def test_a_palette_box_carries_what_the_gui_needs_to_render_it():
    box = next(b for b in palette() if b["id"] == "nav.open_articulation")
    assert box["label"] == REGISTRY["nav.open_articulation"].label
    assert box["actions"] == ["N"]
    names = [p["name"] for p in box["params"]]
    assert "back_off_m" in names


def test_param_widgets_come_from_the_declared_params():
    widgets = {w["name"]: w for w in param_widgets("nav.to_prim")}
    assert widgets["prim_path"]["kind"] == "prim_path"
    assert widgets["which_arm"]["kind"] == "choice"
    assert widgets["which_arm"]["default"] == "Both", "a Choice declares its default; it is not options[0]"
    assert widgets["safety"]["kind"] == "float"


def test_no_floor_default_is_json_safe_and_left_unset_in_composed_steps():
    widgets = {w["name"]: w for w in param_widgets("arm.bowl_place")}

    assert widgets["min_eef_z"] == {"name": "min_eef_z", "kind": "float", "default": None}
    json.dumps(palette(), allow_nan=False)
    assert "else if (p.kind === 'float' && p.default == null) return;" in _composer_source()


def test_a_posted_template_is_validated_before_it_is_accepted():
    payload = {
        "name": "broken",
        "language": "x",
        "roles": [{"name": "target", "object_type": "bowl"}],
        "steps": [{"skill": "arm.telekinesis", "action": "A_r", "params": {}}],
        "subtask_groups": [],
    }
    with pytest.raises(TaskTemplateError, match="arm.telekinesis"):
        template_from_payload(payload)


def test_a_posted_template_with_an_incoherent_sequence_is_refused():
    """Closing a gripper on nothing. The composer must not let you save this."""
    payload = {
        "name": "closes_on_air",
        "language": "x",
        "roles": [{"name": "target", "object_type": "bowl"}],
        "steps": [{"skill": "gripper.set", "action": "G_r", "params": {"grasp": True}}],
        "subtask_groups": [],
        "success": {"all": [{"obj_z": {"role": "@target", "lo": 0.5}}]},
    }
    with pytest.raises(SequenceError, match="closes on nothing"):
        template_from_payload(payload)


def test_a_valid_template_round_trips_from_the_payload():
    from scripts.simvla.test_task_template import bowl_to_drawer
    from task_template import to_json
    import json

    payload = json.loads(to_json(bowl_to_drawer()))
    t = template_from_payload(payload)
    assert [s.skill for s in t.steps] == [s.skill for s in bowl_to_drawer().steps]


# ---------------------------------------------------------------------------------------------
# Task 6: the Scene panel. The composer now edits a task's scene (every placed object + its
# geometry). Same discipline: test the DATA the page builds and the HTTP contract, never the DOM.
# ---------------------------------------------------------------------------------------------

def test_scene_defaults_offer_every_object_type_with_its_defaults():
    from task_composer import scene_defaults_payload
    from scene_spec import OBJECT_TYPES
    payload = scene_defaults_payload()
    assert {e["object_type"] for e in payload} == OBJECT_TYPES
    bowl = next(e for e in payload if e["object_type"] == "bowl")
    assert bowl["size"]["center"] == 1.0
    assert bowl["placement"] == "dishwasher"


def test_scene_defaults_carry_the_surface_options_so_the_gui_need_not_hardcode_them():
    from task_composer import scene_defaults_payload
    from scene_spec import SUPPORT_SURFACES
    for entry in scene_defaults_payload():
        assert entry["surface_options"] == sorted(SUPPORT_SURFACES)


def test_a_posted_template_carries_its_scene():
    from task_composer import template_from_payload
    payload = {
        "name": "t", "language": "l",
        "roles": [{"name": "target", "object_type": "bowl"}],
        "steps": [{"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@target"}}],
        "subtask_groups": [],
        "scene": [
            {"name": "bowl0", "object_type": "bowl", "size": {"center": 1.0, "spread": 0.05},
             "placement": "dishwasher", "lift": {"center": 0.775, "spread": 0.025}},
            {"name": "mug0", "object_type": "mug", "size": {"center": 1.3, "spread": 0.0},
             "placement": "island", "lift": None},
        ],
        "success": {"all": [{"obj_z": {"role": "@target", "lo": 0.5}}]},
    }
    t = template_from_payload(payload)
    assert [o.name for o in t.scene] == ["bowl0", "mug0"]
    assert t.scene[1].size.center == 1.3            # a big mug


def test_a_posted_scene_with_a_bad_placement_is_refused():
    from task_composer import template_from_payload
    from scene_spec import SceneError
    import pytest
    payload = {
        "name": "t", "language": "l",
        "roles": [{"name": "target", "object_type": "bowl"}],
        "steps": [{"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@target"}}],
        "subtask_groups": [],
        "scene": [{"name": "bowl0", "object_type": "bowl", "size": {"center": 1.0, "spread": 0.0},
                   "placement": "ceiling", "lift": None}],
    }
    with pytest.raises(SceneError, match="ceiling"):
        template_from_payload(payload)


# ---------------------------------------------------------------------------------------------
# Step 5: verify at the HTTP layer, since we cannot open a browser here. We drive the two new
# endpoints over a real socket and confirm the served HTML was wired — but the true
# click-a-prim-in-3D drive needs a kitchen scene on the GPU box and is left for Task 8.
# ---------------------------------------------------------------------------------------------

import json
import urllib.error
import urllib.request

import trimesh

from task_composer import ComposerServer, build_preview_html, palette


def _toy_scene():
    """A stand-in kitchen: one support part, one placed object — enough to render the page."""
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
    return scene


def _objects():
    return [{"label": "bowl0", "node_id": "bowl00", "location": "Above cabinet"}]


def _post(url, obj):
    request = urllib.request.Request(
        url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_get_palette_is_exactly_the_palette():
    """GET /palette returns every registered skill and nothing else — the JSON IS palette()."""
    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        body = urllib.request.urlopen(server.url + "palette", timeout=5).read().decode()
        assert json.loads(body) == palette()
    finally:
        server.shutdown()


def test_post_template_accepts_a_valid_task_and_stores_it():
    """POST /template with the reference task -> {"ok": true}, and the server holds the template."""
    from scripts.simvla.test_task_template import bowl_to_drawer
    from task_template import to_json

    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        assert server.template is None
        status, body = _post(server.url + "template", json.loads(to_json(bowl_to_drawer())))
        assert status == 200
        assert body == {"ok": True, "warnings": []}
        assert server.template is not None
        assert [s.skill for s in server.template.steps] == [s.skill for s in bowl_to_drawer().steps]
    finally:
        server.shutdown()


def test_post_template_refuses_an_incoherent_sequence_with_a_400_and_the_reason():
    """A gripper.set closing on nothing must come back as 400 with the SequenceError reason in the
    body so the page can show it inline — not a 200, and not a bare 500 stack trace."""
    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        payload = {
            "name": "closes_on_air",
            "language": "x",
            "roles": [{"name": "target", "object_type": "bowl"}],
            "steps": [{"skill": "gripper.set", "action": "G_r", "params": {"grasp": True}}],
            "subtask_groups": [],
            "success": {"all": [{"obj_z": {"role": "@target", "lo": 0.5}}]},
        }
        status, body = _post(server.url + "template", payload)
        assert status == 400
        assert "closes on nothing" in body["reason"]
        assert server.template is None      # a refused task is not stored
    finally:
        server.shutdown()


def test_the_served_page_wires_the_new_endpoints_and_panes(tmp_path):
    """A substring check on the page, not a rendered DOM: the composer script must reference both
    new endpoints and both new panes so a browser would actually drive them."""
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html",
    ).read_text()

    assert "/palette" in page
    assert "/template" in page
    assert "composer-palette" in page
    assert "composer-sequence" in page
    # It still carries the kitchen preview it is built on.
    assert "simvla-panel" in page
    # and it fills prim_path from a picked prim's sanitized node name.
    assert "bowl00geometry_0" in page


def test_get_scene_defaults_is_exactly_the_scene_defaults_payload():
    """GET /scene_defaults returns the 'add object' menu — the JSON IS scene_defaults_payload()."""
    from task_composer import scene_defaults_payload
    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        body = urllib.request.urlopen(server.url + "scene_defaults", timeout=5).read().decode()
        assert json.loads(body) == scene_defaults_payload()
    finally:
        server.shutdown()


def test_post_template_with_a_valid_scene_stores_it():
    """A posted template carrying a scene round-trips through /template and is stored."""
    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        payload = {
            "name": "t", "language": "l",
            "roles": [{"name": "target", "object_type": "bowl"}],
            "steps": [{"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@target"}}],
            "subtask_groups": [],
            "scene": [
                {"name": "bowl0", "object_type": "bowl", "size": {"center": 1.0, "spread": 0.05},
                 "placement": "dishwasher", "lift": {"center": 0.775, "spread": 0.025}},
                {"name": "mug0", "object_type": "mug", "size": {"center": 1.3, "spread": 0.0},
                 "placement": "island", "lift": None},
            ],
            "success": {"all": [{"obj_z": {"role": "@target", "lo": 0.5}}]},
        }
        status, body = _post(server.url + "template", payload)
        assert status == 200
        assert body == {"ok": True, "warnings": []}
        assert [o.name for o in server.template.scene] == ["bowl0", "mug0"]
    finally:
        server.shutdown()


def test_post_template_refuses_a_bad_scene_with_a_400_and_the_reason():
    """A scene placing a bowl on a surface no layout has must come back as 400 with the SceneError
    reason ('ceiling') in the body — not a 200, and not a bare 500 from an uncaught SceneError."""
    server = ComposerServer("<html><body>x</body></html>", port=0)
    try:
        payload = {
            "name": "t", "language": "l",
            "roles": [{"name": "target", "object_type": "bowl"}],
            "steps": [{"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@target"}}],
            "subtask_groups": [],
            "scene": [{"name": "bowl0", "object_type": "bowl",
                       "size": {"center": 1.0, "spread": 0.0},
                       "placement": "ceiling", "lift": None}],
        }
        status, body = _post(server.url + "template", payload)
        assert status == 400
        assert "ceiling" in body["reason"]
        assert server.template is None      # a refused scene is not stored
    finally:
        server.shutdown()


def test_the_served_page_wires_the_scene_panel(tmp_path):
    """A substring check on the page, not a rendered DOM: the composer script must reference the
    /scene_defaults endpoint and the scene container so a browser would drive the Scene panel."""
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html",
    ).read_text()

    assert "/scene_defaults" in page
    assert "composer-scene" in page
    # added objects are named bottle0, bottle1, ... (not bottle0 then bottle01)
    assert "uniqueNameForType" in page
    # the panel makes clear it edits the task's object list, not the 3D reference kitchen
    assert "not a live edit of the" in page


def test_prim_picking_is_guarded_off_the_drag_canvas(tmp_path):
    """The composer's prim-pick and kitchen_preview.bindDrag share one canvas. A prim click must
    fill prim_path WITHOUT the preview reading the same gesture as select+drag and nudging the
    object. The guard is an explicit armed mode whose overlay keeps the pick gesture off the canvas;
    assert the emitted HTML wires it, so deleting the guard fails here rather than only on the GPU.

    This is a substring/wiring check, in the same spirit as the endpoint-wiring test above: it proves
    the guard is present in the page, not that it runs (no browser executes here — that is Task 8)."""
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html",
    ).read_text()

    # the overlay that intercepts the pick gesture so the drag canvas never sees a pointerdown
    assert "composer-pick-overlay" in page
    # the explicit arm/disarm mode the pick handler is gated on
    assert "armPick" in page
    assert "disarmPick" in page
    assert "picking" in page
    # the pick is armed from a button, not from an implicit click that would fight the drag
    assert "composer-pick-btn" in page


def test_a_field_inside_a_draggable_step_can_hold_focus(tmp_path):
    """A step box is draggable (to reorder), and its prim_path / param fields live inside it. Two
    things would otherwise stop the field from holding focus: (a) mousedown on the field starts a
    drag on the parent instead of placing the cursor, and (b) the step's click handler re-renders the
    sequence, destroying the input mid-click. Assert both guards are wired: a pointerdown handler that
    disables dragging over a form control, and a click handler that skips re-render for form controls.

    A substring/wiring check like the prim-pick test above — it proves the guard is in the page, not
    that a browser runs it. This is the fix for the 'can't type in prim_path unless I keep pressing'
    bug the first real launch surfaced."""
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html",
    ).read_text()

    # the shared test: is the click/pointer target one of the step's interactive controls?
    assert "isFormControl" in page
    # (a) dragging is turned off while the pointer is down on a control, so the field can focus
    assert "pointerdown" in page
    assert "el.draggable = !isFormControl" in page
    # (b) a click that lands on a control does NOT re-select+re-render the step (which drops focus)
    assert "if (isFormControl(e.target)) return;" in page


def test_importing_task_composer_boots_neither_omniverse_nor_the_executor_stack():
    """It is the browser composer, not a GPU job: importing it must not pull the executor /
    Omniverse stack — torch, omni, pxr, isaaclab, tkinter. (numpy/trimesh/PIL come in through
    kitchen_preview's 3D rendering and are allowed — the composer legitimately draws the scene.)

    scene_synthesizer is banned too, and it was added late because its absence is what let a real
    regression through: `from scene_spec import SUPPORT_SURFACES` is a PEP 562 hook that derives
    itself from kitchen_build, so naming it in this module's import list pulled scene_synthesizer in.
    The kitchen generator imports this module (through kitchen_wizard) BEFORE Omniverse boots, and
    scene_synthesizer soft-imports pxr — so that one import line silently unbound every USD name for
    the rest of the process and the generator died at its first export. kitchen_wizard's own
    contract test would catch it transitively; this one catches it in the file that causes it.

    In a CLEAN interpreter — not this one, where the test module's `import skills` (line 11) has
    already pulled torch + isaaclab in, which would make an in-process sys.modules diff blind to a
    future `import torch` regression (and did: the in-process version of this test passed while
    task_composer transitively imports PIL, because PIL was already resident). A subprocess is the
    only honest check. Same pattern as test_task_emit.test_importing_it_boots_nothing."""
    import os
    import subprocess
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import task_composer                                # and nothing else

            banned = ("scene_synthesizer", "omni", "isaaclab", "pxr", "torch", "tkinter")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            assert not pulled, f"importing task_composer pulled in {pulled}"
            print("clean")
        """)],
        cwd=here, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]


def test_launching_the_composer_populates_the_palette():
    """The palette IS the skill REGISTRY, which the @skill decorators fill only on `import skills`.
    task_composer stays import-light (no skills at module load — see the boots-nothing test), so on a
    BARE import the palette is empty. The launch path (composer_page, via _ensure_skills_loaded) must
    fill it, or the composer opens with nothing to drag — which is exactly what shipped the first time.

    A subprocess, because THIS test module imports skills at the top (line 11), so an in-process check
    would see a registry that is already populated and be blind to the regression."""
    import os
    import subprocess
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import task_composer                                # and NOT skills

            assert task_composer.palette() == [], "palette should be empty before skills load"
            task_composer._ensure_skills_loaded()              # what composer_page does first
            n = len(task_composer.palette())
            assert n > 0, "palette is still empty after the launch helper — nothing to drag"
            print("palette", n)
        """)],
        cwd=here, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split()[0] == "palette" and int(proc.stdout.split()[1]) > 0


# ---- save_composed_template: the composer's write-to-templates step ------------------------------

def test_save_composed_template_names_the_file_after_the_task(tmp_path):
    """The file is named after the task itself (the user asked for the task's own name), lands in the
    given dir, and round-trips: what we write is exactly the template the composer handed back."""
    from task_composer import save_composed_template
    from task_template import from_json
    from scripts.simvla.test_task_template import bowl_to_drawer

    t = bowl_to_drawer()
    out = save_composed_template(t, templates_dir=tmp_path)

    assert out == tmp_path / "bowl_to_drawer.json"     # named after t.name, not a generic default
    assert out.exists()
    reloaded = from_json(out.read_text())
    assert reloaded.name == t.name
    assert [s.skill for s in reloaded.steps] == [s.skill for s in t.steps]
    assert [o.name for o in reloaded.scene] == [o.name for o in t.scene]


def test_save_composed_template_slugs_a_spacey_name(tmp_path):
    """A human-typed name with spaces and punctuation becomes a filesystem-safe stem, so a task named
    'Put bowl inside drawer.' does not produce 'Put bowl inside drawer..json'."""
    import dataclasses

    from task_composer import save_composed_template
    from scripts.simvla.test_task_template import bowl_to_drawer

    t = dataclasses.replace(bowl_to_drawer(), name="Put bowl inside drawer.")
    out = save_composed_template(t, templates_dir=tmp_path)

    assert out == tmp_path / "put_bowl_inside_drawer.json"


def test_save_composed_template_refuses_a_task_with_no_usable_name(tmp_path):
    """A blank/whitespace name yields no filename — refuse loudly rather than write '.json' or a file
    named after nothing. This is the ValueError the ComposeDialog turns into a 'name it first' prompt."""
    import dataclasses

    import pytest

    from task_composer import save_composed_template
    from scripts.simvla.test_task_template import bowl_to_drawer

    t = dataclasses.replace(bowl_to_drawer(), name="   ")
    with pytest.raises(ValueError):
        save_composed_template(t, templates_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []      # nothing written on refusal


# ---- grasp thumbnails: pick preferred grasps in the browser, not the Tk chooser ------------------

import urllib.parse

_PNG_SIG = bytes.fromhex("89504e470d0a1a0a")   # a PNG file signature; the server serves bytes verbatim


def _thumbs_dir(tmp_path, names=("segment_01_left", "segment_02_left", "segment_01_right")):
    """A stand-in *_segments_thumbnails folder with tiny files named exactly as BODex renders them."""
    folder = tmp_path / "scale010_grasp_segments_thumbnails"
    folder.mkdir()
    for n in names:
        (folder / (n + ".usda.png")).write_bytes(_PNG_SIG + n.encode())
    return folder


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_parse_thumb_name_is_zero_based_like_the_tk_chooser():
    """segment_01 is index 0 — the exact convention isaaclab.simvla.utils.parse_thumbnail_name uses,
    so a browser pick lands on the same eef_data row plan_arm_grasp indexes. Non-thumbnails parse None."""
    from task_composer import parse_thumb_name
    assert parse_thumb_name("segment_01_left.usda.png") == (0, "left")
    assert parse_thumb_name("segment_05_right.usda.png") == (4, "right")
    assert parse_thumb_name("selected_indices.json") is None
    assert parse_thumb_name("../etc/passwd") is None


def test_grasp_thumbs_listing_lists_parseable_pngs_sorted_hand_then_seg(tmp_path):
    """Every parseable PNG, ordered (hand, seg) the way select_thumbnails lays out its grid."""
    from task_composer import grasp_thumbs_listing
    assert grasp_thumbs_listing(_thumbs_dir(tmp_path)) == [
        {"file": "segment_01_left.usda.png", "seg": 0, "hand": "left"},
        {"file": "segment_02_left.usda.png", "seg": 1, "hand": "left"},
        {"file": "segment_01_right.usda.png", "seg": 0, "hand": "right"},
    ]


def test_write_then_read_grasp_selection_round_trips_in_the_cluster_format(tmp_path):
    """What the browser saves is byte-for-byte what select_thumbnails_cached loads: {"selected_indices":
    [[seg, hand], ...]} with 0-based indices — so the headless run reads exactly this back."""
    from task_composer import grasp_selection, write_grasp_selection
    folder = _thumbs_dir(tmp_path)
    assert write_grasp_selection(folder, [[0, "left"], [4, "right"]]) == 2
    assert json.loads((folder / "selected_indices.json").read_text()) == {
        "selected_indices": [[0, "left"], [4, "right"]]
    }
    assert grasp_selection(folder) == [[0, "left"], [4, "right"]]


def test_write_grasp_selection_refuses_outside_a_thumbnails_folder(tmp_path):
    """The guard that keeps a bad key from dropping selected_indices.json into the goals corpus (or
    anywhere that is not a *_segments_thumbnails dir): refuse, and write nothing."""
    import pytest
    from task_composer import write_grasp_selection
    (tmp_path / "goals").mkdir()
    with pytest.raises(ValueError):
        write_grasp_selection(tmp_path / "goals", [[0, "left"]])
    assert not (tmp_path / "goals" / "selected_indices.json").exists()


def test_write_grasp_selection_refuses_a_bad_hand(tmp_path):
    """A hand that is neither 'left' nor 'right' is refused — plan_arm_grasp only indexes those two."""
    import pytest
    from task_composer import write_grasp_selection
    with pytest.raises(ValueError):
        write_grasp_selection(_thumbs_dir(tmp_path), [[0, "middle"]])


def test_grasp_thumb_path_blocks_traversal(tmp_path):
    """Only a real thumbnail basename that lives directly in the folder resolves; '..'/absolute/missing
    all return None, so GET /grasps/thumb can never be talked into serving another file."""
    from task_composer import grasp_thumb_path
    folder = _thumbs_dir(tmp_path)
    assert grasp_thumb_path(folder, "segment_01_left.usda.png") is not None
    assert grasp_thumb_path(folder, "../selected_indices.json") is None
    assert grasp_thumb_path(folder, "/etc/passwd") is None
    assert grasp_thumb_path(folder, "segment_99_left.usda.png") is None   # parses, but no such file


def test_get_grasps_state_lists_thumbs_and_current_selection(tmp_path):
    """GET /grasps/state reports the thumbnails and any existing pick, so a cached object opens
    pre-selected and an uncached one reads as empty."""
    from task_composer import ComposerServer, write_grasp_selection
    folder = _thumbs_dir(tmp_path)
    write_grasp_selection(folder, [[0, "left"]])
    server = ComposerServer("<html><body>x</body></html>", port=0,
                            grasp_thumbs={"bowl00": str(folder)})
    try:
        status, body = _get(server.url + "grasps/state?key=bowl00")
        st = json.loads(body)
        assert status == 200
        assert st["cached"] is True
        assert st["selected"] == [[0, "left"]]
        assert {"file": "segment_01_left.usda.png", "seg": 0, "hand": "left"} in st["thumbs"]
    finally:
        server.shutdown()


def test_get_grasps_state_404s_an_unknown_key():
    """A key the server was never handed is a 404 — the composer only serves folders it knows."""
    from task_composer import ComposerServer
    server = ComposerServer("<html><body>x</body></html>", port=0, grasp_thumbs={})
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(server.url + "grasps/state?key=nope")
        assert exc.value.code == 404
    finally:
        server.shutdown()


def test_get_grasps_thumb_serves_bytes_and_blocks_traversal(tmp_path):
    """A valid thumbnail comes back as image bytes; a traversal / unknown / missing file is a 404."""
    from task_composer import ComposerServer
    folder = _thumbs_dir(tmp_path)
    server = ComposerServer("<html><body>x</body></html>", port=0,
                            grasp_thumbs={"bowl00": str(folder)})
    try:
        status, body = _get(server.url + "grasps/thumb?key=bowl00&file=segment_01_left.usda.png")
        assert status == 200
        assert body.startswith(_PNG_SIG)
        for bad in ("../selected_indices.json", "/etc/passwd", "segment_99_left.usda.png"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                _get(server.url + "grasps/thumb?key=bowl00&file=" + urllib.parse.quote(bad))
            assert exc.value.code == 404, bad
    finally:
        server.shutdown()


def test_post_grasps_select_writes_the_cache_and_updates_state(tmp_path):
    """POST /grasps/select writes the cluster-format cache and the next state read reflects it —
    the whole point: warm the cache from the browser so the headless run never opens the Tk chooser."""
    from task_composer import ComposerServer
    folder = _thumbs_dir(tmp_path)
    server = ComposerServer("<html><body>x</body></html>", port=0,
                            grasp_thumbs={"bowl00": str(folder)})
    try:
        status, body = _post(server.url + "grasps/select",
                             {"key": "bowl00", "selected": [[0, "left"], [0, "right"]]})
        assert status == 200
        assert body == {"ok": True, "count": 2}
        assert json.loads((folder / "selected_indices.json").read_text()) == {
            "selected_indices": [[0, "left"], [0, "right"]]
        }
        _, st = _get(server.url + "grasps/state?key=bowl00")
        assert json.loads(st)["selected"] == [[0, "left"], [0, "right"]]
    finally:
        server.shutdown()


def test_post_grasps_select_404s_an_unknown_key():
    """A selection for a key the server does not know is refused (404), never written blind."""
    from task_composer import ComposerServer
    server = ComposerServer("<html><body>x</body></html>", port=0, grasp_thumbs={})
    try:
        status, body = _post(server.url + "grasps/select", {"key": "nope", "selected": [[0, "left"]]})
        assert status == 404
        assert body["ok"] is False
    finally:
        server.shutdown()


def test_composer_payload_marks_only_objects_with_grasp_data():
    """An object gets grasp: True exactly when the server was handed a thumbnails folder for it — the
    signal the Grasps panel uses to list it (and only it)."""
    from task_composer import _composer_payload
    scene = _toy_scene()
    objs = [{"label": "bowl0", "node_id": "bowl00", "location": "x"}]
    assert _composer_payload(scene, objs, {"bowl00": "/x_segments_thumbnails"})["objects"][0]["grasp"] is True
    assert _composer_payload(scene, objs, {})["objects"][0]["grasp"] is False


def test_the_served_page_wires_the_grasps_panel(tmp_path):
    """A substring/wiring check: the page must reference the Grasps pane and all three grasp endpoints,
    and mark a handed-in object graspable, so a browser would actually drive the picker."""
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html", grasp_thumbs={"bowl00": "/x_segments_thumbnails"},
    ).read_text()
    assert "composer-grasps" in page
    assert "/grasps/state" in page
    assert "/grasps/thumb" in page
    assert "/grasps/select" in page
    assert '"grasp": true' in page      # bowl00 was handed grasp data


def _composer_source():
    from pathlib import Path
    return (Path(__file__).parent / "task_composer.py").read_text(encoding="utf-8")


def test_save_no_longer_hardcodes_an_empty_roles_list():
    """The whole defect: the page always sent roles: [], so every composed task carried
    literal prim paths and arm.grasp refused them at emit."""
    src = _composer_source()
    assert "roles: []," not in src, "save() still hardcodes an empty roles list"
    assert "roles: composerRoles()," in src, "save() does not send the collected roles"


def test_the_page_defines_the_role_registry_helpers():
    src = _composer_source()
    for fn in ("function objectTypeOf(", "function roleForKey(", "function composerRoles("):
        assert fn in src, f"missing {fn!r}"


def test_fill_prim_path_writes_a_role_reference_not_a_raw_key():
    """fillPrimPath used to assign the raw object key ('bottle00'), which exists in no
    emitted kitchen and which arm.grasp rejects regardless.

    Fix round 2 routed the actual write through setPrimPathParam(step, primParam, value) (the choke
    point that also refreshes the condition panel's role dropdowns — see
    test_every_role_set_change_refreshes_the_condition_role_dropdowns), so the literal assignment
    this test originally pinned no longer appears verbatim; the guarantee it protects (a '@role'
    reference, never the raw key) is unchanged and re-pinned against the new call form.
    """
    src = _composer_source()
    assert "setPrimPathParam(step, primParam, key)" not in src, (
        "fillPrimPath still assigns the raw object key"
    )
    assert "setPrimPathParam(step, primParam, '@' + role);" in src


def test_template_from_payload_accepts_roles_the_page_now_sends():
    """The contract between the page's composerRoles() output and Role(**r). Role(**r)
    raises TypeError on an unexpected field name, so the shape must match exactly."""
    from task_composer import template_from_payload

    t = template_from_payload(
        {
            "name": "grasp_bottle",
            "language": "Pick up the bottle.",
            "roles": [{"name": "bottle", "object_type": "bottle"}],
            "steps": [
                {"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@bottle"}},
            ],
            "subtask_groups": [],
            "scene": [],
            "success": {"all": [{"obj_z": {"role": "@bottle", "lo": 0.5}}]},
        }
    )
    assert [r.name for r in t.roles] == ["bottle"]
    assert t.roles[0].object_type == "bottle"
    assert t.roles[0].articulation_with is None
    assert t.roles[0].handle_of is None
    assert t.steps[0].params["prim_path"] == "@bottle"


def _js_function(name):
    """The source of one top-level helper in the page script.

    The script is indented two spaces inside the IIFE, so a helper runs from `function name(` to the
    first line that is exactly `  }` — nested blocks close deeper and cannot end the slice."""
    src = _composer_source()
    start = src.index("function " + name + "(")
    return src[start:src.index("\n  }", start)]


def test_composer_roles_emits_only_the_roles_a_step_actually_references():
    """The registry is never pruned (re-clicking an object must reuse its role), so it remembers roles
    nothing points at any more — a mis-click corrected by a second click, a step removed with the ✕, a
    task saved earlier from the same open dialog. Emitting one sends a Role the page shows nowhere and
    task_template refuses the save naming it, with no escape but a page reload. Assert the filtering,
    not merely that the function exists."""
    body = _js_function("composerRoles")
    assert "referencedRoleNames()" in body, "composerRoles does not consult the steps at all"
    assert "if (refs.has(name))" in body, "composerRoles pushes unconditionally — orphans ride along"

    refs = _js_function("referencedRoleNames")
    assert "sequence.forEach" in refs, "the reference set must be read from the CURRENT sequence"
    assert "step.params" in refs
    assert "charAt(0) === '@'" in refs, "a role reference is a param whose value starts with '@'"
    assert "refs.set(v.slice(1)" in refs, "the '@' must be stripped to get the role name"


def test_no_suffixed_role_name_is_ever_minted():
    """bottle, bottle_2, ... all carried object_type 'bottle', and task_bind matches on type alone: the
    suffix made a name distinction the binder could not act on, so a two-bottle kitchen reported the
    role ambiguous and skipped every rotation. A role name is now exactly the object type."""
    src = _composer_source()
    assert "_' + n" not in src, "the suffix loop is back"
    assert "base + '_'" not in src, "a role name is still being built from a base plus a suffix"
    body = _js_function("roleForKey")
    assert "for (" not in body, "roleForKey still loops to find a free name"
    assert "var name = objectTypeOf(key);" in body, "a role name must be exactly the object type"


def test_a_second_object_of_a_type_is_refused_with_a_flash_not_given_a_role():
    """Clicking a different object whose type already has a role must NOT register a second role and
    must NOT touch the step — it flashes the reason instead, because binding is by type and the task
    could only ever reference one object of that type.

    Fix round 2: the write itself moved from a literal assignment to
    setPrimPathParam(step, primParam, '@' + role) (the condition-panel refresh choke point); the
    marker used to split "guard" from "the actual write" is updated to match, the assertions below
    are unchanged.
    """
    body = _js_function("fillPrimPath")
    guard = body[: body.index("setPrimPathParam(step, primParam, '@' + role);")]

    assert "roleToKey.has(type)" in guard, "no check for a role already registered for this type"
    assert "!keyToRole.has(key)" in guard, "the check must still let the SAME object reuse its role"
    assert guard.index("roleToKey.has(type)") < guard.index("roleForKey(key)"), (
        "the duplicate-type check must run BEFORE a role is minted"
    )
    # it refuses by flashing an error (the page's `true` = bad) and returning, not by writing a role
    refusal = guard[guard.index("roleToKey.has(type)"):]
    assert "flash(" in refusal and "true);" in refusal, "the refusal is silent — no error is shown"
    assert "return;" in refusal, "the refusal does not return; it would fall through and fill the step"
    assert "Binding matches on type" in refusal, "the message does not explain why a name cannot help"


def test_template_from_payload_rejects_an_unknown_role_field():
    """Guards the field names composerRoles() emits: a typo must fail loudly at the door."""
    import pytest as _pytest
    from task_composer import template_from_payload

    with _pytest.raises(TypeError):
        template_from_payload(
            {
                "name": "bad",
                "language": "",
                "roles": [{"name": "bottle", "objecttype": "bottle"}],
                "steps": [],
                "subtask_groups": [],
                "scene": [],
            }
        )


# ---- the condition palette ---------------------------------------------------------------------

from predicate_contract import REGISTRY as PREDICATES
from task_composer import predicate_param_widgets, predicates_palette

GRASP_SUCCESS = {"all": [
    {"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.2}},
    {"obj_z": {"role": "@target", "lo": 0.90}},
]}


def test_the_condition_palette_is_the_predicate_registry():
    assert [b["id"] for b in predicates_palette()] == sorted(PREDICATES)


def test_a_condition_box_carries_its_phase_so_the_page_can_mark_it():
    box = next(b for b in predicates_palette() if b["id"] == "last_subtask")
    assert box["phase"] == "script"


def test_condition_param_widgets_cover_the_new_kinds():
    widgets = {w["name"]: w for w in predicate_param_widgets("obj_near_prim")}
    assert widgets["role"]["kind"] == "role"
    assert widgets["z_override"]["kind"] == "opt_float"
    assert widgets["body"]["kind"] == "text"
    assert widgets["anchor"]["kind"] == "choice"
    assert widgets["xy_only"]["kind"] == "bool"


def _payload(**over):
    p = {
        "name": "grasp_thing",
        "language": "Grasp the bottle.",
        "roles": [{"name": "target", "object_type": "bottle",
                   "articulation_with": None, "handle_of": None}],
        "steps": [
            {"skill": "arm.grasp", "action": "A_r", "params": {"prim_path": "@target"}},
            {"skill": "gripper.set", "action": "G_r", "params": {"grasp": True}},
        ],
        "scene": [{"name": "bottle0", "object_type": "bottle",
                   "size": {"center": 1.0, "spread": 0.02},
                   "placement": "island", "lift": None}],
        "success": GRASP_SUCCESS,
    }
    p.update(over)
    return p


def test_the_payload_carries_the_composed_conditions():
    t = template_from_payload(_payload(retry={"any": [{"robot_fell": {"z": -0.1}}]}))
    assert t.success == GRASP_SUCCESS
    assert t.retry == {"any": [{"robot_fell": {"z": -0.1}}]}


def test_a_payload_with_no_success_is_refused_with_its_reason():
    with pytest.raises(TaskTemplateError, match="success"):
        template_from_payload(_payload(success=None))


def test_a_payload_with_a_bad_condition_is_refused_with_its_reason():
    with pytest.raises(TaskTemplateError, match="nosuchrole"):
        template_from_payload(_payload(success={"all": [{"obj_z": {"role": "@nosuchrole",
                                                                   "lo": 0.9}}]}))


# ---------------------------------------------------------------------------------------------
# Task 5: the two condition panes in the browser. No JS runtime is available here (no node, no
# browser), so there is no way to actually drive the drag-and-drop — these tests instead attack the
# most likely failure mode of hand-editing a ~950-line HTML/JS string: an id one side of the page
# references that the other side no longer declares, and a payload key that has drifted from the
# name template_from_payload actually reads. Both run against the REAL rendered page / real script
# source (build_preview_html / _composer_source), never a hand-copied excerpt of the string.
# ---------------------------------------------------------------------------------------------

import re


def test_every_getelementbyid_target_exists_in_the_markup(tmp_path):
    """Every `document.getElementById('X')` in the served page's script must find a matching
    `id="X"` somewhere in the same page's markup, or the call returns null and the next line
    (almost always `.value`, `.innerHTML = ...`, or `.addEventListener(...)`) throws. Composer's
    top-level statements run once, synchronously, as the inline <script> is parsed, so one throw
    aborts the WHOLE script: not just the broken pane, everything after it in source order too
    (the palette never loads, nothing drags) -- with no console visible to a test here. This is
    exactly the failure mode of renaming an id in the markup but not the matching JS string
    (or vice versa) while hand-editing the panel.
    """
    page = build_preview_html(
        _toy_scene(), _objects(), {"countertop": "Granite"}, ["base_cabinet/countertop"],
        tmp_path / "composer.html",
    ).read_text()

    referenced = set(re.findall(r"getElementById\(\s*['\"]([^'\"]+)['\"]\s*\)", page))
    declared = set(re.findall(r'id=["\']([^"\']+)["\']', page))

    missing = sorted(referenced - declared)
    assert not missing, (
        "JS calls document.getElementById(...) for id(s) with no matching id=\"...\" in the "
        f"markup: {missing}"
    )
    # A regex pair that matched nothing would pass vacuously. Assert it actually found Task 5's
    # own wiring, so a future refactor that breaks the extraction itself fails loudly here.
    assert {"composer-success", "composer-retry", "composer-predicate-palette"} <= declared
    assert {"composer-success", "composer-retry"} <= referenced


def test_the_save_payload_sends_success_and_retry_keys():
    """template_from_payload (task_composer.py) reads exactly payload.get('success') and
    payload.get('retry') -- no other names. A rename on either side of that contract (the JS
    payload key, or the Python read) drops the composed conditions with no error anywhere: the
    server just treats the field as absent, refusing the save ('no success condition') or silently
    accepting a task with no retry. Pins save()'s payload object literal to the exact two key names
    conditionsToSpec builds. Scoped to save()'s own body (via _js_function) rather than a whole-file
    substring search, so a 'success'/'retry' occurrence elsewhere (a comment, another function)
    cannot make this pass by accident -- the cleaner alternative to a file-wide regex.
    """
    body = _js_function("save")
    assert "success: conditionsToSpec(successRows, 'all')," in body, (
        "save()'s payload does not send a 'success' key built from the composed successRows"
    )
    assert "retry: conditionsToSpec(retryRows, 'any')" in body, (
        "save()'s payload does not send a 'retry' key built from the composed retryRows"
    )


def test_every_role_set_change_refreshes_the_condition_role_dropdowns():
    """Fix rounds 1-2: a condition row's role <select> is built once, from composerRoles(), at the
    moment paramInput renders that row -- nothing refreshed it after that on its own. Tracing every
    writer of `step.params[...]` and every mutator of the state composerRoles() reads
    (keyToRole/roleToKey via roleForKey, and referencedRoleNames() which scans every step param
    string for a leading '@') turns up exactly three role-set-changing sites -- bool/choice/float
    param writers are ruled out because none of those three kinds can ever hold an '@role' string
    (a closed enum, a boolean, a number), and addStep/the reorder drag both leave every existing
    param value untouched:

      1. fillPrimPath           -- a prim click mints/references a role
      2. the prim_path <input> inside paramRow -- hand-typing '@role' (the field's own placeholder
         names this a first-class alternative to clicking a prim) does the same thing by keyboard
      3. the step-delete handler nested in renderSequence -- removing a step can drop a role's last
         reference, shrinking referencedRoleNames()

    (1) and (2) are the only two writers of a prim_path-kind step param, so round 2's fix routes
    both through one choke point, setPrimPathParam(step, name, value), rather than two separate
    call sites that a future third writer could bypass by forgetting the call; (3) is a step
    *removal*, not a param write, so it cannot go through that same setter and keeps its own direct
    call (round 1).

    Static, WHEN-is-it-called checks -- the only kind possible with no JS runtime here. Each site is
    isolated (via _js_function, and further slicing for the two nested anonymous closures) so an
    unrelated 'renderConditions();' or 'setPrimPathParam(' elsewhere in the file cannot make any of
    these pass by accident, and removing any single one of the three call sites turns this red.
    """
    # the choke point itself must actually refresh the panel
    setter = _js_function("setPrimPathParam")
    assert "renderConditions();" in setter, (
        "setPrimPathParam no longer refreshes the condition role dropdowns"
    )

    # site 1: a prim click
    fill = _js_function("fillPrimPath")
    assert "setPrimPathParam(step, primParam, '@' + role);" in fill, (
        "fillPrimPath binds/references a role without routing through setPrimPathParam"
    )

    # site 2: hand-typing '@role' into the prim_path field
    param_row = _js_function("paramRow")
    prim_start = param_row.index("} else {  // prim_path")
    prim_end = param_row.index("\n    }", prim_start)
    prim_path_branch = param_row[prim_start:prim_end]
    assert "setPrimPathParam(step, p.name, this.value)" in prim_path_branch, (
        "hand-typing into the prim_path field does not route through setPrimPathParam"
    )

    # site 3: removing a step (not a param write, so it keeps its own direct call)
    seq_body = _js_function("renderSequence")
    del_start = seq_body.index("x.addEventListener('click', function (e) {")
    del_end = seq_body.index("});", del_start)
    delete_handler = seq_body[del_start:del_end]
    assert "renderConditions();" in delete_handler, (
        "removing a step can drop the last reference to a role, but the step-delete handler does "
        "not refresh the condition role dropdowns"
    )


def test_composer_page_forwards_actions_and_loads_the_skills():
    """composer_page is the only launch path now, so it must still populate the registry — an empty
    palette is exactly what shipping without _ensure_skills_loaded caused the first time.

    Uses _toy_scene() rather than a bare trimesh.Scene(): this trimesh version's scene_to_html
    (via scene.export(file_type="glb")) raises "Can't export empty scenes!" for a Scene with no
    geometry at all, which every other page test in this file already avoids by adding at least
    one box. That constraint is orthogonal to what this test checks (actions forwarding + skill
    loading), so it uses the same non-empty stand-in scene as the rest of the suite.
    """
    import task_composer

    scene = _toy_scene()
    page = task_composer.composer_page(
        scene, [], {}, [], actions=[{"id": "done", "label": "Done"}]
    )
    assert "Done" in page
    assert task_composer.palette(), "the registry must be populated by composer_page"
