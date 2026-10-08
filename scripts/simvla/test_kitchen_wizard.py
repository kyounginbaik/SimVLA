"""The wizard's server contract, exercised over real HTTP.

Same discipline as test_task_composer.py: a live server on port 0, requests through urllib, and
assertions on the DATA the pages are built from — never on the DOM.

Run: ~/miniconda3/envs/env_isaaclab/bin/python -m pytest scripts/simvla/test_kitchen_wizard.py -v
"""

import base64
import json
import pathlib
import re

import trimesh

import kitchen_wizard
import sys
import threading
import time
import urllib.error
import urllib.request

from kitchen_wizard import WizardServer, inject_step_script


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read() or b"{}")


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def test_the_boot_page_follows_the_run_like_every_published_page():
    """The generator constructs this server BEFORE Omniverse boots (~1 min) precisely so the URL and
    the ssh -L line are on screen while it loads — which makes a tab opened on the boot page the
    normal case, not an edge one. Without the step script that tab polls nothing: it sits on
    "Starting…" until someone refreshes by hand, and the head start buys nothing.
    """
    server = WizardServer(port=0)
    try:
        boot = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "SIMVLA_STEP" in boot and "/step" in boot, "the boot page carries no step script"
        assert server.step["id"] == 0 and re.search(r"SIMVLA_STEP\s*=\s*0\b", boot), (
            "the boot page must claim the step it is actually serving, or its first POST is stale"
        )

        server.publish("form", "<html><body>the form</body></html>")
        assert _get(server.url + "step")[1]["id"] == 1, (
            "the poller on the boot page would never see a new step id"
        )
        assert "the form" in urllib.request.urlopen(server.url, timeout=5).read().decode()
    finally:
        server.shutdown()


def test_publish_serves_the_page_and_bumps_the_step():
    server = WizardServer(port=0)
    try:
        first = server.publish("form", "<html><body>one</body></html>")
        assert server.step == {"id": first, "kind": "form"}
        body = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "one" in body

        second = server.publish("progress", "<html><body>two</body></html>")
        assert second == first + 1
        assert server.step["kind"] == "progress"
        assert "two" in urllib.request.urlopen(server.url, timeout=5).read().decode()
    finally:
        server.shutdown()


def test_get_responses_stay_internally_consistent_while_pages_are_swapped():
    """Before the wizard, `_page` was assigned once in __init__ and never touched again, so
    do_GET reading it twice (once for Content-Length, once for the body) was safe. publish() now
    reassigns `_page` repeatedly while the server is live, and the pages can be multi-MB trimesh
    documents, so a concurrent publish() can swap `_page` between those two reads. Every response
    must still be internally consistent: its Content-Length must match the bytes actually
    written, and its body must be exactly one published page -- never a mix of two.

    The window between those two reads is one Python-level statement wide, so hitting it under
    default GIL scheduling is luck. `sys.setswitchinterval` is the standard, black-box way to
    make CPython actually take that chance: it does not touch do_GET, it just makes every thread
    (readers and the publisher below) yield the GIL far more often, so whatever interleaving is
    *possible* is overwhelmingly likely to occur within a bounded number of iterations -- and
    the interval is restored afterwards so it cannot affect any other test.
    """
    server = WizardServer(port=0)
    try:
        short_marker = "S" * 200
        long_marker = "L" * 200_000
        pages = {
            "short": f"<html><body>{short_marker}</body></html>",
            "long": f"<html><body>{long_marker}</body></html>",
        }
        server.publish("form", pages["short"])

        errors = []

        def hammer():
            for _ in range(60):
                try:
                    with urllib.request.urlopen(server.url, timeout=10) as response:
                        declared = int(response.headers["Content-Length"])
                        body = response.read()
                except Exception as exc:  # a failure here would itself be the bug in question
                    errors.append(str(exc))
                    continue
                if len(body) != declared:
                    errors.append(f"Content-Length said {declared} but read {len(body)} bytes")
                    continue
                text = body.decode()
                is_short = short_marker in text
                is_long = long_marker in text
                if is_short == is_long:  # neither page matched, or (impossibly) both did
                    errors.append("body matched neither or both of the published pages")

        readers = [threading.Thread(target=hammer) for _ in range(6)]

        # publish() itself does no I/O, so a fixed handful of calls finishes in microseconds --
        # long before the readers above are done, leaving no overlap to race against. Instead,
        # flip pages for as long as the readers are still running, capped at 20000 so this
        # cannot hang even if something above goes wrong; `stop` lets it exit the moment the
        # readers finish rather than spinning needlessly.
        stop = threading.Event()

        def flip():
            i = 0
            while not stop.is_set() and i < 20_000:
                server.publish("form", pages["short"] if i % 2 == 0 else pages["long"])
                i += 1

        flipper = threading.Thread(target=flip)
        original_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            for t in readers:
                t.start()
            flipper.start()

            for t in readers:
                t.join(timeout=60)
            stop.set()
            flipper.join(timeout=10)
        finally:
            sys.setswitchinterval(original_interval)

        assert not any(t.is_alive() for t in readers), "a reader thread never finished"
        assert not flipper.is_alive(), "the flipper thread never finished"
        assert errors == []
    finally:
        server.shutdown()


def test_the_page_carries_the_step_id_so_it_can_detect_a_change():
    """Every page must know its own step id: that is what /answer is checked against and what the
    poller compares. A page served without it would post answers the server cannot place."""
    server = WizardServer(port=0)
    try:
        step_id = server.publish("form", "<html><body>x</body></html>")
        body = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "window.SIMVLA_STEP = %d" % step_id in body
        assert "/step" in body, "the poller must be present"
    finally:
        server.shutdown()


def test_get_step_reports_the_current_step():
    server = WizardServer(port=0)
    try:
        step_id = server.publish("choice", "<html><body>x</body></html>", options=["ok"])
        status, body = _get(server.url + "step")
        assert status == 200
        assert body == {"id": step_id, "kind": "choice"}
    finally:
        server.shutdown()


def test_answering_unblocks_the_director():
    """The director blocks on wait_answer() while the HTTP thread takes the answer."""
    server = WizardServer(port=0)
    try:
        step_id = server.publish("choice", "<html><body>x</body></html>", options=["yes", "no"])
        answer = {}
        waiter = threading.Thread(target=lambda: answer.update(server.wait_answer()))
        waiter.start()

        status, body = _post(server.url + "answer", {"step": step_id, "id": "yes"})
        assert (status, body) == (200, {"ok": True})

        waiter.join(timeout=5)
        assert not waiter.is_alive(), "wait_answer() never returned"
        assert answer["id"] == "yes"
    finally:
        server.shutdown()


def test_an_answer_from_a_stale_page_is_refused_with_409():
    """The modality the Tk modal used to enforce. A tab left on step 3 must not be able to answer
    step 5 — that is how a second generation used to be started mid-pipeline."""
    server = WizardServer(port=0)
    try:
        stale = server.publish("choice", "<html><body>x</body></html>", options=["yes"])
        server.publish("progress", "<html><body>y</body></html>")

        status, body = _post(server.url + "answer", {"step": stale, "id": "yes"})
        assert status == 409
        assert body["ok"] is False
        assert "out of date" in body["reason"]
    finally:
        server.shutdown()


def test_an_option_the_step_did_not_publish_is_refused():
    server = WizardServer(port=0)
    try:
        step_id = server.publish("choice", "<html><body>x</body></html>", options=["yes", "no"])
        status, body = _post(server.url + "answer", {"step": step_id, "id": "maybe"})
        assert status == 400
        assert "maybe" in body["reason"]
    finally:
        server.shutdown()


def test_progress_is_served_to_the_page():
    server = WizardServer(port=0)
    try:
        server.publish("progress", "<html><body>x</body></html>")  # /progress is sealed during boot
        assert _get(server.url + "progress")[1] == {"label": "", "done": 0, "total": 0}
        server.set_progress("Building rotations — 7 of 12", 7, 24)
        assert _get(server.url + "progress")[1] == {
            "label": "Building rotations — 7 of 12", "done": 7, "total": 24
        }
    finally:
        server.shutdown()


def test_a_new_scene_step_clears_placements_and_template():
    """A fresh server per preview round used to give this for free. One long-lived server has to do
    it, or a re-roll inherits the previous round's drags and the composer's stale template."""
    server = WizardServer(port=0)
    try:
        server.publish("scene", "<html><body>x</body></html>", options=["accept"])
        _post(server.url + "placements",
              {"bowl00": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 1.5, 0.95, 1]})
        assert server.placements != {}

        server.publish("scene", "<html><body>y</body></html>", options=["accept"])
        assert server.placements == {}
        assert server.template is None
    finally:
        server.shutdown()


def test_inject_step_script_appends_inside_the_body():
    html = inject_step_script("<html><body><p>hi</p></body></html>", 3)
    assert html.index("window.SIMVLA_STEP = 3") < html.index("</body>")
    assert "<p>hi</p>" in html


def test_the_banner_names_what_is_being_served():
    """PreviewServer.banner said 'Kitchen preview' unconditionally; the wizard serves the whole run."""
    server = WizardServer(port=0)
    try:
        assert "Kitchen generator ready" in server.banner(title="Kitchen generator")
        assert "ssh -L" in server.banner(title="Kitchen generator")
    finally:
        server.shutdown()


import os

import pytest

import emit_job
from kitchen_build import KITCHEN_BUILDERS, OBJECT_TYPES, available_locations
from kitchen_wizard import (
    BODEX_GROUP, NO_GRASP_GROUP, SetupError, ensure_hf_user, existing_kitchen_numbers,
    form_payload, object_menu, render_form_page, validate_setup,
)

MESHES = [
    "/data/use_data/core_mug_1038e4/mesh/simplified.obj",
    "/data/use_data/core_bowl_12ddb1/mesh/simplified.obj",
]
ISLAND_LOC = available_locations("island")[0].loc


def test_existing_kitchen_numbers_finds_base_usds_only(tmp_path):
    """kitchen_12.usd is a committed kitchen; kitchen_12_03.usd is one of its twelve rotations."""
    for name in ("kitchen_12.usd", "kitchen_102.usd", "kitchen_12_03.usd", "notes.txt"):
        (tmp_path / name).write_text("x")
    assert existing_kitchen_numbers(str(tmp_path)) == [12, 102]


def test_form_payload_offers_every_builder_and_its_own_locations(tmp_path):
    payload = form_payload(str(tmp_path))
    assert payload["kitchen_types"] == sorted(KITCHEN_BUILDERS)
    assert payload["existing"] == []
    island = payload["locations"]["island"]
    assert [p.loc for p in available_locations("island")] == [entry["loc"] for entry in island]
    assert {"loc", "label", "group"} == set(island[0])
    assert "island" not in [entry["loc"] for entry in payload["locations"]["single_wall"]], (
        "a location list is per kitchen type — that is why the Tk menu was rebuilt on every change"
    )


def test_form_payload_warns_when_the_manifest_is_stale(tmp_path, monkeypatch, capsys):
    """The design's only real failure mode ("Manifest older than graspdata_final -> Loud warning
    naming both timestamps") must actually fire from the one place a whole run reads the manifest,
    not just exist as an untested function."""
    bodex_dir = tmp_path / "bodex"
    gd = bodex_dir / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    manifest = {"generated_from": str(bodex_dir), "graspdata_mtime": 1_000_000_000,
                "aperture_m": 0.09, "margin_m": 0.005, "min_candidates": 30, "shrink_floor": 0.7,
                "meshes": {}}
    (bodex_dir / "grasp_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    os.utime(gd, (2_000_000_000, 2_000_000_000))
    monkeypatch.setenv("BODEX_OBJ_DIR", str(bodex_dir))

    kitchen_dir = tmp_path / "kitchens"
    kitchen_dir.mkdir()
    form_payload(str(kitchen_dir))

    out = capsys.readouterr().out
    assert "stale" in out.lower()
    assert "1000000000" in out, "the manifest's own recorded timestamp"
    assert "2000000000" in out, "the live graspdata_final timestamp"
    assert "build_grasp_manifest.py" in out, "the warning must say how to fix it"


def test_form_payload_is_quiet_when_the_manifest_is_current(tmp_path, monkeypatch, capsys):
    bodex_dir = tmp_path / "bodex"
    gd = bodex_dir / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    os.utime(gd, (1_000_000_000, 1_000_000_000))
    manifest = {"generated_from": str(bodex_dir), "graspdata_mtime": 1_000_000_000,
                "aperture_m": 0.09, "margin_m": 0.005, "min_candidates": 30, "shrink_floor": 0.7,
                "meshes": {}}
    (bodex_dir / "grasp_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("BODEX_OBJ_DIR", str(bodex_dir))

    kitchen_dir = tmp_path / "kitchens"
    kitchen_dir.mkdir()
    form_payload(str(kitchen_dir))

    assert "stale" not in capsys.readouterr().out.lower()


def test_an_hf_user_that_is_already_set_is_left_alone(monkeypatch):
    monkeypatch.setenv("HF_USER", "exaFLOPs09")
    assert ensure_hf_user() is None
    assert os.environ["HF_USER"] == "exaFLOPs09"


def test_a_missing_hf_user_is_defaulted_the_way_emit_job_defaults_it(monkeypatch):
    """Unset, it is a KeyError at the composer's import — after the kitchen is already committed."""
    monkeypatch.delenv("HF_USER", raising=False)
    monkeypatch.setenv("USER", "someone")
    assert ensure_hf_user() == "someone"
    assert os.environ["HF_USER"] == "someone"
    assert emit_job.env_defaults()["hf_user"] == "someone", (
        "the emitted sbatch script and this run must agree on who they are"
    )


def test_an_empty_hf_user_counts_as_missing(monkeypatch):
    """`export HF_USER=` leaves it set to '' — still a falsy repo prefix, so still fix it."""
    monkeypatch.setenv("HF_USER", "")
    monkeypatch.setenv("USER", "someone")
    assert ensure_hf_user() == "someone"


def test_hf_user_falls_back_past_an_unset_user_too(monkeypatch):
    """A container or a cron job can have neither; a KeyError at the composer is still the worst
    possible outcome, so something non-empty wins."""
    monkeypatch.delenv("HF_USER", raising=False)
    monkeypatch.delenv("USER", raising=False)
    assert ensure_hf_user() == "simvla"


def _grasp_manifest(**meshes):
    return {"generated_from": "/data", "graspdata_mtime": 0, "aperture_m": 0.09,
            "margin_m": 0.005, "min_candidates": 30, "shrink_floor": 0.7, "meshes": meshes}


def test_the_object_menu_covers_every_placeable_type_exactly_once():
    """A type absent from the menu cannot be placed at all — the form is the only way in."""
    menu = object_menu()
    types = [entry["type"] for entry in menu]
    assert sorted(types) == sorted(OBJECT_TYPES)
    assert len(types) == len(set(types)), "a type in two groups would appear twice in the menu"
    assert {"type", "label", "group"} == set(menu[0])


def test_the_menu_groups_by_what_the_manifest_measured():
    m = _grasp_manifest(a={"type": "mug", "strategy": "single"},
                        b={"type": "nutella", "strategy": "none"})
    groups = {e["type"]: e["group"] for e in object_menu(m)}
    assert groups["mug"] == BODEX_GROUP
    assert groups["nutella"] == NO_GRASP_GROUP


def test_a_type_whose_meshes_disagree_says_so_in_its_label():
    """toaster ships 4 meshes, 2 with grasps: a random pick may or may not be graspable."""
    m = _grasp_manifest(a={"type": "toaster", "strategy": "single"},
                        b={"type": "toaster", "strategy": "none"})
    labels = {e["type"]: e["label"] for e in object_menu(m)}
    assert labels["toaster"] == "toaster — some meshes lack grasps"


def test_an_option_value_is_always_the_bare_type_name():
    """The label is display only; annotating it must not change what gets submitted."""
    m = _grasp_manifest(a={"type": "toaster", "strategy": "single"},
                        b={"type": "toaster", "strategy": "none"})
    assert [e["type"] for e in object_menu(m) if e["type"] == "toaster"] == ["toaster"]


def test_without_a_manifest_the_menu_falls_back_to_the_declared_lists():
    """A clone with no dataset still renders a usable form."""
    groups = {e["type"]: e["group"] for e in object_menu(None)}
    assert groups["mug"] == BODEX_GROUP
    assert groups["nutella"] == NO_GRASP_GROUP, "the hardcoded NO_GRASPDATA fallback"
    assert groups["plate"] == BODEX_GROUP


def test_validate_setup_numbers_objects_per_type_in_row_order():
    """mug0, mug1, bowl0 — the numbering KitchenBuilderApp.add_object_to_list did with obj_dict."""
    parsed = validate_setup({
        "action": "generate",
        "kitchen_num": "12",
        "kitchen_name": "island",
        "objects": [
            {"type": "mug", "loc": ISLAND_LOC},
            {"type": "mug", "loc": ISLAND_LOC},
            {"type": "bowl", "loc": ISLAND_LOC},
        ],
    }, MESHES)
    assert parsed["kitchen_num"] == 12
    assert parsed["kitchen_name"] == "island"
    assert [o["obj_n"] for o in parsed["objects"]] == ["mug0", "mug1", "bowl0"]
    assert [o["mesh"] for o in parsed["objects"]] == [None, None, None], (
        "no mesh named means 'pick a random matching one at build time', today's behaviour"
    )


def test_validate_setup_resolves_random_to_a_real_kitchen_type():
    parsed = validate_setup(
        {"action": "generate", "kitchen_num": "3", "kitchen_name": "random", "objects": []}, MESHES
    )
    assert parsed["kitchen_name"] in KITCHEN_BUILDERS


def test_validate_setup_accepts_an_explicit_mesh():
    """The field the mesh gallery (spec 2) fills in. It must survive to build_kitchen untouched."""
    parsed = validate_setup({
        "action": "generate", "kitchen_num": "1", "kitchen_name": "island",
        "objects": [{"type": "mug", "loc": ISLAND_LOC, "mesh": MESHES[0]}],
    }, MESHES)
    assert parsed["objects"][0]["mesh"] == MESHES[0]


@pytest.mark.parametrize("payload, fragment", [
    ({"action": "generate", "kitchen_num": "", "kitchen_name": "island", "objects": []},
     "non-negative integer"),
    ({"action": "generate", "kitchen_num": "x2", "kitchen_name": "island", "objects": []},
     "non-negative integer"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "chalet", "objects": []},
     "Unknown kitchen type"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [{"type": "hovercraft", "loc": ISLAND_LOC}]}, "unknown object type"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [{"type": "mug", "loc": None}]}, "no placement location"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [{"type": "mug", "loc": 9999}]}, "not available"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [{"type": "mug", "loc": ISLAND_LOC, "mesh": "/nope.obj"}]}, "not in the BODex"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [{"type": "teapot", "loc": ISLAND_LOC}]}, "No BODex mesh matched"),
    ({"action": "wander", "kitchen_num": "1", "kitchen_name": "island", "objects": []},
     "unknown action"),
    # A non-dict row or a non-list `objects` used to escape as an uncaught AttributeError/TypeError
    # -- SetupError is the only exception _handle_setup knows how to turn into a real response.
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": [5, 6]}, "Row 1"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": "not-a-list"}, "objects"),
    ({"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
      "objects": {"type": "mug", "loc": 1}}, "objects"),
    # A digit string long enough to hit CPython's int-string-conversion limit used to raise
    # ValueError straight out of int(), instead of the SetupError every other bad number gets.
    ({"action": "generate", "kitchen_num": "9" * 5000, "kitchen_name": "island", "objects": []},
     "too many digits"),
])
def test_validate_setup_refuses_bad_input_with_a_reason(payload, fragment):
    """Every reason is shown inline on the form, so it has to say what is wrong with WHICH row."""
    with pytest.raises(SetupError) as exc:
        validate_setup(payload, MESHES)
    assert fragment in str(exc.value)


def test_the_session_robot_survives_the_form_round_trip():
    """The robot is picked ONCE, on the step before the form, and every later size comparison is
    drawn against it -- so it has to come back off every submit. The form is re-rendered from
    scratch on each trip out to a gallery, and a field the page does not echo is a field the run
    silently loses on the author's first click.

    Both directions:
      * `robot` arrives on validate_setup's state (what _edit_table and _edit_chair read) AND on
        its generate payload (what the preview opens with);
      * the form's own script carries it in the POST body, and reads it back OUT of DATA.state --
        outside the "did we come back from a gallery" guard, because the very first form has no
        kitchen_name yet and that is exactly the submit that would drop it.

    Read out of the rendered page for the browser half; no JS engine exists on this box.
    """
    import kitchen_preview

    for action, key in (("pick_table", "state"), ("generate", None)):
        parsed = validate_setup({
            "action": action, "kitchen_num": "3", "kitchen_name": "island", "objects": [],
            "robot": "aiworker",
        }, MESHES)
        carried = parsed[key] if key else parsed
        assert carried["robot"] == "aiworker", f"{action} dropped the session's robot: {parsed}"

    # Absent is the DEFAULT, not "no robot": a payload from anything that does not know about this
    # field has to keep drawing the comparison the wizard drew before the field existed.
    bare = validate_setup(
        {"action": "generate", "kitchen_num": "3", "kitchen_name": "island", "objects": []}, MESHES
    )
    assert bare["robot"] == kitchen_preview.DEFAULT_ROBOT
    # And "" really is No robot, which is a state the preview's own selector already has.
    blank = validate_setup({"action": "generate", "kitchen_num": "3", "kitchen_name": "island",
                            "objects": [], "robot": ""}, MESHES)
    assert blank["robot"] == ""

    with pytest.raises(SetupError) as exc:
        validate_setup({"action": "generate", "kitchen_num": "3", "kitchen_name": "island",
                        "objects": [], "robot": "h2"}, MESHES)
    assert "Unknown robot" in str(exc.value)

    page = render_form_page(form_payload("/tmp", {"robot": "anubis"}))
    assert "robot: robot," in page, "the form does not post the robot back"
    assert "var robot = (DATA.state && DATA.state.robot !== undefined) ? DATA.state.robot : null;" \
        in page, "the form reads the robot inside the came-back-from-a-gallery guard, or not at all"
    assert '"robot": "anubis"' in page, "the state the form was given did not reach the page"


def test_compose_on_existing_needs_only_a_number():
    parsed = validate_setup(
        {"action": "compose_existing", "kitchen_num": "12", "objects": []}, MESHES
    )
    assert parsed == {"action": "compose_existing", "kitchen_num": 12,
                      "kitchen_name": None, "seed": None, "objects": []}


def test_the_form_page_carries_the_payload_and_the_buttons(tmp_path):
    (tmp_path / "kitchen_07.usd").write_text("x")
    page = render_form_page(form_payload(str(tmp_path)))
    assert "__FORM_PAYLOAD__" not in page, "the placeholder must be substituted"
    assert '"existing": [7]' in page.replace("'", '"') or "[7]" in page
    assert "/setup" in page
    assert "compose_existing" in page


def test_post_setup_hands_the_parsed_form_to_the_director(tmp_path):
    server = WizardServer(port=0)
    try:
        server.set_setup_context(mesh_files=MESHES)
        step_id = server.publish("form", render_form_page(form_payload(str(tmp_path))))
        answer = {}
        waiter = threading.Thread(target=lambda: answer.update(server.wait_answer()))
        waiter.start()

        status, body = _post(server.url + "setup", {
            "step": step_id, "action": "generate", "kitchen_num": "12",
            "kitchen_name": "island", "objects": [{"type": "mug", "loc": ISLAND_LOC}],
        })
        assert (status, body) == (200, {"ok": True})
        waiter.join(timeout=5)
        assert answer["kitchen_num"] == 12
        assert answer["objects"][0]["obj_n"] == "mug0"
    finally:
        server.shutdown()


def test_post_setup_refuses_bad_input_with_a_400_and_the_reason(tmp_path):
    """400 with the reason in the body, so the page shows it inline and the run stays on the form."""
    server = WizardServer(port=0)
    try:
        server.set_setup_context(mesh_files=MESHES)
        step_id = server.publish("form", "<html><body>x</body></html>")
        status, body = _post(server.url + "setup", {
            "step": step_id, "action": "generate", "kitchen_num": "twelve",
            "kitchen_name": "island", "objects": [],
        })
        assert status == 400
        assert "non-negative integer" in body["reason"]
        assert server.step["kind"] == "form", "a rejected form must not advance the run"
    finally:
        server.shutdown()


@pytest.mark.parametrize("bad_objects", [
    [5, 6],               # a non-dict row
    "not-a-list",         # a truthy non-list `objects`
    {"type": "mug"},      # ditto, a dict this time
])
def test_post_setup_survives_malformed_objects_without_dropping_the_connection(bad_objects):
    """Before the fix, these escaped validate_setup as an uncaught AttributeError/TypeError, so the
    handler thread died mid-request: socketserver closed the socket with no response ever sent, and
    the browser saw nothing at all -- not even a rejection. _post() raising here (instead of
    returning a status/body pair) is exactly that failure mode; it must always get an answer."""
    server = WizardServer(port=0)
    try:
        server.set_setup_context(mesh_files=MESHES)
        step_id = server.publish("form", "<html><body>x</body></html>")
        status, body = _post(server.url + "setup", {
            "step": step_id, "action": "generate", "kitchen_num": "1",
            "kitchen_name": "island", "objects": bad_objects,
        })
        assert 400 <= status < 600
        assert body.get("reason")
        assert server.step["kind"] == "form", "a rejected form must not advance the run"
    finally:
        server.shutdown()


def test_post_setup_survives_an_oversized_kitchen_number_without_dropping_the_connection():
    """A digit string past CPython's int-string-conversion limit used to raise ValueError straight
    out of int(kitchen_num), past the handler's only except clause (SetupError) -- same dropped-
    connection failure as the malformed-objects case above."""
    server = WizardServer(port=0)
    try:
        server.set_setup_context(mesh_files=MESHES)
        step_id = server.publish("form", "<html><body>x</body></html>")
        status, body = _post(server.url + "setup", {
            "step": step_id, "action": "generate", "kitchen_num": "9" * 5000,
            "kitchen_name": "island", "objects": [],
        })
        assert 400 <= status < 600
        assert body.get("reason")
        assert server.step["kind"] == "form", "a rejected form must not advance the run"
    finally:
        server.shutdown()


from kitchen_wizard import render_choice_page, render_done_page, render_progress_page


def test_the_choice_page_renders_one_button_per_option():
    page = render_choice_page(
        title="Choose the prim to rotate",
        text="The twelve rotation USDs differ only in this prim's yaw.",
        options=[{"id": "/world/mug0", "label": "/world/mug0"}, {"id": "", "label": "Cancel"}],
    )
    assert "/world/mug0" in page
    assert "Cancel" in page
    assert "/answer" in page


def test_the_progress_page_polls_progress():
    page = render_progress_page(title="Generating kitchen 12", notice="Committed to disk.")
    assert "/progress" in page
    assert "Committed to disk." in page


def test_ask_publishes_a_choice_and_returns_the_clicked_id():
    server = WizardServer(port=0)
    try:
        picked = {}
        asker = threading.Thread(target=lambda: picked.update(
            {"id": server.ask("Scene ready", "Compose a task?",
                              [{"id": "yes", "label": "Compose"}, {"id": "no", "label": "Finish"}])}
        ))
        asker.start()
        for _ in range(50):                       # the thread has to publish before we can answer
            if server.step["kind"] == "choice":
                break
            time.sleep(0.05)
        assert server.step["kind"] == "choice"

        status, _ = _post(server.url + "answer", {"step": server.step["id"], "id": "no"})
        assert status == 200
        asker.join(timeout=5)
        assert picked["id"] == "no"
    finally:
        server.shutdown()


def test_show_progress_publishes_a_progress_step():
    server = WizardServer(port=0)
    try:
        server.show_progress("Generating kitchen 12")
        assert server.step["kind"] == "progress"
        assert "Generating kitchen 12" in urllib.request.urlopen(server.url, timeout=5).read().decode()
    finally:
        server.shutdown()


# ---- fix round 1: the list layout's Cancel button, script-tag escaping, the poll/finish margin ----


def _script_var(page, name):
    """Parse `var NAME = <json>;` out of a rendered page and decode it, reversing script_json's
    "<" escaping. Assertions here read the DATA a page was built from, not its DOM -- same
    discipline as the rest of this file. Stops at the first ";" (there is a trailing "//" comment
    after it on the same line, not a newline)."""
    match = re.search(r"var " + name + r" = (.*?);", page)
    assert match, f"var {name} not found in page"
    return json.loads(match.group(1).replace("\\u003c", "<"))


def test_the_list_layout_omits_cancel_when_no_option_offers_it():
    """FINDING 1 (round 1 review): the list layout used to hardcode a Cancel button regardless of
    whether the step offered one. A step publishing many real ids and no empty-id option rendered a
    Cancel that answered with an id publish() never allowed -- clicking it always 400'd, silently."""
    options = [{"id": f"/world/obj{i}", "label": f"/world/obj{i}"} for i in range(6)]
    page = render_choice_page(title="Pick a prim", text="x", options=options)
    assert "Cancel" not in page
    assert 'id="cancel"' not in page


def test_the_list_layout_shows_an_offered_cancel_exactly_once():
    """The other half of FINDING 1: a step that DOES include {"id": "", "label": "Cancel"} used to
    get it rendered twice -- once as an ordinary radio row (mixed in with the real choices), once as
    the hardcoded button below. Also checks the ids embedded in the page are exactly what the step
    would publish: the real ids once each, plus the cancel id -- nothing invented, nothing dropped."""
    real = [{"id": f"/world/obj{i}", "label": f"/world/obj{i}"} for i in range(6)]
    options = real + [{"id": "", "label": "Cancel"}]
    page = render_choice_page(title="Pick a prim", text="x", options=options)
    assert page.count("Cancel") == 1

    shown_options = _script_var(page, "OPTIONS")
    shown_cancel = _script_var(page, "CANCEL")
    assert [o["id"] for o in shown_options] == [o["id"] for o in real]
    assert shown_cancel == {"id": "", "label": "Cancel"}
    shown_ids = [o["id"] for o in shown_options] + [shown_cancel["id"]]
    assert len(shown_ids) == len(set(shown_ids)), "no id may be shown twice"


def test_ask_with_the_list_layout_accepts_its_own_cancel_button():
    """End to end against a live server: the exact interaction FINDING 1 flagged as silently
    400-ing. The served page must render Cancel exactly once, and posting the id it advertises must
    be accepted and be what ask() returns to the director."""
    server = WizardServer(port=0)
    try:
        real = [{"id": f"/world/obj{i}", "label": f"/world/obj{i}"} for i in range(6)]
        options = real + [{"id": "", "label": "Cancel"}]
        picked = {}
        # daemon=True: if an assertion below fails before the answer is posted, ask() is left
        # blocked in wait_answer() forever -- daemon keeps that from hanging the whole test process.
        asker = threading.Thread(
            target=lambda: picked.update({"id": server.ask("Pick a prim", "x", options)}),
            daemon=True,
        )
        asker.start()
        for _ in range(50):                       # the thread has to publish before we can answer
            if server.step["kind"] == "choice":
                break
            time.sleep(0.05)
        assert server.step["kind"] == "choice"

        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert page.count("Cancel") == 1

        status, _ = _post(server.url + "answer", {"step": server.step["id"], "id": ""})
        assert status == 200
        asker.join(timeout=5)
        assert picked["id"] == ""
    finally:
        server.shutdown()


def test_notice_publishes_one_button_and_returns_when_answered():
    server = WizardServer(port=0)
    try:
        done = threading.Event()
        # daemon=True: same reasoning as the asker thread above -- a failed assertion here must not
        # leave a blocked non-daemon thread hanging the process.
        notifier = threading.Thread(
            target=lambda: (server.notice("Heads up", "Something happened"), done.set()),
            daemon=True,
        )
        notifier.start()
        for _ in range(50):
            if server.step["kind"] == "choice":
                break
            time.sleep(0.05)
        assert server.step["kind"] == "choice"
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "Something happened" in page

        status, _ = _post(server.url + "answer", {"step": server.step["id"], "id": "ok"})
        assert status == 200
        notifier.join(timeout=5)
        assert done.is_set(), "notice() never returned"
    finally:
        server.shutdown()


def test_finish_publishes_the_done_page_and_returns():
    server = WizardServer(port=0)
    try:
        server.finish("All done", "Kitchen 12 committed to disk.")
        assert server.step["kind"] == "done"
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "Kitchen 12 committed to disk." in page
    finally:
        server.shutdown()


def test_the_poll_interval_and_finish_margin_share_one_constant():
    """FINDING 3 (round 1 review): finish()'s sleep and the step poller's reload interval used to
    be two independent hardcoded numbers, so changing one could silently erode the other's margin."""
    from kitchen_wizard import _FINISH_MARGIN_S, _POLL_INTERVAL_MS

    page = inject_step_script("<html><body></body></html>", 1)
    assert f", {_POLL_INTERVAL_MS});" in page
    assert _FINISH_MARGIN_S > _POLL_INTERVAL_MS / 1000, "the margin must outlast one poll interval"


def test_finish_sleeps_for_exactly_the_derived_margin(monkeypatch):
    import kitchen_wizard

    slept = []
    monkeypatch.setattr(kitchen_wizard.time, "sleep", slept.append)
    server = WizardServer(port=0)
    try:
        server.finish("Done", "Kitchen 12 committed to disk.")
        assert slept == [kitchen_wizard._FINISH_MARGIN_S]
    finally:
        server.shutdown()


def test_choice_page_labels_survive_a_closing_script_tag():
    """FINDING 2 (round 1 review): raw json.dumps does not escape "<", so a label containing the
    literal "</script>" used to close the page's script block early -- the rest of the page then
    parsed as markup instead of JS. These labels are prim paths; this module does not control them."""
    hostile = "</script><script>window.pwned = true;</script>"
    page = render_choice_page(title="x", text="y", options=[{"id": "ok", "label": hostile}])
    assert "</script><script>window.pwned" not in page


def test_form_page_payload_survives_a_closing_script_tag():
    """Same gap, in the setup form's payload."""
    payload = {
        "kitchen_types": ["</script><script>window.pwned = true;</script>"],
        "object_types": [], "locations": {}, "existing": [],
    }
    page = render_form_page(payload)
    assert "</script><script>window.pwned" not in page


# ---- saving the composed task and submitting goal generation ----


from kitchen_wizard import goal_generation_blocked


def _rotations_on_disk(kitchen_dir, kitchen_num):
    for i in range(12):
        (kitchen_dir / f"kitchen_{kitchen_num:02d}_{i:02d}.usd").write_text("x")


def test_goal_generation_is_blocked_without_a_kitchen_number(tmp_path):
    reason, tag = goal_generation_blocked(None, str(tmp_path))
    assert tag == "no_kitchen_num"
    assert "no number" in reason


def test_goal_generation_is_blocked_until_the_rotations_exist(tmp_path):
    reason, tag = goal_generation_blocked(12, str(tmp_path))
    assert tag == "missing_rotations"
    assert "kitchen_12_00.usd" in reason

    _rotations_on_disk(tmp_path, 12)
    assert goal_generation_blocked(12, str(tmp_path)) is None


def test_save_task_without_a_composed_template_is_refused(tmp_path):
    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "save_task", {"step": step_id})
        assert status == 400
        assert "click Save there first" in body["reason"]
    finally:
        server.shutdown()


def test_save_task_writes_the_template_and_offers_goal_generation(tmp_path, monkeypatch):
    """Offered only when the twelve rotations are on disk — the precondition is checked BEFORE the
    question is asked, never after, so a 'yes' can always be honoured."""
    import kitchen_wizard
    from scripts.simvla.test_task_template import bowl_to_drawer

    written = tmp_path / "bowl_to_drawer.json"
    monkeypatch.setattr(kitchen_wizard, "save_composed_template", lambda t: written)
    _rotations_on_disk(tmp_path, 12)

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        with server._lock:
            server._template = bowl_to_drawer()

        status, body = _post(server.url + "save_task", {"step": step_id})
        assert status == 200
        assert str(written) in body["message"]
        assert body["confirm"]["path"] == "/emit_goals"
        assert "kitchen 12" in body["confirm"]["text"]
    finally:
        server.shutdown()


def test_save_task_explains_itself_when_goals_cannot_be_generated(tmp_path, monkeypatch):
    import kitchen_wizard
    from scripts.simvla.test_task_template import bowl_to_drawer

    monkeypatch.setattr(kitchen_wizard, "save_composed_template",
                        lambda t: tmp_path / "bowl_to_drawer.json")
    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))   # no rotations written
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        with server._lock:
            server._template = bowl_to_drawer()

        status, body = _post(server.url + "save_task", {"step": step_id})
        assert status == 200
        assert "confirm" not in body
        assert "Accept & Generate" in body["message"], (
            "pointing at task_emit here would be misleading: it would fail for the same reason"
        )
    finally:
        server.shutdown()


def test_emit_goals_reports_the_job_id(tmp_path, monkeypatch):
    """Also pins the call site to build_emit_sbatch's REAL signature (review fix round 1, test-
    quality note): the other emit_goals tests stand in `lambda **kw: ...`, which absorbs any
    kwargs -- a future rename/drop in either build_emit_sbatch's parameters or the call site would
    not be caught by them. This one checks the kwargs the call site actually sends are a subset of
    what the real function declares, and env_defaults() is left un-mocked so its real keys
    (conda_base/conda_env/hf_user) are exactly what is asserted against."""
    import inspect
    import kitchen_wizard

    real_params = set(inspect.signature(kitchen_wizard.emit_job.build_emit_sbatch).parameters)
    sent = {}

    def capture(**kw):
        unexpected = set(kw) - real_params
        assert not unexpected, f"build_emit_sbatch called with unknown kwargs: {unexpected}"
        sent.update(kw)
        return "#!/bin/bash\n"

    _rotations_on_disk(tmp_path, 12)
    template_file = tmp_path / "t.json"
    template_file.write_text("{}")

    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", capture)
    monkeypatch.setattr(kitchen_wizard.emit_job, "submit_emit_job", lambda script: "918273")

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "emit_goals",
                             {"step": step_id, "template": str(template_file)})
        assert status == 200
        assert "918273" in body["message"]
        assert sent["template_path"] == str(template_file)
        assert sent["kitchens"] == "12"
        assert {"conda_base", "conda_env", "hf_user"} <= set(sent), (
            "the real env_defaults() keys must reach build_emit_sbatch"
        )
    finally:
        server.shutdown()


def test_emit_goals_off_slurm_says_how_to_run_it_by_hand(tmp_path, monkeypatch):
    import kitchen_wizard

    def no_sbatch(script):
        raise FileNotFoundError("sbatch")

    _rotations_on_disk(tmp_path, 12)
    template_file = tmp_path / "t.json"
    template_file.write_text("{}")

    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "env_defaults", dict)
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", lambda **kw: "#!/bin/bash\n")
    monkeypatch.setattr(kitchen_wizard.emit_job, "submit_emit_job", no_sbatch)

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "emit_goals",
                             {"step": step_id, "template": str(template_file)})
        assert status == 400
        assert "task_emit.py" in body["reason"]
    finally:
        server.shutdown()


# ---- fix round 1 (review): /emit_goals re-checks the precondition, both new handlers survive an
# unanticipated exception, and /emit_goals validates the template path it is handed ----


def test_emit_goals_refuses_without_a_save_context(tmp_path, monkeypatch):
    """FINDING 1: /emit_goals is directly POST-able and the confirm round-trip is asynchronous (the
    author can click 'Yes' any time after the offer), so it must re-check
    goal_generation_blocked() itself rather than trust what /save_task saw when it made the offer.
    With set_save_context never called, kitchen_num defaults to None -- this must be refused, not
    silently build kitchens='None' and submit a real Slurm job."""
    import kitchen_wizard

    submitted = []
    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "env_defaults", dict)
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", lambda **kw: "#!/bin/bash\n")
    monkeypatch.setattr(
        kitchen_wizard.emit_job, "submit_emit_job",
        lambda script: submitted.append(script) or "1",
    )

    server = WizardServer(port=0)
    try:
        # set_save_context is deliberately NOT called -- kitchen_num stays at its None default.
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "emit_goals",
                             {"step": step_id, "template": str(tmp_path / "t.json")})
        assert status == 400
        assert "no number" in body["reason"]
        assert submitted == [], "no Slurm job may be submitted once the precondition is blocked"
    finally:
        server.shutdown()


def test_emit_goals_refuses_when_rotations_are_missing(tmp_path, monkeypatch):
    """The other half of FINDING 1: the save context /save_task checked can go stale before the
    author answers the confirm -- e.g. the rotations existed at offer time and are gone by the
    time 'Yes' is clicked. Must be re-checked here, not trusted from /save_task."""
    import kitchen_wizard

    submitted = []
    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "env_defaults", dict)
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", lambda **kw: "#!/bin/bash\n")
    monkeypatch.setattr(
        kitchen_wizard.emit_job, "submit_emit_job",
        lambda script: submitted.append(script) or "1",
    )

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))   # no rotations written
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "emit_goals",
                             {"step": step_id, "template": str(tmp_path / "t.json")})
        assert status == 400
        assert "kitchen_12_00.usd" in body["reason"]
        assert submitted == []
    finally:
        server.shutdown()


@pytest.mark.parametrize("bad_template", [
    None,                                   # omitted from the payload entirely
    "",                                     # blank
    123,                                    # not a string
    "/definitely/not/a/real/path.json",     # does not exist
])
def test_emit_goals_refuses_a_bad_template_path(tmp_path, monkeypatch, bad_template):
    """FINDING 4: not a security boundary -- this server is loopback-only -- but a wrong or missing
    template path here would otherwise only surface as a Slurm failure minutes later. Reject it up
    front instead."""
    import kitchen_wizard

    submitted = []
    _rotations_on_disk(tmp_path, 12)
    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "env_defaults", dict)
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", lambda **kw: "#!/bin/bash\n")
    monkeypatch.setattr(
        kitchen_wizard.emit_job, "submit_emit_job",
        lambda script: submitted.append(script) or "1",
    )

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        body = {"step": step_id}
        if bad_template is not None:
            body["template"] = bad_template
        status, resp = _post(server.url + "emit_goals", body)
        assert status == 400
        assert resp.get("reason")
        assert submitted == []
    finally:
        server.shutdown()


def test_emit_goals_survives_build_emit_sbatch_raising(tmp_path, monkeypatch):
    """FINDING 3: resolve_out_dir/env_defaults/build_emit_sbatch used to run with no try around
    them at all -- any exception from any of the three dropped the connection with no response
    ever sent. Same invariant _handle_setup already established: a POST always gets an answer."""
    import kitchen_wizard

    _rotations_on_disk(tmp_path, 12)
    template_file = tmp_path / "t.json"
    template_file.write_text("{}")

    def boom(**kw):
        raise TypeError("build_emit_sbatch exploded")

    monkeypatch.setattr(kitchen_wizard.emit_job, "resolve_out_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kitchen_wizard.emit_job, "env_defaults", dict)
    monkeypatch.setattr(kitchen_wizard.emit_job, "build_emit_sbatch", boom)

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _post(server.url + "emit_goals",
                             {"step": step_id, "template": str(template_file)})
        assert 400 <= status < 600
        assert body.get("reason")
    finally:
        server.shutdown()


def test_save_task_survives_save_composed_template_raising_an_os_error(tmp_path, monkeypatch):
    """FINDING 2: save_composed_template also does real filesystem I/O (mkdir + write_text), which
    can raise OSError (permissions, a full disk) -- the original handler only ever caught
    ValueError. Same invariant as _handle_setup: a POST must always get a response, never a
    silently dropped connection."""
    import kitchen_wizard
    from scripts.simvla.test_task_template import bowl_to_drawer

    def boom(template):
        raise OSError("disk full")

    monkeypatch.setattr(kitchen_wizard, "save_composed_template", boom)

    server = WizardServer(port=0)
    try:
        server.set_save_context(kitchen_num=12, kitchen_dir=str(tmp_path))
        step_id = server.publish("scene", "<html><body>x</body></html>", options=["done"])
        with server._lock:
            server._template = bowl_to_drawer()

        status, body = _post(server.url + "save_task", {"step": step_id})
        assert 400 <= status < 600
        assert body.get("reason")
    finally:
        server.shutdown()


def test_grasp_thumbs_can_be_set_per_composer_session(tmp_path):
    """ComposerServer took these at construction, once per composer session. This server outlives
    every session, so /grasps/state must see whatever the composer step just set."""
    folder = tmp_path / "mug_segments_thumbnails"
    folder.mkdir()

    server = WizardServer(port=0)
    try:
        server.set_grasp_thumbs({"mug00": str(folder)})
        server.publish("scene", "<html><body>x</body></html>", options=["done"])
        status, body = _get(server.url + "grasps/state?key=mug00")
        assert status == 200
        assert body["thumbs"] == []          # a real folder, with nothing rendered into it yet
    finally:
        server.shutdown()


def test_the_generators_pre_boot_sequence_does_not_reach_scene_synthesizer():
    """Everything the generator does BEFORE AppLauncher must not drag in scene_synthesizer — or
    pxr, omni, isaaclab, torch, tkinter.

    scene_synthesizer is the one that bites, and it bites silently: its usd_import and usd_export
    modules soft-import pxr (`try: from pxr import ... except ImportError: warn`), so importing them
    before AppLauncher has run leaves `Usd` and friends unbound FOREVER — reimporting later is a
    no-op, sys.modules already has them. The run then goes all the way through setup, the build and
    the preview and dies at the commit with `NameError: name 'Usd' is not defined`, which is exactly
    what a browser-driven smoke run caught.

    The subprocess replays kitchen_scene_generator.py's real pre-boot statements — import
    kitchen_wizard, construct the server, import kitchen_preview, import kitchen_gallery — rather
    than just the one import. Constructing WizardServer() is part of the pre-boot window (it is what
    makes the URL printable while Isaac loads) and so is kitchen_preview's auto_open_verdict; both
    were covered here only by accident before, and a __init__ that grew a kitchen_build call would
    have sailed through. kitchen_gallery is here because it is the newest of them and the easiest to
    break: it renders KITCHENS, and the obvious way to write it would have been to import
    kitchen_build and build them itself.

    Call-time reintroduction is a different hazard and this test cannot see it: a handler thread can
    reach scene_synthesizer while the server is live. That is sealed by Handler._sealed() and pinned
    by test_the_boot_step_refuses_everything_that_could_touch_scene_synthesizer below.

    In a CLEAN interpreter, for the same reason test_task_composer's version uses one: this module
    imports kitchen_build at line 275, so an in-process sys.modules check would always pass.
    """
    import os
    import subprocess
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys

            # kitchen_scene_generator.py's pre-boot statements, in order, and nothing else.
            import kitchen_wizard
            server = kitchen_wizard.WizardServer(port=0)
            from kitchen_preview import auto_open_verdict        # noqa: F401
            import kitchen_gallery                               # noqa: F401

            banned = ("scene_synthesizer", "omni", "isaaclab", "pxr", "torch", "tkinter")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            server.shutdown()
            assert not pulled, f"the generator's pre-boot sequence pulled in {pulled}"
            print("clean")
        """)],
        cwd=here, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]


def test_the_boot_step_refuses_everything_that_could_touch_scene_synthesizer():
    """The server is live for the whole ~16s Omniverse boot. Nothing it serves in that window may
    execute code that reaches scene_synthesizer, or pxr is missing and the run dies at the commit.

    Lazy imports alone do not give this: they moved the hazard from import time to call time.
    /scene_defaults calls scene_defaults_payload() -> scene_spec.SUPPORT_SURFACES -> kitchen_build,
    and /template calls validate_scene, and neither carries a step id, so neither is covered by the
    stale-step 409 that happens to protect /setup. A composer tab left open on this port from a
    previous run refetches /scene_defaults the instant this server starts answering.

    So the boot step is default-DENY: the four paths the boot page itself uses, and nothing else.
    """
    def status_of(url, payload=None):
        """The status code, however it comes back — _get raises on a 4xx/5xx and parses JSON."""
        request = urllib.request.Request(url)
        if payload is not None:
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode(), method="POST",
                headers={"Content-Type": "application/json"},
            )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    server = WizardServer(port=0)
    try:
        assert server.step["kind"] == "boot"

        for path in ("scene_defaults", "palette", "predicates", "grasps/state?key=x", "progress"):
            assert status_of(server.url + path) == 503, (
                f"GET /{path} was served during the boot step"
            )
        for path in ("template", "setup", "answer", "placements", "save_task"):
            assert status_of(server.url + path, {"step": 0}) == 503, (
                f"POST /{path} was served during the boot step"
            )

        # The boot page and its poller still work, or the tab never learns the run has started.
        # /progress is not one of them: the boot page's poller only ever fetches /step (see
        # _STEP_SCRIPT) -- /progress is polled by the progress page, never shown during boot.
        assert status_of(server.url) == 200
        assert status_of(server.url + "step") == 200
        assert status_of(server.url + "ping") == 204

        # The seal lifts the moment the director publishes a real step — which the generator only
        # does after AppLauncher has run, so pxr exists by then.
        server.publish("form", "<html><body>x</body></html>")
        assert status_of(server.url + "palette") == 200
    finally:
        server.shutdown()


def test_generate_carries_the_seed_that_reproduces_the_room():
    parsed = validate_setup({
        "action": "generate", "kitchen_num": "12", "kitchen_name": "island",
        "seed": 3, "objects": [],
    }, MESHES)
    assert parsed["seed"] == 3


def test_generate_without_a_seed_is_todays_random_room():
    parsed = validate_setup(
        {"action": "generate", "kitchen_num": "12", "kitchen_name": "island", "objects": []}, MESHES
    )
    assert parsed["seed"] is None


@pytest.mark.parametrize("seed", [True, False])
def test_generate_refuses_a_bool_seed(seed):
    """FIX 5 (fix wave, final review): bool is an int subclass in Python, so the seed check's
    `not isinstance(seed, int)` was true for neither True nor False -- `{"seed": true}` sailed
    through and built with seed 1 (and `{"seed": false}` with seed 0), three lines from the row
    check in _validate_setup that already excludes bool the same way. Same guard, same message
    voice, as that row check."""
    with pytest.raises(SetupError) as exc:
        validate_setup({
            "action": "generate", "kitchen_num": "12", "kitchen_name": "island",
            "seed": seed, "objects": [],
        }, MESHES)
    assert "seed" in str(exc.value).lower() and "whole number" in str(exc.value)


def test_pick_kitchen_returns_the_state_it_was_given():
    """The gallery replaces the form, so everything typed so far has to survive the trip."""
    parsed = validate_setup({
        "action": "pick_kitchen", "kitchen_num": "12", "kitchen_name": "island", "seed": None,
        "objects": [{"type": "mug", "loc": ISLAND_LOC, "mesh": None}],
    }, MESHES)

    assert parsed["action"] == "pick_kitchen"
    assert parsed["row"] is None
    assert parsed["state"]["kitchen_num"] == "12"
    assert parsed["state"]["kitchen_name"] == "island"
    assert parsed["state"]["objects"] == [{"type": "mug", "loc": ISLAND_LOC, "mesh": None}]


def test_browsing_does_not_require_a_kitchen_number_yet():
    """You should be able to look at kitchens before deciding what to call the result. The number
    is still checked on generate."""
    parsed = validate_setup({
        "action": "pick_kitchen", "kitchen_num": "", "kitchen_name": "island", "objects": [],
    }, MESHES)
    assert parsed["state"]["kitchen_num"] == ""


def test_pick_mesh_names_the_row_it_is_choosing_for():
    parsed = validate_setup({
        "action": "pick_mesh", "row": 1, "kitchen_num": "12", "kitchen_name": "island",
        "objects": [
            {"type": "mug", "loc": ISLAND_LOC, "mesh": None},
            {"type": "bowl", "loc": ISLAND_LOC, "mesh": None},
        ],
    }, MESHES)
    assert parsed["row"] == 1


@pytest.mark.parametrize("row", [-1, 2, "one", None, True, False])
def test_pick_mesh_refuses_a_row_that_does_not_exist(row):
    """True/False (fix wave, final review): bool is an int subclass in Python, so
    `isinstance(row, int)` alone is true for both -- without validate_setup's own
    `isinstance(row, bool)` guard, `{"row": false}` would pass as index 0 (0 <= False < 1) and
    silently edit the first row instead of being refused. This parametrize list exercised neither
    bool value before, so that guard had no test pinning it at all."""
    with pytest.raises(SetupError) as exc:
        validate_setup({
            "action": "pick_mesh", "row": row, "kitchen_num": "12", "kitchen_name": "island",
            "objects": [{"type": "mug", "loc": ISLAND_LOC, "mesh": None}],
        }, MESHES)
    assert "row" in str(exc.value).lower()


def test_a_bad_row_is_refused_before_a_gallery_is_built():
    """Building the kitchen gallery costs ~27 seconds. Finding out afterwards that row 1 was
    nonsense would waste all of it."""
    with pytest.raises(SetupError) as exc:
        validate_setup({
            "action": "pick_kitchen", "kitchen_num": "12", "kitchen_name": "island",
            "objects": [{"type": "hovercraft", "loc": ISLAND_LOC}],
        }, MESHES)
    assert "hovercraft" in str(exc.value)


def test_form_payload_carries_a_state_to_prefill_from(tmp_path):
    state = {"kitchen_num": "12", "kitchen_name": "island", "seed": 3, "objects": []}
    assert form_payload(str(tmp_path), state)["state"] == state
    assert form_payload(str(tmp_path))["state"] == {}


def test_the_form_page_offers_both_galleries(tmp_path):
    page = render_form_page(form_payload(str(tmp_path)))
    assert "pick_kitchen" in page, "no way to browse kitchen types"
    assert "pick_mesh" in page, "no way to choose a row's mesh"


def test_the_form_page_embeds_the_state_it_must_restore(tmp_path):
    """After a gallery the form is re-rendered from scratch; the rows the user typed live only in
    this payload."""
    state = {
        "kitchen_num": "12", "kitchen_name": "island", "seed": 3,
        "objects": [{"type": "mug", "loc": ISLAND_LOC, "mesh": "/data/mug.obj"}],
    }
    page = render_form_page(form_payload(str(tmp_path), state))
    assert '"kitchen_num": "12"' in page.replace("\\u003c", "<")
    assert "/data/mug.obj" in page
    assert '"seed": 3' in page


# ---- fix round 1 (review): the checkNumber() regression the prefill introduced, and wiring tests
# with teeth for the two galleries (FINDINGS 1 and 3) ----


def test_the_prefill_checks_the_restored_kitchen_number_for_a_clash(tmp_path):
    """FINDING 1 (fix round 1 review): checkNumber() only ever ran off the #num field's own
    `input` event. The prefill sets `.value` programmatically, which fires no such event, so the
    "kitchen N already exists" warning -- the only overwrite guard anywhere in this flow, there is
    no second check at the accept stage -- silently vanished across a gallery round-trip and only
    came back if the user retyped the number by hand. Pins that the prefill block itself calls
    checkNumber() once it has set the field, so a restored form warns exactly as a typed one does.
    """
    page = render_form_page(form_payload(str(tmp_path)))
    prefill = re.search(r"if \(DATA\.state && DATA\.state\.kitchen_name\) \{(.*?)\n  \}", page, re.S)
    assert prefill, "the prefill block was not found in the page"
    assert "checkNumber();" in prefill.group(1), (
        "the prefill sets #num.value programmatically -- that fires no 'input' event, so "
        "checkNumber() must be called explicitly here or the overwrite warning never reappears"
    )


def test_the_browse_and_choose_buttons_are_wired_to_their_own_actions(tmp_path):
    """FINDING 3 (fix round 1 review): the earlier offers_both_galleries test only checked that the
    substrings "pick_kitchen"/"pick_mesh" appeared SOMEWHERE in the page -- that would still pass
    if the action were only named in a comment, or wired to the wrong element. This pins the actual
    listeners: #browse must call submit('pick_kitchen') with no row, and each row's Choose button
    must call submit('pick_mesh', i) -- its own index, not a fixed one."""
    page = render_form_page(form_payload(str(tmp_path)))

    browse_click = re.search(
        r"getElementById\('browse'\)\.addEventListener\('click', function \(\) \{"
        r"\s*submit\('pick_kitchen'\);\s*\}\);",
        page,
    )
    assert browse_click, "the Browse button is not wired to submit('pick_kitchen')"

    choose_click = re.search(
        r"choose\.addEventListener\('click', function \(\) \{ submit\('pick_mesh', i\); \}\);",
        page,
    )
    assert choose_click, "a row's Choose button is not wired to submit('pick_mesh', i)"


def test_the_prefill_carries_each_rows_mesh_into_the_restored_row(tmp_path):
    """The other half of FINDING 3: DATA.state.objects carrying a mesh path proves nothing about
    the page on its own -- test_the_form_page_embeds_the_state_it_must_restore already showed that
    passes from Task 3's payload embedding alone. This pins that THIS task's prefill loop actually
    reads o.mesh into the row it pushes -- drop that field here and every restored row would show
    'random' regardless of what the user had chosen, with no test catching it."""
    page = render_form_page(form_payload(str(tmp_path)))
    assert "rows.push({type: o.type, loc: o.loc, locLabel: labelFor(o.loc), mesh: o.mesh});" in page


def test_showseed_renders_both_states_and_is_wired_to_fire(tmp_path):
    """The last piece of FINDING 3: the seed line must (a) actually distinguish 'no seed picked yet'
    from 'a layout was picked', and (b) be invoked -- once at load (so a restored seed shows up) and
    once when the kitchen type changes (so an invalidated seed's line updates too), not just be a
    dead function nothing ever calls."""
    page = render_form_page(form_payload(str(tmp_path)))
    assert "(seed === null) ? 'random layout' : ('layout #' + seed" in page, (
        "showSeed must render a different line for 'no seed' than for a picked one"
    )
    assert page.count("showSeed();") >= 2, (
        "showSeed() must be invoked both at load and when the kitchen type changes, "
        "or a restored/invalidated seed never reaches the page"
    )


# ---- final whole-branch review fix wave: FIX 1/2 -- every published page must be well-formed -----
#
# FIX 1 found by inspection: kitchen_gallery._PANEL's <script> was opened but never closed. That is
# not cosmetic -- WizardServer.publish() runs every page (the gallery included) through
# inject_step_script(), which inserts the step script right before the same </body> the panel's own
# markup ends at. With the panel's <script> left open, the step script's markup lands INSIDE it, so
# the first </script> a browser's parser finds is the step script's own -- the combined block becomes
# one unterminated, unparseable mess. On a real page that means: no tile buttons, no click handlers,
# no raycast, window.simvlaPost never defined (so there is no way to answer AT ALL), and the /step
# poller never starts (so the tab never reloads either). The director then blocks in wait_answer()
# forever, with no way out but killing the process.
#
# FIX 2: nothing in this suite (or test_kitchen_gallery.py) would have caught that. Every existing
# assertion is either a substring check or JSON pulled back out of `var DATA = ...` / `var NAME =
# ...` -- both of which pass identically whether or not the surrounding <script> the JS lives in can
# actually be parsed at all. This is the structural invariant those assertions never checked: every
# kind of page the wizard actually publishes must have its <script> opens and </script> closes
# balance, both as rendered by its own render_* function AND after inject_step_script() has added the
# step script every published page carries (that second check is what would have caught FIX 1 --
# render_gallery_page's OWN output already had the 2-opens/1-close defect before injection ever runs,
# but a page-as-page reader has to look at the page inject_step_script produces, since that is what a
# browser is actually served).
#
# Confirmed to fail against the pre-fix kitchen_gallery._PANEL: the gallery page alone counted 2
# <script opens vs 1 </script> close, and after inject_step_script the combined page counted 3 opens
# vs 2 closes -- both mismatches this test catches. All the other page kinds already balanced before
# this fix; they are included so the invariant is pinned for every page kind the wizard actually
# serves, not just the one that broke.
def _published_pages(tmp_path):
    """Every kind of page the wizard puts on screen, rendered, keyed by a readable name.

    Shared by the structural tests below so a new page kind has one place to be registered.
    """
    import trimesh

    import kitchen_build
    import kitchen_gallery
    import kitchen_preview

    mesh_paths = [str(tmp_path / f"mug_{i}.obj") for i in range(2)]
    for path in mesh_paths:
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(path)
    scene, tiles = kitchen_gallery.mesh_tiles(mesh_paths)

    few_options = [{"id": "ok", "label": "OK"}, {"id": "", "label": "Cancel"}]
    many_options = [{"id": f"/world/obj{i}", "label": f"obj{i}"} for i in range(6)]

    return {
        "gallery": kitchen_gallery.render_gallery_page(scene, tiles, title="Pick a mug"),
        "form": render_form_page(form_payload(str(tmp_path))),
        "choice (button row, <=4 options)": render_choice_page(
            title="t", text="x", options=few_options
        ),
        "choice (scrolling list, >4 options)": render_choice_page(
            title="t", text="x", options=many_options
        ),
        "progress": render_progress_page(title="Generating kitchen 12"),
        "done": render_done_page(title="All done", text="Kitchen 12 committed to disk."),
        # The per-piece edit step, which is a 3D page now: a scene through trimesh's viewer with a
        # panel injected, exactly as a gallery is. Registered here rather than only in its own
        # tests because the structural checks below are the ones that catch what nothing else can:
        # a dropped id, an unbalanced <script>, a placeholder that never got substituted. No JS
        # engine exists on this box, so these reads ARE the check that its controls are alive.
        "edit": kitchen_wizard.render_edit_page(
            title="Edit Chair 3", text="uid abc123. Footprint and material.",
            scales=kitchen_build.FURNITURE_SCALES, scale=1.0,
            materials=kitchen_build.FURNITURE_MATERIAL_GROUPS,
            material=kitchen_build.DEFAULT_FURNITURE_MATERIAL,
            piece=trimesh.Scene(trimesh.creation.box(extents=(0.55, 0.5, 0.9))),
            swatches=kitchen_build.FURNITURE_MATERIAL_SWATCHES,
            measured_floor=0.95, floor_note="below 0.95x it stops reading as a chair",
            # WITH A TABLE, because the toggle and the note it drives are markup this page only
            # grows when one is passed -- the no-table state is the one the plain kwargs give.
            table={
                "scene": trimesh.Scene(trimesh.creation.box(extents=(1.4, 0.8, 0.74))),
                "label": "Table 4", "detail": "Long dining",
                "material": kitchen_build.DEFAULT_FURNITURE_MATERIAL,
                "offsets_m": [0.9] * len(kitchen_build.FURNITURE_SCALES),
            },
        ),
        # The first step of the run. Rendered for real here (36 MB of it) rather than stubbed: the
        # structural checks below are the only thing on this box that can see a dead control, and
        # a page nobody renders is a page nobody checks.
        "robot": kitchen_wizard.render_robot_page(
            title="Pick your robot", text="One robot for this run.",
            chosen=kitchen_preview.DEFAULT_ROBOT,
        ),
    }

def test_every_published_page_balances_its_script_tags(tmp_path):
    def _tag_counts(html):
        return html.count("<script"), html.count("</script>")

    for name, page in _published_pages(tmp_path).items():
        opens, closes = _tag_counts(page)
        assert opens == closes, (
            f"{name} page as rendered: {opens} <script> opens vs {closes} </script> closes -- "
            f"an unbalanced page is a SyntaxError in the browser and everything on it is dead"
        )

        injected = inject_step_script(page, 1)
        opens2, closes2 = _tag_counts(injected)
        assert opens2 == closes2, (
            f"{name} page after inject_step_script (what the wizard actually serves): "
            f"{opens2} <script> opens vs {closes2} </script> closes"
        )


def test_no_published_page_leaks_an_unsubstituted_placeholder(tmp_path):
    """Every page is a template with __NAME__ slots filled at render time — the theme, the body,
    the payload. A slot that survives to the browser is either a visible __THEME__ in the text or,
    worse, a stylesheet that never arrived and a page that renders unstyled with no error anywhere.

    The slot names come from the page-building modules themselves rather than a hardcoded list, so
    a new template slot is covered the day it is written. Scoped that way on purpose: the inlined
    three.js bundle has its own __THREE__ / __THREE_DEVTOOLS__ globals, and a bare regex over the
    page flags those every time.
    """
    import re
    from pathlib import Path

    here = Path(__file__).parent
    slots = set()
    for module in ("kitchen_wizard.py", "kitchen_gallery.py", "kitchen_preview.py"):
        source = (here / module).read_text(encoding="utf-8")
        slots |= set(re.findall(r"[\"'](__[A-Z][A-Z0-9_]*__)[\"']", source))
    assert slots, "found no template slots to check — the scan is broken, not the pages"

    for name, page in _published_pages(tmp_path).items():
        leaked = sorted(slot for slot in slots if slot in page)
        assert not leaked, f"{name} page still carries unsubstituted placeholder(s): {leaked}"
        assert "--accent" in page, f"{name} page did not get the shared theme tokens"


def test_every_page_carries_the_phase_strip_once_the_run_has_a_phase(tmp_path):
    """Nothing on screen said what came after the step you were on. The strip is injected in
    inject_step_script rather than in each renderer because that is the one place every page
    passes through — including the 3D pages other modules build."""
    server = WizardServer(port=0)
    try:
        server.publish("form", "<html><body>x</body></html>", phase="setup")
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert "simvla-phases" in page
        assert '<span class="now">set up</span>' in page

        # The phase STICKS: a question or a progress page published without one must not lose it.
        server.publish("choice", "<html><body>y</body></html>", options=["ok"])
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert '<span class="now">set up</span>' in page, "the strip vanished on the next page"

        server.set_phase("build")
        server.publish("progress", "<html><body>z</body></html>")
        page = urllib.request.urlopen(server.url, timeout=5).read().decode()
        assert '<span class="now">build</span>' in page
        assert page.count('class="done"') == 3, "phases before the current one read as done"
    finally:
        server.shutdown()


def test_the_boot_page_has_no_phase_strip():
    """The run has not started, so lighting a phase would be a lie."""
    server = WizardServer(port=0)
    try:
        assert "simvla-phases" not in urllib.request.urlopen(server.url, timeout=5).read().decode()
    finally:
        server.shutdown()


# ---- Task 4: the wizard's table picker ----

from kitchen_build import PLACEMENT_BY_KEY, TABLE_VARIANTS

TABLE_TOP_LOC = PLACEMENT_BY_KEY["table_top"].loc


def test_the_setup_accepts_a_table_variant_or_none():
    """The wizard carries the choice as a variant key. None means the user cancelled the
    picker, which is how you decline a table -- there is no separate checkbox."""
    from kitchen_build import TABLE_VARIANTS

    def payload(table):
        # validate_setup's minimum: a known action, a kitchen number, a kitchen name, rows.
        return {"action": "generate", "kitchen_num": "7", "kitchen_name": "island",
                "objects": [], "table": table}

    assert validate_setup(payload(TABLE_VARIANTS[0].key), [])["table"] == TABLE_VARIANTS[0].key
    assert validate_setup(payload(None), [])["table"] is None


def test_an_unknown_table_variant_is_refused():
    payload = {"action": "generate", "kitchen_num": "7", "kitchen_name": "island",
               "objects": [], "table": "not_a_table"}
    with pytest.raises(SetupError):
        validate_setup(payload, [])


def test_pick_table_returns_the_state_it_was_given():
    """Mirrors test_pick_kitchen_returns_the_state_it_was_given: the gallery replaces the form,
    so everything typed so far -- including a table already chosen -- has to survive the trip."""
    parsed = validate_setup({
        "action": "pick_table", "kitchen_num": "12", "kitchen_name": "island", "seed": None,
        "objects": [], "table": TABLE_VARIANTS[0].key,
    }, MESHES)

    assert parsed["action"] == "pick_table"
    assert parsed["row"] is None
    assert parsed["state"]["kitchen_name"] == "island"
    assert parsed["state"]["table"] == TABLE_VARIANTS[0].key


def test_a_row_aimed_at_the_table_is_refused_without_one():
    """has_table gates 'On the table' the same way for rows as it does for the form's own menu
    -- a row surviving here with no table chosen would place nothing, silently, at build time."""
    with pytest.raises(SetupError) as exc:
        validate_setup({
            "action": "generate", "kitchen_num": "1", "kitchen_name": "island",
            "objects": [{"type": "mug", "loc": TABLE_TOP_LOC}],
        }, MESHES)
    assert "not available" in str(exc.value)


def test_a_row_aimed_at_the_table_is_accepted_once_one_is_chosen():
    parsed = validate_setup({
        "action": "generate", "kitchen_num": "1", "kitchen_name": "island",
        "objects": [{"type": "mug", "loc": TABLE_TOP_LOC}], "table": TABLE_VARIANTS[0].key,
    }, MESHES)
    assert parsed["objects"][0]["loc"] == TABLE_TOP_LOC


def test_form_payload_hides_the_table_location_until_a_table_is_chosen(tmp_path):
    """Task 6's end-to-end check: 'Cancelling the picker leaves a kitchen with no table, and
    On the table does not appear in the placement menu' / 'With a table, On the table appears'.
    This is the server-side half -- form_payload is what DATA.locations comes from."""
    without = form_payload(str(tmp_path))["locations"]["island"]
    assert TABLE_TOP_LOC not in [entry["loc"] for entry in without]

    with_table = form_payload(str(tmp_path), {"table": TABLE_VARIANTS[0].key})["locations"]["island"]
    assert TABLE_TOP_LOC in [entry["loc"] for entry in with_table]


def test_form_payload_carries_the_table_variant_labels(tmp_path):
    """The form shows what is currently chosen (test_showtable below) without importing the
    registry into the page's own JS -- this is the map it reads the label from."""
    variants = form_payload(str(tmp_path))["table_variants"]
    assert variants[TABLE_VARIANTS[0].key] == TABLE_VARIANTS[0].label


def test_form_payload_carries_the_table_data_lines_too(tmp_path):
    """The label is "Table 1" now -- a position in the registry, and nothing about the table. The
    descriptive string it replaced ("Small square") and the Objaverse uid are the DATA LINE, and
    the form has to be handed those separately or a chosen table reads as a bare number.

    Asserted over EVERY offered variant, not just the first: the two halves of the registry get
    their detail from different places (a literal in _PROCEDURAL_TABLES, the manifest's uid), so
    checking one proves nothing about the other.
    """
    from kitchen_build import ALL_TABLE_VARIANTS

    details = form_payload(str(tmp_path))["table_details"]
    assert details == {v.key: v.detail for v in ALL_TABLE_VARIANTS}
    assert all(details.values()), (
        f"a table has an empty data line, so the gallery shows only its number: "
        f"{[k for k, v in details.items() if not v]}"
    )


def test_the_table_button_is_wired_to_pick_table(tmp_path):
    """Mirrors test_the_browse_and_choose_buttons_are_wired_to_their_own_actions: the button has
    to be wired to the actual action, not just have the string appear somewhere on the page."""
    page = render_form_page(form_payload(str(tmp_path)))
    table_click = re.search(
        r"getElementById\('tablebtn'\)\.addEventListener\('click', function \(\) \{"
        r"\s*submit\('pick_table'\);\s*\}\);",
        page,
    )
    assert table_click, "the Table… button is not wired to submit('pick_table')"


def test_showtable_renders_both_states_and_is_wired_to_fire(tmp_path):
    """Mirrors test_showseed_renders_both_states_and_is_wired_to_fire: a line that distinguishes
    'no table' from a chosen one, and is actually invoked rather than dead code.

    The chosen state now goes through tableName(), which pairs the label with its data line
    ("Table 4 · Long dining") -- the renumbering left the label alone unable to say which table
    it is. Both branches are still checked, in the two places they now live.

    The chosen branch also carries the table's own EDIT ("· 1.15×") when there is one, so an
    edited table does not read on the form exactly like an unedited one -- that string is the
    server's, from _chair_edit_line, so this checks the branch and not the wording.
    """
    page = render_form_page(form_payload(str(tmp_path)))
    assert "name ? ('table: ' + name + (edit ? ' · ' + edit : '')) : 'no table'" in page, (
        "showTable must render a different line for 'no table' than for a chosen one"
    )
    assert "return detail ? (label + ' · ' + detail) : label;" in page, (
        "tableName must put the data line beside the label; a bare 'Table 4' names no table"
    )
    assert page.count("showTable();") >= 1, "showTable() must be invoked at load"


def test_the_form_prefill_restores_the_table_choice(tmp_path):
    """The other half of the round trip test_the_prefill_carries_each_rows_mesh_into_the_restored_
    row already pins for rows: a table chosen before a gallery visit must survive it."""
    state = {
        "kitchen_num": "12", "kitchen_name": "island", "seed": None, "objects": [],
        "table": TABLE_VARIANTS[0].key,
    }
    page = render_form_page(form_payload(str(tmp_path), state))
    assert "table = (DATA.state.table === undefined) ? null : DATA.state.table;" in page


# ---- Task 4b: the wizard's chairs ----

from kitchen_build import CHAIR_VARIANTS
from kitchen_wizard import MAX_CHAIRS

#: Real uids from the registry, because that is what validate_setup checks against. A checkout with
#: no chair library offers none, and every test that needs one skips rather than inventing a uid the
#: registry would rightly refuse -- pretending an unknown uid is known is how a cap test comes to
#: pass against a check that never ran.
CHAIR_UIDS = [v.uid for v in CHAIR_VARIANTS]
needs_chairs = pytest.mark.skipif(
    not CHAIR_UIDS, reason="no chair library on this machine (CHAIR_OBJ_DIR / /lustre)"
)


def _chair_payload(chairs, action="generate"):
    """validate_setup's minimum, plus a chair list. Built inline, like the table payloads above."""
    return {"action": action, "kitchen_num": "7", "kitchen_name": "island",
            "objects": [], "chairs": chairs}


def _chair(uid, **edit):
    """One chair as the form posts it: a dict per INSTANCE, defaults filled in.

    The shape changed when a chair gained a footprint scale and a material of its own -- two chairs
    of one uid can now be different objects, so a flat list of uids can no longer describe a
    kitchen. Written once here so the tests below say what they are about rather than repeating
    the shape.
    """
    return dict({"uid": uid, "scale": 1.0, "material": None}, **edit)


@needs_chairs
def test_the_setup_accepts_chairs_up_to_the_cap_and_refuses_one_more():
    """The boundary IS the feature. A test that tried one chair and ten could not see an
    off-by-one, and an off-by-one here is the difference between the cap the default seating was
    measured for (kitchen_build lays out exactly MAX_CHAIRS seats) and one chair with no seat.

    The same uid repeated on purpose: the cap counts chairs, not distinct designs.
    """
    at_the_cap = [_chair(CHAIR_UIDS[0]) for _ in range(MAX_CHAIRS)]
    assert validate_setup(_chair_payload(at_the_cap), [])["chairs"] == at_the_cap, (
        f"{MAX_CHAIRS} chairs is the cap and must be accepted"
    )

    with pytest.raises(SetupError) as exc:
        validate_setup(_chair_payload(at_the_cap + [_chair(CHAIR_UIDS[0])]), [])
    assert str(MAX_CHAIRS) in str(exc.value), (
        f"the refusal has to say what the limit is: {exc.value}"
    )


@needs_chairs
def test_too_many_chairs_is_reported_as_too_many_not_as_a_bad_chair():
    """The cap is checked BEFORE the uids, so a list that is both too long and contains a stranger
    is refused for the reason the author can act on. Checking the uids first would answer
    "Chair 7: unknown chair" -- a message that invites them to swap that one chair and try again,
    which cannot work, because six is the limit whatever the seventh is."""
    seven = ([_chair(CHAIR_UIDS[i % len(CHAIR_UIDS)]) for i in range(MAX_CHAIRS)]
             + [_chair("not-a-chair")])
    with pytest.raises(SetupError) as exc:
        validate_setup(_chair_payload(seven), [])
    assert "unknown chair" not in str(exc.value).lower(), (
        f"an over-long list was refused for its contents, not its length: {exc.value}"
    )
    assert str(MAX_CHAIRS) in str(exc.value), exc.value


def test_no_chairs_at_all_is_the_ordinary_kitchen():
    """Absent and empty both mean "no chairs" -- the default, and the only thing a checkout with
    no chair library can build."""
    assert validate_setup(_chair_payload([]), [])["chairs"] == []
    bare = {"action": "generate", "kitchen_num": "7", "kitchen_name": "island", "objects": []}
    assert validate_setup(bare, [])["chairs"] == []


@pytest.mark.parametrize("chairs, fragment", [
    ([{"uid": "not-a-chair"}], "unknown chair"),
    ([{"uid": None}], "unknown chair"),
    ([{"uid": 12}], "unknown chair"),
    ("a-single-uid-not-in-a-list", "list of chairs"),
    ({"uid": "x"}, "list of chairs"),
    ([None], "expected a chair"),
    ([12], "expected a chair"),
    # The OLD shape, refused by name. A flat uid list is what every call site posted before a
    # chair carried its own scale, and a migration that missed one has to fail loudly here rather
    # than build a kitchen that quietly ignores the author's edit.
    (["a-bare-uid"], "not a bare uid"),
])
def test_a_chair_list_that_cannot_be_built_is_refused_with_a_reason(chairs, fragment):
    """Every reason is shown inline on the form. A bare string is the trap worth naming: it is
    iterable, so a per-item loop with no list check would walk it character by character and
    report 'Chair 1: unknown chair'."""
    with pytest.raises(SetupError) as exc:
        validate_setup(_chair_payload(chairs), [])
    assert fragment in str(exc.value)


@needs_chairs
def test_an_unknown_chair_is_refused_even_when_the_registry_has_chairs():
    with pytest.raises(SetupError) as exc:
        validate_setup(_chair_payload([_chair(CHAIR_UIDS[0]), _chair("not-a-chair")]), [])
    assert "Chair 2" in str(exc.value), (
        f"the reason must name WHICH chair is wrong: {exc.value}"
    )


@needs_chairs
def test_pick_chair_carries_the_chairs_already_placed_through_the_gallery():
    """Mirrors test_pick_table_returns_the_state_it_was_given, and matters more here: the picker
    APPENDS to this list, so a state that lost it would silently restart the count at one."""
    parsed = validate_setup({
        "action": "pick_chair", "kitchen_num": "12", "kitchen_name": "island", "seed": None,
        "objects": [], "table": None, "chairs": [_chair(CHAIR_UIDS[0])],
    }, MESHES)

    assert parsed["action"] == "pick_chair"
    assert parsed["row"] is None
    assert parsed["state"]["chairs"] == [_chair(CHAIR_UIDS[0])]


@needs_chairs
def test_browsing_a_gallery_refuses_a_seventh_chair_before_the_gallery_is_built():
    """The cap applies to every action that carries chairs, not only to generate -- otherwise the
    author browses, picks, and is refused only on the next submit."""
    with pytest.raises(SetupError):
        validate_setup(_chair_payload([_chair(CHAIR_UIDS[0])] * (MAX_CHAIRS + 1),
                                      action="pick_chair"), [])


def test_the_cap_is_never_more_than_the_default_seating_was_laid_out_for():
    """The cap must not exceed kitchen_build's DEFAULT_CHAIR_ROW. They are separate constants --
    the cap is an authoring judgement, the row is geometry -- and only ONE direction is a defect.

    This asserted equality, and equality is not the invariant. A cap ABOVE the row is the measured
    failure: add_chair draws chair i from a max(DEFAULT_CHAIR_ROW, i + 1)-seat ring, so chairs past
    the row come off rings of different radii and land beside neighbours they were never laid out
    against -- at MAX_CHAIRS 8 with the row at 6 that is 34 of 50 chairs rejected. A cap BELOW the
    row is merely a roomier ring with empty seats at the end, which is what ships today (cap 6,
    row 8) and which was swept: five layouts x all fifty mesh tables x six chairs, 0 problems,
    0.0 mm. Pinning equality made that safe state fail.

    Raising the cap to 8 is the next step and this test is what will still hold afterwards.
    """
    from kitchen_build import DEFAULT_CHAIR_ROW

    assert MAX_CHAIRS <= DEFAULT_CHAIR_ROW, (
        f"the wizard allows {MAX_CHAIRS} chairs but the default layout seats only "
        f"{DEFAULT_CHAIR_ROW} -- the chairs past the row come off differently sized rings"
    )


def test_form_payload_carries_the_chair_labels_and_the_cap(tmp_path):
    """The page shows what is placed, and hides the picker at the cap, without importing the
    registry or repeating the number in its own JS."""
    payload = form_payload(str(tmp_path))
    assert payload["max_chairs"] == MAX_CHAIRS
    assert payload["chair_variants"] == {v.uid: v.label for v in CHAIR_VARIANTS}


# ---- chair counts: one row per distinct chair, with + and − ----

needs_two_chairs = pytest.mark.skipif(
    len(CHAIR_UIDS) < 2, reason="need two distinct chairs to tell a grouped row from a flat one"
)


@needs_two_chairs
def test_the_form_groups_repeated_chairs_into_one_row_with_a_count(tmp_path):
    """Six chairs used to be six rows, and a second copy of one meant another trip through a
    twenty-tile gallery. One row per DISTINCT chair, carrying how many of it are placed.

    THE FIXTURE IS THE TEST. First-appearance order is the point — adding a copy must not reorder
    the list under the author's cursor — so the list is built to DISAGREE with the two orders a
    grouping is most likely to fall into by accident:

      * counts descending: the second chair placed is the one with more copies, so a
        `sorted(..., key=-count)` reads (second, first);
      * the registry's own order (and, on this registry, alphabetical by label): the chair placed
        SECOND is the one the registry lists first.

    A list like [A, B, A] — one repeat, the first chair the commonest — agrees with all three and
    would pass against any of them.

    Non-adjacent repeats on purpose too: a grouping that only collapsed neighbours would give four
    rows here.
    """
    state = {"chairs": [_chair(CHAIR_UIDS[1]), _chair(CHAIR_UIDS[0]), _chair(CHAIR_UIDS[1]),
                        _chair(CHAIR_UIDS[0]), _chair(CHAIR_UIDS[0])]}
    rows = form_payload(str(tmp_path), state)["chair_rows"]

    assert [(r["uid"], r["count"]) for r in rows] == [(CHAIR_UIDS[1], 2), (CHAIR_UIDS[0], 3)], rows
    assert [r["label"] for r in rows] == [CHAIR_VARIANTS[1].label, CHAIR_VARIANTS[0].label], (
        f"a row shows the uid where the registry has a label: {rows}"
    )
    assert [r["indices"] for r in rows] == [[0, 2], [1, 3, 4]], (
        f"a row must say WHICH chairs it is showing, or Edit could not change them: {rows}"
    )


@needs_chairs
def test_two_chairs_of_one_design_split_into_two_rows_once_they_differ(tmp_path):
    """The grouping is on the whole spec, not on the uid, and that is what per-piece edits force.

    Keyed by uid alone, these three would be one row with a count of 3, one stepper and one Edit
    button -- for chairs that are no longer the same object. Keyed by the spec they are three rows,
    each of which is exactly "the chairs that are identical", which is the only reading under which
    + ("one more of this") and Edit ("change this") are true statements about a row.
    """
    state = {"chairs": [_chair(CHAIR_UIDS[0]),
                        _chair(CHAIR_UIDS[0], scale=1.15),
                        _chair(CHAIR_UIDS[0], material="cabinet")]}
    rows = form_payload(str(tmp_path), state)["chair_rows"]

    assert [(r["scale"], r["material"], r["count"]) for r in rows] == [
        (1.0, None, 1), (1.15, None, 1), (1.0, "cabinet", 1),
    ], rows
    assert {r["uid"] for r in rows} == {CHAIR_UIDS[0]}, "the three rows are one design"
    # And the unedited one says nothing, so the edited ones are what stands out.
    assert [r["edit"] for r in rows] == ["", "1.15×", "cabinet"], rows


@needs_chairs
def test_an_unedited_kitchen_groups_exactly_as_it_did_before_edits_existed(tmp_path):
    """The grouping key changed from the uid to the whole spec, and this is the promise that
    change carries: every kitchen that predates per-piece edits has one spec per uid, so it groups
    identically. Without it the change would be a silent behaviour change for every existing
    kitchen rather than a new capability."""
    chairs = [_chair(CHAIR_UIDS[0]), _chair(CHAIR_UIDS[1]), _chair(CHAIR_UIDS[0])]
    rows = form_payload(str(tmp_path), {"chairs": chairs})["chair_rows"]
    assert [(r["uid"], r["count"]) for r in rows] == [(CHAIR_UIDS[0], 2), (CHAIR_UIDS[1], 1)], rows


@needs_chairs
def test_a_group_row_knows_when_plus_is_exhausted(tmp_path):
    """The cap counts CHAIRS, not designs, so ONE row of MAX_CHAIRS copies is at the cap exactly as
    MAX_CHAIRS different chairs are — the case a `len(chair_rows) < MAX_CHAIRS` reading of the cap
    would get wrong, and the one the + button is drawn from."""
    at_the_cap = form_payload(str(tmp_path),
                              {"chairs": [_chair(CHAIR_UIDS[0]) for _ in range(MAX_CHAIRS)]})
    assert at_the_cap["chair_rows"][0]["can_add"] is False, at_the_cap["chair_rows"]

    one_short = form_payload(
        str(tmp_path), {"chairs": [_chair(CHAIR_UIDS[0]) for _ in range(MAX_CHAIRS - 1)]})
    assert one_short["chair_rows"][0]["can_add"] is True, one_short["chair_rows"]


def test_no_chairs_at_all_is_no_rows(tmp_path):
    """The empty form, and the checkout with no chair library: no rows, not a row of nothing."""
    assert form_payload(str(tmp_path))["chair_rows"] == []
    assert form_payload(str(tmp_path), {"chairs": []})["chair_rows"] == []


@needs_chairs
def test_the_chair_rows_are_wired_to_add_and_remove(tmp_path):
    """Mirrors test_the_chair_button_is_wired_to_pick_chair. + and − are SUBMITS, not local edits:
    the cap refusal has to come from the server, which is the only thing that can say why."""
    page = render_form_page(form_payload(str(tmp_path), {"chairs": [_chair(CHAIR_UIDS[0])]}))
    for name, action in (("plus", "add_chair"), ("minus", "remove_chair"), ("edit", "edit_chair")):
        assert re.search(
            name + r"\.addEventListener\('click', function \(\) \{\s*"
            + r"submit\('" + action + r"', undefined, rowIndex\);\s*\}\);", page,
        ), f"the {name} button is not wired to submit('{action}', …, that row's INDEX)"
    assert "if (chairRow !== undefined) { body.chair_row = chairRow; }" in page, (
        "submit() does not carry the row index, so all three buttons would name no row"
    )
    assert "plus.disabled = !row.can_add;" in page, "the + button stays live at the cap"


@needs_chairs
@pytest.mark.parametrize("action", ["add_chair", "remove_chair", "edit_chair"])
def test_changing_a_row_names_one_that_is_actually_on_the_form(action):
    """+, − and Edit… name the row they were pressed on, BY INDEX, so it has to be a row the form
    is really showing. Without the check, remove_chair on a row that is not there would silently do
    nothing and add_chair would place a chair the author never picked.

    The index replaced a uid, and that is the shape change per-piece edits force: once one chair of
    a design is edited there are two rows with that uid, so a uid names neither of them. The index
    is exact on both sides because _chair_rows is pure and the form posts back the list it was
    drawn from.
    """
    parsed = validate_setup({
        "action": action, "kitchen_num": "12", "kitchen_name": "island", "seed": None,
        "objects": [], "table": None, "chairs": [_chair(CHAIR_UIDS[0])], "chair_row": 0,
    }, MESHES)
    assert parsed["chair_row"] == 0
    assert parsed["state"]["chairs"] == [_chair(CHAIR_UIDS[0])], (
        "the row actions have to carry the form through, like every other state action"
    )

    # 1 is the sharpest case: one PAST the only row, i.e. the off-by-one a `<=` would let through.
    for row in [None, -1, 1, "0", True, CHAIR_UIDS[0]]:
        with pytest.raises(SetupError) as exc:
            validate_setup({
                "action": action, "kitchen_num": "12", "kitchen_name": "island", "seed": None,
                "objects": [], "table": None, "chairs": [_chair(CHAIR_UIDS[0])], "chair_row": row,
            }, MESHES)
        assert "no such row" in str(exc.value), f"{row!r}: {exc.value}"


@needs_chairs
def test_a_row_index_addresses_the_rows_not_the_chairs():
    """Two chairs of one design that have been edited apart are TWO rows, and the second is row 1 --
    while three identical chairs are ONE row and row 1 does not exist. An index into the flat chair
    list would get both of those backwards."""
    split = [_chair(CHAIR_UIDS[0]), _chair(CHAIR_UIDS[0], scale=1.15)]
    parsed = validate_setup({
        "action": "edit_chair", "kitchen_num": "12", "kitchen_name": "island", "seed": None,
        "objects": [], "table": None, "chairs": split, "chair_row": 1,
    }, MESHES)
    assert parsed["chair_row"] == 1

    same = [_chair(CHAIR_UIDS[0])] * 3
    with pytest.raises(SetupError):
        validate_setup({
            "action": "edit_chair", "kitchen_num": "12", "kitchen_name": "island", "seed": None,
            "objects": [], "table": None, "chairs": same, "chair_row": 1,
        }, MESHES)


@needs_chairs
def test_the_row_indices_the_director_acts_on_come_from_the_same_grouping_the_form_drew():
    """chair_row_indices is the whole contract between the form's row and the state's positions:
    the director rewrites exactly these, and anything else is an edit landing on the wrong chair."""
    chairs = [_chair(CHAIR_UIDS[0]), _chair(CHAIR_UIDS[0], scale=1.15), _chair(CHAIR_UIDS[0])]
    assert kitchen_wizard.chair_row_indices(chairs, 0) == [0, 2]
    assert kitchen_wizard.chair_row_indices(chairs, 1) == [1]
    # No such row is "nothing to do", never an exception and never row 0 by accident.
    for row in (2, -1, None, "0", True):
        assert kitchen_wizard.chair_row_indices(chairs, row) == [], row


def test_the_edit_step_offers_only_what_validate_setup_will_accept():
    """The menu and the check are one decision. An edit page offering a scale the form then refuses
    is a step the author can complete and cannot submit; the reverse leaves a value unreachable."""
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    base = {"action": "generate", "kitchen_num": "7", "kitchen_name": "island", "objects": [],
            "table": TABLE_VARIANTS[0].key}
    for scale in FURNITURE_SCALES:
        for group in FURNITURE_MATERIAL_GROUPS:
            parsed = validate_setup(
                dict(base, table_scale=scale, table_material=group), MESHES
            )
            assert parsed["table_scale"] == scale and parsed["table_material"] == group

    # Every id the page can post is one of those pairs -- on its own for Accept, and again behind
    # every robot the selector offers -- plus Cancel.
    from kitchen_preview import ROBOTS

    prefix = kitchen_wizard.ROBOT_ANSWER_PREFIX
    ids = kitchen_wizard.edit_answer_ids(FURNITURE_SCALES, FURNITURE_MATERIAL_GROUPS)
    assert ids[-1] == "", "the edit step publishes no Cancel id, so its Cancel button is dead"
    pairs = len(FURNITURE_SCALES) * len(FURNITURE_MATERIAL_GROUPS)
    assert len(ids) == pairs * (1 + 1 + len(ROBOTS)) + 1, (
        "the whitelist does not cover every (robot, scale, material) the page can post"
    )
    for answer in ids[:-1]:
        # A robot pick is the same pair behind a robot -- "" for No robot -- so both shapes are
        # checked by stripping the prefix and reading the pair that is left.
        if answer.startswith(prefix):
            robot, _, answer = answer[len(prefix):].partition(":")
            assert robot in ("", *ROBOTS), robot
        raw_scale, _, group = answer.partition(":")
        assert group in FURNITURE_MATERIAL_GROUPS, answer
        assert any(abs(float(raw_scale) - s) < 1e-9 for s in FURNITURE_SCALES), answer


def _edit_page(**overrides):
    """The edit page, rendered over a 0.55 x 0.50 x 0.90 m stand-in piece."""
    import trimesh

    from kitchen_build import (
        FURNITURE_MATERIAL_GROUPS, FURNITURE_MATERIAL_SWATCHES, FURNITURE_SCALES,
    )

    kwargs = dict(
        title="Edit Chair 3", text="uid abc123.", scales=FURNITURE_SCALES, scale=1.0,
        materials=FURNITURE_MATERIAL_GROUPS, material="floor",
        piece=trimesh.Scene(trimesh.creation.box(extents=(0.55, 0.5, 0.9))),
        swatches=FURNITURE_MATERIAL_SWATCHES,
    )
    kwargs.update(overrides)
    return kitchen_wizard.render_edit_page(**kwargs)


def _edit_payload(page):
    return json.loads(re.search(r"var DATA = (\{.*?\});\n", page, re.S).group(1))


def test_the_edit_page_draws_the_piece_and_the_scale_reference_beside_it():
    """The whole point of the step being 3D: the author is choosing a SIZE, and a row of buttons
    reading 1.30x says nothing about whether the chair is too big for the room.

    The piece goes in as ONE node with an identity transform, centred in plan and standing on
    z = 0, because that is what makes the browser's scale.set(s, s, 1) a plan-view resize about the
    piece's own centre rather than a slide across the view.

    The reference beside it is the same one the kitchen preview draws, for whichever robot the
    page's selector is set to -- kitchen_preview.DEFAULT_ROBOT until the author picks another --
    and its RIGHT EDGE has to clear the piece, not the robot's midline: the assembly reaches about
    0.4 m to the robot's right, so anchoring on the robot would stand the dimension line inside
    the table.
    """
    import trimesh

    from kitchen_preview import DEFAULT_ROBOT, ROBOTS, robot_mesh, robot_node, rule_node

    height = ROBOTS[DEFAULT_ROBOT]["height_m"]
    figure, rule = robot_node(DEFAULT_ROBOT), rule_node(DEFAULT_ROBOT)

    piece = trimesh.Scene(trimesh.creation.box(extents=(0.55, 0.5, 0.9)))
    page = _edit_page(piece=piece)
    payload = _edit_payload(page)

    assert payload["node"] == kitchen_wizard._EDIT_PIECE_NODE
    assert [round(v, 6) for v in payload["size"]] == [0.55, 0.5, 0.9]
    assert payload["robot"] == DEFAULT_ROBOT, (
        "the edit page no longer opens on a robot, so it is not a comparison any more"
    )

    # Read back the GLB the page really embeds, so this is geometry and not only JSON.
    glb = base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))
    drawn = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")

    want = {figure: height, rule: height}
    if robot_mesh(DEFAULT_ROBOT) is None:
        want.pop(figure)            # a checkout without the cached asset draws the rule alone
    assert kitchen_wizard._EDIT_PIECE_NODE in drawn.geometry, sorted(drawn.geometry)
    for node in want:
        assert node in drawn.geometry, sorted(drawn.geometry)

    def world_bounds(name):
        transform, geometry = drawn.graph[name]
        mesh = drawn.geometry[geometry].copy()
        mesh.apply_transform(transform)
        return mesh.bounds

    piece_bounds = world_bounds(kitchen_wizard._EDIT_PIECE_NODE)
    assert abs(piece_bounds[0][2]) < 1e-6, "the piece does not stand on z = 0"
    assert abs(piece_bounds[0][0] + piece_bounds[1][0]) < 1e-6, "the piece is not centred in plan"

    # To the LEFT, and clear of the piece even at the biggest scale it can be dragged to. Measured
    # on the RIGHTMOST piece of the reference, whichever that is.
    grown = piece_bounds[0][0] * max(payload["scales"])
    for node, height in want.items():
        bounds = world_bounds(node)
        # 1e-6: the GLB stores vertices as float32, so the round trip is not bit-exact.
        assert abs(bounds[0][2]) < 1e-5, f"{node} does not stand on z = 0"
        assert abs(bounds[1][2] - height) < 1e-5, (
            f"{node} on the page is {bounds[1][2]:.6f} m tall, not {height:.6f} m"
        )
        assert bounds[1][0] < grown, (
            f"{node} is inside the piece at the top of the range: its right edge is "
            f"{bounds[1][0]:.2f}, the piece's left edge at max scale is {grown:.2f}"
        )

    # Its dimension label comes with it: geometry cannot carry text, so a page that drew the rule
    # and forgot the label would show one unexplained stick.
    texts = [item["text"] for item in payload["scale_labels"]]
    assert len(texts) == 1, texts
    assert ROBOTS[DEFAULT_ROBOT]["label"] in texts[0] and f"{height:.2f}" in texts[0], texts


def test_the_edit_page_never_touches_the_scene_it_was_given():
    """The table gallery caches its scenes for the whole run. A robot added to the scene the page
    was handed would stand in the table gallery's tile from then on -- one more of it per visit."""
    import trimesh

    from kitchen_preview import SCALE_NODE_PREFIX

    piece = trimesh.Scene(trimesh.creation.box(extents=(0.55, 0.5, 0.9)))
    before = set(piece.graph.nodes)

    _edit_page(piece=piece)
    _edit_page(piece=piece)

    assert set(piece.graph.nodes) == before, "the edit page mutated the scene it was handed"
    assert not any(n.startswith(SCALE_NODE_PREFIX) for n in piece.geometry), sorted(piece.geometry)


def test_the_edit_page_offers_every_robot_and_no_robot_and_carries_only_the_one_chosen():
    """The preview's selector, on this page: one <select>, No robot first, then every robot in
    kitchen_preview.ROBOTS with the height its dimension line is drawn at.

    ONE ROBOT PER PAGE, checked on the bytes -- a page carries the robot it was rendered for and
    NEITHER of the other two, which is what makes the choice server-side rather than a client-side
    toggle over 25 MB of embedded geometry. No robot carries none of it: 8-18 MB lighter, which is
    the reason that option is here and not only tidiness.
    """
    from kitchen_preview import DEFAULT_ROBOT, ROBOTS, robot_node, rule_node

    payload = _edit_payload(_edit_page())
    assert payload["robot"] == DEFAULT_ROBOT
    assert [c["id"] for c in payload["robot_choices"]] == list(ROBOTS)
    assert [c["height_m"] for c in payload["robot_choices"]] == [
        spec["height_m"] for spec in ROBOTS.values()
    ], "the selector quotes heights the geometry beside it is not drawn at"

    bare = _edit_page(robot="")
    assert _edit_payload(bare)["robot"] == ""
    assert _edit_payload(bare)["scale_labels"] == []
    assert _edit_payload(bare)["robot_choices"], "No robot dropped the selector with the robot"

    # Read on the EMBEDDED GLB, not on the page text: the scale reference's node names live in the
    # base64 payload, so a page that carried a robot it did not admit to would still look clean
    # from the JSON. Matched as bytes against the GLB's own JSON chunk rather than parsed with
    # trimesh -- these are 5-13 MB each and the question is only which names are in there.
    def embedded(page):
        return base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))

    for robot in ROBOTS:
        page = _edit_page(robot=robot)
        glb = embedded(page)
        assert rule_node(robot).encode() in glb, f"the {robot} page draws no dimension line"
        for other in ROBOTS:
            if other != robot:
                assert robot_node(other).encode() not in glb, (
                    f"the {robot} page also carries {other} -- the choice is not server-side"
                )
        # And the label beside the rule names the robot that is really standing there.
        assert ROBOTS[robot]["label"] in _edit_payload(page)["scale_labels"][0]["text"]
        assert len(bare) < len(page), (
            f"No robot is not cheaper than {robot}, so the option buys nothing"
        )

    with pytest.raises(ValueError):
        _edit_page(robot="h2")


def test_picking_a_robot_on_the_edit_page_sends_the_pending_scale_and_material_with_it():
    """THE REGRESSION THIS SELECTOR CAN CAUSE. The slider and the material menu are client state --
    neither is posted until Accept -- while the robot is chosen server-side, so a pick that carried
    only the robot would come back re-rendered at the values the step opened on and silently throw
    away whatever the author had dialled in.

    answerId() is exactly that pending pair, so the id posted is
    "robot:<name>:<scale>:<group>" -- and the director re-publishes from it. That the director
    really does is pinned in test_generator_progress; this is the browser's half, read out of the
    rendered page because no JS engine exists on this box (node, deno and bun are all absent).
    """
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    page = _edit_page()
    prefix = kitchen_wizard.ROBOT_ANSWER_PREFIX

    assert f"answer('{prefix}' + robots.value + ':' + answerId())" in page, (
        "the pick does not carry answerId(), so the re-render loses the pending scale and material"
    )
    # And answerId() is the pair itself, not a re-read of what the page opened on.
    assert "return scaleNow().toFixed(2) + ':' + material;" in page

    # Every id that composition can produce is one the step publishes, so a pick can never be
    # refused with a 400 the author has no way to act on.
    ids = set(kitchen_wizard.edit_answer_ids(FURNITURE_SCALES, FURNITURE_MATERIAL_GROUPS))
    offered = [""] + [c["id"] for c in _edit_payload(page)["robot_choices"]]
    for robot in offered:
        for scale in FURNITURE_SCALES:
            for group in FURNITURE_MATERIAL_GROUPS:
                assert f"{prefix}{robot}:{scale:.2f}:{group}" in ids, (
                    f"the selector can post {robot!r} with {scale} / {group} and the step "
                    f"refuses it"
                )


# --- the robot step -----------------------------------------------------------------------------
# The first page of the run: three robots at true size, turning, and a list to pick from. It is the
# one page that carries every robot at once, which is a deliberate break from the one-at-a-time
# rule the preview and the edit page follow -- the choice being made here IS which robot.


def _robot_page(**overrides):
    kwargs = dict(title="Pick your robot", text="One robot for this run.")
    kwargs.update(overrides)
    return kitchen_wizard.render_robot_page(**kwargs)


def _robot_payload(page):
    return json.loads(re.search(r"var DATA = (\{.*?\});\n", page, re.S).group(1))


def test_the_robot_step_shows_every_robot_at_its_own_height():
    """TRUE SIZE, side by side, each beside a dimension line at its measured standing height --
    which is the whole reason this is not a kitchen_gallery page. kitchen_tiles normalises every
    tile to the same size, and "how tall is it" is the question being asked here.

    The heights are read off the GEOMETRY the page embeds, not off the payload, so a page that
    quoted 1.61 m in the panel and drew something else would fail here.
    """
    import math

    import trimesh

    from kitchen_preview import ROBOTS, robot_mesh, robot_node, rule_node

    page = _robot_page()
    payload = _robot_payload(page)

    assert [t["id"] for t in payload["tiles"]] == list(ROBOTS)
    assert [t["label"] for t in payload["tiles"]] == [s["label"] for s in ROBOTS.values()]
    assert [t["height_m"] for t in payload["tiles"]] == [s["height_m"] for s in ROBOTS.values()]

    glb = base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))
    drawn = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")

    def world_bounds(name):
        transform, geometry = drawn.graph[name]
        mesh = drawn.geometry[geometry].copy()
        mesh.apply_transform(transform)
        return mesh.bounds

    spans = []
    for name, spec in ROBOTS.items():
        wanted = [rule_node(name)] + ([] if robot_mesh(name) is None else [robot_node(name)])
        for node in wanted:
            assert node in drawn.geometry, sorted(drawn.geometry)
            bounds = world_bounds(node)
            # 1e-5: the GLB stores vertices as float32, so the round trip is not bit-exact.
            assert abs(float(bounds[0][2])) < 1e-4, f"{node} does not stand on z = 0"
            assert abs(float(bounds[1][2]) - spec["height_m"]) < 1e-4, (
                f"{node} is {bounds[1][2]:.6f} m tall on the page, not {spec['height_m']:.6f} m"
            )
        assembly = [world_bounds(n) for n in wanted]
        lo = min(float(b[0][0]) for b in assembly)
        hi = max(float(b[1][0]) for b in assembly)
        # THE SLOT IS THE TURNING CIRCLE, because these robots rotate: a robot spins about its own
        # plan centre, so what it can reach is its half-DIAGONAL, not the half-span it stands in.
        # Checked on the drawn geometry, so a layout that packed them on their static footprints
        # (each robot passing through its neighbour's dimension line twice a revolution) fails
        # here rather than in the browser, where nothing on this box can look.
        mesh = robot_mesh(name)
        if mesh is not None:
            figure = world_bounds(robot_node(name))
            centre = (float(figure[0][0]) + float(figure[1][0])) / 2.0
            turn = math.hypot(*[float(v) for v in mesh.extents[:2]]) / 2.0
            lo, hi = min(lo, centre - turn), max(hi, centre + turn)
        spans.append((lo, hi))

    for (_lo, hi), (next_lo, _next_hi) in zip(spans, spans[1:]):
        assert next_lo > hi, f"the robots sweep into each other as they turn: {spans}"

    # Its label is drawn for each, or a page of three unexplained sticks.
    texts = [item["text"] for item in payload["scale_labels"]]
    assert len(texts) == len(ROBOTS), texts
    for name, spec in ROBOTS.items():
        assert any(spec["label"] in t and f"{spec['height_m']:.2f}" in t for t in texts), texts


def test_the_robot_step_publishes_every_answer_its_list_can_produce():
    """The galleries' rule on a page with no paging: the server refuses an id the step did not
    publish, so every button on this panel has to be in robot_answer_ids()."""
    from kitchen_preview import ROBOTS

    ids = kitchen_wizard.robot_answer_ids()
    assert ids == list(ROBOTS) + [""], ids
    for tile in _robot_payload(_robot_page())["tiles"]:
        assert tile["id"] in ids, tile


def test_the_robots_turn_and_stop_turning_for_a_reader_who_asked_them_to():
    """A slow turntable, which is the cheap and honest version of "make it move": these are static
    meshes with no rig, and turning a node is the only motion that is not a lie about them.

    prefers-reduced-motion is respected, and it is not decoration: this page's animation is a
    large object rotating continuously across most of the viewport, which is exactly what that
    preference exists for. Nothing else changes -- the same three robots, the same list.

    Read out of the rendered page: no JS engine exists on this box (node, deno and bun are all
    absent), so what is checked here is that the code is present and reaches the right nodes.
    """
    from kitchen_preview import robot_mesh, robot_node, rule_node, sanitize_three_name

    page = _robot_page()
    payload = _robot_payload(page)

    assert "(prefers-reduced-motion: reduce)" in page
    assert "if (reduced || !turning.length) {" in page, "the preference is read and then ignored"
    assert "node.rotation.z = angle;" in page, "the robots do not turn about their own up axis"
    assert "window.requestAnimationFrame(frame);" in page

    # The FIGURE turns; the ruler beside it does not. A rule that swung round with the robot would
    # stop measuring anything.
    turning = {t["node"] for t in payload["tiles"]}
    for name in payload["tiles"]:
        assert sanitize_three_name(rule_node(name["id"])) not in turning, name
    for tile in payload["tiles"]:
        if robot_mesh(tile["id"]) is not None:
            assert tile["node"] == sanitize_three_name(robot_node(tile["id"])), tile


def test_the_robot_step_is_the_one_page_that_carries_all_three():
    """Every other page in this wizard embeds ONE robot, chosen server-side, because each is
    5-13 MB of GLB. This one cannot: it is where that choice is made, and showing one at a time
    would be asking the question while hiding the answer.

    So the cost is measured rather than asserted away. 36 MB at the time of writing (5.2 + 8.4 +
    12.9 MB of GLB, base64'd), paid once at the top of a run that then costs a minute of Isaac
    boot. The bound below is a tripwire for a fourth robot or an undecimated re-export, not a
    budget anybody has room in.
    """
    from kitchen_preview import ROBOTS, robot_mesh, robot_node

    page = _robot_page()
    glb = base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))
    for name in ROBOTS:
        if robot_mesh(name) is not None:
            assert robot_node(name).encode() in glb, f"{name} is missing from the page"
    assert len(page) < 60_000_000, f"the robot step is {len(page):,} bytes"


def test_the_robot_step_marks_the_one_the_run_already_has():
    """Cancel on this step keeps the current robot (the galleries' convention), so the page has to
    say which one that is -- otherwise Cancel is a choice made blind."""
    page = _robot_page(chosen="anubis")

    assert _robot_payload(page)["chosen"] == "anubis"
    assert "if (t.id === DATA.chosen) { b.className = 'on'; }" in page
    assert "Cancel keeps " in page
    assert _robot_payload(_robot_page())["chosen"] == ""


def test_the_robot_step_refuses_a_robot_it_has_no_registry_entry_for():
    """The same guard render_edit_page has, and for the same reason: a name no robot answers to
    would otherwise fail deep inside scale_reference_extents with a KeyError."""
    with pytest.raises(ValueError):
        _robot_page(robots=["h2"])


# --- the chair edit page's table --------------------------------------------------------------
# A chair's size only means something against the table it goes with, so the chair's edit page
# draws the chosen table -- the EDITED one -- at the seat the build would put the chair in. It is
# the first display-only FURNITURE any page has added, so the containment claims that were written
# for the robot apply to it, and the tests below hold them the same way.


def _a_table(**overrides):
    """The `table` argument, over a 1.40 x 0.80 x 0.74 m stand-in top."""
    import trimesh

    from kitchen_build import DEFAULT_FURNITURE_MATERIAL, FURNITURE_SCALES

    table = dict(
        scene=trimesh.Scene(trimesh.creation.box(extents=(1.4, 0.8, 0.74))),
        label="Table 4", detail="Long dining", material=DEFAULT_FURNITURE_MATERIAL,
        # One per offered scale, rising with it: the ring seats a bigger chair further out.
        offsets_m=[0.4 + 0.5 * s for s in FURNITURE_SCALES],
    )
    table.update(overrides)
    return table


def test_the_chair_edit_page_seats_the_chosen_table_where_the_ring_would_put_it():
    """The table goes in AT THE SEAT, not beside the chair: the standoff comes from the director's
    reading of kitchen_build's own seating ring (one offset per slider step) and the page stands
    the table's plan centre that far in +y from the chair.

    +y and not -x, where the robot is, because that is where the ring's one-seat arrangement puts
    a chair -- at the middle of the table's -y side, the side away from the counter run. The
    camera looks from -y, so the reader sees the chair at the near edge with the work surface
    behind it.
    """
    import trimesh

    from kitchen_build import FURNITURE_SCALES

    table = _a_table()
    page = _edit_page(table=table, scale=1.0, scales=FURNITURE_SCALES)
    payload = _edit_payload(page)

    assert payload["table"]["node"] == kitchen_wizard._EDIT_TABLE_NODE
    assert payload["table"]["label"] == "Table 4" and payload["table"]["detail"] == "Long dining"
    assert payload["table"]["offsets_m"] == table["offsets_m"], (
        "the page did not carry the ring's per-scale standoffs, so the browser cannot re-seat it"
    )
    assert [round(v, 6) for v in payload["table"]["size"]] == [1.4, 0.8, 0.74]

    # Read back the GLB the page really embeds: this is a placement, so it has to be geometry.
    glb = base64.b64decode(re.search(r'base64_data\s*=\s*"([A-Za-z0-9+/=]+)"', page).group(1))
    drawn = trimesh.load(trimesh.util.wrap_as_stream(glb), file_type="glb")
    transform, geometry = drawn.graph[kitchen_wizard._EDIT_TABLE_NODE]
    mesh = drawn.geometry[geometry].copy()
    mesh.apply_transform(transform)

    at_one = table["offsets_m"][list(FURNITURE_SCALES).index(1.0)]
    centre = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    assert abs(float(centre[1]) - at_one) < 1e-5, (
        f"the table's plan centre is at y={centre[1]:.3f}, not at the seat's {at_one:.3f}"
    )
    assert abs(float(centre[0])) < 1e-5, "the table is not centred on the chair in x"
    assert abs(float(mesh.bounds[0][2])) < 1e-5, "the table does not stand on z = 0"
    # The offset is a NODE transform, which is what the browser moves as the slider is dragged --
    # baked into the vertices it would be nothing three.js could reposition.
    assert abs(float(transform[1][3]) - at_one) < 1e-5, (
        f"the offset is not on the node: {transform[1][3]}"
    )


def test_the_chair_edit_page_moves_the_table_as_the_chair_is_resized():
    """The build reflows the ring when the chair grows (kitchen_build._seats_around_table takes
    the chair's footprint), so the page has to as well -- a table that stayed put would show a
    1.30x chair standing in it and a 0.70x one marooned half a metre away.

    The rule is NOT re-derived in the browser: the offsets are enumerated per slider step, exactly
    as the scales themselves are, so there is one seating rule in this project and it is
    kitchen_build's. Read out of the rendered page; there is no JS engine on this box.
    """
    page = _edit_page(table=_a_table())

    assert "var y = DATA.table.offsets_m[Math.min(index, DATA.table.offsets_m.length - 1)];" in page
    assert "tableNode.position.y = y;" in page
    assert "seatTable();" in page, "the slider does not re-seat the table"


def test_the_chair_edit_page_says_so_when_no_table_has_been_chosen():
    """The toggle stays, disabled, and says why. A control that vanished would leave the author
    wondering whether the feature exists; one that stayed live would do nothing.

    And the reason comes from the CALLER, because the same page draws the table's own edit step,
    where "the table it goes with" is not a question about anything -- there the block is not
    drawn at all. One page, two pieces, and the difference is a sentence rather than a branch.
    """
    prompt = "Pick one with Table… on the setup form."
    page = _edit_page(table_prompt=prompt)     # a chair, with no table chosen yet
    payload = _edit_payload(page)

    assert payload["table"] is None
    assert payload["table_prompt"] == prompt
    assert 'id="showtable"' in page and 'id="tablenote"' in page and 'id="tablesection"' in page
    assert "showtable.disabled = true;" in page
    assert "if (!DATA.table_prompt) {" in page, (
        "the table block is drawn on a page that has no table question to ask"
    )

    # The table's own edit step: no table, no prompt, and the block hidden.
    bare = _edit_page()
    assert _edit_payload(bare)["table_prompt"] == ""
    assert "document.getElementById('tablesection').style.display = 'none';" in bare


def test_the_table_on_the_edit_page_is_named_as_the_ruler_it_is():
    """CONTAINMENT. This is display-only furniture, and the four places nothing display-only may
    reach are all checked by NAME -- "no node whose name starts with SCALE_NODE_PREFIX reaches
    `objects` / the placement gate / the USD export / _label_supports" (see
    test_kitchen_preview's four containment tests). Naming the table with that prefix enrols it in
    all four rather than starting a second mechanism that has to be kept in step.

    Nothing here can reach a committed kitchen today: this page builds its own scene from nothing
    and only renders it. That is exactly the state the robot was in before render_page started
    drawing it into a copy of the kitchen, which is why the name matters now and not later.
    """
    from kitchen_preview import SCALE_NODE_PREFIX

    assert kitchen_wizard._EDIT_TABLE_NODE.startswith(SCALE_NODE_PREFIX)
    payload = _edit_payload(_edit_page(table=_a_table()))
    assert payload["table"]["node"].startswith(SCALE_NODE_PREFIX)


def test_the_edit_page_never_touches_the_table_scene_it_was_given():
    """The same claim the piece already has. _edit_scene_for_table builds fresh, but the table
    gallery caches its scenes for the whole run and the two go through the same builder -- a page
    that mutated what it was handed is one refactor away from putting a robot in every tile."""
    import trimesh

    from kitchen_preview import SCALE_NODE_PREFIX

    table = _a_table()
    before = set(table["scene"].graph.nodes)

    _edit_page(table=table)
    _edit_page(table=table)

    assert set(table["scene"].graph.nodes) == before, "the page mutated the table it was handed"
    assert not any(n.startswith(SCALE_NODE_PREFIX) for n in table["scene"].geometry), (
        sorted(table["scene"].geometry)
    )


def test_the_table_is_drawn_client_side_and_the_robot_is_not():
    """Two display-only things on one page, and they are toggled differently ON PURPOSE.

    A robot is 5-13 MB of GLB, so it is chosen SERVER-side and only the chosen one is embedded. A
    table is a few hundred KB, so it is embedded once and shown or hidden in the browser -- which
    costs no round trip, and, decisively, adds nothing to the step's answer whitelist: the robot
    selector already multiplies that list by four (kitchen_wizard.edit_answer_ids), and a
    server-side table toggle would double it again for a control that changes nothing about the
    answer.
    """
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    with_table = _edit_page(table=_a_table())
    without = _edit_page()

    assert "tableNode.visible = showtable.checked;" in with_table, "the toggle is not client-side"
    assert kitchen_wizard.ROBOT_ANSWER_PREFIX + "' + robots.value" in with_table, (
        "the robot is no longer chosen server-side"
    )
    # And the whitelist does not grow for it: nothing about the toggle is an answer.
    ids = kitchen_wizard.edit_answer_ids(FURNITURE_SCALES, FURNITURE_MATERIAL_GROUPS)
    assert not any("table" in i for i in ids), (
        f"the table toggle reached the step's answer ids: {[i for i in ids if 'table' in i]}"
    )
    assert len(with_table) > len(without), "the table was never embedded"
    # And it is small against the robot standing beside it, which is the reason for the split.
    robot_only = len(without) - len(_edit_page(robot=""))
    assert len(with_table) - len(without) < robot_only, (
        f"the table costs {len(with_table) - len(without)} bytes against the robot's {robot_only}"
    )


def test_the_edit_page_still_draws_when_there_is_nothing_to_draw():
    """piece=None is a supported state (a chair whose OBJ has gone missing must still be given a
    material), and No robot is now a supported state too -- together they leave an empty scene,
    which trimesh refuses to export. The step must survive that: the panel IS the decision."""
    page = kitchen_wizard.render_edit_page(
        title="Edit Chair 3", text="uid abc123.", scales=[1.0], scale=1.0,
        materials=["floor"], material="floor", piece=None, robot="",
    )
    assert 'id="robot"' in page and 'id="accept"' in page and 'id="scale-slider"' in page
    assert "--accent" in page, "the fallback page arrived without the shared theme"


def test_the_edit_page_resizes_all_three_axes():
    """The live preview must show what the commit will BUILD, and the builder is uniform now --
    kitchen_build.FURNITURE_SCALE_IS_UNIFORM. This assertion used to pin the opposite
    (`..., scaleNow(), 1`); it is reversed deliberately, and the page has to keep SAYING what that
    costs, because "the work surface has left the 0.74 m the chair library is drawn against" went
    from something the code prevented to something only the author can notice."""
    page = _edit_page()

    assert "piece.scale.set(scaleNow(), scaleNow(), scaleNow());" in page, (
        "the live resize is not a uniform scale on the piece's root"
    )
    assert "0.74" in page and "height" in page


def test_the_edit_pages_slider_can_only_produce_an_offered_scale():
    """The slider's value is an INDEX into the published list, so no amount of dragging can post a
    scale the step did not publish -- the property that stops a control answering something the
    director would have to refuse with a 400."""
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    page = _edit_page()
    payload = _edit_payload(page)

    assert payload["scales"] == list(FURNITURE_SCALES)
    assert "slider.max = DATA.scales.length - 1;" in page
    assert "index = Number(this.value);" in page
    assert "return scaleNow().toFixed(2) + ':' + material;" in page
    ids = set(kitchen_wizard.edit_answer_ids(FURNITURE_SCALES, FURNITURE_MATERIAL_GROUPS))
    for scale in FURNITURE_SCALES:
        assert f"{scale:.2f}:floor" in ids, f"the slider can reach {scale} and the step refuses it"


def test_the_edit_page_marks_the_measured_floor_and_still_offers_below_it():
    """SHOWN, NOT WITHHELD. Where the measured-safe region ends is drawn on the slider's own
    travel, because that is the question -- a sentence cannot say WHERE on the drag it happens --
    and every step below it is still reachable."""
    from kitchen_build import FURNITURE_SCALES

    page = _edit_page(measured_floor=0.95, floor_note="below 0.95x it is under the 0.35 m bar")
    payload = _edit_payload(page)

    assert payload["measured_floor"] == 0.95
    assert "0.35" in payload["floor_note"]
    assert payload["scales"][0] == FURNITURE_SCALES[0] == 0.70, (
        "the marked page stopped offering the bottom of the range"
    )
    assert "shade.style.width" in page, "nothing draws the band on the slider"
    assert 'id="band"' in page

    # And a piece with nothing measured about it gets no mark and no claim.
    unmarked = _edit_payload(_edit_page(measured_floor=None))
    assert unmarked["measured_floor"] is None and unmarked["floor_note"] == ""


def test_the_edit_page_calls_its_material_preview_an_approximation():
    """MDL materials do not exist in this preview -- it is a trimesh scene in three.js, and the MDL
    is bound on the USD stage at commit. A swatch that implied "this is the wood you will get"
    would be worse than no swatch at all, so the page says what it is where the author reads it."""
    from kitchen_build import FURNITURE_MATERIAL_SWATCHES

    page = _edit_page()
    payload = _edit_payload(page)

    assert payload["swatches"] == dict(FURNITURE_MATERIAL_SWATCHES)
    assert "Approximate" in page
    assert "MDL" in page and "commit" in page
    # The tint is applied to the piece's own meshes, and follows the select.
    assert "m.material.color.set(hex);" in page


def test_the_edit_page_shows_what_the_piece_currently_has():
    """Re-opening Edit… on an edited piece must show the edit, not the defaults -- otherwise the
    button reads as "reset" and there is no way to see what a piece is set to."""
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    page = kitchen_wizard.render_edit_page(
        title="Edit Chair 3", text="uid abc123.", scales=FURNITURE_SCALES, scale=1.15,
        materials=FURNITURE_MATERIAL_GROUPS, material="cabinet",
    )
    payload = json.loads(re.search(r"var DATA = (\{.*?\});\n", page, re.S).group(1))
    assert payload["scale"] == 1.15 and payload["material"] == "cabinet"
    assert payload["scales"] == list(FURNITURE_SCALES)
    assert payload["materials"] == list(FURNITURE_MATERIAL_GROUPS)
    # And the page says why there is no height control, on the page rather than only in the source.
    assert "0.74" in page and "height" in page


def test_the_edit_page_answers_with_both_values_at_once():
    """One Accept, carrying both choices. Two separate answers would be two steps, and a Cancel
    between them would leave the piece half-edited."""
    from kitchen_build import FURNITURE_MATERIAL_GROUPS, FURNITURE_SCALES

    page = kitchen_wizard.render_edit_page(
        title="t", text="x", scales=FURNITURE_SCALES, scale=1.0,
        materials=FURNITURE_MATERIAL_GROUPS, material="floor",
    )
    assert "return scaleNow().toFixed(2) + ':' + material;" in page, (
        "the edit page's answer id is not the scale and the material together"
    )
    assert re.search(
        r"getElementById\('accept'\)\.addEventListener\('click', function \(\) \{ "
        r"answer\(answerId\(\)\); \}\);", page,
    ), "Use it is not wired to the answer id"
    assert re.search(
        r"getElementById\('cancel'\)\.addEventListener\('click', function \(\) \{ "
        r"answer\(''\); \}\);", page,
    ), "Cancel does not answer the empty id the galleries use"


def test_the_chair_button_is_wired_to_pick_chair(tmp_path):
    """Mirrors test_the_table_button_is_wired_to_pick_table: the button has to be wired to the
    actual action, not just have the string appear somewhere on the page."""
    page = render_form_page(form_payload(str(tmp_path)))
    chair_click = re.search(
        r"getElementById\('chairbtn'\)\.addEventListener\('click', function \(\) \{"
        r"\s*submit\('pick_chair'\);\s*\}\);",
        page,
    )
    assert chair_click, "the Chair… button is not wired to submit('pick_chair')"


def test_the_form_posts_the_chairs_it_is_holding(tmp_path):
    """The list only reaches the server if submit() puts it in the body. Without this the picker
    would append to a variable nothing ever sent, and every kitchen would build with no chairs."""
    page = render_form_page(form_payload(str(tmp_path)))
    assert re.search(r"\bchairs:\s*chairs\b", page), (
        "submit()'s body does not carry the chair list"
    )


@needs_chairs
def test_the_form_prefill_restores_the_chairs(tmp_path):
    """The other half of the round trip: chairs picked before a gallery visit must survive it,
    or the sixth trip through the picker would come back holding one chair."""
    state = {
        "kitchen_num": "12", "kitchen_name": "island", "seed": None, "objects": [],
        "table": None, "chairs": [_chair(CHAIR_UIDS[0], scale=1.15, material="cabinet")],
    }
    page = render_form_page(form_payload(str(tmp_path), state))
    assert "chairs = (DATA.state.chairs || []).slice();" in page
    assert CHAIR_UIDS[0] in page, "the prefilled state does not carry the chair uid to the page"
    # And the EDIT with it: a gallery trip that restored the uid and dropped the scale would put
    # a chair back on the form at a size the author never chose.
    assert '"scale": 1.15' in page and '"material": "cabinet"' in page, (
        "the prefilled state lost the chair's own scale/material"
    )


def test_a_checkout_with_no_chair_library_offers_none_and_still_builds(tmp_path):
    """The case this machine cannot reach in-process, and the one most likely to be missed.

    CHAIR_VARIANTS is derived at IMPORT from the manifest under CHAIR_OBJ_DIR, so emptying the
    registry means emptying it before kitchen_build is imported -- a fresh interpreter with
    CHAIR_OBJ_DIR pointed at a directory with no manifest in it. chair_manifest.load() returns an
    empty skeleton rather than raising, which is what makes this a supported state instead of a
    broken checkout.

    Three claims in one subprocess, because they share that one import: the registry is empty, the
    form offers no chairs (so the page can disable its own picker), and a kitchen with no chairs
    still validates -- while any uid at all is refused, since there is no such chair to place.
    """
    import json
    import os
    import subprocess
    import sys
    import textwrap

    here = os.path.dirname(os.path.abspath(__file__))
    empty = tmp_path / "no_chair_library"
    empty.mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(f"""
            import json
            import kitchen_wizard
            from kitchen_build import CHAIR_VARIANTS

            payload = kitchen_wizard.form_payload({str(tmp_path)!r})
            base = {{"action": "generate", "kitchen_num": "1", "kitchen_name": "island",
                     "objects": []}}
            none_at_all = kitchen_wizard.validate_setup(dict(base, chairs=[]), [])
            refused = None
            try:
                kitchen_wizard.validate_setup(dict(base, chairs=[{{"uid": "anything"}}]), [])
            except kitchen_wizard.SetupError as exc:
                refused = str(exc)
            print("RESULT " + json.dumps({{
                "offered": len(CHAIR_VARIANTS),
                "form_offers": payload["chair_variants"],
                "max_chairs": payload["max_chairs"],
                "chairs": none_at_all["chairs"],
                "refused": refused,
            }}))
        """)],
        cwd=here, capture_output=True, text=True,
        env=dict(os.environ, CHAIR_OBJ_DIR=str(empty)),
    )
    assert proc.returncode == 0, f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT "))
    result = json.loads(line[len("RESULT "):])

    assert result["offered"] == 0, "CHAIR_OBJ_DIR did not empty the registry; the test proved nothing"
    assert result["form_offers"] == {}, "the form offered chairs that do not exist here"
    assert result["max_chairs"] == MAX_CHAIRS, "the cap vanished with the library"
    assert result["chairs"] == [], "a kitchen with no chairs stopped validating"
    assert result["refused"] and "unknown chair" in result["refused"], (
        f"a uid was accepted with no library to place it from: {result['refused']!r}"
    )


def test_showchairs_hides_the_picker_at_the_cap_and_with_nothing_to_offer(tmp_path):
    """Mirrors test_showtable_renders_both_states_and_is_wired_to_fire. Two states, because they
    have different causes: at the cap there is nothing left to add, and with no chair library
    there was never anything to add -- and a live button in either case is a guaranteed refusal."""
    page = render_form_page(form_payload(str(tmp_path)))
    assert "btn.disabled = (offered === 0) || (chairs.length >= max);" in page, (
        "the Chair… button is not disabled at the cap and with an empty library"
    )
    assert "'no chair library on this machine'" in page
    assert page.count("showChairs();") >= 1, "showChairs() must be invoked at load"


def test_every_getelementbyid_resolves_on_the_page_that_asks_for_it(tmp_path):
    """Every id the browser JS reaches for is really in the markup of a page that runs that JS.

    This is the failure mode a re-skin produces and nothing else here can catch: rename or drop
    one id while restructuring the markup and the control it belongs to goes dead silently. No
    JS engine exists on this box -- node, deno and bun are all absent -- so the page cannot be
    executed to find out. Reading the rendered HTML back is the substitute.

    Scoped per page rather than over the union: `list` and `ok` live only on the scrolling-list
    choice page, `detail` and `fill` only on progress, so asserting every id on every page would
    be wrong. The rule is the real one -- a page's own script must not query an id that page
    does not have.
    """
    source = pathlib.Path(kitchen_wizard.__file__).read_text()
    wanted = set(re.findall(r"getElementById\('([A-Za-z0-9_-]+)'\)", source))
    assert wanted, "no getElementById calls found -- this test has lost its subject"

    for name, page in _published_pages(tmp_path).items():
        present = re.findall(r'\bid="([A-Za-z0-9_-]+)"', page)
        duplicated = sorted({i for i in present if present.count(i) > 1})
        assert not duplicated, (
            f"{name} page repeats id(s) {duplicated} -- getElementById returns the first, so a "
            "control silently drives the wrong element"
        )
        # Only the ids this page's own inline script actually asks for.
        asked = {i for i in wanted if f"getElementById('{i}')" in page}
        missing = sorted(i for i in asked if i not in set(present))
        assert not missing, (
            f"{name} page's script calls getElementById for {missing}, which the page does not "
            "contain -- that control is dead and no other test can see it"
        )
