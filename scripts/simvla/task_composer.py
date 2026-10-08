"""SimVLA: compose a task by dragging skills, in a browser.

The palette IS the registry. That is only possible because a skill is declared in one place: before
the contract, the GUI's skill list, the goal writer's registry and the executor's dispatch table
were three lists that could silently disagree — and did, which is how `G_b` came to be a skill the
GUI could author and the executor could not run.

Extends kitchen_preview.PreviewServer: the same stdlib http.server on 127.0.0.1, the same three.js
scene, the same raycast picking — so it works over the SSH tunnel already in use. Adds two
endpoints: GET /palette (the registry, as JSON, for the left pane) and POST /template (the composed
sequence, validated by BOTH validate_template and validate_sequence before it is accepted, so an
incoherent script is refused with its reason at authoring time rather than an hour into a GPU run).

Imports the same stack as its siblings: stdlib + skill_contract + task_template + task_validate +
kitchen_preview. It does NOT import skills or torch — REGISTRY is populated by whichever process
imported skills first (the GUI, the tests). Importing this module boots neither Omniverse nor the
executor stack.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import kitchen_preview
from kitchen_preview import PreviewServer, object_node_names
from predicate_contract import REGISTRY as PREDICATE_REGISTRY
from predicate_contract import OptFloat, RoleRef, Text
from scene_spec import (
    OBJECT_TYPES,
    SceneError,
    default_object,
    scene_from_dicts,
)
# SUPPORT_SURFACES is deliberately NOT imported here: it is a PEP 562 module __getattr__ on
# scene_spec that derives itself from kitchen_build, so merely naming it in this import list
# imports kitchen_build -> scene_synthesizer at THIS module's import time. The kitchen generator
# imports this module (through kitchen_wizard) before Omniverse boots, and scene_synthesizer's
# usd_import/usd_export soft-import pxr -- so pulling it in that early silently unbinds their USD
# names for the rest of the process. Read it inside scene_defaults_payload() instead, which only
# runs once the composer page is being built. See kitchen_wizard's module docstring.
from skill_contract import REGISTRY, Bool, Choice, Float, PrimPath
import task_template
from task_template import Role, TaskTemplate, TaskTemplateError, TemplateStep, validate_template
from task_template import template_warnings
from task_validate import SequenceError, validate_sequence


#: Where composed task templates are written — the same dir goal_generator --scene reads.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def template_slug(name: str) -> str:
    """A filesystem-safe stem derived from the task's own name ('Bowl to drawer' -> 'bowl_to_drawer')."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip()).strip("_").lower()


def save_composed_template(template: TaskTemplate, templates_dir=TEMPLATES_DIR) -> Path:
    """Write a composed TaskTemplate to <templates_dir>/<slug(name)>.json, named after the task itself.

    Raises ValueError if the task has no name that yields a usable filename. Returns the Path written.
    """
    slug = template_slug(getattr(template, "name", ""))
    if not slug:
        raise ValueError(
            "This task has no usable name — name it in the composer before saving."
        )
    templates_dir = Path(templates_dir)
    templates_dir.mkdir(parents=True, exist_ok=True)
    out = templates_dir / f"{slug}.json"
    out.write_text(task_template.to_json(template), encoding="utf-8")
    return out


# --- grasp thumbnails: capture the grasp preference at authoring time -----------------------------
# The uncached-grasp problem: plan_arm_grasp calls select_thumbnails_cached(folder), which on a cache
# MISS opens a Tk chooser — fine at a desktop, a silent hang on the headless cluster. These helpers let
# the browser composer capture that same preference up front (over the SSH tunnel it already uses),
# writing the identical selected_indices.json the cluster then reads. The index/hand convention is
# copied verbatim from isaaclab.simvla.utils.parse_thumbnail_name: filenames are 1-based
# (segment_01_...), the cached index is 0-based, so segment_01_left -> [0, "left"]. Getting this wrong
# would index the wrong grasp pose in eef_data. The regex carries no path separators, so a `file`
# param that matches it cannot traverse out of its folder.

_THUMB_RE = re.compile(r"segment_(\d+)_(left|right)\.usda\.png$")


def parse_thumb_name(filename: str):
    """'segment_05_right.usda.png' -> (4, 'right'); None if it does not match. 0-based, like
    isaaclab.simvla.utils.parse_thumbnail_name (filenames are 1-based, the stored index is not)."""
    m = _THUMB_RE.match(filename or "")
    if not m:
        return None
    return (int(m.group(1)) - 1, m.group(2))


def grasp_thumbs_listing(folder) -> list[dict]:
    """[{file, seg, hand}] for every parseable thumbnail PNG in folder, sorted (hand, seg) the same
    way select_thumbnails orders its grid, so the browser shows lefts then rights in segment order."""
    folder = Path(folder)
    out = []
    for name in sorted(p.name for p in folder.glob("*.png")):
        parsed = parse_thumb_name(name)
        if parsed:
            out.append({"file": name, "seg": parsed[0], "hand": parsed[1]})
    out.sort(key=lambda t: (t["hand"], t["seg"]))
    return out


def grasp_selection(folder) -> list[list]:
    """The cached [[seg, hand], ...] from selected_indices.json, or [] if absent/corrupt — the same
    read select_thumbnails_cached does, so a folder already chosen shows its picks pre-selected."""
    cache = Path(folder) / "selected_indices.json"
    if not cache.exists():
        return []
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return [list(item) for item in data.get("selected_indices", [])]


def write_grasp_selection(folder, selected) -> int:
    """Write selected_indices.json in the exact shape select_thumbnails_cached loads: a list of
    [seg:int, hand:'left'|'right']. Refuses any folder that is not a *_segments_thumbnails dir, so a
    bad key can never drop a file into the goals corpus or anywhere else. Returns the count written."""
    folder = Path(folder)
    if not folder.name.endswith("_segments_thumbnails"):
        raise ValueError(f"refusing to write a grasp selection outside a thumbnails folder: {folder}")
    cleaned = []
    for item in selected:
        seg, hand = int(item[0]), item[1]
        if hand not in ("left", "right"):
            raise ValueError(f"bad hand {hand!r} (expected 'left' or 'right')")
        cleaned.append([seg, hand])
    (folder / "selected_indices.json").write_text(
        json.dumps({"selected_indices": cleaned}, indent=2), encoding="utf-8"
    )
    return len(cleaned)


def grasp_thumb_path(folder, filename):
    """The absolute path of `filename` inside `folder`, or None unless it is a real thumbnail basename
    that lives directly in folder — the guard that keeps GET /grasps/thumb from serving arbitrary files."""
    if not _THUMB_RE.match(filename or ""):
        return None
    p = Path(folder) / filename
    if p.parent.resolve() != Path(folder).resolve() or not p.is_file():
        return None
    return p


def param_widgets(skill_id: str) -> list[dict]:
    """One dict per declared param. Returning data rather than widgets is what makes this testable."""
    widgets = []
    for p in REGISTRY[skill_id].params:
        if isinstance(p, Choice):
            widgets.append({"name": p.name, "kind": "choice", "default": p.default,
                            "options": list(p.options)})
        elif isinstance(p, Float):
            default = float(p.default)
            widgets.append({"name": p.name, "kind": "float",
                            "default": default if math.isfinite(default) else None})
        elif isinstance(p, Bool):
            widgets.append({"name": p.name, "kind": "bool", "default": bool(p.default)})
        elif isinstance(p, PrimPath):
            widgets.append({"name": p.name, "kind": "prim_path", "default": ""})
        else:
            raise TypeError(f"{skill_id}: no widget for param {p!r}")
    return widgets


def palette() -> list[dict]:
    """The draggable skill boxes. Sorted by id, so the page is stable across runs."""
    return [
        {
            "id": sid,
            "label": REGISTRY[sid].label,
            "actions": list(REGISTRY[sid].actions),
            "runtime": REGISTRY[sid].is_runtime,
            "params": param_widgets(sid),
        }
        for sid in sorted(REGISTRY)
    ]


def predicate_param_widgets(pid: str) -> list[dict]:
    """One dict per declared param, in the same vocabulary as param_widgets so the page renders
    both palettes with one code path."""
    widgets = []
    for p in PREDICATE_REGISTRY[pid].params:
        if isinstance(p, RoleRef):
            widgets.append({"name": p.name, "kind": "role", "default": ""})
        elif isinstance(p, OptFloat):
            widgets.append({"name": p.name, "kind": "opt_float", "default": None})
        elif isinstance(p, Text):
            widgets.append({"name": p.name, "kind": "text", "default": p.default})
        elif isinstance(p, Choice):
            widgets.append({"name": p.name, "kind": "choice", "default": p.default,
                            "options": list(p.options)})
        elif isinstance(p, Float):
            widgets.append({"name": p.name, "kind": "float", "default": float(p.default)})
        elif isinstance(p, Bool):
            widgets.append({"name": p.name, "kind": "bool", "default": bool(p.default)})
        else:
            raise TypeError(f"{pid}: no widget for param {p!r}")
    return widgets


def predicates_palette() -> list[dict]:
    """The draggable condition boxes. Sorted by id, so the page is stable across runs."""
    return [
        {
            "id": pid,
            "label": PREDICATE_REGISTRY[pid].label,
            "phase": PREDICATE_REGISTRY[pid].phase,
            "params": predicate_param_widgets(pid),
        }
        for pid in sorted(PREDICATE_REGISTRY)
    ]


def scene_defaults_payload() -> list[dict]:
    """The 'add object' menu: each type's default row, plus the surface options.

    One entry per OBJECT_TYPES (sorted, so the page is stable across runs), each being
    default_object("<type>0", type).to_dict() with a "surface_options" list bolted on so the GUI
    renders a new row's placement dropdown from the same source of truth instead of hardcoding the
    surface labels — which is exactly how the palette avoids hardcoding the skill list.
    """
    from scene_spec import SUPPORT_SURFACES      # lazy on purpose; see the import block above

    surfaces = sorted(SUPPORT_SURFACES)
    return [
        {**default_object(f"{t}0", t).to_dict(), "surface_options": surfaces}
        for t in sorted(OBJECT_TYPES)
    ]


def template_from_payload(payload: dict) -> TaskTemplate:
    """What the page POSTs, turned into a template — and refused if it cannot work.

    Role(**r) is deliberate: a role key the dataclass does not accept (a typo, a stale field) raises
    TypeError here rather than being silently swallowed, so a malformed role fails loudly at the door
    instead of vanishing from the task.
    """
    t = TaskTemplate(
        name=payload["name"],
        language=payload.get("language", ""),
        roles=[Role(**r) for r in payload.get("roles", [])],
        steps=[
            TemplateStep(
                skill=s["skill"],
                action=s["action"],
                params=s.get("params", {}),
                language=s.get("language", ""),
            )
            for s in payload.get("steps", [])
        ],
        subtask_groups=[list(g) for g in payload.get("subtask_groups", [])],
        scene=scene_from_dicts(payload.get("scene", [])),
        success=payload.get("success"),
        retry=payload.get("retry"),
    )
    validate_template(t)      # the skill/action/param schema AND the scene
    validate_sequence(t)      # can this script actually run?
    return t


# --- the page --------------------------------------------------------------------------------
# Built on top of kitchen_preview.render_page: it produces the three.js kitchen and its own panel,
# and we inject a second panel (the palette + the sequence) plus a script that (a) lists the palette
# from /palette, (b) lets you drag skills into a sequence and reorder them, (c) renders each step's
# params from its widget kinds, (d) fills the selected step's prim_path when you click a prim in the
# 3D view, and (e) POSTs the sequence to /template, showing the reason inline if it is refused.

def _composer_payload(scene, objects, grasp_thumbs=None) -> dict:
    """The object node -> key map the prim picker needs, sanitized the way three.js will see it.

    Same treatment kitchen_preview gives its objects: an object's graph nodes are matched against a
    clicked mesh's ancestor chain, so they must be the sanitized three.js names, not the raw trimesh
    ones. `key` is what a click fills the selected step's prim_path with — the picked prim's id.
    `grasp` is True when this object has BODex grasp thumbnails to choose from, so the Grasps panel
    lists exactly those objects (and no others).
    """
    grasp_thumbs = grasp_thumbs or {}
    return {
        "objects": [
            {
                "label": obj["label"],
                "key": obj["node_id"],
                "nodes": object_node_names(scene, obj["node_id"]),
                "grasp": obj["node_id"] in grasp_thumbs,
            }
            for obj in objects
        ]
    }


def render_page(scene, objects, materials, supports, joints=None, grasp_thumbs=None,
                actions=None) -> str:
    """The composer page as a string: the kitchen preview, plus the palette + sequence panes."""
    base = kitchen_preview.render_page(scene, objects, materials, supports, joints, actions=actions)
    panel = _COMPOSER_PANEL.replace(
        "__COMPOSER_PAYLOAD__",
        kitchen_preview.script_json(_composer_payload(scene, objects, grasp_thumbs)),
    )
    return base.replace("</body>", panel + "\n</body>")


def composer_page(scene, objects, materials, supports, joints=None, grasp_thumbs=None,
                  actions=None) -> str:
    """The composer page, with the skill registry populated first.

    The one entry point that puts the composer in front of a user, and therefore the one place the
    @skill decorators must have run: GET /palette IS the registry, so without this the composer
    opens with an empty palette and nothing to drag.
    """
    _ensure_skills_loaded()
    return render_page(scene, objects, materials, supports, joints, grasp_thumbs, actions)


def build_preview_html(scene, objects, materials, supports, out_path=None, joints=None,
                       grasp_thumbs=None) -> Path:
    """Write the composer page to a file and return its path."""
    if out_path is None:
        out_path = Path(tempfile.mkdtemp(prefix="simvla_task_composer_")) / "composer.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_page(scene, objects, materials, supports, joints, grasp_thumbs), encoding="utf-8"
    )
    return out_path


class ComposerServer(PreviewServer):
    """PreviewServer plus GET /palette and POST /template.

    /palette returns exactly palette() — every registered skill, nothing else. /template turns the
    posted payload into a TaskTemplate, validates it (schema AND sequence), and either stores it and
    answers {"ok": true} or answers 400 with the validation reason in the body so the page can show
    it inline. A validation failure never comes back as 200, and never as a bare 500 stack trace.
    """

    def __init__(self, page: str, port: int | None = None, grasp_thumbs=None):
        self._template = None            # the last ACCEPTED template
        #: {object key -> absolute *_segments_thumbnails folder} for objects with BODex grasp data.
        #: The resolver lives in the generator (it has the stage); this server only serves/writes the
        #: folders it is handed, so task_composer stays stdlib-only.
        self._grasp_thumbs = dict(grasp_thumbs or {})
        super().__init__(page, port)

    @property
    def template(self):
        """Whatever /template last accepted, or None."""
        with self._lock:
            return self._template

    def _make_handler(self):
        server = self
        base = PreviewServer._make_handler(self)   # the /ping, /, /placements handler

        class Handler(base):
            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/palette":
                    self._write_json(200, palette())
                    return
                if path == "/predicates":
                    self._write_json(200, predicates_palette())
                    return
                if path == "/scene_defaults":
                    self._write_json(200, scene_defaults_payload())
                    return
                if path == "/grasps/state":
                    self._handle_grasps_state()
                    return
                if path == "/grasps/thumb":
                    self._handle_grasps_thumb()
                    return
                base.do_GET(self)

            def do_POST(self):
                if self.path == "/template":
                    self._handle_template()
                    return
                if urlparse(self.path).path == "/grasps/select":
                    self._handle_grasps_select()
                    return
                base.do_POST(self)

            # ---- grasp thumbnails: list / serve / write the per-object selection ----
            def _grasp_folder(self):
                """The thumbnails folder for the ?key= object, or None if the key is unknown."""
                key = (parse_qs(urlparse(self.path).query).get("key") or [""])[0]
                return server._grasp_thumbs.get(key)

            def _handle_grasps_state(self):
                folder = self._grasp_folder()
                if not folder:
                    self._write_json(404, {"ok": False, "reason": "unknown object key"})
                    return
                selected = grasp_selection(folder)
                self._write_json(200, {
                    "cached": bool(selected),
                    "thumbs": grasp_thumbs_listing(folder),
                    "selected": selected,
                })

            def _handle_grasps_thumb(self):
                folder = self._grasp_folder()
                fname = (parse_qs(urlparse(self.path).query).get("file") or [""])[0]
                path = grasp_thumb_path(folder, fname) if folder else None
                if not path:
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle_grasps_select(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, TypeError) as exc:
                    self._write_json(400, {"ok": False, "reason": f"malformed request: {exc}"})
                    return
                folder = server._grasp_thumbs.get(payload.get("key"))
                if not folder:
                    self._write_json(404, {"ok": False, "reason": "unknown object key"})
                    return
                try:
                    count = write_grasp_selection(folder, payload.get("selected", []))
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    self._write_json(400, {"ok": False, "reason": f"bad selection: {exc}"})
                    return
                self._write_json(200, {"ok": True, "count": count})

            def _handle_template(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, TypeError) as exc:
                    self._write_json(400, {"ok": False, "reason": f"malformed request: {exc}"})
                    return

                try:
                    template = template_from_payload(payload)
                except (TaskTemplateError, SequenceError, SceneError) as exc:
                    # The reason travels back in the body so the page can show it next to the
                    # sequence — refused, not silently accepted, and not a bare 500 traceback.
                    # SceneError is raised by validate_scene (a bad placement / type / dup name),
                    # so a bad scene is a clean 400 here, not an uncaught 500.
                    self._write_json(400, {"ok": False, "reason": str(exc)})
                    return
                except (KeyError, TypeError) as exc:
                    # Assumption: these are payload-SHAPE errors — a missing key or a wrong type in
                    # the untrusted POST body — so 400 (blame the input) is right. The tradeoff:
                    # if a future bug INSIDE validate_template/validate_sequence raised KeyError or
                    # TypeError, it would be misattributed to the user here rather than surfacing.
                    self._write_json(400, {"ok": False, "reason": f"malformed template: {exc}"})
                    return

                with server._lock:
                    server._template = template
                self._write_json(200, {"ok": True, "warnings": template_warnings(template)})

            def _write_json(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler


def _ensure_skills_loaded() -> None:
    """Populate the skill REGISTRY — the palette IS the registry, and the @skill decorators fill it
    on import of `skills`.

    This module deliberately does NOT import `skills` at module load: `skills` pulls torch +
    isaaclab, and `import task_composer` must stay light so the executor-adjacent import contract (and
    test_importing_task_composer_boots_neither_...) holds. But the registry has to be populated
    *before the browser asks GET /palette*, or the composer opens with an empty palette and nothing to
    drag. composer_page is the one entry point that puts the composer in front of a user, so load them
    here — lazily, when the page is built, not at import time. (Shipping without this is exactly why
    the first launch showed no skills.)"""
    import skills  # noqa: F401 — the @skill decorators register into REGISTRY as a side effect
    if not REGISTRY:
        raise RuntimeError(
            "the skill REGISTRY is empty after importing skills — the composer palette would be "
            "blank. Something is wrong with the skill declarations."
        )


_COMPOSER_PANEL = """
<style>
#composer-panel{position:fixed;top:0;left:0;width:340px;max-height:100vh;overflow-y:auto;
  background:var(--scrim);color:var(--fg);box-sizing:border-box;padding:14px 16px;z-index:9999;
  font:13px/1.45 var(--mono)}
#composer-panel h2{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:16px 0 6px;font-weight:600}
#composer-panel h2:first-child{margin-top:0}
#composer-panel input[type=text],#composer-panel input[type=number],#composer-panel select{
  width:100%;box-sizing:border-box;background:var(--panel);color:var(--fg);border:1px solid var(--edge-bright);
  border-radius:0;padding:3px 5px;font:12px system-ui}
#composer-palette .box{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;padding:5px 8px;
  margin:4px 0;cursor:grab;font-size:12px}
#composer-palette .box:active{cursor:grabbing}
#composer-palette .box .a{color:var(--muted);font-size:10px;float:right}
#composer-palette .box.runtime{border-left:3px solid var(--accent)}
#composer-predicate-palette .box{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;
  padding:5px 8px;margin:4px 0;cursor:grab;font-size:12px}
#composer-predicate-palette .box:active{cursor:grabbing}
#composer-predicate-palette .box.script{border-left:3px solid var(--warn)}
#composer-panel h3{font-size:12px;margin:10px 0 4px;font-weight:600;color:var(--fg)}
#composer-conditions .cond{min-height:28px;border:1px dashed var(--edge-bright);border-radius:0;padding:4px}
#composer-conditions .row{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;
  padding:4px 6px;margin:3px 0;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#composer-conditions .row.script{border-left:3px solid var(--warn)}
#composer-conditions .neg{color:var(--err);font-weight:bold}
#composer-sequence{min-height:40px;border:1px dashed var(--edge-bright);border-radius:0;padding:4px}
#composer-sequence.over{border-color:var(--accent);background:var(--accent-wash)}
#composer-sequence .step{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;padding:6px 8px;
  margin:5px 0;cursor:grab}
#composer-sequence .step.sel{border-color:var(--warn);box-shadow:0 0 0 1px var(--warn) inset}
#composer-sequence .step .hd{display:flex;justify-content:space-between;align-items:center;gap:6px}
#composer-sequence .step .hd b{font-weight:600;font-size:12px}
#composer-sequence .step .x{cursor:pointer;color:var(--muted);padding:0 4px}
#composer-sequence .step .p{display:flex;align-items:center;gap:6px;margin-top:4px}
#composer-sequence .step .p span{color:var(--muted);font-size:11px;width:80px;flex:none;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
#composer-scene .obj{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;padding:6px 8px;
  margin:5px 0}
#composer-scene .obj .hd{display:flex;justify-content:space-between;align-items:center;gap:6px}
#composer-scene .obj .hd b{font-weight:600;font-size:12px}
#composer-scene .obj .hd .t{color:var(--muted);font-size:10px}
#composer-scene .obj .x{cursor:pointer;color:var(--muted);padding:0 4px}
#composer-scene .obj .p{display:flex;align-items:center;gap:6px;margin-top:4px}
#composer-scene .obj .p span{color:var(--muted);font-size:11px;width:88px;flex:none;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
#composer-scene .obj .p .pair{display:flex;gap:5px;flex:1}
#composer-scene-add{margin-left:6px;flex:none}
#composer-scene-add-type{flex:1}
#composer-panel .note{color:var(--muted);font-size:11px;margin-top:6px}
#composer-panel .err{color:var(--err);font-size:12px;margin-top:8px;white-space:pre-wrap}
#composer-panel .ok{color:var(--ok);font-size:12px;margin-top:8px}
#composer-panel button{background:var(--panel-2);color:var(--fg);border:1px solid var(--edge-bright);border-radius:0;
  padding:5px 10px;cursor:pointer;font-size:12px}
#composer-panel button.save{background:var(--ok-dim);border-color:var(--ok)}
#composer-panel button#composer-pick-btn.arm{background:var(--warn-dim);border-color:var(--warn);color:var(--warn)}
/* Raised above the 3D canvas only while a prim is being picked, so the picking gesture never
   reaches the canvas — where kitchen_preview.bindDrag would otherwise read it as select+drag and
   nudge the object. Below the panel (z 9999) so the panel's own buttons stay clickable. */
#composer-pick-overlay{position:fixed;inset:0;z-index:9998;cursor:crosshair;background:transparent}
#composer-grasps .g{background:var(--panel);border:1px solid var(--edge-bright);border-radius:0;padding:6px 8px;
  margin:5px 0;display:flex;align-items:center;gap:8px}
#composer-grasps .g b{font-weight:600;font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
#composer-grasps .g .pill{font-size:10px;padding:1px 6px;border-radius:0;flex:none}
#composer-grasps .g .pill.ok{background:var(--ok-dim);color:var(--ok)}
#composer-grasps .g .pill.no{background:var(--warn-dim);color:var(--warn)}
#composer-grasps .g button{flex:none;padding:3px 8px;font-size:11px}
#composer-grasp-modal{position:fixed;inset:0;z-index:10000;background:var(--scrim);
  display:none;align-items:center;justify-content:center}
#composer-grasp-box{background:var(--bg);border:1px solid var(--edge-bright);border-radius:0;
  width:min(920px,94vw);max-height:90vh;display:flex;flex-direction:column;
  font:13px/1.45 var(--mono);color:var(--fg)}
#composer-grasp-box .top{display:flex;align-items:center;gap:10px;padding:12px 16px;
  border-bottom:1px solid var(--panel-2)}
#composer-grasp-box .top b{font-size:13px;flex:1}
#composer-grasp-box .top .count{color:var(--muted);font-size:12px}
#composer-grasp-grid{overflow-y:auto;padding:12px 16px;display:grid;
  grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px}
#composer-grasp-grid .cell{border:2px solid var(--panel-2);border-radius:0;padding:4px;cursor:pointer;
  background:var(--panel);text-align:center}
#composer-grasp-grid .cell.sel{border-color:var(--accent);background:var(--accent-wash)}
#composer-grasp-grid .cell img{width:100%;height:auto;display:block;border-radius:0;background:var(--bg)}
#composer-grasp-grid .cell .cap{font-size:10px;color:var(--muted);margin-top:3px}
#composer-grasp-box .btm{display:flex;align-items:center;gap:8px;padding:12px 16px;
  border-top:1px solid var(--panel-2)}
#composer-grasp-box .btm .sp{flex:1}
#composer-grasp-box .btm .msg{font-size:12px}
</style>
<div id="composer-panel">
  <h2>Task</h2>
  <div class="p"><input type="text" id="composer-name" placeholder="task name (e.g. bowl_to_drawer)"></div>
  <div class="p" style="margin-top:5px"><input type="text" id="composer-language"
    placeholder="language / dataset label (e.g. Put bowl inside drawer.)"></div>

  <h2>Skills (drag into the sequence)</h2>
  <div id="composer-palette"><div class="note">Loading the palette from /palette…</div></div>

  <h2>Conditions (drag into Success / Retry)</h2>
  <div id="composer-conditions">
    <div id="composer-predicate-palette"><div class="note">Loading the condition palette from /predicates…</div></div>
    <h3>Success — all of</h3>
    <div id="composer-success" class="cond"></div>
    <div class="note">Drag a condition here. A task cannot be saved without one.</div>
    <h3>Retry — any of</h3>
    <div id="composer-retry" class="cond"></div>
    <div class="note">Leave empty for the default: target dropped below 0.3 m, or the robot fell.</div>
  </div>

  <h2>Sequence</h2>
  <div id="composer-sequence"><div class="note" id="composer-empty">Drag skills here to build the task.</div></div>
  <button id="composer-pick-btn">Pick a prim for this step</button>
  <div class="note" id="composer-pickhint">Select a step above, then arm this and click a prim in the
    3D view to fill its prim_path. Arming raises an overlay so the click fills the role without moving
    the object. Esc (or click again) to cancel.</div>

  <h2>Scene (this task's objects)</h2>
  <div id="composer-scene"><div class="note">Loading object defaults from /scene_defaults…</div></div>
  <div class="p" style="margin-top:6px">
    <select id="composer-scene-add-type"></select>
    <button id="composer-scene-add">Add object</button>
  </div>
  <div class="note" id="composer-scene-hint"><b>This is the task's object list, not a live edit of the
    3D kitchen.</b> Adding an object adds a row here (below) and it is placed on every kitchen when you
    generate goals — it does <b>not</b> spawn a mesh in this reference preview. Each object carries a
    size (uniform scale, center &amp; per-kitchen spread), a placement surface, and — for a manipulated
    object — a lift height. Defaults are pre-filled per type. A role that manipulates a type with no
    matching object here is refused on save.</div>

  <h2>Grasps (pick preferred grasps per object)</h2>
  <div id="composer-grasps"><div class="note">Objects with BODex grasp data appear here.</div></div>
  <div class="note">An object marked <b>⚠ none chosen</b> would open a chooser mid-run on the headless
    cluster and hang. Pick its grasps here once — that writes the cache the run reads. Objects marked
    <b>✓</b> are already chosen; click to review or change them.</div>

  <h2>Save</h2>
  <button class="save" id="composer-save">Validate &amp; save task</button>
  <div id="composer-msg"></div>
  <div class="note">Saving POSTs to /template, which refuses an incoherent script (e.g. a gripper
    closing on nothing) with the reason shown here — it is never silently accepted.</div>
</div>
<div id="composer-grasp-modal">
  <div id="composer-grasp-box">
    <div class="top">
      <b id="composer-grasp-title">Pick grasps</b>
      <span class="count" id="composer-grasp-count"></span>
      <button id="composer-grasp-close">Close</button>
    </div>
    <div id="composer-grasp-grid"></div>
    <div class="btm">
      <button id="composer-grasp-all">Select all</button>
      <button id="composer-grasp-none">Clear</button>
      <span class="sp"></span>
      <span class="msg" id="composer-grasp-msg"></span>
      <button class="save" id="composer-grasp-save">Save selection</button>
    </div>
  </div>
</div>
<script>
(function () {
  var COMPOSER = __COMPOSER_PAYLOAD__;

  // node (sanitized three.js name) -> the object key we fill a prim_path with
  var nodeToKey = new Map();
  COMPOSER.objects.forEach(function (o) {
    o.nodes.forEach(function (n) { nodeToKey.set(n, o.key); });
  });

  var boxesById = new Map();     // skill id -> palette box (label, actions, params)
  var sequence = [];             // [{skill, action, params:{}}]
  var selected = -1;             // index of the selected step, or -1

  var paletteEl = document.getElementById('composer-palette');
  var seqEl = document.getElementById('composer-sequence');
  var msgEl = document.getElementById('composer-msg');
  var pickBtn = document.getElementById('composer-pick-btn');

  function loadPalette() {
    fetch('/palette', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (boxes) {
        paletteEl.innerHTML = '';
        boxes.forEach(function (box) {
          boxesById.set(box.id, box);
          var el = document.createElement('div');
          el.className = 'box' + (box.runtime ? ' runtime' : '');
          el.draggable = true;
          el.dataset.skill = box.id;
          var a = document.createElement('span');
          a.className = 'a';
          a.textContent = box.actions.join(' ');
          el.appendChild(a);
          el.appendChild(document.createTextNode(box.label));
          el.title = box.id;
          el.addEventListener('dragstart', function (e) {
            e.dataTransfer.setData('text/plain', 'palette:' + box.id);
          });
          paletteEl.appendChild(el);
        });
      })
      .catch(function (e) {
        paletteEl.innerHTML = '<div class="err">Could not load /palette (' +
          (e && e.message ? e.message : 'network error') + ').</div>';
      });
  }

  // ---- the condition palette: the predicate registry, same shape as the skill palette ----
  var predsById = new Map();
  var successRows = [];        // [{id, params:{}, neg:false}]
  var retryRows = [];

  fetch('/predicates', { cache: 'no-store' })
    .then(function (r) { return r.json(); })
    .then(function (boxes) {
      boxes.forEach(function (b) { predsById.set(b.id, b); });
      renderPredicatePalette(boxes);
      renderConditions();
    })
    .catch(function (e) {
      flash('Could not load /predicates (' + (e && e.message ? e.message : 'network') + ').', true);
    });

  function renderPredicatePalette(boxes) {
    var el = document.getElementById('composer-predicate-palette');
    el.innerHTML = '';
    boxes.forEach(function (b) {
      var box = document.createElement('div');
      box.className = 'box' + (b.phase === 'script' ? ' script' : '');
      box.textContent = b.label;
      box.draggable = true;
      box.addEventListener('dragstart', function (e) {
        e.dataTransfer.setData('text/plain', 'pred:' + b.id);
      });
      el.appendChild(box);
    });
  }

  // Named defaultCondParams, not defaultParams: a `function defaultParams(box)` for SKILL step
  // params is already declared just below. Two same-named function declarations in one scope do
  // not overload -- the later one wins for every call site, so addStep's defaultParams(box) would
  // start calling predsById.get(box) (a box object, not a predicate id) and throw.
  function defaultCondParams(id) {
    var out = {};
    (predsById.get(id).params || []).forEach(function (w) {
      if (w.kind === 'opt_float' || w.kind === 'text') return;   // omitted means unbounded/default
      out[w.name] = w.default;
    });
    return out;
  }

  function addCondition(list, id) {
    list.push({ id: id, params: defaultCondParams(id), neg: false });
    renderConditions();
  }

  function conditionDrop(el, list) {
    el.addEventListener('dragover', function (e) { e.preventDefault(); });
    el.addEventListener('drop', function (e) {
      e.preventDefault();
      var data = e.dataTransfer.getData('text/plain') || '';
      if (data.indexOf('pred:') === 0) addCondition(list, data.slice('pred:'.length));
    });
  }

  function renderConditionList(el, list) {
    el.innerHTML = '';
    list.forEach(function (row, i) {
      var box = predsById.get(row.id);
      var el2 = document.createElement('div');
      el2.className = 'row' + (box.phase === 'script' ? ' script' : '');

      var neg = document.createElement('button');
      neg.textContent = row.neg ? 'NOT' : 'not';
      neg.className = row.neg ? 'neg' : '';
      neg.title = 'negate this condition';
      neg.addEventListener('click', function () { row.neg = !row.neg; renderConditions(); });
      el2.appendChild(neg);

      var name = document.createElement('span');
      name.textContent = box.label;
      el2.appendChild(name);

      (box.params || []).forEach(function (w) { el2.appendChild(paramInput(row, w)); });

      var del = document.createElement('button');
      del.textContent = '×';
      del.addEventListener('click', function () { list.splice(i, 1); renderConditions(); });
      el2.appendChild(del);

      el.appendChild(el2);
    });
  }

  // A role param is a dropdown of the roles declared above, so '@target' cannot be mistyped.
  function paramInput(row, w) {
    var wrap = document.createElement('label');
    wrap.textContent = w.name + ' ';
    var input;
    if (w.kind === 'role') {
      input = document.createElement('select');
      composerRoles().forEach(function (r) {
        var o = document.createElement('option');
        o.value = '@' + r.name; o.textContent = '@' + r.name;
        input.appendChild(o);
      });
      input.value = row.params[w.name] || '';
      input.addEventListener('change', function () { row.params[w.name] = this.value; });
    } else if (w.kind === 'choice') {
      input = document.createElement('select');
      w.options.forEach(function (opt) {
        var o = document.createElement('option'); o.value = opt; o.textContent = opt;
        input.appendChild(o);
      });
      input.value = row.params[w.name];
      input.addEventListener('change', function () { row.params[w.name] = this.value; });
    } else if (w.kind === 'bool') {
      input = document.createElement('input'); input.type = 'checkbox';
      input.checked = !!row.params[w.name];
      input.addEventListener('change', function () { row.params[w.name] = this.checked; });
    } else if (w.kind === 'text') {
      input = document.createElement('input'); input.type = 'text'; input.size = 14;
      input.value = row.params[w.name] == null ? '' : row.params[w.name];
      input.addEventListener('change', function () {
        // An empty text param is OMITTED, not sent as "" — obj_near_prim.body defaults to body 0.
        if (this.value === '') delete row.params[w.name];
        else row.params[w.name] = this.value;
      });
    } else {                                   // float, opt_float
      input = document.createElement('input'); input.type = 'number';
      input.step = '0.01'; input.size = 6;
      input.value = row.params[w.name] == null ? '' : row.params[w.name];
      input.addEventListener('change', function () {
        // An empty optional bound is OMITTED: obj_z with only `hi` means "below hi".
        if (this.value === '') delete row.params[w.name];
        else row.params[w.name] = parseFloat(this.value);
      });
    }
    wrap.appendChild(input);
    return wrap;
  }

  function renderConditions() {
    renderConditionList(document.getElementById('composer-success'), successRows);
    renderConditionList(document.getElementById('composer-retry'), retryRows);
  }

  // The single choke point for every write to a prim_path-kind step param -- the only param kind
  // that can ever hold an '@role' reference (bool/choice/float values are booleans, closed enums,
  // and numbers respectively; none can carry one). A plain `step.params[name] = value` is exactly
  // what let a role change go unnoticed by the condition panel (fix rounds 1-2): a click via
  // fillPrimPath and hand-typing in paramRow's prim_path <input> are the only two places that ever
  // write one, and both must route through here so a future third writer cannot reintroduce the
  // bug by simply forgetting to call renderConditions() itself.
  function setPrimPathParam(step, name, value) {
    step.params[name] = value;
    renderConditions();
  }

  conditionDrop(document.getElementById('composer-success'), successRows);
  conditionDrop(document.getElementById('composer-retry'), retryRows);

  // Rows -> the spec grammar. A negated row becomes {"not": leaf}.
  function conditionsToSpec(list, op) {
    if (!list.length) return null;
    var items = list.map(function (row) {
      var leaf = {}; leaf[row.id] = row.params;
      return row.neg ? { not: leaf } : leaf;
    });
    var out = {}; out[op] = items;
    return out;
  }

  function defaultParams(box) {
    var params = {};
    box.params.forEach(function (p) {
      if (p.kind === 'prim_path') params[p.name] = '';
      else if (p.kind === 'float' && p.default == null) return;
      else params[p.name] = p.default;
    });
    return params;
  }

  function addStep(skillId, atIndex) {
    var box = boxesById.get(skillId);
    if (!box) return;
    var step = { skill: skillId, action: box.actions[0], params: defaultParams(box) };
    if (atIndex == null || atIndex < 0 || atIndex > sequence.length) sequence.push(step);
    else sequence.splice(atIndex, 0, step);
    selected = sequence.indexOf(step);
    renderSequence();
  }

  // True for the interactive controls that live inside a draggable step. Clicks and pointer-downs
  // on these must edit the control, not drag/re-select the step (else the field can't hold focus).
  function isFormControl(t) {
    return !!t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' ||
                   t.tagName === 'TEXTAREA' || t.tagName === 'OPTION');
  }

  function renderSequence() {
    seqEl.innerHTML = '';
    if (!sequence.length) {
      var empty = document.createElement('div');
      empty.className = 'note';
      empty.textContent = 'Drag skills here to build the task.';
      seqEl.appendChild(empty);
      return;
    }
    sequence.forEach(function (step, i) {
      var box = boxesById.get(step.skill) || { label: step.skill, actions: [step.action], params: [] };
      var el = document.createElement('div');
      el.className = 'step' + (i === selected ? ' sel' : '');
      el.draggable = true;
      el.dataset.index = i;

      var hd = document.createElement('div');
      hd.className = 'hd';
      var name = document.createElement('b');
      name.textContent = (i + 1) + '. ' + box.label;
      hd.appendChild(name);

      // action picker (a skill may declare several channels: A_r / A_l / A_b)
      var actSel = document.createElement('select');
      box.actions.forEach(function (act) {
        var opt = document.createElement('option');
        opt.value = act; opt.textContent = act;
        if (act === step.action) opt.selected = true;
        actSel.appendChild(opt);
      });
      actSel.style.width = 'auto';
      actSel.addEventListener('change', function () { step.action = this.value; });
      hd.appendChild(actSel);

      var x = document.createElement('span');
      x.className = 'x';
      x.textContent = '\\u2715';
      x.title = 'remove';
      x.addEventListener('click', function (e) {
        e.stopPropagation();
        sequence.splice(i, 1);
        if (selected >= sequence.length) selected = sequence.length - 1;
        renderSequence();
        // Removing a step can drop the last reference to a role (referencedRoleNames shrinks), so
        // a condition row's role <select> -- built once from composerRoles() when it rendered --
        // would otherwise go stale and still offer a role no longer declared.
        renderConditions();
      });
      hd.appendChild(x);
      el.appendChild(hd);

      box.params.forEach(function (p) {
        el.appendChild(paramRow(step, p));
      });

      el.addEventListener('click', function (e) {
        // Clicking INTO a field (prim_path, action select, a number) edits it — it must not
        // re-select+re-render the step, because renderSequence() rebuilds this DOM and would drop
        // the focus/cursor you just placed. (Typing still writes to the right step: each input's
        // own handler is bound to this step object, selected or not.) Only a click on the step body
        // selects it.
        if (isFormControl(e.target)) return;
        selected = i; renderSequence();
      });
      el.addEventListener('pointerdown', function (e) {
        // A text field inside a draggable box can't be focused — the browser starts a drag instead
        // of placing the cursor. Turn dragging OFF while the pointer is down on a form control, and
        // back ON for the rest of the box, so reordering by dragging the header still works.
        el.draggable = !isFormControl(e.target);
      });
      el.addEventListener('dragstart', function (e) {
        e.dataTransfer.setData('text/plain', 'move:' + i);
      });
      seqEl.appendChild(el);
    });
  }

  // Render one param input from its widget kind, wired back into the step's params.
  function paramRow(step, p) {
    var row = document.createElement('div');
    row.className = 'p';
    var label = document.createElement('span');
    label.textContent = p.name;
    label.title = p.name;
    row.appendChild(label);

    var input;
    if (p.kind === 'bool') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!step.params[p.name];
      input.addEventListener('change', function () { step.params[p.name] = this.checked; });
    } else if (p.kind === 'choice') {
      input = document.createElement('select');
      (p.options || []).forEach(function (o) {
        var opt = document.createElement('option');
        opt.value = o; opt.textContent = o;
        if (o === step.params[p.name]) opt.selected = true;
        input.appendChild(opt);
      });
      input.addEventListener('change', function () { step.params[p.name] = this.value; });
    } else if (p.kind === 'float') {
      input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      input.value = step.params[p.name];
      input.addEventListener('input', function () {
        var v = parseFloat(this.value);
        step.params[p.name] = isNaN(v) ? this.value : v;
      });
    } else {  // prim_path
      input = document.createElement('input');
      input.type = 'text';
      input.value = step.params[p.name] || '';
      input.placeholder = 'click a prim, or @role, or /World/...';
      input.dataset.primPath = p.name;
      // Hand-typing '@role' (the field's own placeholder names this as a first-class alternative
      // to clicking a prim) changes the role set exactly like fillPrimPath does, so it must go
      // through the same choke point -- not a plain step.params[p.name] = this.value assignment.
      input.addEventListener('input', function () { setPrimPathParam(step, p.name, this.value); });
    }
    row.appendChild(input);
    return row;
  }

  // ---- drag a palette skill in, and reorder steps within the sequence ----
  function indexFromEvent(e) {
    var steps = Array.prototype.slice.call(seqEl.querySelectorAll('.step'));
    for (var i = 0; i < steps.length; i++) {
      var box = steps[i].getBoundingClientRect();
      if (e.clientY < box.top + box.height / 2) return Number(steps[i].dataset.index);
    }
    return sequence.length;
  }

  seqEl.addEventListener('dragover', function (e) { e.preventDefault(); seqEl.classList.add('over'); });
  seqEl.addEventListener('dragleave', function () { seqEl.classList.remove('over'); });
  seqEl.addEventListener('drop', function (e) {
    e.preventDefault();
    seqEl.classList.remove('over');
    var data = e.dataTransfer.getData('text/plain') || '';
    var at = indexFromEvent(e);
    if (data.indexOf('palette:') === 0) {
      addStep(data.slice('palette:'.length), at);
    } else if (data.indexOf('move:') === 0) {
      var from = Number(data.slice('move:'.length));
      var step = sequence[from];
      sequence.splice(from, 1);
      if (from < at) at -= 1;
      sequence.splice(at, 0, step);
      selected = sequence.indexOf(step);
      renderSequence();
    }
  });

  // ---- a click declares a ROLE, never a literal prim path ----
  // arm.grasp refuses a literal: the run config must know whether the target is a rigid body and
  // which env.scene.rigid_objects key to dereference, and only a @role carries that. A literal also
  // names the wrong prim -- the page keys objects by the in-memory node id ('bottle00'), while the
  // rotation USDs that task_emit loads hold 'bottle0'.
  var keyToRole = new Map();   // object key ('bottle00') -> role name ('bottle')
  var roleToKey = new Map();   // role name -> the ONE object key that declared it

  function objectTypeOf(key) {
    var obj = COMPOSER.objects.filter(function (o) { return o.key === key; })[0];
    var label = (obj && obj.label) || key;
    return label.replace(/[0-9]+$/, '') || label;
  }

  // A role name IS its object type. It cannot be anything else: task_bind matches a role to a scene
  // object by TYPE alone, so a second role of the same type ('bottle_2') would name a distinction the
  // binder cannot make — it would report the type as ambiguous and skip every rotation. Minting one is
  // therefore refused up in fillPrimPath, and this only ever runs for a type that is still free.
  function roleForKey(key) {
    if (keyToRole.has(key)) return keyToRole.get(key);   // same object in a later step: one role
    var name = objectTypeOf(key);
    keyToRole.set(key, name);
    roleToKey.set(name, key);
    return name;
  }

  // Every '@name' any step currently points at. The registry above is deliberately never pruned (so
  // re-clicking an object reuses its role), which means it also remembers roles nothing references
  // any more: a mis-click corrected by a second click, a step deleted with the ✕, or a previous task
  // saved from this same still-open dialog. Emitting those sends a Role the page shows nowhere, and
  // task_template refuses the save naming it — with no way out but a reload.
  function referencedRoleNames() {
    var refs = new Map();
    sequence.forEach(function (step) {
      var params = step.params || {};
      Object.keys(params).forEach(function (k) {
        var v = params[k];
        if (typeof v === 'string' && v.charAt(0) === '@') refs.set(v.slice(1), true);
      });
    });
    return refs;
  }

  function composerRoles() {
    var refs = referencedRoleNames();
    var out = [];
    keyToRole.forEach(function (name, key) {
      if (refs.has(name)) out.push({ name: name, object_type: objectTypeOf(key) });
    });
    return out;
  }

  // ---- click a prim in the 3D view to fill the selected step's prim_path ----
  function fillPrimPath(key) {
    if (selected < 0 || selected >= sequence.length) {
      flash('Select a step first, then click a prim to fill its prim_path.', true);
      return;
    }
    var step = sequence[selected];
    var box = boxesById.get(step.skill);
    var primParam = (box.params.filter(function (p) { return p.kind === 'prim_path'; })[0] || {}).name;
    if (!primParam) {
      flash('Step ' + (selected + 1) + ' has no prim_path to fill.', true);
      return;
    }
    // A different object of a type some other object already claimed: refuse it. Naming it is the one
    // thing that would NOT help — binding resolves a role by type, so both names would point at the
    // same ambiguous set and the task would bind to neither.
    var type = objectTypeOf(key);
    if (!keyToRole.has(key) && roleToKey.has(type)) {
      flash('@' + type + ' is already the role for another ' + type + '. Binding matches on type, so ' +
            'a task can reference only one ' + type + ' — reuse @' + type + ' here, or compose on a ' +
            'kitchen that has a single ' + type + '.', true);
      return;
    }
    var role = roleForKey(key);
    // setPrimPathParam (not a plain assignment): a role can be newly minted or newly referenced
    // right here, and it is the choke point that keeps every condition row's role <select> current.
    setPrimPathParam(step, primParam, '@' + role);
    renderSequence();
    flash('Filled ' + primParam + ' of step ' + (selected + 1) + ' with @' + role + '.', false);
  }

  function ownerKey(node) {
    for (var cur = node; cur; cur = cur.parent) {
      var key = nodeToKey.get(cur.name);
      if (key) return key;
    }
    return null;
  }

  // ---- prim picking is an EXPLICIT armed mode, and the mode is the guard ----
  // The prim-pick click and kitchen_preview.bindDrag's object-drag both live on the SAME canvas
  // (bindDrag binds pointerdown/move/up on renderer.domElement). If picking were a bare click
  // listener on that canvas — as it once was — a pick would ALSO drive bindDrag's
  // pointerdown->select+drag and, on any cursor motion before release, pointermove->dropOnto,
  // physically nudging the object and re-POSTing placements. We do NOT fight that on the canvas
  // (listener order there is not guaranteed, and the two scripts are separate closures). Instead,
  // arming raises a transparent overlay ABOVE the canvas: the picking gesture lands on the overlay,
  // so the canvas never sees a pointerdown and bindDrag is never triggered. When NOT armed there is
  // no overlay at all, so the preview's own orbit/drag work exactly as before. Touches only this
  // file — kitchen_preview.py is unchanged.
  var picking = false;
  var pickOverlay = null;

  function selectedPrimParam() {
    if (selected < 0 || selected >= sequence.length) return null;
    var box = boxesById.get(sequence[selected].skill);
    if (!box) return null;
    return (box.params.filter(function (p) { return p.kind === 'prim_path'; })[0] || {}).name || null;
  }

  function onPickClick(event) {
    if (typeof scene === 'undefined' || !scene || typeof camera === 'undefined') return;
    var raycaster = new THREE.Raycaster();
    var pointer = new THREE.Vector2();
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    var meshes = [];
    scene.traverse(function (o) { if (o.isMesh && o.visible) meshes.push(o); });
    var hits = raycaster.intersectObjects(meshes, false);
    for (var i = 0; i < hits.length; i++) {
      var key = ownerKey(hits[i].object);
      if (key) { fillPrimPath(key); disarmPick(); return; }   // filled the role; leave the object put
    }
    flash('No prim under the cursor — click directly on an object, or press Esc to cancel.', true);
  }

  function armPick() {
    if (picking) { disarmPick(); return; }                      // the button toggles
    if (typeof renderer === 'undefined' || !renderer) { flash('The 3D view is not ready yet.', true); return; }
    if (selected < 0 || selected >= sequence.length) {
      flash('Select a step first, then pick a prim for it.', true);
      return;
    }
    var primParam = selectedPrimParam();
    if (!primParam) { flash('Step ' + (selected + 1) + ' has no prim_path to fill.', true); return; }

    picking = true;
    pickBtn.classList.add('arm');
    pickBtn.textContent = 'Click a prim… (Esc to cancel)';
    if (!pickOverlay) {
      pickOverlay = document.createElement('div');
      pickOverlay.id = 'composer-pick-overlay';
      pickOverlay.addEventListener('click', onPickClick);
      document.body.appendChild(pickOverlay);
    }
    pickOverlay.style.display = 'block';
    flash('Click a prim to fill ' + primParam + ' of step ' + (selected + 1) + '.', false);
  }

  function disarmPick() {
    picking = false;
    pickBtn.classList.remove('arm');
    pickBtn.textContent = 'Pick a prim for this step';
    if (pickOverlay) pickOverlay.style.display = 'none';
  }

  function flash(text, bad) {
    msgEl.innerHTML = '<div class="' + (bad ? 'err' : 'ok') + '">' + text + '</div>';
  }

  // ---- the Scene panel: add/remove objects and edit each one's geometry ----
  // The scene is authored the same way the sequence is: pure client-side rows whose ONLY output is
  // the JSON POSTed to /template. Defaults for every type (and the placement surfaces) come from
  // /scene_defaults so nothing here hardcodes what scene_spec already declares — the geometry
  // analogue of "the palette IS the registry".
  var sceneDefaults = new Map();   // object_type -> its default row {name,object_type,size,placement,lift}
  var surfaceOptions = [];         // sorted SUPPORT_SURFACES, for every placement dropdown
  var sceneRows = [];              // [{name, object_type, size:{center,spread}, placement, lift:{center,spread}|null}]
  var sceneEl = document.getElementById('composer-scene');
  var sceneAddType = document.getElementById('composer-scene-add-type');

  function clone(v) { return JSON.parse(JSON.stringify(v)); }

  // First free <type><n> starting at 0: bottle0, bottle1, bottle2 — not bottle0 then bottle01.
  function uniqueNameForType(type) {
    var taken = {};
    sceneRows.forEach(function (r) { taken[r.name] = true; });
    for (var i = 0; ; i++) { if (!taken[type + i]) return type + i; }
  }

  function loadSceneDefaults() {
    fetch('/scene_defaults', { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (entries) {
        sceneDefaults.clear();
        sceneAddType.innerHTML = '';
        entries.forEach(function (e) {
          sceneDefaults.set(e.object_type, e);
          surfaceOptions = e.surface_options || surfaceOptions;
          var opt = document.createElement('option');
          opt.value = e.object_type; opt.textContent = e.object_type;
          sceneAddType.appendChild(opt);
        });
        renderScene();
      })
      .catch(function (e) {
        sceneEl.innerHTML = '<div class="err">Could not load /scene_defaults (' +
          (e && e.message ? e.message : 'network error') + ').</div>';
      });
  }

  function addObject(type) {
    var def = sceneDefaults.get(type);
    if (!def) return;
    var row = clone(def);
    delete row.surface_options;                  // that list is menu metadata, not a field of the object
    row.name = uniqueNameForType(type);          // bottle0, bottle1, bottle2, ...
    sceneRows.push(row);
    renderScene();
  }

  // A role that manipulates a type the scene lacks is refused on save; nudge the author toward that.
  function objectTypeCounts() {
    var counts = {};
    sceneRows.forEach(function (r) { counts[r.object_type] = (counts[r.object_type] || 0) + 1; });
    return counts;
  }

  function extentPair(row, field, onchange) {
    var wrap = document.createElement('div');
    wrap.className = 'pair';
    ['center', 'spread'].forEach(function (k) {
      var input = document.createElement('input');
      input.type = 'number'; input.step = 'any';
      input.title = field + ' ' + k;
      input.placeholder = k;
      input.value = row[field][k];
      input.addEventListener('input', function () {
        var v = parseFloat(this.value);
        row[field][k] = isNaN(v) ? 0 : v;
        if (onchange) onchange();
      });
      wrap.appendChild(input);
    });
    return wrap;
  }

  function renderScene() {
    sceneEl.innerHTML = '';
    if (!sceneRows.length) {
      var empty = document.createElement('div');
      empty.className = 'note';
      empty.textContent = 'No objects yet — add one below. An empty scene falls back to the ' +
        'generator default.';
      sceneEl.appendChild(empty);
      return;
    }
    sceneRows.forEach(function (row, i) {
      var el = document.createElement('div');
      el.className = 'obj';

      var hd = document.createElement('div');
      hd.className = 'hd';
      var nameIn = document.createElement('input');
      nameIn.type = 'text'; nameIn.value = row.name; nameIn.title = 'object name';
      nameIn.style.flex = '1';
      nameIn.addEventListener('input', function () { row.name = this.value; });
      hd.appendChild(nameIn);
      var t = document.createElement('span');
      t.className = 't'; t.textContent = row.object_type;
      hd.appendChild(t);
      var x = document.createElement('span');
      x.className = 'x'; x.textContent = '\\u2715'; x.title = 'remove';
      x.addEventListener('click', function () { sceneRows.splice(i, 1); renderScene(); });
      hd.appendChild(x);
      el.appendChild(hd);

      var sizeRow = document.createElement('div');
      sizeRow.className = 'p';
      var sizeLbl = document.createElement('span');
      sizeLbl.textContent = 'size (c / spread)'; sizeRow.appendChild(sizeLbl);
      sizeRow.appendChild(extentPair(row, 'size'));
      el.appendChild(sizeRow);

      var placeRow = document.createElement('div');
      placeRow.className = 'p';
      var placeLbl = document.createElement('span');
      placeLbl.textContent = 'placement'; placeRow.appendChild(placeLbl);
      var placeSel = document.createElement('select');
      surfaceOptions.forEach(function (s) {
        var opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        if (s === row.placement) opt.selected = true;
        placeSel.appendChild(opt);
      });
      placeSel.addEventListener('change', function () { row.placement = this.value; });
      placeRow.appendChild(placeSel);
      el.appendChild(placeRow);

      var liftRow = document.createElement('div');
      liftRow.className = 'p';
      var liftLbl = document.createElement('span');
      liftLbl.textContent = 'lift (manipulated)'; liftRow.appendChild(liftLbl);
      var liftChk = document.createElement('input');
      liftChk.type = 'checkbox'; liftChk.checked = row.lift != null; liftChk.style.flex = 'none';
      liftChk.title = 'lifted (manipulation target)';
      liftChk.addEventListener('change', function () {
        row.lift = this.checked ? { center: 0.775, spread: 0.025 } : null;
        renderScene();
      });
      liftRow.appendChild(liftChk);
      if (row.lift) liftRow.appendChild(extentPair(row, 'lift'));   // pair only exists when lifted
      el.appendChild(liftRow);

      sceneEl.appendChild(el);
    });
  }

  // Rows -> the SceneObject dicts scene_spec.scene_from_dicts expects. lift is null unless set.
  function sceneToPayload() {
    return sceneRows.map(function (r) {
      return {
        name: r.name,
        object_type: r.object_type,
        size: { center: r.size.center, spread: r.size.spread },
        placement: r.placement,
        lift: r.lift ? { center: r.lift.center, spread: r.lift.spread } : null
      };
    });
  }

  function save() {
    var payload = {
      name: (document.getElementById('composer-name').value || '').trim(),
      language: (document.getElementById('composer-language').value || '').trim(),
      roles: composerRoles(),
      steps: sequence.map(function (s) { return { skill: s.skill, action: s.action, params: s.params }; }),
      subtask_groups: [],
      scene: sceneToPayload(),
      success: conditionsToSpec(successRows, 'all'),
      retry: conditionsToSpec(retryRows, 'any')
    };
    fetch('/template', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (r) {
      return r.json().then(function (body) { return { status: r.status, body: body }; });
    }).then(function (res) {
      if (res.status === 200 && res.body.ok) {
        var warn = (res.body.warnings || []);
        flash(warn.length ? ('Saved, but: ' + warn.join(' ')) : 'Saved. The task validates and can run.',
              warn.length > 0);
      } else {
        // The reason came back in the body (a 400, not a 200, and not a bare 500) — show it inline.
        flash(res.body && res.body.reason ? res.body.reason : ('Refused (' + res.status + ').'), true);
      }
    }).catch(function (e) {
      flash('Could not reach /template (' + (e && e.message ? e.message : 'network error') + ').', true);
    });
  }

  // ---- the Grasps panel: pick preferred grasps per object, warming the cache the run reads ----
  // Only objects the server handed a thumbnails folder (o.grasp === true) appear. "Pick grasps" opens
  // a modal grid of that object's thumbnails; clicking a tile toggles it; Save POSTs the [[seg,hand]]
  // list to /grasps/select, which writes the same selected_indices.json plan_arm_grasp reads. This is
  // the browser replacement for the Tk chooser that hangs on the headless cluster.
  var graspEl = document.getElementById('composer-grasps');
  var graspObjects = COMPOSER.objects.filter(function (o) { return o.grasp; });
  var graspModal = document.getElementById('composer-grasp-modal');
  var graspGrid = document.getElementById('composer-grasp-grid');
  var graspTitle = document.getElementById('composer-grasp-title');
  var graspCount = document.getElementById('composer-grasp-count');
  var graspMsg = document.getElementById('composer-grasp-msg');
  var graspOpen = null;              // {key, sel:Set('seg|hand'), thumbs:[...]} while the modal is up

  function selKey(t) { return t.seg + '|' + t.hand; }

  function renderGraspsPanel() {
    graspEl.innerHTML = '';
    if (!graspObjects.length) {
      var none = document.createElement('div');
      none.className = 'note';
      none.textContent = 'No objects in this kitchen carry BODex grasp data.';
      graspEl.appendChild(none);
      return;
    }
    graspObjects.forEach(function (o) {
      var row = document.createElement('div');
      row.className = 'g';
      var name = document.createElement('b');
      name.textContent = o.label; name.title = o.key;
      row.appendChild(name);
      var pill = document.createElement('span');
      pill.className = 'pill no'; pill.textContent = '…';
      pill.dataset.key = o.key;
      row.appendChild(pill);
      var btn = document.createElement('button');
      btn.textContent = 'Pick grasps';
      btn.addEventListener('click', function () { openGraspModal(o); });
      row.appendChild(btn);
      graspEl.appendChild(row);
      refreshPill(o.key);
    });
  }

  function refreshPill(key) {
    var pill = graspEl.querySelector('.pill[data-key="' + cssEsc(key) + '"]');
    if (!pill) return;
    fetch('/grasps/state?key=' + encodeURIComponent(key), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (st) {
        var n = (st.selected || []).length;
        if (n > 0) { pill.className = 'pill ok'; pill.textContent = '✓ ' + n + ' chosen'; }
        else { pill.className = 'pill no'; pill.textContent = '⚠ none chosen'; }
      })
      .catch(function () { pill.className = 'pill no'; pill.textContent = '?'; });
  }

  // CSS.escape isn't in every browser we tunnel to; a minimal escaper for the key attribute selector.
  function cssEsc(s) { return String(s).replace(/["\\\\]/g, '\\\\$&'); }

  function openGraspModal(o) {
    graspTitle.textContent = 'Pick grasps — ' + o.label;
    graspGrid.innerHTML = '<div class="note">Loading thumbnails…</div>';
    graspMsg.textContent = '';
    graspModal.style.display = 'flex';
    fetch('/grasps/state?key=' + encodeURIComponent(o.key), { cache: 'no-store' })
      .then(function (r) { return r.json(); })
      .then(function (st) {
        var sel = new Set((st.selected || []).map(function (p) { return p[0] + '|' + p[1]; }));
        graspOpen = { key: o.key, sel: sel, thumbs: st.thumbs || [] };
        renderGraspGrid();
      })
      .catch(function (e) {
        graspGrid.innerHTML = '<div class="err">Could not load grasps (' +
          (e && e.message ? e.message : 'network error') + ').</div>';
      });
  }

  function renderGraspGrid() {
    graspGrid.innerHTML = '';
    if (!graspOpen.thumbs.length) {
      graspGrid.innerHTML = '<div class="note">No thumbnail images for this object.</div>';
      updateGraspCount();
      return;
    }
    graspOpen.thumbs.forEach(function (t) {
      var cell = document.createElement('div');
      cell.className = 'cell' + (graspOpen.sel.has(selKey(t)) ? ' sel' : '');
      var img = document.createElement('img');
      img.loading = 'lazy';
      img.src = '/grasps/thumb?key=' + encodeURIComponent(graspOpen.key) +
                '&file=' + encodeURIComponent(t.file);
      cell.appendChild(img);
      var cap = document.createElement('div');
      cap.className = 'cap';
      cap.textContent = 'Seg ' + (t.seg + 1) + ' · ' + t.hand;   // label 1-based, like the Tk chooser
      cell.appendChild(cap);
      cell.addEventListener('click', function () {
        var k = selKey(t);
        if (graspOpen.sel.has(k)) graspOpen.sel.delete(k); else graspOpen.sel.add(k);
        cell.classList.toggle('sel');
        updateGraspCount();
      });
      graspGrid.appendChild(cell);
    });
    updateGraspCount();
  }

  function updateGraspCount() {
    graspCount.textContent = graspOpen.sel.size + ' / ' + graspOpen.thumbs.length + ' selected';
  }

  function closeGraspModal() { graspModal.style.display = 'none'; graspOpen = null; }

  function saveGraspSelection() {
    if (!graspOpen) return;
    var selected = [];
    graspOpen.thumbs.forEach(function (t) {                     // preserve grid order (hand, seg)
      if (graspOpen.sel.has(selKey(t))) selected.push([t.seg, t.hand]);
    });
    graspMsg.textContent = 'Saving…';
    fetch('/grasps/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: graspOpen.key, selected: selected })
    }).then(function (r) {
      return r.json().then(function (body) { return { status: r.status, body: body }; });
    }).then(function (res) {
      if (res.status === 200 && res.body.ok) {
        var key = graspOpen.key;
        closeGraspModal();
        refreshPill(key);
      } else {
        graspMsg.innerHTML = '<span class="err">' +
          (res.body && res.body.reason ? res.body.reason : ('Refused (' + res.status + ').')) + '</span>';
      }
    }).catch(function (e) {
      graspMsg.innerHTML = '<span class="err">Could not save (' +
        (e && e.message ? e.message : 'network error') + ').</span>';
    });
  }

  document.getElementById('composer-grasp-save').addEventListener('click', saveGraspSelection);
  document.getElementById('composer-grasp-close').addEventListener('click', closeGraspModal);
  document.getElementById('composer-grasp-all').addEventListener('click', function () {
    if (!graspOpen) return;
    graspOpen.thumbs.forEach(function (t) { graspOpen.sel.add(selKey(t)); });
    renderGraspGrid();
  });
  document.getElementById('composer-grasp-none').addEventListener('click', function () {
    if (!graspOpen) return;
    graspOpen.sel.clear();
    renderGraspGrid();
  });
  graspModal.addEventListener('click', function (e) { if (e.target === graspModal) closeGraspModal(); });

  document.getElementById('composer-save').addEventListener('click', save);
  document.getElementById('composer-scene-add').addEventListener('click', function () {
    if (sceneAddType.value) addObject(sceneAddType.value);
  });
  pickBtn.addEventListener('click', armPick);
  // Esc cancels an armed pick without picking anything, so the overlay can never get stuck up; and
  // Esc closes the grasp modal (when no pick is armed) so it can never trap the user either.
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    if (picking) disarmPick();
    else if (graspOpen) closeGraspModal();
  });
  loadPalette();
  loadSceneDefaults();
  renderGraspsPanel();
  renderSequence();
  // Picking needs no pre-wiring: the button checks that scene/renderer/camera exist when armed,
  // and the overlay it raises is what keeps the gesture off the drag canvas.
})();
</script>
"""
