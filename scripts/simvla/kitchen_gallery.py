"""SimVLA: gallery pages — pick a kitchen layout or an exact object mesh by looking at it.

A gallery is the same kind of page as the preview: one trimesh.Scene rendered by
trimesh.viewer.scene_to_html, with a small panel injected into it. That is not a stylistic choice --
there is no server-side renderer available here (trimesh's offscreen render wants a GL context and
hangs headless), and three.js exists only inside trimesh's own inlined bundle, so a page that draws
3D has to be one of these.

Imports trimesh + stdlib only: no isaaclab, omni, pxr, scene_synthesizer or kitchen_build. Callers
that need a kitchen BUILT hand the built scene in (see kitchen_tiles), which keeps this module
importable -- and therefore testable -- on a login node. mesh_orientation is the one project
import (see _stood_up) and is safe under that rule: a lookup table and some pure functions over
grasp_manifest, which is json and os.
"""

from __future__ import annotations

import math
import os

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

from kitchen_preview import (
    SPRITE_LABEL_JS,
    THEME_CSS,
    sanitize_three_name,
    scene_page,
    script_json,
)

#: Tiles per page in the object gallery. 24 x ~71 KB of GLB is ~1.7 MB and about a second to
#: build, which is the point where paging beats waiting.
DEFAULT_PER_PAGE = 24

#: A tile is scaled to this fraction of its cell, so neighbours never touch and a click near an
#: edge cannot land on the wrong tile.
_FILL = 0.8


def tile_label(path: str) -> str:
    """The object's name, from a BODex path like `<root>/use_data/core_mug_1038e4ea/mesh/simplified.obj`.

    The directory two levels up is the object; the filename is `simplified.obj` for all 360 of them
    and so tells the user nothing.

    Empty segments are dropped before counting: an absolute path's leading separator otherwise
    produces a leading '' that throws the "-3" index off by one -- '/data/mug.obj' normalises to
    parts ['', 'data', 'mug.obj'], length 3, and parts[-3] was '', a blank button in the gallery
    list. A path with fewer than three real segments falls back to showing itself in full, which is
    never blank. Same fix as shortMesh() in kitchen_wizard.py's form page -- the two must agree.
    """
    parts = [p for p in os.path.normpath(str(path)).split(os.sep) if p != ""]
    return parts[-3] if len(parts) >= 3 else str(path)


def _cell_origin(index: int, per_row: int, cell: float):
    """Row-major grid position: columns run +X, rows run -Y, so tile 0 is at the origin."""
    row, column = divmod(index, max(1, per_row))
    return np.array([column * cell, -row * cell, 0.0])


def _normalised(mesh, cell: float):
    """A copy centred on the origin and scaled so its longest side is `_FILL` of a cell.

    Display only: the tile's id carries the source path, and the mesh is loaded fresh at build
    time, so nothing scaled here reaches the generated scene.
    """
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    longest = float(max(mesh.extents)) if len(mesh.extents) else 0.0
    if longest > 0:
        mesh.apply_scale(cell * _FILL / longest)
    return mesh


#: Tiles are shown with a light neutral material. Without one, a mesh exports to glTF carrying no
#: material at all and three.js falls back to a dark default -- the first galleries rendered as
#: black silhouettes, which is useless for telling one mug from another. Slightly warm, so it does
#: not read as the same grey as the page chrome behind it.
_TILE_COLOR = [198, 196, 190, 255]


def _tile_material(index: int):
    """A material of its own for tile `index`.

    The distinct NAME keeps each tile's material its own: trimesh's glTF exporter deduplicates
    materials that compare equal, so tiles sharing one colour collapse to a single glTF material
    that every mesh then points at — measured, with one shared name 6 meshes exported 1 material.
    Nothing recolours a tile today (selection is drawn in the panel, not in the view), so this
    buys nothing at the moment; it means any later per-tile visual change cannot leak across the
    whole gallery.
    """
    return PBRMaterial(
        name=f"tile_{index:03d}",
        baseColorFactor=list(_TILE_COLOR),
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )


def _stood_up(mesh, up):
    """A copy rotated so `up` (in MESH coordinates) points at +Z, or the mesh itself when up is
    None.

    This is the picker's half of mesh_orientation.resolved_up: the placer stands the object on
    that vector, and drawing the tile in the raw file frame instead is what let the gallery show
    a bottle lying down that the scene then stood up. Only the up axis is matched, not the yaw —
    the placer aligns a `front` as well, but a tile is a free-orbit view of one object, so which
    way it faces answers no question the picker is being asked.

    The SMALLEST rotation that does it, built here rather than taken from trimesh.geometry.
    align_vectors, which is free to pick any rotation carrying one vector onto the other and picks
    a 120° turn about (1,1,1) for up=+Y — an axis cycle. That lands the object upright either way,
    but it shuffles the two horizontal extents, and those are two thirds of the size the panel
    prints. Turning about `up × Z` keeps the untouched axis untouched.
    """
    if up is None:
        return mesh
    up = np.array(up, dtype=float)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    if not np.any(axis):                    # already +Z, or flat upside down: no cross to turn about
        if up[2] > 0:
            return mesh
        axis = np.array([1.0, 0.0, 0.0])
    angle = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
    tilted = mesh.copy()
    tilted.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    return tilted


def _add_tile(scene, mesh, index: int, per_row: int, cell: float):
    """Place one normalised mesh in its cell and return the raw node name it was given."""
    node = f"tile_{index:03d}"
    transform = trimesh.transformations.translation_matrix(_cell_origin(index, per_row, cell))
    tile = _normalised(mesh, cell)
    tile.visual = trimesh.visual.TextureVisuals(material=_tile_material(index))
    scene.add_geometry(tile, node_name=node, geom_name=node, transform=transform)
    return node


def mesh_tiles(paths, *, per_row: int = 6, cell: float = 1.0, grasp_backed=None, obj_type=None):
    """A scene of one normalised mesh per path, and what each tile is.

    Every mesh is stood on the up vector the PLACER will use for it — mesh_orientation.resolved_up,
    the same call kitchen_build._object_pose makes. `obj_type` is the row's type, which decides
    that vector for a mesh the table does not list; without one the core_* fallback applies, which
    is what the placer would do with a type it does not recognise either. There is deliberately no
    way to ask for the raw file frame: it is never what the user is choosing.

    `grasp_backed` is the set of paths with real BODex grasps; None means nobody told us (no
    manifest), and the tile says nothing rather than claiming every mesh is ungraspable.

    A path that will not load is left out rather than raising: one corrupt file among 148 should
    cost that tile, not the gallery.
    """
    from mesh_orientation import resolved_up      # a dict and two rules; imports no Omniverse

    scene = trimesh.Scene()
    tiles = []
    for path in paths:
        try:
            mesh = trimesh.load(str(path), force="mesh")
        except Exception:
            continue
        # A malformed .obj does not always raise: trimesh can hand back an empty Trimesh
        # (0 vertices) whose `.extents` attribute exists but is None, so `hasattr` alone
        # does not catch it -- and a None extents crashes _normalised() below.
        if not hasattr(mesh, "extents") or mesh.extents is None:
            continue
        mesh = _stood_up(mesh, resolved_up(obj_type, str(path)))
        # Read AFTER standing it up, so the panel's "9.4 x 8.1 x 11.2 cm" is width x depth x
        # height on the counter and its third number is the height in the picture beside it.
        size = [float(v) for v in mesh.extents]   # before _add_tile normalises it away
        node = _add_tile(scene, mesh, len(tiles), per_row, cell)
        tiles.append({
            "id": str(path),
            "size": size,
            "node_raw": node,
            "node": sanitize_three_name(node),
            "label": tile_label(path),
            "grasps": None if grasp_backed is None else (str(path) in grasp_backed),
        })
    return scene, tiles


def kitchen_tiles(named_kitchens, *, per_row: int = 3, cell: float = 1.0, details=None):
    """A scene of one normalised kitchen per (name, seed, scene), and what each tile is.

    The caller builds the kitchens: building needs kitchen_build, which reaches scene_synthesizer,
    which this module must not import. Each kitchen is flattened to a single mesh -- a tile shows a
    silhouette, not a working scene, and 30-odd separate geometries per tile would bloat the page
    for nothing.

    `details` is an optional {seed: str} of data lines -- what the tile IS, when its label is only
    a number. Keyed by the SECOND element of each tuple (a variant key, a chair uid, a layout
    seed), not by the composite tile id this builds from it, so a caller can hand over the map it
    already has. The furniture galleries pass one ("Long dining", "uid e7cc55"; see
    kitchen_build.FURNITURE_NUMBERING); the five-layout kitchen gallery, whose labels are the
    thing itself, passes none. A keyword rather than a fourth element of each tuple, because that
    3-tuple is the shape every caller and every test stub of this function already builds.
    """
    details = details or {}
    scene = trimesh.Scene()
    tiles = []
    for name, seed, kitchen_scene in named_kitchens:
        mesh = kitchen_scene.to_geometry()
        size = [float(v) for v in mesh.extents]
        node = _add_tile(scene, mesh, len(tiles), per_row, cell)
        tiles.append({
            "id": f"{name}:{seed}",
            "size": size,
            "node_raw": node,
            "node": sanitize_three_name(node),
            "label": str(name).replace("_", " "),
            "detail": str(details.get(seed, "")),
        })
    return scene, tiles


def paginate(items, page, per_page: int = DEFAULT_PER_PAGE):
    """(the page's items, the clamped 1-based page, the total pages).

    Clamps rather than raising: a Next click from a page that has since shrunk is a stale button,
    not a bug, and it must land somewhere real.
    """
    items = list(items)
    pages = max(1, math.ceil(len(items) / per_page))
    page = min(max(1, int(page)), pages)
    start = (page - 1) * per_page
    return items[start:start + per_page], page, pages


def gallery_options(tiles, page: int, pages: int) -> list[str]:
    """Every id this page can answer with — tiles, the page turns that exist, and Cancel.

    The step publishes exactly this list. The server refuses an id it did not publish, so anything
    clickable that is missing here answers 400 and looks broken to the user.
    """
    options = [str(t["id"]) for t in tiles]
    if page > 1:
        options.append(f"page:{page - 1}")
    if page < pages:
        options.append(f"page:{page + 1}")
    options.append("")                      # Cancel
    return options


def render_gallery_page(scene, tiles, *, title: str, subtitle: str = "",
                        page: int = 1, pages: int = 1, labels_in_view: bool = False,
                        per_row: int = 6, units: str = "cm") -> str:
    """The gallery as a page: the tile scene, plus a panel to click, page and cancel.

    labels_in_view draws each tile's label into the 3D view as well as the list. On for the five
    kitchen layouts, where the name IS the thing being chosen and there is room for it; off for a
    24-tile mesh page, where the labels are long hashes that would collide with each other.
    """
    payload = {
        "tiles": [
            {"id": t["id"], "node": t["node"], "label": t["label"], "size": t.get("size", []),
             # What the tile IS, when its label is only a number ("Table 7 · Long dining"). Absent
             # for mesh_tiles, whose labels are filenames; "" then, and the panel prints nothing.
             "detail": t.get("detail", ""),
             # Absent for kitchen_tiles (grasps is a mesh-gallery concept); .get() turns that into
             # the same JSON null the panel already treats as "nobody told us".
             "grasps": t.get("grasps")}
            for t in tiles
        ],
        # Every tile is drawn at the same size, so the panel is the only place the real one can be
        # read -- and for a grasping task it is the number that decides whether the gripper closes.
        "units": str(units),
        # So the arrow keys can move up and down a row, not just along one.
        "per_row": int(per_row),
        "page": int(page),
        "pages": int(pages),
        "title": str(title),
        "subtitle": str(subtitle),
        "labels": bool(labels_in_view),
    }
    panel = (
        _PANEL.replace("__THEME__", THEME_CSS)
        .replace("__SPRITE_LABEL__", SPRITE_LABEL_JS)
        .replace("__GALLERY__", script_json(payload))
    )
    return scene_page(scene, panel)
_PANEL = """
<style>
__THEME__
/* Deliberately still a fixed 280px, unlike kitchen_preview's #simvla-panel. That panel widened
   because it carries long, unpredictable data (object labels, material names, joint paths).
   This one's own content is short by construction -- TABLE_VARIANTS' longest label is "Square,
   box legs" (18 chars) -- so it does not truncate at 280px. What it overlays does not scale the
   same way: render_gallery_page draws its tiles (and, with labels_in_view, their labels) in the
   3D canvas UNDER this position:fixed panel, and frame_at_elevation frames the whole scene
   without knowing the panel is there. _pick_table alone shows 12 tiles at per_row=4 -- widening
   this panel the way the preview's was widened would cover more of that grid's right column on
   a laptop-width screen for no truncation this panel actually has. */
#gal{position:fixed;top:0;right:0;width:280px;max-height:100vh;overflow-y:auto;
  background:var(--panel);color:var(--fg);box-sizing:border-box;padding:16px 18px;z-index:9999;
  font:12.5px/1.5 var(--ui);border-left:1px solid var(--edge)}
#gal h2{font-size:14px;margin:0 0 3px;font-weight:620;letter-spacing:-.01em}
#gal .sub{color:var(--fg-2);font-size:11.5px}
#gal .note{color:var(--muted);font-size:11px;line-height:1.5}
/* The key line is a legend for the keyboard, which is data about the keys themselves. */
#gal .keys{font-family:var(--mono);font-size:10.5px;letter-spacing:.02em}
#gal .row{display:flex;gap:7px;margin:11px 0;align-items:center}
/* "page 2 of 4" is a count, so mono and tabular: the digits must not shuffle as pages turn. */
#gal #gal-page{font-family:var(--mono);font-size:11px;font-variant-numeric:tabular-nums}
#gal button{font:inherit;font-size:12px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:5px 10px;cursor:pointer}
#gal button:hover:enabled{background:var(--edge);color:var(--accent)}
#gal button:disabled{opacity:.4;cursor:default}
#gal button:focus-visible{outline:1px solid var(--accent);outline-offset:1px}
/* Brass, spent once on this panel: the primary action. The highlighted TILE below gets a brass
   EDGE rather than a brass fill, so the two never compete for the same "this one" reading. */
#gal #gal-use:enabled{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);
  font-weight:640}
#gal #gal-use:enabled:hover{background:var(--accent-lift);color:var(--accent-fg)}
#gal .tiles{display:flex;flex-direction:column;gap:3px;margin-top:7px}
#gal .tiles button{text-align:left;font-size:11.5px;background:var(--bg);border-color:var(--edge);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#gal .tiles button.on{border-color:var(--accent);color:var(--accent);background:var(--accent-wash)}
#gal .status{color:var(--muted);font-size:11px;margin-top:9px;min-height:15px;
  font-family:var(--mono)}
</style>
<div id="gal">
  <h2 id="gal-title"></h2>
  <div class="sub" id="gal-sub"></div>
  <div class="note">Click a tile to highlight it, then Use it. Or pick from the list.</div>
  <div class="note keys">←→↑↓ move · Enter use · Esc cancel · PgUp/PgDn page</div>
  <div class="row">
    <button id="gal-prev">Prev</button>
    <span class="sub" id="gal-page"></span>
    <button id="gal-next">Next</button>
  </div>
  <div class="tiles" id="gal-tiles"></div>
  <div class="row">
    <button id="gal-use" disabled>Use this one</button>
    <button id="gal-cancel">Cancel</button>
  </div>
  <div class="status" id="gal-status"></div>
</div>
<script>
(function () {
  var DATA = __GALLERY__;
  var byNode = {};
  DATA.tiles.forEach(function (t) { byNode[t.node] = t; });

  document.getElementById('gal-title').textContent = DATA.title;
  document.getElementById('gal-sub').textContent = DATA.subtitle;
  document.getElementById('gal-page').textContent = 'page ' + DATA.page + ' of ' + DATA.pages;

  function setStatus(text) { document.getElementById('gal-status').textContent = text; }

  // The wizard keeps exactly one pending answer per step, and the director does not wake from
  // wait_answer() and republish a fresh step until it has processed the first one -- around a
  // second for a full 24-tile page. Every tile button, Prev, Next, Cancel AND a raycast hit on
  // the 3D view can all call answer(), so without a guard two rapid clicks (two tiles, or a tile
  // then Cancel) can both land inside that window, both get 200, and the single answer slot is
  // last-write-wins: the user sees the first click succeed and the director silently acts on the
  // second, with nothing on screen saying so. `answering` closes that window -- once an answer is
  // in flight, everything clickable here is refused until the POST resolves, and a non-200
  // response reopens it, same as _CHOICE_BODY in kitchen_wizard.py.
  var answering = false;

  function setBusy(busy) {
    answering = busy;
    document.querySelectorAll('#gal button').forEach(function (b) { b.disabled = busy; });
    if (!busy) {
      // A blanket re-enable would also re-enable a page turn that does not exist.
      document.getElementById('gal-prev').disabled = DATA.page <= 1;
      document.getElementById('gal-next').disabled = DATA.page >= DATA.pages;
    }
  }

  function answer(id, what) {
    if (answering) { return; }
    setBusy(true);
    setStatus(what ? ('picking ' + what + '…') : 'going back…');
    window.simvlaPost('/answer', {id: id}).then(function (r) {
      // On 200 the director moves on and the step poller reloads this tab.
      if (r.status !== 200) {
        setStatus((r.body && r.body.reason) || ('rejected (' + r.status + ')'));
        setBusy(false);
      }
    });
  }

  // ---- selection: what the view's click sets, and what "Use this one" commits ----
  var selected = null;
  var buttonOf = {};
  var useButton = document.getElementById('gal-use');

  // Selection is shown in the PANEL only -- the highlighted row, the Use button's label and the
  // status line. The tile in the 3D view is deliberately left alone: you are picking by looking at
  // the shape and its real colour, and washing the selected one in an emissive tint changes the
  // very thing being judged.
  function mark(tile, on) {
    var button = buttonOf[tile.id];
    if (button) { button.className = on ? 'on' : ''; }
  }

  function select(tile) {
    if (selected) { mark(selected, false); }
    selected = tile;
    mark(tile, true);
    useButton.disabled = false;
    useButton.textContent = 'Use ' + tile.label;
    describe(tile, 'selected: ');
  }

  function commitSelection() {
    if (selected) { answer(selected.id, selected.label); }
  }

  useButton.addEventListener('click', commitSelection);

  var list = document.getElementById('gal-tiles');
  DATA.tiles.forEach(function (t) {
    var b = document.createElement('button');
    b.textContent = t.label;
    b.title = t.id;
    // The list is the deliberate way in: a named row you aimed at, so it commits directly.
    b.addEventListener('click', function () { answer(t.id, t.label); });
    buttonOf[t.id] = b;
    list.appendChild(b);
  });

  var prev = document.getElementById('gal-prev');
  var next = document.getElementById('gal-next');
  prev.disabled = DATA.page <= 1;
  next.disabled = DATA.page >= DATA.pages;
  prev.addEventListener('click', function () { answer('page:' + (DATA.page - 1), ''); });
  next.addEventListener('click', function () { answer('page:' + (DATA.page + 1), ''); });
  document.getElementById('gal-cancel').addEventListener('click', function () { answer('', ''); });

  // Clicking the 3D itself. `scene`, `camera`, `renderer` and THREE are globals from trimesh's
  // viewer, which is how kitchen_preview's panel raycasts too.
  var raycaster = new THREE.Raycaster();
  var pointer = new THREE.Vector2();

  function tileOf(object) {
    for (var node = object; node; node = node.parent) {
      if (byNode[node.name]) { return byNode[node.name]; }
    }
    return null;
  }

  // "9.4 × 8.1 × 11.2 cm" — every tile is drawn the same size, so this is the only place the real
  // one can be read, and for a grasping task it is the number that decides whether the gripper
  // closes on the thing.
  function sizeOf(tile) {
    if (!tile.size || tile.size.length !== 3) { return ''; }
    var factor = DATA.units === 'cm' ? 100 : 1;
    var digits = DATA.units === 'cm' ? 1 : 2;
    return tile.size.map(function (v) { return (v * factor).toFixed(digits); }).join(' × ')
      + ' ' + DATA.units;
  }

  // The data line: the label, then what the thing IS, then how big it really is. `detail` leads
  // the measurements because for the furniture galleries the label is only a number and the
  // detail is the short uid — the one string a scene is reproducible from.
  function describe(tile, prefix) {
    var size = sizeOf(tile);
    var grasp = (tile.grasps === null || tile.grasps === undefined)
      ? '' : (tile.grasps ? '  ·  BODex grasps' : '  ·  NO grasp data');
    setStatus(prefix + tile.label + (tile.detail ? '  ·  ' + tile.detail : '')
      + (size ? '  ·  ' + size : '') + grasp);
  }

  function tileUnder(event) {
    var box = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - box.left) / box.width) * 2 - 1;
    pointer.y = -((event.clientY - box.top) / box.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    var meshes = [];
    scene.traverse(function (o) { if (o.isMesh) { meshes.push(o); } });
    var hits = raycaster.intersectObjects(meshes, false);
    return hits.length ? tileOf(hits[0].object) : null;
  }

  // Sweeping the grid to find a shape should cost nothing: hovering names the tile and its real
  // size, and touches neither the selection nor the answer.
  var hovered = null;

  function onViewHover(event) {
    if (answering) { return; }
    var tile = tileUnder(event);
    if (tile === hovered) { return; }
    hovered = tile;
    if (tile) {
      describe(tile, '');
      renderer.domElement.style.cursor = 'pointer';
    } else {
      renderer.domElement.style.cursor = '';
      if (selected) { describe(selected, 'selected: '); } else { setStatus(''); }
    }
  }

  function onViewClick(event) {
    // A click here SELECTS and never commits. You drag this same view to orbit, and a drag that
    // ends a pixel from where it started is a click -- when that committed, arranging the camera
    // picked a tile by accident and the run moved on. Nothing is sent until "Use this one".
    // Not a button, so setBusy()'s disabling would not stop this on its own.
    if (answering) { return; }
    var tile = tileUnder(event);
    if (tile) { select(tile); }
  }

  renderer.domElement.addEventListener('click', onViewClick);
  renderer.domElement.addEventListener('mousemove', onViewHover);

  // ---- the whole gallery from the keyboard ----
  function step(delta) {
    if (!DATA.tiles.length) { return; }
    var at = selected ? DATA.tiles.indexOf(selected) : -1;
    var next = at < 0 ? 0 : Math.min(DATA.tiles.length - 1, Math.max(0, at + delta));
    select(DATA.tiles[next]);
    var button = buttonOf[DATA.tiles[next].id];
    if (button && button.scrollIntoView) { button.scrollIntoView({block: 'nearest'}); }
  }

  function onKeyDown(event) {
    if (answering || event.metaKey || event.ctrlKey || event.altKey) { return; }
    var per = DATA.per_row || 6;
    var handled = true;
    if (event.key === 'ArrowRight') { step(1); }
    else if (event.key === 'ArrowLeft') { step(-1); }
    else if (event.key === 'ArrowDown') { step(per); }
    else if (event.key === 'ArrowUp') { step(-per); }
    else if (event.key === 'Enter') { commitSelection(); }
    else if (event.key === 'Escape') { answer('', ''); }
    else if (event.key === 'PageDown' && DATA.page < DATA.pages) {
      answer('page:' + (DATA.page + 1), '');
    }
    else if (event.key === 'PageUp' && DATA.page > 1) { answer('page:' + (DATA.page - 1), ''); }
    else { handled = false; }
    // Only swallow the keys we acted on: the browser's own shortcuts, and typing into anything
    // that lands here later, must keep working.
    if (handled) { event.preventDefault(); }
  }

  document.addEventListener('keydown', onKeyDown);

  // ---- labels drawn into the view, for galleries that ask for them ----
__SPRITE_LABEL__

  // simvlaLabelSprite above is kitchen_preview.SPRITE_LABEL_JS, shared with the preview's
  // dimension lines: one canvas-sprite mechanism for drawing text in a 3D page, not two. What is
  // left here is the only part that is a GALLERY decision -- how big the plaque is relative to the
  // tile it names, and where it sits.
  function addLabel(tile) {
    var object = scene.getObjectByName(tile.node);
    if (!object) { return; }
    var bounds = new THREE.Box3().setFromObject(object);
    var size = bounds.getSize(new THREE.Vector3());
    var centre = bounds.getCenter(new THREE.Vector3());
    var width = Math.max(size.x, size.z) * 0.95 || 1;
    var sprite = simvlaLabelSprite(tile.label, width);
    // Above the tile in world terms, and in front of it when the view looks straight down.
    sprite.position.set(centre.x, bounds.max.y + width * 0.2, centre.z);
    scene.add(sprite);
  }

  if (DATA.labels) { DATA.tiles.forEach(addLabel); }
})();
</script>
"""
