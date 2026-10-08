"""SimVLA: interactive 3D preview of a generated kitchen scene.

Renders a trimesh.Scene into a single HTML page (three.js and the geometry both
inlined, nothing fetched from the network) with an x-ray control, so objects inside
a closed refrigerator or behind wall cabinets can still be inspected.

Imports only trimesh + stdlib: no isaaclab, omni, pxr or scene_synthesizer.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from trimesh.viewer import scene_to_html

# Fixed by default so one SSH tunnel keeps working across re-rolls; override with
# SIMVLA_PREVIEW_PORT.
DEFAULT_PREVIEW_PORT = 8777


def _is_remote_session() -> bool:
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))


def find_vscode_helper():
    """VS Code Remote's `browser.sh`, if this machine has one.

    It runs `code --openExternal <url>`, which opens the page in the browser on the *user's*
    machine (their real, GPU-having default browser) and forwards the port automatically.

    Found on disk rather than through $BROWSER, because $BROWSER is only exported into VS
    Code's own integrated terminal. Run the generator from any other shell on the same box —
    a plain SSH session, tmux — and $BROWSER is simply absent, at which point Python's
    webbrowser falls back to the one browser the remote box does have: an X11-forwarded
    Firefox with no GPU, which renders a blank page. Locating the helper ourselves means the
    preview opens correctly no matter which terminal launched it.
    """
    home = Path.home()
    candidates = [
        *(home / ".vscode-server" / "cli" / "servers").glob("*/server/bin/helpers/browser.sh"),
        *(home / ".vscode-server" / "bin").glob("*/bin/helpers/browser.sh"),
    ]
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        return None
    # newest install wins — old ones linger after VS Code updates
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _socket_is_live(path: str) -> bool:
    """Can we actually connect? VS Code leaves dozens of dead sockets behind."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    try:
        sock.connect(path)
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _vscode_ipc_sockets():
    """Live VS Code CLI sockets, newest first. The helper reaches VS Code through one of these.

    $VSCODE_IPC_HOOK_CLI is tried first but is NOT trusted on its own: a shell that has been
    open across a VS Code reconnect carries a stale value, and this box currently has 17 dead
    sockets to 14 live ones. Trying only the stale one — and giving up when it fails — is how
    the preview ended up telling the user to open the URL by hand while a perfectly good live
    socket sat right next to it. So: gather every candidate, then keep only the ones that
    actually accept a connection.
    """
    seen = set()
    live = []
    for path in _ipc_socket_candidates():
        if path in seen:
            continue
        seen.add(path)
        if _socket_is_live(path):
            live.append(path)
    return live


def _ipc_socket_candidates():
    """Every socket worth trying: $VSCODE_IPC_HOOK_CLI first, then what's on disk, newest first."""
    candidates = []
    if os.environ.get("VSCODE_IPC_HOOK_CLI"):
        candidates.append(os.environ["VSCODE_IPC_HOOK_CLI"])
    try:
        found = sorted(
            Path(f"/run/user/{os.getuid()}").glob("vscode-ipc-*.sock"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates += [str(p) for p in found]
    except OSError:
        pass
    return candidates


def _open_via_vscode(url: str) -> bool:
    """Open `url` in the user's own browser through VS Code Remote. True if it worked."""
    helper = find_vscode_helper()
    if helper is None:
        print("[preview] no VS Code Remote helper on this machine.")
        return False

    sockets = _vscode_ipc_sockets()
    if not sockets:
        # Classic cluster shape: ~/.vscode-server lives on a shared home, so the helper is
        # visible from every node — but the CLI sockets live in /run/user/<uid>, which is
        # local tmpfs per machine. On a compute node VS Code was never connected to, the
        # helper is there and there is no socket to reach it through. Nothing we can do:
        # there is no path from here to the user's browser.
        print(
            f"[preview] VS Code's browser helper is on this machine ({socket.gethostname()}) "
            f"but no live CLI socket is — VS Code is not connected to this host, so it cannot "
            f"open a browser for you. Connect VS Code to this host directly (Remote-SSH, with "
            f"a ProxyJump through your login node) and both the browser and the port forward "
            f"become automatic."
        )
        return False

    for sock in sockets[:6]:
        env = dict(os.environ, VSCODE_IPC_HOOK_CLI=sock)
        try:
            done = subprocess.run(
                [str(helper), url],
                env=env,
                timeout=20,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            return True

    print(
        f"[preview] VS Code's helper failed on all {min(len(sockets), 6)} live socket(s). "
        f"Open the URL yourself."
    )
    return False


def auto_open_verdict():
    """Should we launch a browser ourselves, and why? -> (open_it, reason)

    The trap this exists to avoid: on a remote workstation, a bare webbrowser.open() launches
    whatever browser the *remote* box has, forwarded over X11 — no GPU, so WebGL fails and the
    user gets a blank white page. Better to hand over the URL than to open something that
    cannot render.

    But VS Code Remote changes the picture: its helper opens the page on the user's own
    machine. If we can reach that helper — via $BROWSER, or by finding it on disk — opening is
    exactly the right thing even over SSH.
    """
    forced = os.environ.get("SIMVLA_PREVIEW_OPEN")
    if forced == "0":
        return False, "SIMVLA_PREVIEW_OPEN=0"
    if forced == "1":
        return True, "SIMVLA_PREVIEW_OPEN=1"

    if os.environ.get("BROWSER"):
        return True, "$BROWSER is set — it opens the page on your machine, not this one"

    if find_vscode_helper() is not None:
        return True, "VS Code Remote — opening it on your machine, not this one"

    if _is_remote_session():
        return False, (
            "this is an SSH session and no VS Code helper was found, so the only browser here "
            "would be an X11-forwarded one — no GPU, no WebGL, blank page"
        )

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False, "no DISPLAY"

    return True, "local desktop"

# three.js: PropertyBinding.sanitizeNodeName strips these from every glTF node name.
_THREE_RESERVED = re.compile(r"[\[\]\.:/]")
_WHITESPACE = re.compile(r"\s")


def sanitize_three_name(name: str) -> str:
    """Mirror THREE.PropertyBinding.sanitizeNodeName, which GLTFLoader applies to every node.

    trimesh's 'bowl00/geometry_0' arrives in three.js as 'bowl00geometry_0'.
    """
    return _THREE_RESERVED.sub("", _WHITESPACE.sub("_", name))


#: The generator's one visual identity, as CSS custom properties, shared by every surface the run
#: puts on screen: the wizard's own pages (kitchen_wizard._SHELL) and the gallery panel injected
#: into a 3D page (kitchen_gallery._PANEL). One block so the two cannot drift apart.
#:
#: Deliberately single-theme rather than light/dark aware. It is an ARCHITECT'S DARK TABLE: a
#: neutral ink ramp (--bg -> --panel -> --panel-2 -> --raise) for the ground the work sits on, brass
#: for the one thing currently chosen, and every colour stated outright so it holds whatever the
#: surrounding page does. Amber warns, red refuses, steel annotates.
#:
#: Two faces, split by JOB, not by taste: --ui carries labels, headings and prose; --mono is
#: reserved for DATA -- dimensions, seeds, uids, mesh paths, counts. Both are system stacks, never a
#: webfont: every page here is served off 127.0.0.1 and must stay self-contained (see
#: test_page_is_self_contained), so a linked font would be a request that fails and a page that
#: silently reflows.
#:
#: This replaced a phosphor terminal -- black ground, green monospace, everything at 12px. That was
#: internally consistent but spent its one accent everywhere: green was the selection AND the body
#: text AND the headings AND the borders, so a selected tile had nothing left to say. Here brass is
#: spent ONCE, on the current selection and the primary action, and on nothing else.
#:
#: THE TOKEN NAMES ARE THE CONTRACT, not the values. Four surfaces style themselves out of this one
#: block -- kitchen_wizard._SHELL, kitchen_gallery._PANEL, kitchen_preview._PANEL and
#: task_composer's own panel -- and test_kitchen_wizard asserts --accent reaches every published
#: page. Re-skinning means changing values here; renaming a token means visiting all four.
THEME_CSS = """
:root{
  --bg:#0E1116; --panel:#151A21; --panel-2:#1B2129; --raise:#222932;
  --fg:#E8EBEF; --fg-2:#98A3B2; --muted:#66717F;
  --edge:#2A323D; --edge-bright:#39434F;
  --accent:#C89B4A; --accent-fg:#12151A; --accent-wash:rgba(200,155,74,.10);
  --accent-lift:#DCAE5C;
  --steel:#6E9CC4;
  --ok:#5FA97E; --ok-dim:#16281e;
  --warn:#C98A3E; --warn-dim:#2c2113;
  --err:#D4675E; --err-dim:#2e1917;
  --scrim:rgba(14,17,22,.94);
  --ui:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,"DejaVu Sans Mono",monospace;
}
"""

#: The 3D viewport's ground, as a CSS/hex colour.
#:
#: trimesh's bundled viewer template hardcodes `scene.background = new THREE.Color(0xffffff)`, so
#: every page scene_to_html produces -- the kitchen preview and every gallery -- put this module's
#: near-black terminal panel on a stark white void. That is not a neutral backdrop either: a white
#: ground washes out the pale woods and white goods a kitchen is mostly made of, and the chair and
#: table libraries are lit flat, so a light chair on white loses its silhouette exactly when you
#: are trying to judge its shape.
#:
#: Lifted off THEME_CSS's --bg on purpose rather than matching it: at true black a dark chair loses
#: its outline the same way a pale one does against white. This sits dark enough to read as the same
#: surface as the panel and light enough to hold both.
#:
#: RE-DERIVED FOR THE INK PALETTE, and it had to be. The old value (#16211c) was a green-tinted
#: near-black picked to match the phosphor theme's --bg (#05070a); dropped unchanged onto the ink
#: ramp it reads as a green cast behind every asset -- most visible on the pale woods the table and
#: chair libraries are mostly made of, which is exactly what the ground exists not to distort.
#:
#: Derived by keeping the two things the old value was chosen for and re-solving them on the new
#: ramp: (a) the SAME lift off --bg in perceptual lightness -- old was L* 2.2 -> 11.4, +9.2, and
#: this is L* 4.2 -> 13.4, +9.2; (b) a hue on the ink ramp itself, so it reads as the same surface
#: family as the panel rather than as a second, differently-tinted dark. In the ramp it lands
#: between --panel-2 (#1B2129, L* 12.0) and --raise (#222932, L* 16.3), which is where the old one
#: sat relative to its own panel tones.
#:
#: Override without editing code -- SIMVLA_VIEWER_BG='#808080' for a neutral grey, or 'white' to
#: get trimesh's original back.
VIEWER_BG = os.environ.get("SIMVLA_VIEWER_BG", "#1C232C")

#: The exact statement to rewrite in trimesh's minified template. Pinned as the whole statement,
#: not the bare colour: '0xffffff' appears four times in that file (light colours among them) and
#: rewriting those would change the lighting, not the ground.
_TRIMESH_BG_ANCHOR = "scene.background=new THREE.Color(0xffffff);"


def set_viewer_background(html: str, colour: str | None = None) -> str:
    """Repaint the 3D viewport of a page trimesh's scene_to_html produced.

    Raises if the anchor is gone, which is the point: this patches a vendored, minified template
    we do not control, so a trimesh upgrade that renames it must fail loudly here rather than
    silently restoring the white ground on every gallery in the run.
    """
    if colour is None:
        colour = VIEWER_BG
    if html.count(_TRIMESH_BG_ANCHOR) != 1:
        raise RuntimeError(
            "trimesh's viewer template no longer sets the background where this expects it "
            f"({_TRIMESH_BG_ANCHOR!r} found {html.count(_TRIMESH_BG_ANCHOR)} times, want 1). "
            "Re-read the template and update _TRIMESH_BG_ANCHOR."
        )
    return html.replace(
        _TRIMESH_BG_ANCHOR, f"scene.background=new THREE.Color('{colour}');"
    )


def script_json(payload) -> str:
    """JSON safe to embed in a <script> block.

    json.dumps alone is not enough: it does not escape '<', so a value containing '</script>'
    ends the block early and everything after it is parsed as live markup. These payloads carry
    filesystem paths, prim paths and object labels, so that is data, not a hypothetical.
    """
    return json.dumps(payload).replace("<", "\\u003c")


def object_node_names(scene, node_id: str) -> list[str]:
    """Graph nodes owned by node_id, named the way three.js will see them.

    Matching keeps the '/' boundary so that node 'mug100' (object mug10) is not
    mistaken for a child of object mug1.
    """
    owned = [n for n in scene.graph.nodes if n == node_id or n.startswith(node_id + "/")]
    return sorted(sanitize_three_name(n) for n in owned)


#: How far above the horizon a 3D page opens, in degrees. 0 is a straight-on side view, 90 the
#: plan view trimesh defaults to. The default is the one angle that hides orientation: from directly
#: overhead a standing bottle is a small circle and a fallen one is a long shape, so "compact" reads
#: as fallen and "elongated" as upright -- the judgement inverts. Both the preview and the tile
#: galleries open here.
ELEVATION_DEG = 32.0


def frame_at_elevation(scene, degrees: float = ELEVATION_DEG) -> None:
    """Point the opening camera DOWN AT the scene rather than straight down ON it.

    Sets only where the view starts; the page is still orbitable. A trimesh that disagrees about
    set_camera keeps its own default rather than losing the page -- a camera is a nicety, a page
    that will not render is not.
    """
    try:
        bounds = scene.bounds
        if bounds is None:
            return
        span = float(max(bounds[1][i] - bounds[0][i] for i in range(3)))
        scene.set_camera(
            angles=(math.radians(90.0 - degrees), 0.0, 0.0),
            distance=max(span * 1.15, 1e-3),
            center=bounds.mean(axis=0),
        )
    except Exception:
        pass


def scene_page(scene, panel: str) -> str:
    """A trimesh.Scene as a standalone 3D page with `panel` injected before </body>.

    The one shape every 3D page in this wizard has: frame the camera, render through trimesh's
    bundled viewer template, repaint its hardcoded white ground, inject a panel. Written once here
    rather than three times over, because the ORDER matters and is not obvious -- set_viewer_background
    has to run on scene_to_html's output (it rewrites a statement inside the vendored template) and
    the panel has to go in after that (it carries a <style> block the template must not have
    swallowed). See kitchen_gallery.render_gallery_page and kitchen_wizard.render_edit_page, both of
    which are exactly this plus their own panel.
    """
    frame_at_elevation(scene)
    return set_viewer_background(scene_to_html(scene)).replace("</body>", panel + "\n</body>")


#: The ONE way a 3D page in this run draws text into the 3D view.
#:
#: There is no text primitive in a trimesh scene and no font in the page, so a label has to be
#: rasterised in the browser. A canvas texture on a THREE.Sprite is how kitchen_gallery has drawn
#: its tile labels since labels_in_view existed; this is that code, lifted here so the preview's
#: dimension lines reuse the mechanism instead of being a third way of doing the same thing.
#: kitchen_gallery imports THEME_CSS, scene_page and script_json from this module already, so the
#: dependency runs the way it already ran.
#:
#: A SPRITE rather than extruded geometry, and the reason is not tidiness: a sprite always faces
#: the camera, so a dimension label stays readable whichever way the reader orbits the view, and
#: depthTest:false keeps it legible when the thing it annotates is behind a cabinet. Extruded text
#: would need a font file -- a network fetch, on a page that must stay self-contained.
#:
#: The canvas is a fixed 256x64, so the plaque is always 4:1 whatever the text says and `width` is
#: its long side in world units. fillText's maxWidth argument squeezes anything that would overflow
#: rather than letting it run off the edge; every label drawn today fits without it (the longest is
#: "single wall" in the layout gallery and "H1 · 1.80 m" here, both about 190 of the 236 usable
#: pixels), so it changes nothing now and stops a longer one from silently losing its end.
SPRITE_LABEL_JS = """
  function simvlaLabelSprite(text, width) {
    var canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(16,18,22,0.82)';
    ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#f2f4f6';
    ctx.font = 'bold 30px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 34, 236);

    var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false
    }));
    sprite.scale.set(width, width * 0.25, 1);
    sprite.renderOrder = 999;
    return sprite;
  }
"""

#: The prefix every piece of the scale reference is named with -- the robot, and its dimension rule.
#: ONE prefix, because the containment checks are "no node whose name starts with this reaches
#: `objects` / the placement gate / the USD export / _label_supports", and a check written that way
#: covers the next piece somebody adds as well as today's two.
#:
#: Deliberately not a valid-looking asset name. Nothing in kitchen_build ever produces a node with
#: this prefix, so a prim carrying it in an exported stage is unambiguous evidence of a leak rather
#: than a coincidence.
SCALE_NODE_PREFIX = "simvla_scale_"

#: The scene-graph nodes one robot and its dimension line are added under. Per robot, so a page
#: rendered for a different robot cannot be confused with this one -- and both still carry
#: SCALE_NODE_PREFIX, which is what every containment check matches on.
def robot_node(robot: str) -> str:
    return SCALE_NODE_PREFIX + robot


def rule_node(robot: str) -> str:
    return SCALE_NODE_PREFIX + "rule_" + robot


#: The three robots this project has assets for, and their standing heights in metres. MEASURED,
#: never recalled -- a wrong robot height in a scale reference is worse than no robot at all.
#:
#: PROVENANCE. Each number is the z extent of the VISIBLE geometry of the very USD that the
#: project's own robot config spawns. The asset for each was chosen by reading what the code loads,
#: never by name -- there are seven Anubis USDs in this tree and two RB-Y1 model trees, each
#: holding both an rby1a and an rby1m:
#:
#:   Anubis     source/isaaclab_assets/data/Robots/anubis_simvla.usd
#:              (isaaclab_assets/robots/anubis_wheels.py:16)     z 0.000003 .. 1.194271 -> 1.1942683
#:   AI Worker  source/isaaclab_assets/data/Robots/MM/aiworker/ffw_bg2.usd
#:              (isaaclab_assets/robots/aiworker_BG2.py:12)       z 0.000000 .. 1.611375 -> 1.611375
#:   RB-Y1      $SIMVLA_RBY1M_DIR/models/rby1a/urdf/model/model.usd
#:              z 0.000000 .. 1.470000 -> 1.470000
#:
#: RB-Y1 IS THE rby1A, ON THE USER'S INSTRUCTION, and this disagrees with the code: RBY1_CFG
#: (isaaclab_assets/robots/rby1.py:13) spawns .../models/rby1m/... The config is production robot
#: configuration and was deliberately left alone; only the drawing follows the instruction. The
#: `rby1m` tree is used rather than the `rby1m_new` one because RBY1_CFG's own path pattern points
#: there -- an ASSUMPTION, recorded rather than hidden: the two trees' model.usd files are
#: byte-identical and their mesh payloads differ by about 1 MB in 86 MB, so something under them
#: does differ.
#:
#: Measured with the visible geometry itself, cross-checked against UsdGeom.BBoxCache(purposes=
#: [default, render], useExtentsHint=False) over each stage's DEFAULT PRIM. Every one of those
#: choices is load-bearing:
#:   * the default prim, because a walk of the whole stage would pick up anything a sibling adds;
#:   * default+render purposes only, because the collision hulls are `guide` -- Anubis ships 404,350
#:     faces of them and AI Worker 646,996, and RB-Y1's are 5 cm TALLER than its visible robot
#:     (1.519927 m against 1.470000), so a bbox over all purposes would put the line 5 cm wrong;
#:   * the GEOMETRY rather than BBoxCache's answer where the two differ, which is Anubis by 2.95 mm:
#:     its meshes carry authored `extent` attributes padded past their own points, BBoxCache reads
#:     those, and the renderer draws the points. The line has to measure the robot.
#:
#: ANUBIS IS MEASURED IN THE POSE ITS CONFIG SPAWNS IT IN; the other two are measured in the pose
#: their USD is authored in. Anubis alone, on the user's instruction, and the difference is 52 cm:
#: ANUBIS_CFG.init_state (anubis_wheels.py:35) folds both elbows to 2.356 rad, which drops the robot
#: from the 1.715914 m V-shape its asset is authored in -- wrists overhead at 1.63 m -- to 1.1942683
#: m with the arms tucked forward and the mast the tallest thing left. A reference standing in the
#: authored pose sizes furniture against a robot no scene ever contains. The pose is READ from that
#: config rather than restated, by scripts/tools/build_robot_reference.spawn_pose_of, and applied by
#: forward kinematics over the USD's own UsdPhysics joints (spawn_transforms), so tuning the spawn
#: pose and rebuilding the cache is the whole of what it takes to move this number.
#:
#: NO PUBLISHED FIGURE WAS AVAILABLE to compare any of these against -- there is no spec sheet in
#: either asset tree and this machine has no network. Each measurement therefore ships unreconciled,
#: as the G1 it replaces did. RB-Y1's 1.470000 m is exact to seven digits with its torso at the
#: authored posture -- that column is prismatic on the real robot, so a running scene can be taller.
ROBOTS = {
    "anubis": {"label": "Anubis", "height_m": 1.1942683},
    "rby1": {"label": "RB-Y1", "height_m": 1.470000},
    "aiworker": {"label": "AI Worker", "height_m": 1.611375},
}

#: The heights alone, which is what the dimension line and its label are drawn against. A VIEW of
#: ROBOTS rather than a second list, so the two cannot drift.
ROBOT_HEIGHTS_M = {name: spec["height_m"] for name, spec in ROBOTS.items()}

#: Which robot a page draws when it has no chooser on it -- the wizard's edit step, which sizes one
#: piece of furniture against a robot and has no room for a selector. RB-Y1 because it is the
#: shortest of the three, so a counter judged reachable beside it is reachable beside all of them.
DEFAULT_ROBOT = "rby1"

#: Where the cached robots live, and the environment variable that moves them. The GLBs are built by
#: scripts/tools/build_robot_reference.py and kept on /lustre, never in git -- the same rule the
#: chair and table libraries follow (chair_manifest.DEFAULT_CHAIR_OBJ_DIR), for the same reason:
#: they are assets this install has, not something the repo carries.
ROBOT_REFERENCE_DIR = "robot_reference"
ROBOT_REFERENCE_ENV = "SIMVLA_ROBOT_REFERENCE_DIR"


def robot_reference_path(robot: str) -> Path:
    """The cached `robot`: under $SIMVLA_ROBOT_REFERENCE_DIR, else the /lustre default."""
    directory = os.environ.get(ROBOT_REFERENCE_ENV) or ROBOT_REFERENCE_DIR
    return Path(directory) / f"{robot}_scale_reference.glb"


def robot_mesh(robot: str):
    """The cached `robot`, standing on z = 0 and facing the reader. None if it is not cached.

    THE ROBOT'S OWN GEOMETRY AT SOURCE RESOLUTION, not a proxy of it: 262k-647k faces and 5-13 MB
    of GLB. It costs that much only on a page whose author has picked this robot out of the page's
    selector -- the reference is off by default and nothing is embedded until it is chosen -- so
    fidelity is what the bytes are spent on. scripts/tools/build_robot_reference.py made it and
    records what from, including the per-link decimation it measured and did not take.

    IT ARRIVES IN THE ASSET'S OWN LIVERY, per link: AI Worker white against dark grey grippers,
    RB-Y1 mid grey with near-white and near-black trim, Anubis silver with dark accents. That livery
    is most of what makes a robot recognisable at a glance. The colours ride in the GLB as
    per-vertex colour, which is the one form that survives both trimesh's glTF loader and this
    page's own re-export.

    NONE, NOT AN EXCEPTION, on a checkout without the asset: every page must still render. The
    reference then draws the dimension line alone, which still answers the question it is here for
    -- how high this robot comes against this counter.

    TURNED TO FACE THE READER, a quarter turn about z, and the same turn for all three because the
    builder has already checked that each one faces +x (see build_robot_reference.FacingWitness).
    The page's camera stands on the -Y side looking toward +Y: frame_at_elevation sets
    angles=(radians(90-32), 0, 0), so the camera's own +Z in world is
    R_x(58 deg) . (0, 0, 1) = (0, -0.848, +0.530), and a camera looks down its own -Z. Untouched,
    the robot opens in profile. It is not re-centred here: the builder centres it in plan, because
    Anubis is authored 2.0 m away from its own origin.
    """
    path = robot_reference_path(robot)
    if not path.is_file():
        return None
    # CACHED ON (path, mtime), because one page render asks for this three times -- the footprint,
    # the inset and the geometry itself -- and parsing 13 MB of glTF three times is seconds of wait
    # per wizard step. Keyed on the file rather than on the robot name so that moving
    # $SIMVLA_ROBOT_REFERENCE_DIR, or rebuilding the cache, is picked up; the miss path returns
    # None without touching the cache, so a robot built after a cold-cache page still appears.
    # The mesh is SHARED between pages: nothing here mutates it (add_geometry stores the transform
    # on the scene graph, and trimesh's exporters do not write to their input).
    return _load_robot_mesh(robot, str(path), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=8)
def _load_robot_mesh(robot: str, path: str, _mtime_ns: int):
    import trimesh
    from trimesh.visual.material import PBRMaterial

    mesh = trimesh.load(path, force="mesh")
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(-math.pi / 2.0, (0.0, 0.0, 1.0))
    )
    # A material AND the asset's colours, which takes both halves because neither alone works.
    # Without a material trimesh exports the mesh to glTF carrying none, and three.js falls back to
    # a near-black default -- the same trap kitchen_gallery._tile_material documents. Without the
    # colours the robot is one flat tone and stops looking like itself. So the loaded per-vertex
    # colours are carried across onto a material whose baseColorFactor is WHITE, because glTF
    # multiplies the two and any tint here would mute the livery it is drawn to show.
    colours = mesh.visual.vertex_colors.copy()
    mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        name=f"simvla_{robot}", baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0, roughnessFactor=0.75, doubleSided=True,
    ))
    mesh.visual.vertex_attributes["color"] = colours
    return mesh


#: A dimension rule's cross-section and how long its end ticks are, in metres. Ticks rather than
#: arrowheads: that is what an architectural dimension line uses, and two boxes cost 24 faces.
_RULE_THICKNESS_M = 0.012
_RULE_TICK_M = 0.14

#: The gap left between the robot's right edge and the dimension line's tick, in metres. The offset
#: itself is MEASURED off the robot that will be drawn rather than stated, because the three are
#: 0.58, 0.70 and 0.71 m across and one constant would stand the line inside two of them.
_RULE_GAP_M = 0.05

#: Where the line stands on a page whose robot is not cached, in metres -- there is no robot to
#: measure, and a line drawn at x = 0 would still be a line the author can read a height off.
_RULE_OFFSET_M = 0.32

#: How far above the top of the rule its number is drawn, in metres. Clear of both the rule's own
#: top tick and the robot's head.
_LABEL_RISE_M = 0.09

#: How wide a label plaque is drawn, in metres. The sprite is the gallery's 256x64 canvas, so it is
#: always 4:1 whatever the text says; this is its long side.
_LABEL_WIDTH_M = 0.40


def dimension_rule(height_m: float, *, tick_m: float = _RULE_TICK_M,
                   thickness_m: float = _RULE_THICKNESS_M):
    """A dimension line `height_m` tall: a thin vertical bar with a flat tick at each end.

    EXACTLY `height_m` FROM z = 0 TO z = height_m, by construction and with no normalising step:
    the ticks are laid INSIDE the span (the floor tick occupies [0, thickness], the top one
    [height - thickness, height]) rather than centred on the ends, so the mesh's own z extent is the
    dimension it claims. A tick centred on each end would make a rule labelled 1.8 m measure 1.812.

    The text is NOT here -- there is no text primitive in a trimesh scene. It is drawn in the
    browser as a canvas sprite; see SPRITE_LABEL_JS.
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial

    height = float(height_m)
    thickness = float(thickness_m)
    bar = trimesh.creation.box(extents=(thickness, thickness, height))
    bar.apply_translation((0.0, 0.0, height / 2.0))
    parts = [bar]
    for z_centre in (thickness / 2.0, height - thickness / 2.0):
        tick = trimesh.creation.box(extents=(float(tick_m), thickness, thickness))
        tick.apply_translation((0.0, 0.0, z_centre))
        parts.append(tick)
    rule = trimesh.util.concatenate(parts)
    # THEME_CSS's --fg-2: a drawn line, lighter than the steel robot it annotates and not the brass
    # accent, which this theme spends only on the current selection and the primary action.
    rule.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
        name="simvla_scale_rule", baseColorFactor=[152, 163, 178, 255],
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True,
    ))
    return rule


def _scale_pieces(robot: str = DEFAULT_ROBOT):
    """Everything `robot`'s scale reference draws: [(node name, mesh, x offset in metres)].

    ONE list, read by both scale_reference_extents and add_scale_reference, so the footprint the
    page reserves cannot drift from the geometry it then stands there. The rule is always drawn; the
    robot only on an install that has the cached asset (see robot_mesh).
    """
    mesh = robot_mesh(robot)
    offset = (_RULE_OFFSET_M if mesh is None
              else float(mesh.bounds[1][0]) + _RULE_TICK_M / 2.0 + _RULE_GAP_M)
    pieces = [] if mesh is None else [(robot_node(robot), mesh, 0.0)]
    pieces.append((rule_node(robot), dimension_rule(ROBOT_HEIGHTS_M[robot]), offset))
    return pieces


def scale_reference_extents(robot: str = DEFAULT_ROBOT):
    """(x_lo, x_hi, y_lo, y_hi): the whole assembly's plan footprint, relative to the robot's own
    standing point at (0, 0).

    Measured off the meshes that will actually be drawn rather than re-derived from the layout
    constants, so it cannot drift from them -- an assembly that reports a footprint it does not have
    is how the robot ends up half inside a cabinet.
    """
    import numpy as np

    boxes = []
    for _node, mesh, offset in _scale_pieces(robot):
        # A copy: trimesh hands back a read-only cached array, and this shifts it into place.
        bounds = np.array(mesh.bounds, dtype=float)
        bounds[:, 0] += float(offset)
        boxes.append(bounds)
    low = min(float(b[0][0]) for b in boxes), min(float(b[0][1]) for b in boxes)
    high = max(float(b[1][0]) for b in boxes), max(float(b[1][1]) for b in boxes)
    return low[0], high[0], low[1], high[1]


def scale_reference_inset(robot: str = DEFAULT_ROBOT) -> float:
    """How far inside a scene's plan bounds the assembly's CENTRE has to stay to fit inside them.

    Half the assembly's longer plan span, plus 5 cm. clearest_floor_spot uses this as its inset, so
    that "the clearest square" is a square the whole reference fits in rather than one its midline
    fits in.
    """
    x_lo, x_hi, y_lo, y_hi = scale_reference_extents(robot)
    return max(x_hi - x_lo, y_hi - y_lo) / 2.0 + 0.05


def add_scale_reference(scene, at=(0.0, 0.0), robot: str = DEFAULT_ROBOT):
    """Stand `robot`'s scale reference in `scene`, centred in plan on `at`. -> (nodes, labels)

    `nodes` are three.js node names -- the robot and its dimension rule. `labels` are
    {"text", "at": [x, y, z], "width"} for the browser to draw as sprites, because a trimesh scene
    has no text primitive.

    MUTATES `scene`, so every caller has to have decided already that this scene is a throwaway
    render copy. render_page copies before calling this; render_edit_page builds the scene it passes
    from nothing. There is deliberately no "add it and take it out again" path: a remove that is
    skipped by an exception is how a robot ends up in a committed kitchen.
    """
    import trimesh

    pieces = _scale_pieces(robot)
    x_lo, x_hi, y_lo, y_hi = scale_reference_extents(robot)
    # `at` is the assembly's plan CENTRE, not the robot's feet: the robot stands at one end of the
    # row and the rule beside it, so anchoring on the robot would push the rule off the clear spot
    # that was chosen. The robot therefore stands at `at` minus the assembly's own centre.
    origin = (
        float(at[0]) - (x_lo + x_hi) / 2.0,
        float(at[1]) - (y_lo + y_hi) / 2.0,
    )

    nodes = []
    rule_offset = _RULE_OFFSET_M
    for node, mesh, offset in pieces:
        scene.add_geometry(
            mesh, node_name=node, geom_name=node,
            transform=trimesh.transformations.translation_matrix(
                [origin[0] + offset, origin[1], 0.0]
            ),
        )
        nodes.append(sanitize_three_name(node))
        if node == rule_node(robot):
            rule_offset = offset
    height = ROBOT_HEIGHTS_M[robot]
    labels = [{
        "text": f"{ROBOTS[robot]['label']} · {height:.2f} m",
        "at": [origin[0] + rule_offset, origin[1], height + _LABEL_RISE_M],
        "width": _LABEL_WIDTH_M,
    }]
    return nodes, labels


def clearest_floor_spot(scene, inset_m: float | None = None, robot: str = DEFAULT_ROBOT):
    """Where on this scene's floor plan the scale reference can stand without standing in anything.

    A coarse grid over the plan bounds, scored by distance to the nearest geometry's plan-view
    bounding box, best square wins. Bounding boxes rather than the meshes themselves because this
    is a display decision made once per page: the boxes overstate a table's footprint slightly,
    which errs toward the open floor, and they cost nothing against a real kitchen's 40-odd
    geometries where a proper distance query would not.

    The alternative -- standing the reference at the middle of the room -- was tried and looks
    broken: the middle of a kitchen is where the table is, so the ruler comes up embedded in it.

    `inset_m` keeps the reference's own footprint inside the plan bounds. Without it the clearest
    square is a CORNER of the bounding box on three of the five layouts (measured: l_shaped,
    single_wall and peninsula all answered exactly min-x), which stands the reference half outside
    the room -- the walls are added at commit and are not in this scene, so nothing else stops it.
    None means scale_reference_inset(), i.e. half the whole assembly's longer plan span plus a
    margin, measured off the meshes rather than stated: the robots are 0.51-0.71 m across and the
    rule stands 0.37-0.47 m further out again, so the assembly is about a metre wide and a stated
    constant would be wrong for two of the three. The inset is dropped rather than allowed to
    invert on a scene smaller than twice itself.

    Falls back to the plan centre if the scene has no bounds at all, which is the empty-scene case
    and not worth an exception on a page that is otherwise fine.
    """
    import numpy as np

    bounds = getattr(scene, "bounds", None)
    if bounds is None:
        return (0.0, 0.0)
    low, high = np.asarray(bounds, dtype=float)
    centre = ((low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0)
    inset = scale_reference_inset(robot) if inset_m is None else float(inset_m)
    if high[0] - low[0] > 2 * inset and high[1] - low[1] > 2 * inset:
        low = low + np.array([inset, inset, 0.0])
        high = high - np.array([inset, inset, 0.0])

    boxes = []
    for name in scene.graph.nodes_geometry:
        try:
            transform, geometry = scene.graph[name]
            mesh = scene.geometry[geometry].copy()
            mesh.apply_transform(transform)
            boxes.append((mesh.bounds[0][:2], mesh.bounds[1][:2]))
        except Exception:
            continue
    if not boxes:
        return centre

    def clearance(x, y):
        worst = float("inf")
        for lo, hi in boxes:
            # Distance from (x, y) to the box: 0 inside it, the euclidean gap outside.
            dx = max(lo[0] - x, 0.0, x - hi[0])
            dy = max(lo[1] - y, 0.0, y - hi[1])
            worst = min(worst, float(np.hypot(dx, dy)))
        return worst

    best, best_score = centre, -1.0
    steps = 24
    for i in range(steps + 1):
        for j in range(steps + 1):
            x = low[0] + (high[0] - low[0]) * i / steps
            y = low[1] + (high[1] - low[1]) * j / steps
            score = clearance(x, y)
            if score > best_score:
                best, best_score = (x, y), score
    return best


#: The node ids whose objects drag against a mathematical ground plane (pickGround) instead of
#: raycasting the support meshes (pickSurface): the furniture the author places on the FLOOR.
#: Written here as node ids, matching the two names kitchen_build.build_kitchen puts in `objects`
#: -- "table" from add_table and "chair_<n>" from add_chair.
#:
#: A NODE-ID test rather than "this object has no support", even though support=None is what both
#: entries carry and reads like the same statement. It is not: a caller that builds its object
#: list by hand omits the key entirely (kitchen_usd_load._composer_objects does exactly that, for
#: every object it loads back out of a committed USD), and `not obj.get("support")` cannot tell
#: "stands on the floor" from "nobody said". Reading it as floor-dragged would put every
#: composer-loaded object on the ground plane.
#:
#: fullmatch, never a substring test, so a fixture whose name merely contains "table" or "chair"
#: is not swept in -- the same rule and the same reason as kitchen_build._is_furniture.
_FLOOR_DRAGGED_NODE = re.compile(r"table|chair_\d+")


def render_page(scene, objects, materials, supports, joints=None, actions=None,
                scale_reference: str | None = None) -> str:
    """The preview page as a string.

    scene     -- trimesh.Scene
    objects   -- [{"label": "bowl0", "node_id": "bowl00", "location": "Above cabinet",
                   "support": "countertop_base_cabinet"}]  ("support" optional)
    materials -- {group: display name}, e.g. {"countertop": "Granite_Dark"}
    joints    -- kitchen_build.collect_joints(kitchen), or None for no joint sliders.
                 Inspection only: the page never posts joint values back, so opening a
                 drawer to look inside can never leak into the exported USD.
    supports  -- [node_id, ...] scene-graph node names of every support surface
                 (countertops, island top, fridge shelves) a drag may land on. Expanded
                 through object_node_names the same way objects are, so the JS gets the
                 sanitized three.js node names it can match against mesh ancestor chains.
    actions   -- [{"id": "accept", "label": "Accept & Generate", "path": "/answer"}] or None.
                 Renders a button row at the top of the panel; each button POSTs {"id": ...}
                 through window.simvlaPost (injected by kitchen_wizard, which stamps the step id).
                 "path" defaults to "/answer". None means no row, which is what every caller
                 outside the wizard wants.
    scale_reference -- which robot to stand in the room, and whether to offer the choice at all.
                 THREE STATES, in one argument:
                   None    -- no reference and no selector. Every caller that has not asked for
                              one, so nothing changes for them at all.
                   ""      -- the selector, with no robot chosen. Nothing is drawn and nothing is
                              embedded; this is what the wizard's preview opens on.
                   a name  -- that robot (a key of ROBOTS) and a dimension line at its measured
                              standing height, plus the selector, set to it.

    THE CHOICE IS SERVER-SIDE, and that is not a style preference. Each robot is 5-13 MB of GLB
    (see robot_mesh), so a client-side switch between three would have to embed all three -- 25 MB
    of page, paid on every wizard step by an author who never opens the reference. The selector
    instead POSTs `robot:<name>` back like any other action and the caller re-renders this page for
    the chosen robot, so a page carries exactly one robot and a page with none carries none.

    THE SCALE REFERENCE GOES INTO A COPY OF `scene`, NEVER INTO `scene`. The caller's scene is the
    object the wizard goes on to export to USD -- run_wizard holds one `kitchen` across the preview
    loop and commits that very scene -- so a robot added here would be committed into the kitchen,
    would be measured by fixture_overlaps against every fixture in the room, and would arrive in
    the composer as a prim nobody placed. A copy costs one scene duplication per page render
    (measured at well under the GLB encode this function then does) and makes the containment
    unconditional rather than dependent on a later removal that an exception could skip.
    """
    if scale_reference not in (None, "") and scale_reference not in ROBOTS:
        raise ValueError(f"{scale_reference!r} is not one of {sorted(ROBOTS)}")
    if scale_reference:
        scene = scene.copy()
        scale_nodes, scale_labels = add_scale_reference(
            scene, clearest_floor_spot(scene, robot=scale_reference), robot=scale_reference
        )
    else:
        scale_nodes, scale_labels = [], []

    support_nodes = sorted(
        {n for node_id in supports for n in object_node_names(scene, node_id)}
    )
    payload = {
        "objects": [
            {
                "label": obj["label"],
                "location": obj.get("location", ""),
                "key": obj["node_id"],                              # what we POST back to python
                "root": sanitize_three_name(obj["node_id"]),        # the three.js node name
                "nodes": object_node_names(scene, obj["node_id"]),  # it and its descendants
                # The support this object was placed on, expanded and sanitized like the rest.
                # pickSurface() prefers a hit on it over a nearer hit on some other support
                # (see the drag-disambiguation note there). Absent for callers that build the
                # object list by hand; the JS then falls back to plain nearest-hit.
                "support": object_node_names(scene, obj["support"]) if obj.get("support") else [],
            }
            for obj in objects
        ],
        "materials": dict(materials),
        "supports": support_nodes,
        # Objects that drag on a mathematical ground plane instead of raycasting
        # supportNodeNames — the table and every chair (see pickGround() in the JS below). Keyed
        # by label, like homeSupportNames, because that's what `selected` holds when the drag
        # handler checks membership. Node ids and labels are the same string for the table today
        # (kitchen_build.add_table always returns "table") and DIFFERENT for a chair ("chair_0"
        # vs "chair 0"), which is exactly why this reads obj["label"] rather than the node id.
        "floor_dragged": [
            obj["label"] for obj in objects if _FLOOR_DRAGGED_NODE.fullmatch(obj["node_id"])
        ],
        "joints": [
            dict(j, node=sanitize_three_name(j["child"]))   # the three.js name of the moving part
            for j in (joints or [])
        ],
        "actions": [
            {"id": a["id"], "label": a["label"], "path": a.get("path", "/answer")}
            for a in (actions or [])
        ],
        # The selector: every robot that can be picked, which one is picked now, and what the
        # dimension line says -- so the panel quotes the height the geometry beside it is drawn at
        # rather than restating it. [] on a page rendered with scale_reference=None, which is also
        # how the JS knows not to build a selector at all.
        "robot_choices": ([] if scale_reference is None else
                          [{"id": name, "label": spec["label"], "height_m": spec["height_m"]}
                           for name, spec in ROBOTS.items()]),
        "robot": scale_reference or "",
        # EVERY node of the scale reference -- the robot and its dimension rule. The JS keeps
        # these meshes out of shellMeshes (or Solid/Ghost/Hidden would turn them back on behind the
        # checkbox's back) and hides them until the box is ticked. [] on a page without one, which
        # is also how the JS knows not to build a toggle at all.
        "scale_nodes": scale_nodes,
        # The text beside the rules, which cannot be geometry: a trimesh scene has no text
        # primitive. Drawn in the browser as canvas sprites at these scene-frame positions, by the
        # same mechanism the kitchen gallery already labels its tiles with (see SPRITE_LABEL_JS).
        "scale_labels": scale_labels,
    }

    return scene_page(scene, _PANEL.replace("__THEME__", THEME_CSS)
                                   .replace("__SPRITE_LABEL__", SPRITE_LABEL_JS)
                                   .replace("__PAYLOAD__", script_json(payload)))


def build_preview_html(scene, objects, materials, supports, out_path=None, joints=None) -> Path:
    """Write the preview page to a file and return its path."""
    if out_path is None:
        out_path = Path(tempfile.mkdtemp(prefix="simvla_kitchen_preview_")) / "preview.html"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_page(scene, objects, materials, supports, joints), encoding="utf-8"
    )
    return out_path


def _column_major_to_matrix(values):
    """THREE.Matrix4.toArray() (column-major, 16 floats) -> a row-major 4x4."""
    if len(values) != 16:
        raise ValueError(f"expected a 16-element matrix, got {len(values)}")
    columns = [values[i : i + 4] for i in range(0, 16, 4)]
    return [[float(columns[c][r]) for c in range(4)] for r in range(4)]


class PreviewServer:
    """Serves the preview page on 127.0.0.1 and collects the placements it posts back.

    Runs in a daemon thread so the director's main thread is free to block on wait_answer()'s
    threading.Event (and, before that, on Isaac's boot) while still serving requests, and so the
    process can exit without waiting for serve_forever() to return. Bound to loopback only.
    """

    def __init__(self, page: str, port: int | None = None):
        self._page = page.encode("utf-8")
        self._lock = threading.Lock()
        self._placements = {}

        # A stable port by default, so an SSH tunnel set up once keeps working across
        # re-rolls and runs. Falls back to an ephemeral port if it is already taken
        # (e.g. a second generator, or a previous server still winding down).
        if port is None:
            port = int(os.environ.get("SIMVLA_PREVIEW_PORT", DEFAULT_PREVIEW_PORT))
        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        except OSError:
            print(
                f"[preview] port {port} is busy; using a random one instead. "
                f"Set SIMVLA_PREVIEW_PORT to pick another fixed port."
            )
            self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/"

    @property
    def placements(self):
        """{node_id: row-major 4x4} — whatever the page last posted."""
        with self._lock:
            return dict(self._placements)

    def open(self) -> bool:
        """Open the preview in a browser that can actually render it. True if one was launched.

        Order matters, and the last rule is the important one:

        1. $BROWSER — under VS Code Remote this is the helper that opens the page on the user's
           own machine. Reaching around it to find a "better" browser would defeat it.
        2. The VS Code helper found on disk — same thing, but works from any shell on the box,
           not just VS Code's integrated terminal (which is the only place $BROWSER is set).
        3. A real local desktop — prefer Chrome/Chromium over Firefox.
        4. Otherwise: refuse. On an SSH session the only browser left is an X11-forwarded one
           with no GPU; launching it produces a blank white page and a confused user. Say we
           could not, and let the caller print the URL.
        """
        if os.environ.get("BROWSER"):
            webbrowser.open(self.url)
            return True

        if _open_via_vscode(self.url):
            return True

        if _is_remote_session():
            return False        # never launch the GPU-less X11 browser

        for name in ("chrome", "google-chrome", "chromium", "chromium-browser"):
            try:
                webbrowser.get(name).open(self.url)
                return True
            except webbrowser.Error:
                continue
        webbrowser.open(self.url)
        return True

    def banner(self, opened: bool = False, reason: str = "", title: str = "Kitchen preview") -> str:
        """What the console says when the page comes up.

        `title` names what is being served: the preview, or (from the wizard) the whole run.
        """
        port = self._httpd.server_address[1]
        host = socket.gethostname()
        head = f"\n  {title} ready:  {self.url}\n"
        if opened:
            return head + f"  Opening it in your browser ({reason}).\n"
        return (
            head
            + f"  Not opening a browser for you: {reason}.\n"
            + f"\n  Forward the port from your own machine, then browse to "
            f"http://127.0.0.1:{port}/ :\n"
            + f"      ssh -L {port}:127.0.0.1:{port} {host}\n"
            + f"\n  On a cluster where {host} is a compute node behind a login node, jump "
            f"through it\n  in one command (no nested ssh needed):\n"
            + f"      ssh -J <you>@<login-node> -L {port}:127.0.0.1:{port} <you>@{host}\n"
            + f"\n  Better: connect VS Code to {host} directly (Remote-SSH + ProxyJump) and "
            f"both the\n  browser and the port forward happen by themselves.\n"
        )

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # keep the generator's console readable

            def do_GET(self):
                # The page probes this on load. Serving the page proves the browser can READ
                # from us; only a round-trip like this proves it can WRITE back — and if it
                # cannot, every drag is silently discarded, so the user has to know up front
                # rather than after twenty minutes of arranging a scene.
                if self.path == "/ping":
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                if self.path not in ("/", "/index.html"):
                    self.send_error(404)
                    return
                # Snapshot under the lock: the wizard's publish() reassigns _page repeatedly
                # while the server is live (this used to be write-once at __init__), and these
                # pages can be multi-MB trimesh documents, so wfile.write() below can block long
                # enough for a concurrent publish() to swap _page mid-response. Reading it twice
                # without a snapshot would then send a Content-Length that does not match the
                # bytes actually written.
                with server._lock:
                    page = server._page
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

            def do_POST(self):
                if self.path != "/placements":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length) or b"{}")
                    parsed = {
                        node_id: _column_major_to_matrix(values)
                        for node_id, values in body.items()
                    }
                except (ValueError, AttributeError) as exc:
                    self.send_error(400, str(exc))
                    return

                with server._lock:
                    server._placements = parsed

                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

        return Handler


_PANEL = """
<style>
__THEME__
/* Responsive on purpose, not a bigger fixed number: this panel carries object labels with
   locations, a material name per group, and a joint list, several of which are long enough to
   truncate at a fixed width. clamp() grows it on a wide display without eating the 3D view --
   the preview -- on a laptop, and the 280px floor keeps today's narrowest case unchanged. If
   you are about to put a bare `width:280px` back, that is the truncation bug this line fixed. */
#simvla-panel{position:fixed;top:0;right:0;width:clamp(280px, 26vw, 460px);max-height:100vh;
  overflow-y:auto;background:var(--scrim);color:var(--fg);box-sizing:border-box;padding:16px 18px;
  z-index:9999;font:13px/1.55 var(--ui);border-left:1px solid var(--edge)}
/* Named sections, in mono at micro size: the section names are the only thing on this panel that
   is chrome rather than content, and small caps mono is what separates them from the prose. */
#simvla-panel h2{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);
  margin:20px 0 7px;font-weight:600;font-family:var(--mono)}
#simvla-panel h2:first-child{margin-top:0}
#simvla-panel label{display:block;margin:4px 0;cursor:pointer;font-size:12.5px}
#simvla-panel table{width:100%;border-collapse:collapse}
/* Data, so mono: these are material names baked into the USD, read character by character. */
#simvla-panel td{padding:3px 0;vertical-align:top;font-family:var(--mono);font-size:11.5px}
/* Left unchanged: every group name in kitchen_build.MATERIALS is short ("white plastic" is the
   longest at 13 chars), so nowrap here never forces a wrap wider than a couple of characters --
   it was the long material *values* in the other column running out of room, which the wider
   panel above now gives space to. */
#simvla-panel td.k{color:var(--muted);padding-right:10px;white-space:nowrap}
#simvla-panel .note{color:var(--muted);font-size:11.5px;margin-top:9px;line-height:1.5}
#simvla-panel .warn{color:var(--err);font-size:12px;margin-top:9px;
  border-left:2px solid var(--err);padding-left:9px}
#simvla-panel .loc{color:var(--muted);font-family:var(--mono);font-size:11px}
#simvla-panel .row{display:flex;gap:7px;margin-bottom:7px}
/* The panel's buttons used to be whatever the browser draws by default -- this panel is injected
   into a trimesh page that has no shell CSS of its own, so nothing was styling them and a native
   grey chrome button sat on the ink. */
#simvla-panel button{font:inherit;font-size:12.5px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:6px 12px;cursor:pointer}
#simvla-panel button:hover:enabled{background:var(--edge);color:var(--accent)}
#simvla-panel button:disabled{opacity:.4;cursor:default}
#simvla-panel button:focus-visible,#simvla-panel input:focus-visible,
#simvla-panel select:focus-visible{outline:1px solid var(--accent);outline-offset:1px}
/* The robot selector. Same treatment the edit page gives its two menus, so the one <select> the
   preview has does not arrive as an unstyled browser default on a dark panel. */
#simvla-panel select{font:inherit;font-size:12px;background:var(--raise);color:var(--fg);
  border:1px solid var(--edge-bright);border-radius:0;padding:4px 6px;margin-left:6px}
/* The one primary action a preview offers -- Accept -- is the first button in the actions row. */
#simvla-panel #simvla-actions button:first-child{background:var(--accent);color:var(--accent-fg);
  border-color:var(--accent);font-weight:640}
#simvla-panel #simvla-actions button:first-child:hover:enabled{background:var(--accent-lift);
  color:var(--accent-fg)}
#simvla-panel details{margin:3px 0;border-left:1px solid var(--edge);padding-left:9px}
#simvla-panel summary{cursor:pointer;color:var(--fg-2);font-size:12px;padding:3px 0}
#simvla-panel .j{display:flex;align-items:center;gap:7px;margin:3px 0}
/* Kept even at the new max width: joint names here are `child` node paths with the group
   prefix stripped (see buildJointUI below), which can still run several path segments deep on
   a real kitchen asset. Untested against a real generated scene (this module only imports
   trimesh + stdlib, so there is no built kitchen to measure against here) -- at 460px this
   ellipsis may now only be doing work for the narrowest screens, but removing it risks the
   name colliding with the slider/readout it shares a flex row with, so it stays. */
#simvla-panel .j span{color:var(--muted);font-size:11px;flex:1;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-family:var(--mono)}
#simvla-panel .j input{flex:1;min-width:0}
#simvla-panel .j b{color:var(--fg);font-weight:400;font-size:11px;width:44px;text-align:right;
  font-family:var(--mono);font-variant-numeric:tabular-nums}
#simvla-panel #simvla-actions button{flex:1 1 auto}
#simvla-splash{position:fixed;inset:0;z-index:10000;display:flex;align-items:center;
  justify-content:center;background:var(--bg);color:var(--fg);
  font:14px/1.6 var(--ui);text-align:center}
#simvla-splash .box{max-width:520px;padding:0 24px}
#simvla-splash .t{font-size:17px;font-weight:620;margin-bottom:6px;letter-spacing:-.01em}
#simvla-splash .d{color:var(--fg-2)}
#simvla-splash .e{color:var(--err);text-align:left;margin-top:14px;font-family:var(--mono);
  font-size:12px}
#simvla-splash code{background:var(--panel-2);border:1px solid var(--edge);padding:1px 5px;
  border-radius:0;font-family:var(--mono);font-size:12px}
</style>
<div id="simvla-splash"><div class="box">
  <div class="t" id="simvla-splash-t">Loading the kitchen…</div>
  <div class="d" id="simvla-splash-d">Decoding the scene. This is a single self-contained page,
    so the first paint can take a few seconds.</div>
</div></div>
<div id="simvla-panel">
  <div id="simvla-actions" class="row"></div>
  <div id="simvla-answer" class="note"></div>
  <h2>Kitchen shell</h2>
  <label><input type="radio" name="simvla-shell" value="solid" checked> Solid</label>
  <label><input type="radio" name="simvla-shell" value="ghost"> Ghost (see through)</label>
  <label><input type="radio" name="simvla-shell" value="hidden"> Hidden</label>
  <div class="note">Ghost or Hidden to check objects inside the refrigerator or behind wall cabinets.</div>
  <div id="simvla-figure-row"></div>

  <h2>Placed objects</h2>
  <label><input type="checkbox" id="simvla-hl" checked> Highlight objects</label>
  <div id="simvla-objects"></div>
  <div id="simvla-warn"></div>

  <h2>Selection</h2>
  <div id="simvla-selected" class="note">Click an object to select it, then drag it onto a surface.</div>
  <label>Yaw <input type="range" id="simvla-yaw" min="-180" max="180" value="0" step="1" disabled></label>
  <button id="simvla-reset">Reset placements</button>
  <div id="simvla-status" class="note"></div>

  <h2>Joints</h2>
  <div class="row">
    <button id="simvla-open-all">Open all</button>
    <button id="simvla-close-all">Close all</button>
  </div>
  <div id="simvla-joints"></div>
  <div class="note">Doors, drawers and shelves, driven between their real limits. This is for
    looking only &mdash; joint positions are <b>not</b> saved to the USD, so opening a drawer to
    check inside cannot leak into the generated scene.</div>

  <h2>Materials</h2>
  <table id="simvla-materials"></table>
  <div class="note">Materials are Omniverse MDL and are not rendered here &mdash; this preview shows
    geometry only. These are the names that will be baked into the USD.</div>
</div>
<script>
(function () {
  var DATA = __PAYLOAD__;
  DATA.joints = DATA.joints || [];   // a page rendered without joints must still work

  var nodeToLabel = new Map();
  DATA.objects.forEach(function (o) {
    o.nodes.forEach(function (n) { nodeToLabel.set(n, o.label); });
  });

  var supportNodeNames = new Set(DATA.supports);

  // Labels that drag on the floor (a mathematical plane, see pickGround() below) instead of
  // raycasting supportNodeNames -- currently just the table. A table stands on the floor, and
  // there is no floor in this scene: the floor comes from the env cfg and the walls are added
  // later at commit, so intersecting a plane avoids adding geometry that would then export
  // into the USD. Read from the payload rather than hardcoded, so renaming the node in
  // kitchen_build cannot silently disable the drag.
  var floorDragged = new Set(DATA.floor_dragged || []);
  var groundPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

  // label -> the node names of the support THIS object was placed on (a subset of
  // supportNodeNames). Used by pickSurface() to break the tie when a drag ray crosses
  // several supports. Empty when the caller supplied no "support" field.
  var homeSupportNames = new Map();
  DATA.objects.forEach(function (o) {
    homeSupportNames.set(o.label, new Set(o.support || []));
  });

  var objectMeshes = new Map();  // label -> [mesh]
  var shellMeshes = [];
  // Everything the scale reference draws -- the robot's meshes, the dimension rule's meshes and
  // the label sprite -- kept OUT of shellMeshes and out of objectMeshes: none of it is part of the
  // kitchen and none of it is something the author placed. Empty on a page rendered without one.
  // One list, so the toggle cannot turn half of it on.
  var figureMeshes = [];
  var scaleNodeNames = new Set(DATA.scale_nodes || []);
  // Every mesh that is a valid drop surface. NOT a subset of shellMeshes: the table's top is a
  // support and also belongs to an object, so it is raycast for drops but never shell-toggled.
  var supportMeshes = [];
  var pristine = new Map();      // mesh -> original material state

  function ownerOf(node) {
    for (var cur = node; cur; cur = cur.parent) {
      var label = nodeToLabel.get(cur.name);
      if (label) return label;
    }
    return null;
  }

  function isSupportMesh(node) {
    for (var cur = node; cur; cur = cur.parent) {
      if (supportNodeNames.has(cur.name)) return true;
    }
    return false;
  }

  // Is this mesh part of `names` (a set of scene-graph node names)? Same ancestor walk as
  // isSupportMesh, against an arbitrary set instead of the global support set.
  function isUnder(node, names) {
    if (!names || !names.size) return false;
    for (var cur = node; cur; cur = cur.parent) {
      if (names.has(cur.name)) return true;
    }
    return false;
  }

  // Is this mesh part of the scale reference -- the robot, or its dimension rule? An ancestor walk,
  // like ownerOf/isSupportMesh: each piece arrives as one node with a geometry child under it, so
  // the mesh itself is not the node the payload names.
  //
  // Reads the WHOLE scale_nodes set rather than any one piece's name. A test that knew only about
  // the robot would leave the dimension line permanently visible, selectable and shell-toggled --
  // exactly the leak this separation exists to prevent.
  function isFigureMesh(node) {
    if (!scaleNodeNames.size) return false;
    for (var cur = node; cur; cur = cur.parent) {
      if (scaleNodeNames.has(cur.name)) return true;
    }
    return false;
  }

  function classify() {
    scene.traverse(function (o) {
      if (!o.isMesh) return;
      o.material = o.material.clone();  // don't let a toggle leak across shared materials
      pristine.set(o, {
        color: o.material.color ? o.material.color.clone() : null,
        emissive: o.material.emissive ? o.material.emissive.clone() : null,
        opacity: o.material.opacity,
        transparent: o.material.transparent,
        depthWrite: o.material.depthWrite
      });
      // Before ownerOf/shellMeshes, and before the support test: the scale reference is not
      // kitchen geometry, so the shell toggle must not own it (Solid would set visible = true and
      // undo the checkbox) and a drag must never be able to land on it.
      if (isFigureMesh(o)) {
        // VISIBLE, unlike everything else in this branch used to be: a robot is on this page only
        // because the author picked it out of the selector, so hiding it behind a second control
        // would make picking one do nothing. The branch still exists for containment -- the shell
        // radios and the drag raycaster must never own the reference.
        figureMeshes.push(o);
        return;
      }

      var label = ownerOf(o);
      if (label) {
        if (!objectMeshes.has(label)) objectMeshes.set(label, []);
        objectMeshes.get(label).push(o);
      } else {
        // A support mesh (countertop/island/shelf) is also shell geometry: it must keep
        // getting shown/hidden/ghosted by the Solid/Ghost/Hidden toggle exactly as before.
        // This stays in the else branch: object meshes were never toggled, and moving them
        // into shellMeshes would start ghosting the very objects the user is dragging.
        shellMeshes.push(o);
      }

      // Deliberately OUTSIDE the branch: a support is a support even when it also belongs to
      // an object. The table is exactly that -- it is in DATA.objects so it can be selected
      // and dragged, AND its `tabletop` is in DATA.supports so things can be placed on it.
      // While this sat inside the else, ownerOf() claimed every table mesh first, `tabletop`
      // never reached supportMeshes, and pickSurface() (which raycasts supportMeshes only)
      // could never hit it: dragging the loc-42 mug whose homeSupportNames is {tabletop}
      // silently no-opped, or dropped it onto whatever other support the ray crossed. Before
      // the table became a user-placed object it was a Fixture, absent from DATA.objects, so
      // its top WAS a support mesh -- this keeps that true now that it is both.
      if (isSupportMesh(o)) supportMeshes.push(o);
    });
  }

  function applyShell(mode) {
    shellMeshes.forEach(function (m) {
      var was = pristine.get(m);
      if (mode === 'hidden') { m.visible = false; return; }
      m.visible = true;
      if (mode === 'ghost') {
        m.material.transparent = true;
        m.material.opacity = 0.22;
        m.material.depthWrite = false;
      } else {
        m.material.transparent = was.transparent;
        m.material.opacity = was.opacity;
        m.material.depthWrite = was.depthWrite;
      }
      m.material.needsUpdate = true;
    });
    render();
  }

  function applyHighlight(on) {
    objectMeshes.forEach(function (meshes) {
      meshes.forEach(function (m) {
        var was = pristine.get(m);
        if (on) {
          if (m.material.color) m.material.color.setHex(0xff7043);
          if (m.material.emissive) m.material.emissive.setHex(0x4a1a00);
        } else {
          if (m.material.color && was.color) m.material.color.copy(was.color);
          if (m.material.emissive && was.emissive) m.material.emissive.copy(was.emissive);
        }
        m.material.needsUpdate = true;
      });
    });
    render();
  }

  function buildUI() {
    buildActions();
    var list = document.getElementById('simvla-objects');
    if (!DATA.objects.length) {
      list.innerHTML = '<div class="note">No objects placed.</div>';
    }
    DATA.objects.forEach(function (o) {
      var row = document.createElement('label');
      var box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = true;
      box.addEventListener('change', function () {
        (objectMeshes.get(o.label) || []).forEach(function (m) { m.visible = box.checked; });
        render();
      });
      row.appendChild(box);
      row.appendChild(document.createTextNode(' ' + o.label + ' '));
      var loc = document.createElement('span');
      loc.className = 'loc';
      loc.textContent = '(' + o.location + ')';
      row.appendChild(loc);
      list.appendChild(row);
    });

    var table = document.getElementById('simvla-materials');
    Object.keys(DATA.materials).sort().forEach(function (group) {
      var tr = table.insertRow();
      var k = tr.insertCell(); k.className = 'k'; k.textContent = group;
      tr.insertCell().textContent = DATA.materials[group];
    });

    var missing = DATA.objects.filter(function (o) { return !objectMeshes.has(o.label); });
    if (missing.length) {
      warnings.push(
        'Could not locate ' + missing.map(function (o) { return o.label; }).join(', ') +
        ' in the 3D scene. Highlight and per-object visibility will not work for them.'
      );
    }
    renderWarnings();

    document.querySelectorAll('input[name="simvla-shell"]').forEach(function (radio) {
      radio.addEventListener('change', function () { if (radio.checked) applyShell(radio.value); });
    });
    document.getElementById('simvla-hl').addEventListener('change', function () {
      applyHighlight(this.checked);
    });
    buildRobotSelector();
  }

__SPRITE_LABEL__

  // The dimension labels, in the 3D view. Sprites can only be made once THREE exists and the GLB
  // has loaded, so they are created here rather than shipped as geometry -- and they are pushed
  // into figureMeshes so the reference's text is contained exactly like its geometry.
  //
  // Parented to the gltf 'world' node, which is what carries trimesh's scene frame, so `at` is in
  // the same metres as everything else the page draws. Falling back to `scene` is safe: trimesh
  // writes that node with no matrix of its own (verified against the exported glTF), so the two
  // frames coincide today -- the parenting is what keeps that true if it ever stops being.
  function buildScaleLabels() {
    var root = scene.getObjectByName('world') || scene;
    (DATA.scale_labels || []).forEach(function (item) {
      var sprite = simvlaLabelSprite(item.text, item.width);
      sprite.position.set(item.at[0], item.at[1], item.at[2]);
      root.add(sprite);
      figureMeshes.push(sprite);
    });
  }

  // The robot selector, and the choice is SERVER-SIDE: picking one re-renders this page carrying
  // that robot's geometry. Each robot is 5-13 MB of GLB at source resolution, so a client-side
  // switch would have to embed all three and charge every author 25 MB for a control most of them
  // never touch. It POSTs `robot:<id>` through the same sendDecision() the button row uses, so a
  // pick flushes any pending drags first and cannot land on a scene the generator never saw.
  //
  // Built in JS rather than written into the markup so that a page rendered WITHOUT a reference has
  // no dead control on it -- the same reason the Cancel button on a choice page is only created
  // when the step published a cancel id.
  function buildRobotSelector() {
    buildScaleLabels();
    var choices = DATA.robot_choices || [];
    if (!choices.length) return;
    var row = document.getElementById('simvla-figure-row');
    var label = document.createElement('label');
    label.appendChild(document.createTextNode('Scale reference '));
    var select = document.createElement('select');
    select.id = 'simvla-robot';
    [{id: '', label: 'No robot'}].concat(choices).forEach(function (c) {
      var option = document.createElement('option');
      option.value = c.id;
      option.textContent = c.height_m ? c.label + ' · ' + c.height_m.toFixed(2) + ' m' : c.label;
      select.appendChild(option);
    });
    select.value = DATA.robot || '';
    select.addEventListener('change', function () {
      select.disabled = true;
      // sendDecision() resolves on its own error path too, so this cannot leave the control stuck.
      sendDecision({id: 'robot:' + select.value}).then(function () { select.disabled = false; });
    });
    label.appendChild(select);
    row.appendChild(label);
    var note = document.createElement('div');
    note.className = 'note';
    note.textContent = 'Stands the chosen robot on the clearest patch of floor, with a dimension '
      + 'line at its measured standing height. Drawn here only: never placed, never checked for '
      + 'clashes, never exported. Picking one re-renders this page, because only the robot you '
      + 'choose is sent with it.';
    row.appendChild(note);
  }

  var raycaster = new THREE.Raycaster();
  var pointer = new THREE.Vector2();
  var roots = new Map();      // label -> Object3D
  var initial = new Map();    // label -> Matrix4 clone
  var yaws = new Map();       // label -> degrees
  var selected = null;
  var dragging = false;
  var warnings = [];          // strings shown in #simvla-warn, collected across setup steps
  var placementEnabled = true;

  function renderWarnings() {
    document.getElementById('simvla-warn').innerHTML = warnings.map(function (w) {
      return '<div class="warn">' + w + '</div>';
    }).join('');
  }

  // The gltf root node carries trimesh's 'world' frame; report matrices relative to it.
  function worldFrame() { return scene.getObjectByName('world'); }

  function matrixInSceneFrame(obj) {
    var m = new THREE.Matrix4();
    var root = worldFrame();
    if (root) m.copy(root.matrixWorld).invert();
    return m.multiply(obj.matrixWorld);
  }

  // Object3D.position/.matrix are LOCAL to the object's parent, not world space. A placed
  // object's real parent is whatever furniture node it landed on (e.g. a countertop that
  // itself carries a rotation+translation), so root.position is not the world-frame
  // coordinate the drag/yaw math below needs. Reparent every draggable root directly under
  // the gltf 'world' node with attach(), which preserves world transform while rewriting
  // the local one — after this, world's own transform is identity (it is trimesh's base
  // frame), so root.position/.matrix genuinely are world-space and dropOnto()/setYaw() are
  // valid. THREE.Object3D.rotateOnWorldAxis explicitly assumes a parent with no rotation;
  // without this reparenting that assumption is violated for anything sitting on a rotated
  // countertop/island/shelf.
  function findRoots() {
    var world = worldFrame();
    if (!world) {
      placementEnabled = false;
      warnings.push(
        "Could not find the scene's 'world' node — drag positioning is unavailable. " +
        'Objects can still be viewed, hidden and highlighted.'
      );
    }
    DATA.objects.forEach(function (o) {
      var root = scene.getObjectByName(o.root);
      if (!root) return;
      if (world) {
        world.attach(root);           // re-parent, preserving root's current world transform
        root.updateMatrixWorld(true);
      }
      roots.set(o.label, root);
      // Snapshot AFTER the attach: the local matrix means "relative to world" only now.
      initial.set(o.label, root.matrix.clone());
      yaws.set(o.label, 0);
    });
  }

  // ---- joints: drive a door/drawer between its real limits ----------------------------
  // At q=0 the moving part's local matrix IS the joint origin, so the part is driven by
  //     child.matrix = origin * T(q)
  // with T a rotation about `axis` (revolute) or a translation along it (prismatic).
  // The parts are real scene nodes, so opening a door updates matrixWorld for everything
  // under it — which is what makes it safe to drag an object while the fridge is open.
  // Nothing here is ever posted back: joint state is inspection-only, by design.
  var jointNodes = new Map();   // joint.name -> {node, origin}

  function initJoints() {
    DATA.joints.forEach(function (j) {
      var node = scene.getObjectByName(j.node);
      if (!node) return;
      var origin = new THREE.Matrix4();
      origin.set.apply(origin, [].concat.apply([], j.origin));   // row-major, as Matrix4.set wants
      jointNodes.set(j.name, { node: node, origin: origin });
    });
  }

  function setJoint(j, q) {
    var entry = jointNodes.get(j.name);
    if (!entry) return;
    var axis = new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize();
    var motion = new THREE.Matrix4();
    if (j.type === 'revolute') {
      motion.makeRotationAxis(axis, q);
    } else {
      motion.makeTranslation(axis.x * q, axis.y * q, axis.z * q);
    }
    var m = entry.origin.clone().multiply(motion);
    // matrixAutoUpdate rebuilds .matrix from position/quaternion/scale every frame, so the
    // matrix has to be decomposed back into them or the next frame would undo this.
    m.decompose(entry.node.position, entry.node.quaternion, entry.node.scale);
    entry.node.updateMatrixWorld(true);
  }

  function jointLabel(j, q) {
    return j.type === 'revolute'
      ? (q * 180 / Math.PI).toFixed(0) + '\\u00B0'
      : (q * 100).toFixed(1) + ' cm';
  }

  function buildJointUI() {
    var host = document.getElementById('simvla-joints');
    if (!DATA.joints.length) {
      host.innerHTML = '<div class="note">This kitchen has no movable joints.</div>';
      document.getElementById('simvla-open-all').disabled = true;
      document.getElementById('simvla-close-all').disabled = true;
      return;
    }

    var groups = new Map();
    DATA.joints.forEach(function (j) {
      if (!groups.has(j.group)) groups.set(j.group, []);
      groups.get(j.group).push(j);
    });

    var sliders = [];
    Array.from(groups.keys()).sort().forEach(function (group) {
      var items = groups.get(group);
      var det = document.createElement('details');
      var sum = document.createElement('summary');
      sum.textContent = group + ' (' + items.length + ')';
      det.appendChild(sum);

      items.forEach(function (j) {
        var row = document.createElement('div');
        row.className = 'j';

        var name = document.createElement('span');
        name.textContent = j.child.indexOf('/') >= 0 ? j.child.split('/').slice(1).join('/') : j.child;
        name.title = j.name;

        var slider = document.createElement('input');
        slider.type = 'range';
        slider.min = j.lower; slider.max = j.upper;
        slider.step = (j.upper - j.lower) / 100;
        slider.value = j.lower;

        var readout = document.createElement('b');
        readout.textContent = jointLabel(j, j.lower);

        slider.addEventListener('input', function () {
          var q = Number(this.value);
          setJoint(j, q);
          readout.textContent = jointLabel(j, q);
          render();
        });

        row.appendChild(name); row.appendChild(slider); row.appendChild(readout);
        det.appendChild(row);
        sliders.push({ j: j, slider: slider, readout: readout });
      });
      host.appendChild(det);
    });

    function setAll(pick) {
      sliders.forEach(function (s) {
        var q = pick(s.j);
        s.slider.value = q;
        s.readout.textContent = jointLabel(s.j, q);
        setJoint(s.j, q);
      });
      render();
    }
    document.getElementById('simvla-open-all')
      .addEventListener('click', function () { setAll(function (j) { return j.upper; }); });
    document.getElementById('simvla-close-all')
      .addEventListener('click', function () { setAll(function (j) { return j.lower; }); });
  }

  function setPointer(event) {
    pointer.x = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
  }

  function pickObject(event) {
    setPointer(event);
    var meshes = [];
    objectMeshes.forEach(function (list) {
      list.forEach(function (m) { if (m.visible) meshes.push(m); });
    });
    var hits = raycaster.intersectObjects(meshes, false);
    return hits.length ? ownerOf(hits[0].object) : null;
  }

  function pickSurface(event, label) {
    setPointer(event);
    // Only supportMeshes (labeled countertops/island top/fridge shelves) are raycast
    // here, not all shellMeshes and not objectMeshes: raycasting the whole shell would
    // let the ray land on whatever is nearest along it — e.g. a closed fridge door in
    // front of the shelf it hides — dropping the object outside the compartment it's
    // meant to go in. Restricting to supportMeshes guarantees a hit is always a real
    // placement surface. Not filtered by .visible: three.js's Raycaster does not skip
    // invisible objects, and requiring visibility here would make dragging a silent
    // no-op in Hidden mode, which is exactly the mode used to see inside the fridge.
    var hits = raycaster.intersectObjects(supportMeshes, false);
    if (!hits.length) return null;

    // Restricting to supports is necessary but not sufficient: supports occlude each other
    // too. The default camera looks down from well above the scene, so a ray aimed into the
    // fridge pierces, in order, 'refrigerator/top' (an exterior support, loc 8) and then
    // every shelf below it. Plain nearest-hit therefore drags an object off its own shelf and
    // onto the roof of the fridge — the same class of bug the support restriction fixed, one
    // level in.
    //
    // So: if the ray crosses the support this object is actually resting on, that hit wins,
    // however many nearer supports are stacked in front of it. Dragging within a surface then
    // stays on that surface, which is also what apply_placements() assumes (it keeps the
    // object on its existing support edge and only rewrites the transform). Moving an object
    // to a DIFFERENT support still works: as soon as the cursor leaves the current support's
    // silhouette the preference cannot fire and nearest-hit takes over again.
    //
    // hits are distance-sorted by THREE.Raycaster, so the first match is also the nearest
    // point on the home support.
    var home = homeSupportNames.get(label);
    if (home && home.size) {
      for (var i = 0; i < hits.length; i++) {
        if (isUnder(hits[i].object, home)) return hits[i].point;
      }
    }
    return hits[0].point;
  }

  // The table's drag surface: no geometry to raycast (see floorDragged's comment above), so
  // intersect the ray against a mathematical z=0 plane instead. Reuses setPointer()'s NDC
  // conversion rather than converting the pointer event a second time.
  function pickGround(event) {
    setPointer(event);
    var point = new THREE.Vector3();
    return raycaster.ray.intersectPlane(groundPlane, point) ? point : null;
  }

  // Drop the object so its bottom-centre sits on `point`.
  function dropOnto(root, point) {
    var box = new THREE.Box3().setFromObject(root);
    var centre = box.getCenter(new THREE.Vector3());
    var bottom = new THREE.Vector3(centre.x, centre.y, box.min.z);  // Z-up
    root.position.add(point.clone().sub(bottom));
    root.updateMatrixWorld(true);
  }

  function setYaw(label, degrees) {
    var root = roots.get(label);
    if (!root) return;
    var delta = THREE.MathUtils.degToRad(degrees - (yaws.get(label) || 0));
    var axis = new THREE.Vector3(0, 0, 1);                          // Z-up

    // Rotate about the object's own origin — no pivot translation. Recomputing an AABB
    // centre on every input event (the old approach) makes the pivot wander for any
    // asset that isn't centrally symmetric (e.g. a mug with a handle), since the AABB
    // centre shifts as the shape turns; that showed up as several millimeters of drift
    // per slider nudge, enough to walk an object off a shelf edge over repeated
    // adjustments. The BODex assets are already origin-aligned by asset_generator's
    // origin=("com","bottom","com")/("com","com","bottom"), so the object's own origin
    // is the correct, path-independent spin axis.
    root.rotateOnWorldAxis(axis, delta);
    root.updateMatrixWorld(true);
    yaws.set(label, degrees);
    render();
  }

  function setStatus(text, bad) {
    var el = document.getElementById('simvla-status');
    el.textContent = text;
    el.className = bad ? 'warn' : 'note';
  }

  // Serving the page only proves the browser can READ from the generator. Placements travel
  // the other way, and if that direction is broken every drag is silently thrown away. So
  // check it on load, not after the user has spent twenty minutes arranging a scene.
  function checkConnection() {
    fetch("/ping", { method: "GET", cache: "no-store" })
      .then(function (r) {
        if (r.ok || r.status === 204) {
          setStatus('Connected — drags will be saved.', false);
        } else {
          setStatus('Generator answered ' + r.status + ' — drags may not be saved.', true);
        }
      })
      .catch(function (e) {
        setStatus('NOT connected to the generator (' + (e && e.message ? e.message : 'network error')
          + ') — drags will NOT be saved. Reload the page from the URL the generator printed.', true);
      });
  }

  // Reports success/failure through setStatus exactly as before -- the fire-and-forget call sites
  // (resetPlacements, endDrag, the yaw slider's change handler) rely on that line alone and don't
  // need anything more. But a failure now also REJECTS the promise this returns, instead of always
  // fulfilling: the non-2xx branch throws (so it is not ALSO caught by the network-error handler --
  // that one is the second argument to .then below, not a trailing .catch, precisely so it only
  // fires when fetch() itself rejects and never re-catches this throw with the wrong reason), and
  // the network-error branch rethrows after reporting. Without this, sendDecision() below could not
  // tell a successful flush from a failed one, and a click could accept a scene the generator was
  // never told about -- exactly the bug this exists to prevent.
  function postPlacements() {
    var payload = {};
    DATA.objects.forEach(function (o) {
      var root = roots.get(o.label);
      if (root) payload[o.key] = matrixInSceneFrame(root).toArray();
    });
    return fetch("/placements", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(function (r) {
      if (r.ok || r.status === 204) {
        setStatus('Placements saved.', false);
        return;
      }
      setStatus('Generator rejected the placements (' + r.status + ').', true);
      throw new Error('placements rejected (' + r.status + ')');
    }, function (e) {
      setStatus('NOT saved — lost the generator (' + (e && e.message ? e.message : 'network error')
        + '). Do not click Accept; reload the page first.', true);
      throw e;
    });
  }

  // The wizard's button row. Each click's flush -> decision -> re-enable flow lives in
  // sendDecision(), so its shape can be reasoned about (and tested, with no browser here) as one
  // unit rather than as an inline handler.
  function buildActions() {
    var row = document.getElementById('simvla-actions');
    if (!DATA.actions || !DATA.actions.length) { row.style.display = 'none'; return; }

    DATA.actions.forEach(function (a, i) {
      var b = document.createElement('button');
      b.textContent = a.label;
      if (i === 0) { b.style.fontWeight = '600'; }
      b.addEventListener('click', function () { sendDecision(a); });
      row.appendChild(b);
    });
  }

  // One action's click flow: flush placements, and only send the decision if that flush actually
  // reached the generator. postPlacements() now rejects (see above) on a rejected or unreachable
  // flush, which skips straight to the .catch below instead of sending the decision -- a click
  // landing right after a drag must not accept a scene the generator was never told about.
  // Invariant: this row must never end a click disabled without saying why -- both the success
  // continuation and the terminal .catch re-enable it.
  function sendDecision(a) {
    var row = document.getElementById('simvla-actions');
    var out = document.getElementById('simvla-answer');
    row.querySelectorAll('button').forEach(function (x) { x.disabled = true; });
    out.textContent = '';
    return postPlacements().then(function () {
      // Reached only on a successful flush -- a rejected postPlacements() jumps straight to the
      // .catch below, so the decision is never sent for a scene the generator never saw.
      return window.simvlaPost(a.path || '/answer', {id: a.id});
    }).then(function (r) {
      if (r.body && r.body.message) { out.textContent = r.body.message; }
      if (r.status !== 200) {
        out.textContent = (r.body && r.body.reason) || ('rejected (' + r.status + ')');
      }
      if (r.body && r.body.confirm) { askConfirm(r.body.confirm); }
      // A step that ends here is replaced by the poller; re-enable so a page that stays
      // (Save Task) is usable again.
      row.querySelectorAll('button').forEach(function (x) { x.disabled = false; });
    }).catch(function (e) {
      // Either the flush failed (postPlacements() already explained why on the status line above,
      // and its own message says not to click Accept) or the decision POST itself dropped mid-
      // flight (the wizard's POST helper does not catch a raw fetch() rejection, only a bad JSON
      // body). Either way the decision was NOT sent -- say so here too, and always re-enable: a
      // click must never leave the row stuck disabled with no way to tell why.
      out.textContent = 'Could not send the decision (' + (e && e.message ? e.message : 'error')
        + '). Check the status line above; fix the connection, then try again.';
      row.querySelectorAll('button').forEach(function (x) { x.disabled = false; });
    });
  }

  // A follow-up yes/no the server asked for, e.g. "also generate goals now?". Generic on purpose:
  // this panel knows nothing about goals.
  function askConfirm(confirm) {
    var out = document.getElementById('simvla-answer');
    var wrap = document.createElement('div');
    wrap.className = 'row';
    var text = document.createElement('div');
    text.textContent = confirm.text;
    var yes = document.createElement('button');
    yes.textContent = 'Yes';
    var no = document.createElement('button');
    no.textContent = 'No';
    yes.addEventListener('click', function () {
      wrap.remove();
      window.simvlaPost(confirm.path, confirm.body || {}).then(function (r) {
        out.textContent = (r.body && (r.body.message || r.body.reason)) || ('done (' + r.status + ')');
      });
    });
    no.addEventListener('click', function () { wrap.remove(); });
    wrap.appendChild(text);
    wrap.appendChild(yes);
    wrap.appendChild(no);
    out.appendChild(wrap);
  }

  function select(label) {
    selected = label;
    var slider = document.getElementById('simvla-yaw');
    var caption = document.getElementById('simvla-selected');
    if (!label) {
      slider.disabled = true;
      caption.textContent = 'Click an object to select it, then drag it onto a surface.';
    } else {
      slider.disabled = false;
      slider.value = yaws.get(label) || 0;
      caption.textContent = 'Selected: ' + label + ' — drag to move, slider to spin.';
    }
  }

  function resetPlacements() {
    DATA.objects.forEach(function (o) {
      var root = roots.get(o.label);
      var m = initial.get(o.label);
      if (!root || !m) return;
      root.matrix.copy(m);
      root.matrix.decompose(root.position, root.quaternion, root.scale);
      root.updateMatrixWorld(true);
      yaws.set(o.label, 0);
    });
    select(selected);
    render();
    // Fire-and-forget: postPlacements() already reports success/failure via setStatus. Catch here
    // only to avoid an unhandled promise rejection now that a failure rejects the promise too --
    // there is nothing more for this call site to do with it.
    postPlacements().catch(function () {});
  }

  function bindDrag() {
    var canvas = renderer.domElement;

    canvas.addEventListener('pointerdown', function (event) {
      if (!placementEnabled) return;
      var label = pickObject(event);
      if (!label) return;
      select(label);
      dragging = true;
      controls.enabled = false;   // don't orbit while dragging
      // The control panel is a fixed-position overlay above the canvas; without capture,
      // releasing the pointer button over the panel never fires the canvas's 'pointerup'
      // and the drag would stay stuck 'on' forever. Capture routes move/up here regardless
      // of what element is under the cursor.
      canvas.setPointerCapture(event.pointerId);
      render();
    });

    canvas.addEventListener('pointermove', function (event) {
      if (!dragging || !selected) return;
      var point = floorDragged.has(selected) ? pickGround(event) : pickSurface(event, selected);
      if (!point) return;         // cursor is not over any surface: leave it where it is
      dropOnto(roots.get(selected), point);
      render();
    });

    function endDrag(event) {
      if (!dragging) return;
      dragging = false;
      controls.enabled = true;
      if (event && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      // Fire-and-forget: postPlacements() already reports success/failure via setStatus. Catch
      // here only to avoid an unhandled promise rejection now that a failure rejects the promise
      // too -- there is nothing more for this call site to do with it.
      postPlacements().catch(function () {});
    }

    canvas.addEventListener('pointerup', endDrag);
    // pointercancel fires if the browser/OS interrupts the gesture (e.g. a touch turns
    // into a scroll, or the window loses focus mid-drag); it must run the same teardown
    // or 'dragging' sticks true with controls.enabled stuck false.
    canvas.addEventListener('pointercancel', endDrag);

    document.getElementById('simvla-yaw').addEventListener('input', function () {
      if (!placementEnabled) return;
      if (selected) setYaw(selected, Number(this.value));
    });
    document.getElementById('simvla-yaw').addEventListener('change', function () {
      // Fire-and-forget, same as endDrag/resetPlacements above: postPlacements() already reports
      // via setStatus; catch here only to avoid an unhandled promise rejection now that a failure
      // rejects the promise too.
      postPlacements().catch(function () {});
    });
    document.getElementById('simvla-reset').addEventListener('click', resetPlacements);
  }

  // A blank page must never just sit there looking blank: say what went wrong.
  function splashFail(title, detail) {
    var s = document.getElementById('simvla-splash');
    if (!s) return;
    document.getElementById('simvla-splash-t').textContent = title;
    document.getElementById('simvla-splash-d').innerHTML = detail;
  }

  function webglAvailable() {
    try {
      var c = document.createElement('canvas');
      return !!(window.WebGLRenderingContext &&
                (c.getContext('webgl') || c.getContext('experimental-webgl')));
    } catch (e) { return false; }
  }

  if (!webglAvailable()) {
    // The usual cause: the page was opened in a browser forwarded over X11, which has no
    // GPU. Nothing will ever render there — the fix is to open the URL locally, not to wait.
    splashFail('This browser cannot render 3D (no WebGL).',
      'The preview needs WebGL. If you opened this in a browser forwarded over SSH/X11, ' +
      'it has no GPU and will never render.<br><br>Open the URL in a browser on your <b>own</b> ' +
      'machine instead. Forward the port first:<br><code>ssh -L 8777:127.0.0.1:8777 &lt;host&gt;</code>' +
      '<br>then open <code>http://127.0.0.1:8777/</code> locally.');
    return;
  }

  // The template loads the GLB asynchronously, so wait for the scene graph to exist.
  var tries = 0;
  var timer = setInterval(function () {
    if (++tries > 400) {   // ~20s
      clearInterval(timer);
      splashFail('The kitchen did not load.',
        'The 3D scene never finished decoding after 20 seconds. Check the browser console ' +
        'for an error, and try reloading the page.');
      return;
    }
    if (typeof scene === 'undefined' || !scene) return;
    var meshes = 0;
    scene.traverse(function (o) { if (o.isMesh) meshes++; });
    if (!meshes) return;

    clearInterval(timer);
    var splash = document.getElementById('simvla-splash');
    if (splash) splash.remove();      // loaded: get out of the way
    classify();
    if (!supportMeshes.length) {
      // Dragging with nothing to raycast against would otherwise be a silent no-op:
      // pointermove's pickSurface() always returns null, so the object just never moves
      // with no feedback at all. Disable placement outright and say why, instead.
      placementEnabled = false;
      warnings.push(
        'No labeled support surfaces were found in this scene — drag positioning is ' +
        'unavailable. Objects can still be viewed, hidden and highlighted.'
      );
    }
    findRoots();
    initJoints();
    buildUI();
    buildJointUI();
    bindDrag();
    select(null);
    applyHighlight(true);
    checkConnection();
    render();
  }, 50);
})();
</script>
"""
