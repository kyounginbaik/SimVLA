"""Build the cached robot the kitchen preview stands beside a counter. One robot per run.

    conda run --no-capture-output -n env_isaaclab python scripts/tools/build_robot_reference.py rby1

WHAT THIS PRODUCES: one GLB on /lustre per robot -- the robot's own visible geometry, in its own
livery, standing on z = 0 and measuring exactly its asset's own height -- plus a sidecar JSON
recording where it came from and what was measured on it. scripts/simvla/kitchen_preview.py loads
that file and draws it; it never converts anything itself (it imports trimesh + stdlib only, and
pxr is not importable in the environment the preview runs in). kitchen_preview owns the registry,
the paths and the heights; this script imports them rather than restating them, so the two cannot
drift.

WHY IT IS CACHED AND NOT COMMITTED: the same rule the chair and table libraries follow -- assets
live under /lustre (see chair_manifest.DEFAULT_CHAIR_OBJ_DIR) and are deliberately out of git.
A checkout without the cache still renders every page; see kitchen_preview.robot_mesh.

THE PIPELINE:

  1. OPEN the very USD the project's own robot config spawns (see ROBOTS, and kitchen_preview.ROBOTS
     for which line of which config chose each). All three are LOCAL files; two of them are 1.6 KB
     stage roots that reference their geometry out of a sibling configuration/*_base.usd, so the
     stage is opened and the references resolved rather than the file being parsed for meshes.
  2. CONVERT every Mesh prim under the stage's default prim whose computed purpose is default or
     render AND whose computed visibility is not `invisible`, with its local-to-world transform
     baked in, AND the diffuse colour of the material each one binds. Both filters are load-bearing
     and both are MEASURED, not assumed: the collision hulls are `guide` (Anubis 404,350 faces of
     them, AI Worker 646,996), and RB-Y1's guide geometry is 5 cm TALLER than its visible robot.
  3. WELD each prim's own vertices. The assets arrive fully unwelded -- three vertices per triangle,
     no sharing -- which triples the GLB for nothing. Per prim, never across prims: a weld across
     two links would average their two colours into one seam vertex.
  4. NO REMESH AND NO DECIMATION. See NO_REDUCTION.
  5. POSE ANUBIS as its own IsaacLab config spawns it, by forward kinematics over the USD's own
     UsdPhysics joints. Anubis alone -- see ROBOT_SOURCES["anubis"]["spawn_pose"] and spawn_pose_of.
  6. CHECK THE FACING against a per-robot witness measured off the prim names (see FacingWitness),
     then turn the robot to face +x -- which all three already do -- centre it in plan, stand it on
     z = 0 and scale it so its z extent is exactly the height measured in step 2.

It re-execs itself with Isaac's vendored pxr on PYTHONPATH when pxr is not already importable,
which is the normal case -- no Isaac boot, no GPU, no Omniverse app. It refuses to overwrite an
existing cache; delete the GLB to rebuild.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simvla"))
from kitchen_preview import ROBOTS, ROBOT_HEIGHTS_M, robot_reference_path    # noqa: E402

#: WHERE EACH ROBOT'S FRONT IS, pinned by a measurement rather than by eye, because a robot drawn
#: back-to-front is a robot nobody recognises and the page has no second viewpoint to correct it.
#:
#: Each entry says: the prims whose path contains `part` have their mean position on the `sign` side
#: of the robot's own plan centre along `axis`, by at least `margin_m`. That fact is what makes
#: "this robot faces +x" a measurement. The evidence differs per robot because the ASSETS differ:
#:
#:   AI Worker  arm_l_* mean y = +0.226, arm_r_* = -0.228. Its left arm is on +y, so with z up the
#:              front is +x -- ROS REP-103, which every URDF-derived asset here follows.
#:   RB-Y1      link_left_arm_* mean y = +0.221, link_right_arm_* = -0.222. Same argument.
#:   Anubis     has no left/right ARM in any prim name -- its arms are arm1 (on -y) and arm2 (on
#:              +y), and which is which is exactly the question. So the witness is the BASE instead:
#:              the mast carrying both arms is mounted at x = 0.9149 while the base drum's plan
#:              centre is x = 1.0124, and sliced by height the drum runs 0.055 m past the mast on
#:              the -x side and 0.250 m past it on the +x side. That is a mobile manipulator's
#:              mast-at-the-back layout, so the front is +x.
#:
#:              CORROBORATED BY A SECOND, INDEPENDENT SIGNAL, because a mast-position argument on
#:              its own is a heuristic: Anubis's GRIPPERS are named per finger, and both of them put
#:              fingerR at lower y than fingerL (gripper1 -2.0179 against -1.9821, gripper2 -1.4634
#:              against -1.4276). Right on -y and left on +y is the same REP-103 statement the other
#:              two robots make in their link names, reached from the other end of the robot. The
#:              mast is what is asserted below because it is the robust one -- it cannot move with a
#:              joint angle, while a rotated wrist would swing the fingers onto another axis.
#:
#: If a future asset revision moves any of this, the build stops rather than caching a robot facing
#: the wrong way.
FacingWitness = tuple  # (prim path substring, axis index, sign, minimum margin in metres)

#: The robots, and the exact asset each one is built from. Every path was chosen by reading what the
#: project's own robot configs spawn, never by name -- Anubis has seven USDs in this tree and RB-Y1
#: has two model trees each holding an rby1a and an rby1m.
ROBOT_SOURCES = {
    # isaaclab_assets/robots/anubis_wheels.py:16
    "anubis": {
        "usd": "source/isaaclab_assets/data/Robots/anubis_simvla.usd",
        "witness": ("arm1_base_link", 0, -1, 0.05),
        # ANUBIS ALONE IS POSED, on the user's instruction, and the other two keep the pose their
        # USD is authored in. `spawn_pose` is (config source, symbol): its absence is what leaves a
        # robot unposed, so adding a second one is a decision somebody has to write down here.
        "spawn_pose": ("source/isaaclab_assets/isaaclab_assets/robots/anubis_wheels.py",
                       "ANUBIS_CFG"),
    },
    # isaaclab_assets/robots/aiworker_BG2.py:12
    "aiworker": {
        "usd": "source/isaaclab_assets/data/Robots/MM/aiworker/ffw_bg2.usd",
        "witness": ("arm_l_link", 1, +1, 0.15),
    },
    # isaaclab_assets/robots/rby1.py:13 spawns $SIMVLA_RBY1M_DIR/models/rby1m/... -- this draws the
    # rby1A on the user's instruction. ASSUMPTION, stated because it is not derivable from the code:
    # the non-`_new` tree is the current one, because RBY1_CFG's own path pattern points there. The
    # two trees' model.usd files are byte-identical; their mesh payloads differ by 1 MB in 86 MB.
    "rby1": {
        "usd": os.path.join(
            os.environ.get("SIMVLA_RBY1M_DIR", "SIMVLA_RBY1M_DIR"),
            "models/rby1a/urdf/model/model.usd",
        ),
        "witness": ("link_left_arm", 1, +1, 0.15),
    },
}

#: WHY NOTHING IS REDUCED, said plainly because the previous version of this file reduced hard and
#: the result was rejected twice for looking like an approximate mesh rather than like the robot.
#:
#: The reference is opt-in: kitchen_preview embeds a robot only once the author picks one from the
#: page's selector, so an author laying out a kitchen pays none of this. That makes fidelity the
#: thing to spend on and page bytes the thing to report.
#:
#: MEASURED, per-link quadric decimation at the page's own camera (azim 180, elev -32), against the
#: full-resolution robot:
#:
#:      x1.0    the robot.
#:      x0.5    indistinguishable on all three, for half the GLB.
#:      x0.25   AI Worker's head shell and RB-Y1's torso start to facet.
#:      x0.12   plainly faceted on all three: polygonal heads, polygonal base cylinders.
#:
#: So x0.5 was available and is not taken. Nothing is reduced at all, because "indistinguishable at
#: one camera and one resolution" is a claim about the render this was judged with, and the page can
#: be orbited and zoomed. The welding in step 3 is a pure win (it changes no vertex position) and it
#: is where the real saving was: Anubis 25.2 MB unwelded -> 8.4 MB welded.
#:
#: NOT A VOXEL REMESH, which is what this file used to do and is the specific reason it was rejected:
#: voxelising at 4 mm and marching the distance transform melts flat panels and hard edges into
#: organic surfaces, and left a ~2 cm banding across broad panels that no smoothing setting removed.
#: Its justification was that quadric decimation shattered the MERGED mesh -- true, and irrelevant
#: once each link is treated as the closed mesh it already is.
NO_REDUCTION = True

#: Isaac's vendored pxr, the same glob scripts/simvla/test_chair_placement.py resolves.
_USD_LIBS_GLOB = "extscache/omni.usd.libs-*"
_REEXEC_FLAG = "SIMVLA_ROBOT_REFERENCE_REEXEC"


def reexec_with_pxr() -> None:
    """Re-run this script with Isaac's USD libraries on the path, unless pxr already imports.

    The conversion needs pxr and nothing else out of Isaac -- no app, no renderer, no GPU. So
    CUDA_VISIBLE_DEVICES is left ALONE here, unlike the export probe in test_chair_placement.py:
    that one boots enough of Omniverse to want the variable gone, this one opens a stage and reads
    points. Inside a SLURM allocation the variable is how the card was handed over, and clearing it
    is not this script's business.
    """
    if importlib.util.find_spec("pxr") is not None:
        return
    if os.environ.get(_REEXEC_FLAG):
        raise SystemExit(
            "pxr is still not importable after re-exec -- Isaac's omni.usd.libs is on PYTHONPATH "
            "but the extension module would not load."
        )
    spec = importlib.util.find_spec("isaacsim")
    libs = next(
        (c for root in (spec.submodule_search_locations if spec else [])
         for c in sorted(Path(root).glob(_USD_LIBS_GLOB))
         if (c / "pxr").is_dir() and (c / "bin").is_dir()),
        None,
    )
    if libs is None:
        raise SystemExit(
            "no isaacsim in this environment, so there is no pxr to read the USD with. Run this "
            "under `conda run -n env_isaaclab`."
        )
    env = dict(os.environ)
    env[_REEXEC_FLAG] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(libs), env.get("PYTHONPATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(libs / "bin"), env.get("LD_LIBRARY_PATH", "")])
    print(f"[robot] re-exec with pxr from {libs}")
    sys.stdout.flush()          # execve does not flush; without this the line is lost on a pipe
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _triangles(counts, indices) -> np.ndarray:
    """Fan-triangulate USD faces; faceVertexCounts may hold quads. Same shape as
    kitchen_usd_load._triangles, which is the loader this repo already trusts for USD faces."""
    tris = []
    at = 0
    for n in counts:
        face = indices[at:at + n]
        at += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int64)


#: A material name that records its own colour as six hex digits, e.g. material_C0C0C0.
_HEX_NAMED_MATERIAL = re.compile(r"material_([0-9A-Fa-f]{6})$")


def _diffuse_rgb(prim):
    """The bound material's diffuse colour for `prim`, as floats in 0..1.

    THE LIVERY IS WHAT MAKES THE ROBOT RECOGNISABLE, so it is read rather than invented. MDL first:
    these are OmniPBR shaders, whose colour is `diffuse_color_constant` on the mdl output. Mid grey
    for anything unbound, so an asset revision that adds a prim gets a plausible robot rather than a
    black one. None of the three carries a texture map or a displayColor primvar -- checked, not
    assumed -- so this constant is the whole of what each asset says about its own colour.

    THE NAME IS READ WHEN THE CONSTANT IS DEGENERATE, and that is Anubis-shaped rather than general:
    every one of the seven Anubis USDs in this tree binds a material NAMED material_C0C0C0 -- HTML
    silver -- and then authors (1e-6, 1e-6, 1e-6) into it, so 96.2% of the robot arrives pure black
    and renders as a silhouette. The other two robots have no such disagreement: RB-Y1 names its
    materials material_<r>_<g>_<b> in decimal and every constant matches its name, and AI Worker's
    are unnamed. So an all-but-zero constant under a name that records a colour is taken as the
    converter having lost the value it had just written into the name.
    """
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        return (0.5, 0.5, 0.5)
    shade = UsdShade.Material(material.GetPrim())
    shader = shade.ComputeSurfaceSource("mdl")[0] or shade.ComputeSurfaceSource()[0]
    colour = shader.GetInput("diffuse_color_constant").Get() if shader else None
    if colour is None:
        return (0.5, 0.5, 0.5)
    rgb = tuple(float(c) for c in colour)
    if max(rgb) < 2.0 / 255.0:
        named = _HEX_NAMED_MATERIAL.match(material.GetPrim().GetName())
        if named:
            return tuple(int(named.group(1)[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return rgb


def convert_usd(path):
    """The visible geometry of `path`'s default prim, one welded Trimesh per Mesh prim.

    -> (parts, bbox_height_m), where parts is [(prim path, Trimesh in world metres, (r, g, b))].
    `bbox_height_m` is UsdGeom.BBoxCache's own answer over the same prim and purposes, kept so the
    caller can check the geometry it is about to draw against USD's own idea of what is renderable
    rather than against a number in a comment.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    default = stage.GetDefaultPrim()
    if not default:
        raise SystemExit(f"{path} has no default prim, so there is nothing to measure or convert")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    aligned = cache.ComputeWorldBound(default).ComputeAlignedRange()
    bbox_height = float(aligned.GetMax()[2] - aligned.GetMin()[2])

    parts = []
    traversal = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined)
    for prim in Usd.PrimRange(default, traversal):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if str(imageable.ComputePurpose()) not in ("default", "render"):
            continue
        if str(imageable.ComputeVisibility()) == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        part = trimesh.Trimesh(vertices=np.asarray(points, dtype=float),
                               faces=_triangles(list(counts), list(indices)), process=False)
        part.apply_transform(np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=float,
        ).T)                    # USD's Gf.Matrix4d is row-major with translation in the last ROW
        part.merge_vertices()   # within this prim only: see the module docstring, step 3
        parts.append((str(prim.GetPath()), part, _diffuse_rgb(prim)))

    if not parts:
        raise SystemExit(f"{path} carries no visible default/render-purpose Mesh geometry")
    faces = sum(len(p.faces) for _n, p, _c in parts)
    colours = {c for _n, _p, c in parts}
    print(f"[robot] converted: {faces} faces from {len(parts)} visible mesh prims in "
          f"{len(colours)} colours")
    return parts, bbox_height


def spawn_pose_of(config_path: Path, symbol: str) -> dict:
    """`symbol`.init_state.joint_pos, read out of `config_path`'s SOURCE. -> {joint name: value}.

    WHY THE SOURCE AND NOT THE VALUE. The pose has to come from the config, because a copy of
    nineteen floats drifts the moment somebody tunes the spawn pose and the reference then draws a
    robot the simulator no longer spawns. But `import isaaclab_assets.robots.anubis_wheels` is not
    available to this script: the module's first line is `import isaaclab.sim`, which pulls
    Omniverse -- MEASURED here, not assumed, and it fails at `ModuleNotFoundError: No module named
    'carb'`, carb being the Carbonite runtime that only exists once an Isaac app has bootstrapped.
    Booting Omniverse to read a dict of constants would also undo this script's whole premise that
    it needs pxr and nothing else. So the literal is parsed instead.

    ast.literal_eval, NOT eval and not a regex: the dict is a literal in the source and
    literal_eval is the reader that says so, refusing anything with a name or a call in it. A config
    that computed its pose would raise here rather than being silently half-read.
    """
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    call = next(
        (node.value for node in tree.body
         if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
         and any(getattr(target, "id", None) == symbol for target in node.targets)),
        None,
    )
    if call is None:
        raise SystemExit(f"{config_path} has no `{symbol} = <call>(...)` to read a spawn pose from")
    init = next((kw.value for kw in call.keywords if kw.arg == "init_state"), None)
    if not isinstance(init, ast.Call):
        raise SystemExit(f"{symbol} in {config_path} passes no init_state=<call>(...)")
    joint_pos = next((kw.value for kw in init.keywords if kw.arg == "joint_pos"), None)
    if joint_pos is None:
        raise SystemExit(f"{symbol}.init_state in {config_path} sets no joint_pos")
    pose = {str(name): float(value) for name, value in ast.literal_eval(joint_pos).items()}
    print(f"[robot] spawn pose: {len(pose)} joints from {symbol} in {config_path.name}")
    return pose


def _rigid(pos, rot) -> np.ndarray:
    """A USD (translation, rotation) pair as a 4x4 column-vector matrix, the convention numpy and
    trimesh use. Transposed for the same reason convert_usd transposes: Gf.Matrix4d is row-major
    with the translation in the last ROW."""
    from pxr import Gf

    return np.array(Gf.Matrix4d().SetTransform(Gf.Rotation(rot), Gf.Vec3d(*pos)), dtype=float).T


def _joint_motion(prim, value: float) -> np.ndarray:
    """What a joint's own value does to its child frame, in the joint's frame. -> 4x4.

    The axis is UsdPhysics's `physics:axis` token, which names a coordinate axis OF THE JOINT FRAME
    -- so this is applied between the two local frames and never in world.

    RADIANS AND METRES, because that is what an IsaacLab ArticulationCfg.InitialStateCfg holds and
    the value handed in comes from one. USD's own degrees convention for revolute joints is not in
    play: nothing here reads a limit or a drive target off the stage, only the axis token.
    """
    from pxr import UsdPhysics

    motion = np.eye(4)
    if value == 0.0:
        return motion
    if prim.IsA(UsdPhysics.RevoluteJoint):
        axis = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
        cos, sin = np.cos(value), np.sin(value)
        motion[:3, :3] = {
            "X": ((1, 0, 0), (0, cos, -sin), (0, sin, cos)),
            "Y": ((cos, 0, sin), (0, 1, 0), (-sin, 0, cos)),
            "Z": ((cos, -sin, 0), (sin, cos, 0), (0, 0, 1)),
        }[axis]
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        axis = str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get())
        motion[:3, 3] = np.eye(3)["XYZ".index(axis)] * value
    else:
        raise SystemExit(
            f"{prim.GetName()} is set to {value} by the config but is a {prim.GetTypeName()}, "
            "which has no value to set"
        )
    return motion


def spawn_transforms(usd_path, joint_pos: dict) -> dict:
    """Where `joint_pos` puts each of `usd_path`'s bodies, relative to where the USD authors it.

    -> {body prim path: 4x4 rigid transform}, each one carrying that body's AUTHORED world pose to
    its posed one, which is exactly what has to be applied to geometry already baked into world
    space by convert_usd.

    THE KINEMATICS AND THE GEOMETRY COME OUT OF ONE FILE. anubis_simvla.usd carries all 24 of its
    UsdPhysics joints with both body relationships resolved, so nothing here has to reconcile a
    URDF's link names against a USD's prim names -- the four Anubis URDFs in this tree would have
    needed exactly that, and a mesh attached to the wrong link is a defect no test of the FK itself
    would catch. Every visible mesh prim in this asset lives under `/<default>/<body>/visuals/...`,
    so the body a mesh belongs to is its own prim path (see apply_pose).

    THE JOINT CONVENTION, and it is CHECKED rather than trusted -- see
    test_kitchen_preview.test_the_anubis_reference_is_posed_by_its_own_config, which recovers each
    link's motion from the DRAWN vertices and compares it against the config's own numbers. A
    UsdPhysics joint names one frame on each body, (localPos0, localRot0) on body0 and
    (localPos1, localRot1) on body1, and constrains them to coincide but for its own degree of
    freedom. So

        world(body1) = world(body0) . frame0 . motion(value) . frame1^-1

    and at value = 0 that must reproduce the transforms the stage authors. It does, over all 24
    joints, to 4e-6 m -- which also establishes that THE AUTHORED POSE IS THIS ROBOT'S ZERO POSE.

    A JOINT WITH NO body0 IS NOT KINEMATICS. anubis_simvla.usd's `root_joint` is a fixed joint with
    an empty body0 pinning the robot to the static world frame, and the asset places that frame 2.0
    m from the stage origin. Treated as a joint it would drag the whole robot back to the origin;
    it is skipped, and the body it holds is the root the walk starts from, keeping its authored
    place. The three base joints -- base_prismatic_x_joint, base_prismatic_y_joint and
    base_revolute_z_joint -- are the dummy mobile-base mechanism and the config sets all three to
    zero, so they contribute identity and the robot stands where its own asset stands.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(metres - 1.0) > 1e-9:
        raise SystemExit(
            f"{usd_path} is authored at {metres} metres per unit, so the config's prismatic joint "
            "values -- which are metres -- cannot be applied to it unconverted"
        )

    joints = {}                 # child body path -> (joint prim, parent body path, frame0, frame1)
    traversal = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined)
    for prim in Usd.PrimRange(default, traversal):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue            # a grounding joint, not kinematics -- see the docstring
        joints[body1[0].pathString] = (
            prim, body0[0].pathString,
            _rigid(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            _rigid(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        )

    bodies = set(joints) | {parent for _j, parent, _f0, _f1 in joints.values()}
    authored = {
        path: np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()), dtype=float).T
        for path in bodies
    }
    posed = {path: authored[path] for path in bodies if path not in joints}

    # BREADTH-FIRST OVER WHATEVER TREE THE ASSET HAS, rather than a hardcoded chain: Anubis branches
    # at base_link into two arms and again at each wrist into two fingers, and a future revision may
    # branch elsewhere. A pass that resolves nothing means the joints do not reach every body.
    set_by_config = set()
    while len(posed) < len(bodies):
        resolved = 0
        for child, (prim, parent, frame0, frame1) in joints.items():
            if child in posed or parent not in posed:
                continue
            name = prim.GetName()
            if name in joint_pos:
                set_by_config.add(name)
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise SystemExit(
                    f"{name} moves this robot but the spawn pose does not set it, so it would be "
                    "drawn at zero while the simulator spawns it wherever the config says"
                )
            posed[child] = (posed[parent] @ frame0
                            @ _joint_motion(prim, joint_pos.get(name, 0.0))
                            @ np.linalg.inv(frame1))
            resolved += 1
        if not resolved:
            raise SystemExit(
                f"{sorted(bodies - set(posed))} are jointed to nothing this walk can reach -- the "
                f"asset's joint graph is not the tree this forward kinematics assumes"
            )

    unset = sorted(set(joint_pos) - set_by_config)
    if unset:
        raise SystemExit(
            f"the spawn pose names {unset}, which no joint in {usd_path} is called -- the config "
            "and the asset have been renamed apart, and the pose drawn would be the wrong one"
        )
    return {path: posed[path] @ np.linalg.inv(authored[path]) for path in bodies}


def apply_pose(parts, transforms: dict) -> None:
    """Move each of `parts` by its own body's transform, in place. See spawn_transforms.

    Each mesh is matched to the body it hangs under by prim path, and a mesh under no known body is
    an ERROR rather than a mesh left where it was: geometry that silently keeps the authored pose
    while the rest of the robot moves is a broken robot, and it is what an asset revision adding a
    part would produce.
    """
    for path, mesh, _rgb in parts:
        owners = [body for body in transforms if path.startswith(body + "/")]
        if len(owners) != 1:
            raise SystemExit(
                f"{path} hangs under {len(owners)} of this asset's jointed bodies ({owners}), so "
                "there is no one pose to draw it in"
            )
        mesh.apply_transform(transforms[owners[0]])
    print(f"[robot] posed {len(parts)} mesh prims over {len(transforms)} jointed bodies")


def z_extent(parts) -> float:
    """The height of `parts` taken together -- the number the dimension line has to measure."""
    low = min(float(mesh.bounds[0][2]) for _n, mesh, _c in parts)
    high = max(float(mesh.bounds[1][2]) for _n, mesh, _c in parts)
    return high - low


def check_facing(parts, witness):
    """Assert the per-robot facing witness, and return the margin it measured. See FacingWitness."""
    part_name, axis, sign, margin_m = witness
    centres = np.array([p.bounds.mean(axis=0) for _n, p, _c in parts])
    low = np.array([p.bounds[0] for _n, p, _c in parts]).min(axis=0)
    high = np.array([p.bounds[1] for _n, p, _c in parts]).max(axis=0)
    plan_centre = (low + high) / 2.0
    picked = [c for (name, _p, _c), c in zip(parts, centres) if part_name in name]
    if not picked:
        raise SystemExit(
            f"no visible prim's path contains {part_name!r}, so the facing of this robot is not "
            "measured and it would be drawn in whatever direction the asset happens to use"
        )
    measured = float(sign) * (np.mean(picked, axis=0)[axis] - plan_centre[axis])
    if measured < margin_m:
        raise SystemExit(
            f"{part_name!r} sits {measured:.4f} m on the expected side of the plan centre along "
            f"{'xyz'[axis]}, against the {margin_m} m this robot's facing was pinned by. The asset's "
            "pose has changed; re-measure before drawing it."
        )
    print(f"[robot] facing witness: {part_name} is {measured:.4f} m "
          f"{'+' if sign > 0 else '-'}{'xyz'[axis]} of the plan centre, so the front is +x")
    return measured


def assemble(parts):
    """One Trimesh carrying every part, with each part's colour on each of its vertices.

    -> (mesh, uint8 (n, 4) vertex colours). NOT welded across parts and NOT processed: two links
    that touch keep their own vertices, so the colour boundary between them stays exactly where the
    asset puts it rather than being averaged across a shared vertex.
    """
    meshes = [p for _n, p, _c in parts]
    rows = [np.tile(c, (len(p.vertices), 1)) for _n, p, c in parts]
    whole = trimesh.util.concatenate(meshes)
    rgb = np.rint(np.concatenate(rows, axis=0) * 255.0)
    if len(rgb) != len(whole.vertices):
        raise SystemExit(
            f"{len(rgb)} colour rows against {len(whole.vertices)} vertices -- the per-prim colours "
            "no longer line up with the concatenated geometry"
        )
    return whole, np.concatenate(
        [rgb, np.full((len(rgb), 1), 255.0)], axis=1
    ).astype(np.uint8)


def normalise(mesh, height_m: float):
    """Centre `mesh` in plan, stand it on z = 0, and scale it so its z extent is exactly `height_m`.

    CENTRED IN PLAN because none of the three assets is: Anubis is authored 2.0 m from its own
    origin (its plan centre is at x 1.012, y -1.723) and the other two are a few millimetres off.
    The dimension line stands at a fixed offset from this node and has to clear the ROBOT.
    """
    out = mesh.copy()
    low, high = out.bounds
    out.apply_translation((
        -(float(low[0]) + float(high[0])) / 2.0,
        -(float(low[1]) + float(high[1])) / 2.0,
        -float(low[2]),
    ))
    out.apply_scale(float(height_m) / float(out.extents[2]))
    out.apply_translation((0.0, 0.0, -float(out.bounds[0][2])))
    return out


def main(argv) -> int:
    if len(argv) != 1 or argv[0] not in ROBOT_SOURCES:
        raise SystemExit(f"usage: build_robot_reference.py {{{'|'.join(ROBOT_SOURCES)}}}")
    name = argv[0]
    source = ROBOT_SOURCES[name]
    glb_path = robot_reference_path(name)
    if glb_path.exists():
        print(f"[robot] {glb_path} exists already; delete it to rebuild.")
        return 0

    usd = Path(source["usd"])
    if not usd.is_absolute():
        usd = Path(__file__).resolve().parents[2] / usd
    if not usd.is_file():
        raise SystemExit(f"{usd} is not on this machine, so there is nothing to build {name} from")

    reexec_with_pxr()                       # replaces this process when pxr is missing

    started = time.time()
    parts, bbox_height = convert_usd(usd)
    authored_height = z_extent(parts)

    # THE GEOMETRY IS THE HEIGHT, and BBoxCache is the cross-check rather than the answer. The two
    # agree to 1e-7 on AI Worker and RB-Y1 and disagree by 2.95 mm on Anubis, whose meshes carry
    # authored `extent` attributes padded a few millimetres past their own points -- BBoxCache reads
    # those, the renderer draws the points. The line beside the robot has to measure the robot.
    #
    # The check is one-sided for exactly that reason: geometry TALLER than BBoxCache means the
    # purpose/visibility filter above let something in that USD does not consider renderable, which
    # is the G1 bug this file was written around (two invisible cubes 0.47 mm below the soles).
    # Geometry SHORTER is either padded extents or a filter that dropped a limb, so it is bounded
    # too, at a centimetre -- an asset losing a real part loses far more than that.
    #
    # BEFORE POSING, AND ONLY EVER BEFORE IT, because the question this pair asks is WHICH PRIMS the
    # filter kept, not where they are; BBoxCache has one answer and it is the authored one. Moving a
    # limb changes the height by half a metre and would drown the millimetre this is looking for.
    if authored_height > bbox_height + 1e-6:
        raise SystemExit(
            f"the converted geometry is {authored_height:.7f} m but BBoxCache measures "
            f"{bbox_height:.7f} m over the same prim and purposes -- the purpose or visibility "
            "filter is letting through something USD does not consider renderable."
        )
    if bbox_height - authored_height > 0.01:
        raise SystemExit(
            f"BBoxCache measures {bbox_height:.7f} m against {authored_height:.7f} m of geometry "
            "-- more than padded extents can account for, so this conversion has dropped a part."
        )

    # THE POSE THE SIMULATOR SPAWNS, for the robot that asked for it. Anubis's config folds both
    # arms 135 degrees at the elbow, which is half a metre of height and a different footprint, so
    # a reference drawn in the authored pose is a reference to a robot no scene contains.
    spawn_pose = source.get("spawn_pose")
    joint_pos = None
    if spawn_pose is not None:
        config, symbol = spawn_pose
        joint_pos = spawn_pose_of(Path(__file__).resolve().parents[2] / config, symbol)
        apply_pose(parts, spawn_transforms(usd, joint_pos))

    witness_m = check_facing(parts, source["witness"])
    mesh, colours = assemble(parts)
    height = float(mesh.extents[2])
    recorded = float(ROBOT_HEIGHTS_M[name])
    if abs(height - recorded) > 5e-7:
        raise SystemExit(
            f"this asset measures {height:.7f} m, but kitchen_preview.ROBOT_HEIGHTS_M[{name!r}] "
            f"records {recorded:.6f} m. One of them is stale; do not ship a dimension line against "
            "a height nobody measured."
        )

    final = normalise(mesh, height)
    # ColorVisuals, NOT a material, and the choice is measured rather than stylistic: trimesh can
    # export a mesh carrying both (a PBRMaterial plus visual.vertex_attributes["color"]), but its
    # glTF LOADER drops the attribute on the way back in, so a GLB written that way arrives in
    # kitchen_preview as a plain grey robot. Written as vertex colours the livery survives the load
    # AND the page's own re-export. The material the browser needs is put back by
    # kitchen_preview.robot_mesh, which is the one place that knows the page's lighting.
    final.visual = trimesh.visual.ColorVisuals(final, vertex_colors=colours)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    scene.add_geometry(final, node_name=name, geom_name=name)
    glb_path.write_bytes(scene.export(file_type="glb"))
    glb_path.with_suffix(".json").write_text(json.dumps({
        "robot": name,
        "source_usd": str(usd),
        "source_faces": int(len(mesh.faces)),
        "source_prims": len(parts),
        "source_height_m": height,
        "authored_height_m": authored_height,
        "bbox_height_m": bbox_height,
        # WHAT POSE THIS WAS DRAWN IN, so a cache on disk can be told from one built before the
        # config was tuned. null for the two robots that keep their asset's authored pose.
        "spawn_pose": None if joint_pos is None else {
            "config": spawn_pose[0], "symbol": spawn_pose[1], "joint_pos": joint_pos,
        },
        "source_colours": sorted({tuple(int(v) for v in np.rint(np.asarray(c) * 255.0))
                                  for _n, _p, c in parts}),
        "facing": "+x",
        "facing_witness": list(source["witness"]),
        "facing_witness_m": float(witness_m),
        "reduction": "none -- see build_robot_reference.NO_REDUCTION",
        "faces": int(len(final.faces)),
        "vertices": int(len(final.vertices)),
        "height_m": float(final.extents[2]),
        "glb_bytes": glb_path.stat().st_size,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "built_by": "scripts/tools/build_robot_reference.py",
    }, indent=1), encoding="utf-8")
    print(f"[robot] {ROBOTS[name]['label']}: {len(final.faces)} faces, {height:.6f} m -> "
          f"{glb_path} ({glb_path.stat().st_size / 1e6:.2f} MB) in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
