"""Read a committed kitchen USD back into what the task composer draws.

The composer renders a trimesh.Scene held in memory. A kitchen that was generated earlier
exists only as USD on disk, and the export is one-way: scene_synthesizer.usd_import exposes
helpers but no whole-scene loader, and trimesh has no USD reader. Nor can the scene be rebuilt
from kitchen_data_<N>.json, which records only kitchen_type and per-object mesh paths while
placement uses randomized jitter. So of the kitchens on disk, none could be authored against.

This module closes that. It is strictly READ-ONLY -- it never writes USD.

It imports pxr, so it CANNOT be imported on a login node and must not be imported by
kitchen_build.py, which is deliberately Omniverse-free so its tests run without a GPU.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

SIMVLA_REPO_ROOT = os.environ.get("SIMVLA_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
KITCHEN_DIR = os.path.join(
    os.path.expanduser(os.environ.get(
        "SIMVLA_ASSETS_DIR",
        os.path.join(SIMVLA_REPO_ROOT, "source/isaaclab_assets/data"),
    )),
    "Kitchen",
)


class LoadedKitchen:
    """Quacks like the scene_synthesizer.Scene the rest of the pipeline passes around.

    match_support_nodes(), collect_joints() and composer_page() all reach through a `.scene`
    attribute to the trimesh scene. Wrapping rather than subclassing keeps this module free of
    any scene_synthesizer dependency.
    """

    def __init__(self, scene):
        self.scene = scene


def rotation_usd_path(kitchen_num: int) -> str:
    """Rotation 00, never the base kitchen_<N>.usd.

    Isaac-Kitchen-v<N>-00 is the canonical registered env, and only the rotations carry the
    post-rename object naming: rotate_and_register_envs renames /world/bottle00 -> /world/bottle0
    when it writes them, so the base USD's names do not match what task_emit binds against.
    """
    return f"{KITCHEN_DIR}/kitchen_{kitchen_num:02d}_00.usd"


def _node_name(prim) -> str:
    """A prim path as the in-memory scene would have named it: '/world/x/y' -> 'x/y'."""
    path = str(prim.GetPath())
    return path[len("/world/"):] if path.startswith("/world/") else path.lstrip("/")


def _local_transform(prim) -> np.ndarray:
    """The prim's OWN transform (parent-relative, not local-to-world) as a numpy 4x4.

    USD's Gf.Matrix4d is ROW-major with translation in the last ROW; numpy and trimesh expect
    the last COLUMN. The transpose is mandatory -- omitting it produces plausible-looking but
    wrong placements.

    Local, not world, because load_kitchen_scene reproduces USD's transform hierarchy: each prim
    becomes a graph node carrying this matrix, and the graph composes the chain. Baking world
    transforms into vertices instead is what made joint sliders inert -- see load_kitchen_scene.
    """
    m = UsdGeom.Xformable(prim).GetLocalTransformation()
    return np.array(m, dtype=float).T


def _triangles(counts, indices) -> np.ndarray:
    """Fan-triangulate USD faces. faceVertexCounts may hold quads, not only triangles."""
    tris = []
    at = 0
    for n in counts:
        face = indices[at : at + n]
        at += n
        for k in range(1, n - 1):
            tris.append((face[0], face[k], face[k + 1]))
    return np.asarray(tris, dtype=np.int64)


def _mesh_from_mesh_prim(prim):
    """A LOCAL-space trimesh.Trimesh for one UsdGeom.Mesh prim, or None if it carries no geometry.

    USD `points` are already in the prim's own frame, so they are used as authored: the node this
    mesh hangs off carries the prim's local transform and the scene graph composes the rest.
    """
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return None
    faces = _triangles(list(counts), list(indices))
    if len(faces) == 0:
        return None
    verts = np.asarray(points, dtype=float)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


#: UsdGeomCube's schema-defined fallback for `size` when the attribute is unauthored.
_CUBE_DEFAULT_SIZE = 2.0


def _mesh_from_cube_prim(prim):
    """A LOCAL-space trimesh.Trimesh for one UsdGeom.Cube prim: a `size`-edge cube on the origin.

    scene_synthesizer builds furniture carcasses, shelves and panels out of boxes, and USD stores
    an analytic box as a Cube prim (a `size` edge length, centred on the local origin) rather than
    a Mesh -- UsdGeom.Mesh only covers imported .obj assets (bottles, mugs...). A traversal that
    looks for Mesh alone silently drops every one of these: on kitchen 1313 that is 436 of 498
    renderable prims, which is why a loaded fridge showed only its (Mesh) door and handle, never
    its (Cube) body. Non-uniform shape comes entirely from the prim's own transform, which the
    prim's graph node carries, so the cube arrives as the right box once the graph composes it.
    """
    size = UsdGeom.Cube(prim).GetSizeAttr().Get()
    if size is None:
        size = _CUBE_DEFAULT_SIZE
    return trimesh.creation.box(extents=(size, size, size))


#: UsdGeomCylinder's schema-defined fallbacks for unauthored attributes.
_CYLINDER_DEFAULT_RADIUS = 1.0
_CYLINDER_DEFAULT_HEIGHT = 2.0

#: A 16-sided prism. Only 4 cylinders exist on kitchen 1313 (hinge-style hardware), so this is a
#: click target and a placeholder body, not a physics collider -- a coarse prism is visually
#: indistinguishable at kitchen scale and costs 64 triangles total, not thousands.
_CYLINDER_SECTIONS = 16

#: trimesh.creation.cylinder always builds along local +Z, but UsdGeom.Cylinder's `axis` token can
#: be X, Y or Z, so the Z-aligned prism is rotated onto the requested axis. That rotation is part
#: of the prim's own shape, not its placement, so it stays baked into the mesh while the prim's
#: local transform stays on its graph node. Only the Z -> axis mapping matters here -- a cylinder
#: is rotationally symmetric about its own axis, so any rotation achieving that one mapping
#: produces an identical mesh.
_CYLINDER_AXIS_ROTATION = {
    "Z": np.eye(4),
    "X": np.array(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
    "Y": np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    ),
}


def _mesh_from_cylinder_prim(prim):
    """A LOCAL-space trimesh.Trimesh approximating one UsdGeom.Cylinder prim as a 16-sided prism,
    or None if its axis token is not one of X/Y/Z."""
    cylinder = UsdGeom.Cylinder(prim)
    radius = cylinder.GetRadiusAttr().Get()
    height = cylinder.GetHeightAttr().Get()
    axis_token = cylinder.GetAxisAttr().Get()
    if radius is None:
        radius = _CYLINDER_DEFAULT_RADIUS
    if height is None:
        height = _CYLINDER_DEFAULT_HEIGHT
    rotation = _CYLINDER_AXIS_ROTATION.get(axis_token or "Z")
    if rotation is None:
        print(f"[usd_load] {prim.GetPath()} has unrecognized cylinder axis {axis_token!r}; skipping.")
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=_CYLINDER_SECTIONS)
    cyl.apply_transform(rotation)
    return cyl


def _mesh_from_prim(prim):
    """A local-space trimesh.Trimesh for one renderable Gprim (Mesh, Cube or Cylinder), or None if
    it carries no geometry or is a Gprim type this loader does not yet convert (none exist on
    kitchen 1313 -- see load_kitchen_scene's traversal, which only calls this for IsA(Gprim))."""
    if prim.IsA(UsdGeom.Mesh):
        return _mesh_from_mesh_prim(prim)
    if prim.IsA(UsdGeom.Cube):
        return _mesh_from_cube_prim(prim)
    if prim.IsA(UsdGeom.Cylinder):
        return _mesh_from_cylinder_prim(prim)
    print(f"[usd_load] {prim.GetPath()} is a {prim.GetTypeName()} Gprim with no converter; skipping.")
    return None


_WORLD_PATH = "/world"


def _paths_to_geometry(stage) -> set:
    """Every prim path from /world down to each renderable Gprim -- the intermediate Xforms too.

    These are exactly the prims that must become scene-graph nodes. Restricting to this set (rather
    than every prim on the stage) keeps out the parts of the USD that carry no geometry at all --
    Looks/Material/Shader, the physics scene, the joint prims themselves -- while guaranteeing that
    every geometry-bearing branch is complete, so no mesh loses its transform chain.
    """
    needed = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        path = prim.GetPath()
        while path.pathString not in (_WORLD_PATH, "/", ""):
            needed.add(path.pathString)
            path = path.GetParentPath()
    return needed


def load_kitchen_scene(kitchen_num: int) -> trimesh.Scene:
    """Rotation 00 as a trimesh.Scene with USD's transform HIERARCHY reproduced.

    Every prim on the path to geometry becomes a graph node named by its prim path minus '/world/'
    ('refrigerator', 'refrigerator/door', 'refrigerator/door/handle'), carrying that prim's own
    LOCAL transform and parented to its parent prim's node; the graph composes the chain. Each
    Gprim's geometry hangs off its own node with local-space vertices.

    That hierarchy is load-bearing, not tidiness. An earlier version baked world transforms into
    vertices and added every mesh at identity, so only Gprims were nodes -- and a joint's child
    body (refrigerator/door, base_cabinet/drawer_0_0) is an Xform owning no geometry itself, so it
    was not a node at all. The preview drives a part by name (getObjectByName(j.child), then
    child.matrix = origin @ T(q)), found nothing, and all 27 of kitchen 1313's joint sliders moved
    nothing. With the hierarchy the child IS a node whose local matrix is exactly the `origin`
    load_kitchen_joints reports, so setting its matrix swings the door and its whole subtree --
    handle, shelves, anything parented under it -- comes along, which is also why the geometry must
    stay in local space: baked world vertices would not follow their parent.

    All three Gprim types are converted, not just Mesh: scene_synthesizer builds furniture
    carcasses, shelves and panels as analytic UsdGeom.Cube, and small hardware as UsdGeom.Cylinder
    -- only imported .obj assets (bottles, mugs...) come in as UsdGeom.Mesh. On kitchen 1313, Mesh
    alone is 58 of 498 renderable prims (12%); a Mesh-only traversal is what rendered a fridge with
    only its door and handle, no body.

    Node names are full prim paths and need no de-duplication: prim paths are unique by
    construction, and nothing is collapsed. A free-standing object is reachable under the bare name
    the composer binds against ('bottle0') because its wrapper Xform is now a node in its own
    right, with object_node_names() resolving its subtree.
    """
    path = rotation_usd_path(kitchen_num)
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(f"Could not open kitchen USD: {path}")

    scene = trimesh.Scene()
    # trimesh's own root frame is already called 'world', which is what the USD's /world prim maps
    # to -- and the preview's worldFrame() looks that node up by name. A node cannot have an edge
    # to itself, so if /world ever carried a transform of its own it is folded into its children
    # here instead of being silently dropped (it is identity on today's kitchens).
    world_prim = stage.GetPrimAtPath(_WORLD_PATH)
    root_matrix = _local_transform(world_prim) if world_prim else np.eye(4)
    node_by_path = {_WORLD_PATH: scene.graph.base_frame}

    wanted = _paths_to_geometry(stage)
    for prim in stage.Traverse():   # depth-first, so a parent is always registered before its child
        prim_path = str(prim.GetPath())
        if prim_path not in wanted:
            continue
        parent_node = node_by_path.get(str(prim.GetParent().GetPath()))
        if parent_node is None:
            # Only reachable if a Gprim's ancestor chain leaves /world or passes through a
            # non-Xformable prim. Dropping the branch would lose geometry silently, and
            # test_geometry_count_matches_total_renderable_gprims counts Gprims straight off the
            # USD, so it would fail -- but say so here too rather than only in a test.
            print(f"[usd_load] {prim_path} has no loaded parent node; skipping its subtree.")
            continue

        name = _node_name(prim)
        matrix = _local_transform(prim)
        if parent_node == scene.graph.base_frame:
            matrix = root_matrix @ matrix
        mesh = _mesh_from_prim(prim) if prim.IsA(UsdGeom.Gprim) else None
        if mesh is None:
            # A pure Xform (or a Gprim with no usable geometry): a node with a transform and no
            # mesh. Its descendants still need it, and a joint child is usually exactly this.
            scene.graph.update(frame_from=parent_node, frame_to=name, matrix=matrix)
        else:
            scene.add_geometry(
                mesh, node_name=name, geom_name=name,
                parent_node_name=parent_node, transform=matrix,
            )
        node_by_path[prim_path] = name
    return scene


def load_kitchen_data(kitchen_num: int) -> dict:
    """kitchen_data_<N>.json: {"kitchen_type": ..., "<obj_n>": "<mesh path>", ...}."""
    path = f"{KITCHEN_DIR}/bodex/kitchen_data_{kitchen_num:02d}.json"
    with open(path) as f:
        return json.load(f)


def load_kitchen_objects(kitchen_num: int, scene) -> list[dict]:
    """The composer's object list: [{"label", "node_id", "location"}, ...].

    node_id IS the USD prim name, not the in-memory f"{obj_n}0" form. The loaded scene's nodes
    are named after prim paths, so using the USD name is what makes object_node_names() resolve;
    inventing the in-memory name would give the composer a key matching nothing. The composer
    derives a role's object type from `label` by stripping trailing digits, so "bottle0" still
    yields the type "bottle".

    `location` is not recoverable -- no placement loc is persisted anywhere -- and is display-only
    in the preview, so it is left empty.
    """
    nodes = set(scene.graph.nodes)
    objects = []
    for label in load_kitchen_data(kitchen_num):
        if label == "kitchen_type":
            continue
        if label not in nodes:
            print(f"[usd_load] {label} is in kitchen_data but not in the USD; skipping.")
            continue
        objects.append({"label": label, "node_id": label, "location": ""})
    return objects


_AXIS_VECTORS = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}


def _axis_in_child_frame(axis, local_rot1) -> list[float]:
    """The joint's axis expressed in the CHILD body's frame, which is the frame that drives it.

    UsdPhysics' `physics:axis` token is expressed in the JOINT's frame, and the joint frame is
    attached to the child body by physics:localPos1/localRot1. The preview drives a part with
    child.matrix = origin @ T(q), where T(q) rotates or translates about `axis` in the CHILD's
    local frame -- so the token vector has to be rotated by localRot1 first. Skipping that is
    exactly what made some sliders swing about the wrong axis or in the wrong direction while
    others looked fine: on kitchen 1313, 10 of 27 joints have an identity localRot1 (those worked)
    and 17 do not (those were wrong). collect_joints has no equivalent step because
    scene_synthesizer's edge metadata already stores the axis in the child's frame.

    The direction is localRot1 * v, not its inverse: localRot1 IS the joint frame expressed in the
    child's frame, so it maps a joint-frame vector into the child's. The inverse is a plausible-
    looking wrong answer -- for these 90-degree rotations it yields a different axis, not a
    detectably broken one -- so it was settled physically rather than by inspection: a drawer must
    slide along the direction its cabinet opens (the normal of its corpus's back panel, taken from
    geometry alone). All 7 of kitchen 1313's drawers open outward under this convention, 0 of 7
    under either the inverse or the raw token. See test_drawers_slide_out_of_their_cabinet.

    An absent localRot1 means an unrotated joint frame, so the token vector is already right.
    """
    if local_rot1 is None:
        return [float(v) for v in axis]
    quat = Gf.Quatd(
        float(local_rot1.GetReal()), Gf.Vec3d(*[float(v) for v in local_rot1.GetImaginary()])
    )
    rotated = np.array(Gf.Rotation(quat).TransformDir(Gf.Vec3d(*axis)), dtype=float)
    norm = float(np.linalg.norm(rotated))
    if norm < 1e-9:                       # a degenerate quaternion; the token is the best answer
        return [float(v) for v in axis]
    return [float(v) for v in rotated / norm]


def _local_matrix(prim) -> list[list[float]]:
    """The prim's own local transform as a plain nested list, for JSON.

    This IS the joint origin the preview needs. The preview drives a part with
    child.matrix = origin @ T(q), where origin is the child's local matrix at q=0 -- and the
    kitchen is exported with its doors and drawers closed, so the exported local transform
    already is that matrix. (If a kitchen were ever exported with a joint open, this assumption
    breaks and origin would have to be composed from physics:localPos1/localRot1 against the
    child's frame. test_joint_origin_is_the_child_rest_pose below is what would catch that.)

    It is the SAME matrix load_kitchen_scene puts on the child's graph node, which is what makes
    the slider a no-op at q=0 instead of a jump: the preview overwrites a node matrix with an
    equal-at-rest one.
    """
    return [[float(v) for v in row] for row in _local_transform(prim)]


def load_kitchen_joints(kitchen_num: int) -> list[dict]:
    """Drivable joints in collect_joints()'s shape, reconstructed from UsdPhysics prims.

    Wired straight through to the composer: _compose_on_existing forwards these to composer_page,
    which renders one slider per joint and drives the part named by `child`. That works because
    load_kitchen_scene reproduces the USD hierarchy, so every `child` here -- refrigerator/door,
    base_cabinet/drawer_0_0, all Xforms owning no geometry of their own -- is a real scene node
    whose local matrix is the `origin` below. (Before that rework the scene was flattened to one
    identity-placed node per Gprim, those children were absent, and all 27 sliders on kitchen 1313
    moved nothing; test_joint_children_are_scene_nodes_so_the_sliders_drive_something guards it.)
    Driving a joint is inspection only -- the preview never posts joint values back, so opening a
    drawer to look inside cannot leak into the exported USD, which keeps this read-only.

    collect_joints reads joint data off trimesh scene-graph edge metadata, which does not
    survive export -- but the equivalent physics data does, as real UsdPhysics joints carrying
    an axis token, limits, and local frames. Joints with no range are dropped, matching
    collect_joints, which excludes parts that cannot move.

    Three conversions are needed to land in collect_joints' conventions, and skipping any of them
    yields joints that look well-formed and behave wrongly:
      - revolute limits are stored in DEGREES by UsdPhysics, and are converted to radians here;
      - `axis` is stored in the JOINT's frame and is rotated into the CHILD's frame by
        _axis_in_child_frame, because that is the frame the preview's T(q) acts in;
      - `name` is the joint prim's full path minus '/world/', because the preview keys its joint
        map by it and USD leaf names repeat across furniture (see the note at the field itself).
    """
    stage = Usd.Stage.Open(rotation_usd_path(kitchen_num))
    if stage is None:
        raise FileNotFoundError(f"Could not open kitchen USD: {rotation_usd_path(kitchen_num)}")

    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            kind, joint = "revolute", UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            kind, joint = "prismatic", UsdPhysics.PrismaticJoint(prim)
        else:
            continue

        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            continue
        lower, upper = float(lower), float(upper)
        if kind == "revolute":
            lower, upper = np.deg2rad(lower), np.deg2rad(upper)
        if upper <= lower:
            continue

        targets = joint.GetBody1Rel().GetTargets()
        if not targets:
            continue
        child_prim = stage.GetPrimAtPath(targets[0])
        if not child_prim:
            continue
        child = _node_name(child_prim)

        axis_token = joint.GetAxisAttr().Get()
        axis = _AXIS_VECTORS.get(axis_token)
        if axis is None:
            print(f"[usd_load] {prim.GetPath()} has unrecognized axis token {axis_token!r}; skipping.")
            continue
        axis = _axis_in_child_frame(axis, joint.GetLocalRot1Attr().Get())

        joints.append(
            {
                # The joint prim's PATH minus '/world/', not its leaf name. The preview keys its
                # joint map by name (jointNodes.set(j.name, ...)), and a JS Map overwrites on a
                # duplicate key, so leaf names silently merged sliders across furniture: kitchen
                # 1313's 27 joints share only 14 leaf names -- 'corpus_to_door_0_0' alone occurs 7
                # times, once per wall cabinet -- so 13 sliders drove some other cabinet's door
                # while the row label, which comes from j.child, still read correctly. Prim paths
                # are unique by construction (the same argument that let _simplify_names go), and
                # this derivation matches _node_name's, so the name reads as
                # 'base_cabinet/corpus_to_drawer_0_0' -- which is also exactly how collect_joints
                # names the same joint on a freshly built kitchen.
                "name": _node_name(prim),
                "type": kind,
                "child": child,
                "group": child.split("/")[0],
                "axis": axis,
                "origin": _local_matrix(child_prim),
                "lower": float(lower),
                "upper": float(upper),
            }
        )
    return joints


def load_kitchen_for_composer(kitchen_num: int):
    """Everything the composer needs for a kitchen already on disk.

    Returns (loaded, objects, joints, kitchen_data) -- FOUR values.

    Three of the composer's five inputs come from here -- `loaded.scene`, `objects` and `joints`,
    the last of which the caller forwards so doors and drawers get working sliders (they are
    drivable because load_kitchen_scene reproduces the USD hierarchy). The remaining two the
    loader deliberately never produces, and the caller supplies both as empty:

    materials={} -- USD stores MDL file references and mapping them back to pick_materials()'s
    display names is lossy, so the kitchen renders untextured.

    supports=[] -- `supports` feeds only the preview's drag-to-place raycast, and this loader is
    READ-ONLY. An object dragged onto a surface would render somewhere it is not in the USD on
    disk, while the authored task still binds to its real position at emit; a view that disagrees
    with the file is worse than no dragging. (The surfaces are largely absent from the USD anyway:
    only 2 of 21 placement locations resolve, because the file holds the visible/collidable meshes
    and not labelled surfaces like refrigerator/shelf_3. Offering just those two would be the
    misleading case.) Clicking is unaffected -- it resolves through the objects list, never
    through supports.
    """
    scene = load_kitchen_scene(kitchen_num)
    loaded = LoadedKitchen(scene)
    objects = load_kitchen_objects(kitchen_num, scene)
    joints = load_kitchen_joints(kitchen_num)
    return loaded, objects, joints, load_kitchen_data(kitchen_num)
