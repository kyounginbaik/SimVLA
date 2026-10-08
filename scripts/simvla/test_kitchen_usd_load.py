"""Tests for reading a committed kitchen USD back into a trimesh scene.

Needs pxr, so it CANNOT run on a login node. Run with:
    export HF_USER=exaFLOPs09
    srun --gres=gpu:1 ~/miniconda3/envs/env_isaaclab/bin/python \
        -m pytest scripts/simvla/test_kitchen_usd_load.py -v
"""

import os
from collections import Counter

import pytest

# A kitchen known to exist on disk with a bottle in it.
KITCHEN = 1313
OBJECT_LABEL = "bottle0"


@pytest.fixture(scope="module")
def loaded():
    # pxr is not a plain conda package here -- it is assembled by Omniverse Kit's extension
    # system at app boot. goal_generator.py (the tool that produced the committed kitchen USDs)
    # boots it the same way, at import time, before anything does `from pxr import ...`:
    # `AppLauncher({"headless": True})` must run first, or `import kitchen_usd_load` itself
    # raises ModuleNotFoundError: pxr, even under `srun --gres=gpu:1`.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True})
    _ = app_launcher.app  # force Omniverse up before pxr/omni become importable

    import kitchen_usd_load
    if not os.path.exists(kitchen_usd_load.rotation_usd_path(KITCHEN)):
        pytest.skip(f"kitchen {KITCHEN} rotation 00 not on disk")
    return kitchen_usd_load.load_kitchen_scene(KITCHEN)


def test_the_scene_has_geometry(loaded):
    """Measured on kitchen_1102_00.usd: 74 mesh prims, ~12.5k triangles. Any kitchen should
    yield dozens of meshes; zero means the traversal or the mesh reader is broken."""
    assert len(loaded.geometry) > 10, f"only {len(loaded.geometry)} meshes loaded"
    total_faces = sum(len(m.faces) for m in loaded.geometry.values())
    assert total_faces > 1000, f"only {total_faces} faces total"


def test_geometry_count_matches_total_renderable_gprims(loaded):
    """The bug this test exists to catch: the loader used to traverse for UsdGeom.Mesh only, so
    it captured 58 of kitchen 1313's 498 renderable Gprims (12%) -- a user opening this kitchen in
    the composer saw a fridge with only its (Mesh) door and handle, no (Cube) body, and cabinets
    with only a countertop or a door and handle, no carcass. scene_synthesizer builds furniture
    carcasses, shelves and panels out of analytic UsdGeom.Cube (436 of them here) and small
    hardware out of UsdGeom.Cylinder (4); only imported .obj assets (bottles, mugs) come in as
    Mesh. This counts Gprims directly off the USD, independent of the loader, so it fails again if
    a whole Gprim type is ever silently dropped -- checking loaded.geometry alone would not have
    caught the original bug, since 58 meshes still looked like "dozens of meshes" to
    test_the_scene_has_geometry above."""
    import kitchen_usd_load
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(kitchen_usd_load.rotation_usd_path(KITCHEN))
    total_gprims = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Gprim))
    assert total_gprims > 100, f"only {total_gprims} Gprims in the USD -- fixture kitchen changed?"
    loaded_count = len(loaded.geometry)
    # Equality, not a tolerance. An earlier `>= total_gprims - 5` defeated the whole point: kitchen
    # 1313 has exactly 4 Cylinder prims, so losing Cylinder support completely (498 -> 494) still
    # passed. 498 of 498 load today, so any slack here is slack for a silently dropped type.
    assert loaded_count == total_gprims, (
        f"loaded {loaded_count} meshes but the USD has {total_gprims} renderable Gprims -- "
        f"a Gprim type is being skipped, or two prims collapsed onto one node name"
    )


def test_a_cube_furniture_body_resolves_to_a_real_node_with_volume(loaded):
    """A named, non-door, non-handle furniture body that is a UsdGeom.Cube in the USD --
    refrigerator/corpus/shelf_3, an actual fridge shelf -- must load as real, solid geometry, not
    merely as a name. This is the concrete instance of the bug report: Cube prims were skipped
    entirely, so every cabinet's and fridge's carcass, shelves and panels were invisible even
    though their doors and handles (Mesh assets) rendered fine."""
    name = "refrigerator/corpus/shelf_3"
    assert name in loaded.graph.nodes, f"{name} is not a scene node"
    assert name in loaded.geometry, f"{name} has no geometry attached"
    mesh = loaded.geometry[name]
    assert mesh.volume > 1e-6, f"{name} has near-zero volume ({mesh.volume}) -- looks degenerate"


def test_every_geometry_is_a_real_node(loaded):
    """The composer matches clicked meshes against scene-graph node names, so every geometry
    must be reachable as a node."""
    nodes = set(loaded.graph.nodes)
    for name in loaded.geometry:
        assert name in nodes, f"{name} has geometry but is not a graph node"


def test_node_names_are_prim_paths_without_the_world_prefix(loaded):
    """kitchen_build.match_support_nodes() fnmatches against these names, so they must look
    like the in-memory ones ('countertop_base_cabinet', 'refrigerator/shelf_3')."""
    names = list(loaded.geometry)
    assert not any(n.startswith("/") for n in names), f"absolute paths leaked: {names[:3]}"
    assert not any(n.startswith("world/") for n in names), f"world/ prefix leaked: {names[:3]}"


def test_the_object_is_present_under_its_post_rename_name_and_is_clickable(loaded):
    """Rotation USDs hold /world/bottle0 (rotate_and_register_envs renames from bottle00).
    The loader must NOT invent the in-memory name.

    The bare name being a node is what let _simplify_names go: an object is exported as an Xform
    wrapper around one nested Mesh, so under the old flattening the only node was
    'bottle0/simplified_obj/simplified_obj' and a rename step had to fake 'bottle0' into existence.
    With the hierarchy the wrapper is a node in its own right. It owns no geometry itself, so the
    click path is also asserted here: object_node_names() must expand it to a subtree that includes
    the geometry-bearing descendant, since that is the node a raycast hit reports.
    """
    import kitchen_preview
    nodes = set(loaded.graph.nodes)
    assert OBJECT_LABEL in nodes, f"{OBJECT_LABEL} not among {sorted(nodes)[:8]}"

    clickable = kitchen_preview.object_node_names(loaded, OBJECT_LABEL)
    assert kitchen_preview.sanitize_three_name(OBJECT_LABEL) in clickable
    with_geometry = [
        n for n in loaded.graph.nodes
        if (n == OBJECT_LABEL or n.startswith(OBJECT_LABEL + "/")) and n in loaded.geometry
    ]
    assert with_geometry, f"nothing under {OBJECT_LABEL} carries geometry -- a click cannot hit it"
    assert all(
        kitchen_preview.sanitize_three_name(n) in clickable for n in with_geometry
    ), f"object_node_names({OBJECT_LABEL}) misses its own meshes: {with_geometry}"


def test_geometry_is_placed_in_world_space(loaded):
    """The transpose trap: USD matrices are row-major with translation in the last ROW, numpy
    uses the last COLUMN. Transposing wrongly collapses everything toward the origin or
    scatters it. A kitchen spans metres in x/y and stands above z=0."""
    lo, hi = loaded.bounds
    assert hi[0] - lo[0] > 1.0, f"x extent only {hi[0] - lo[0]:.3f} m"
    assert hi[1] - lo[1] > 1.0, f"y extent only {hi[1] - lo[1]:.3f} m"
    assert hi[2] > 0.5, f"nothing above z=0.5 (max z {hi[2]:.3f})"


def test_every_object_in_kitchen_data_survives_into_the_composer_list():
    """No object may be silently dropped between kitchen_data and the composer's clickable list.

    Objects are the ONLY thing clickable in a loaded kitchen (materials, supports and joints are
    all passed empty), so a dropped object is a role the user cannot author. load_kitchen_objects
    drops any label whose prim is missing from the scene with nothing but a print -- invisible
    behind a Tk GUI -- and _compose_on_existing then opens a composer with fewer targets than the
    kitchen has, or none at all.

    Asserting the converse (that each returned node_id is a real node, and that label == node_id)
    is what an earlier version of this test did, and it could not fail: load_kitchen_objects
    filters to labels already present in `nodes` and assigns both fields from the same variable.
    This direction -- every non-kitchen_type key of kitchen_data appears in `objects` -- is the one
    that exercises the filter, and it fails if the filter ever drops anything.
    """
    import kitchen_usd_load
    scene = kitchen_usd_load.load_kitchen_scene(KITCHEN)
    kitchen_data = kitchen_usd_load.load_kitchen_data(KITCHEN)
    objects = kitchen_usd_load.load_kitchen_objects(KITCHEN, scene)
    expected = {k for k in kitchen_data if k != "kitchen_type"}
    assert expected, f"kitchen {KITCHEN}'s kitchen_data lists no objects -- fixture changed?"
    got = {o["node_id"] for o in objects}
    assert got == expected, (
        f"objects dropped between kitchen_data and the composer: missing {sorted(expected - got)}, "
        f"unexpected {sorted(got - expected)}"
    )
    assert OBJECT_LABEL in expected, f"{OBJECT_LABEL} is no longer in kitchen_data -- fixture changed?"


def test_the_loader_produces_no_drag_targets():
    """A loaded kitchen is READ-ONLY, so drag-to-place must be off entirely.

    `supports` feeds exactly one thing: the preview's drag raycast. Since the loader never
    writes USD, an object dragged onto a surface would render somewhere it is not in the file,
    while the authored task still binds to its real position at emit. Offering the two surfaces
    that happen to survive in the USD would be the misleading case, so the loader produces none
    and load_kitchen_for_composer returns FOUR values, not five.
    """
    import kitchen_usd_load
    result = kitchen_usd_load.load_kitchen_for_composer(KITCHEN)
    assert len(result) == 4, (
        f"expected (loaded, objects, joints, kitchen_data), got {len(result)} values"
    )
    loaded, objects, joints, kitchen_data = result
    assert objects, "no objects loaded"
    assert joints, "no joints reconstructed"
    assert kitchen_data["kitchen_type"], "kitchen_type missing"
    assert loaded.scene.geometry, "no geometry on the wrapper's scene"


def test_joints_match_collect_joints_shape_and_are_drivable():
    """Measured on kitchen_1102_00.usd: 26 revolute + 7 prismatic. The preview drives a part
    with child.matrix = origin @ T(q), so origin must be a 4x4 and the range must be non-empty."""
    import kitchen_usd_load
    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints reconstructed"
    for j in joints:
        assert set(j) == {"name", "type", "child", "group", "axis", "origin", "lower", "upper"}
        assert j["type"] in ("revolute", "prismatic")
        assert j["upper"] > j["lower"], f"{j['name']} has no range to drive"
        assert len(j["axis"]) == 3
        assert len(j["origin"]) == 4 and len(j["origin"][0]) == 4
        assert j["group"] == j["child"].split("/")[0]
    assert any("refrigerator" in j["child"] for j in joints), "the fridge door is missing"


def test_joint_children_are_scene_nodes_so_the_sliders_drive_something(loaded):
    """Every joint's child must be a node the preview can find AND drive without a jump.

    This is the inversion of a test that used to assert the opposite. The loader once flattened the
    USD -- world transforms baked into vertices, one identity node per Gprim -- so a joint's child
    body (refrigerator/door, base_cabinet/drawer_0_0: Xforms owning no geometry themselves) was not
    a node at all. kitchen_preview's setJoint() does
    `var node = scene.getObjectByName(j.node); if (!node) return;` and buildJointUI only reports
    "no movable joints" when the list is EMPTY, so all 27 of kitchen 1313's sliders moved nothing
    and said nothing. load_kitchen_scene now reproduces the hierarchy, so they are real nodes.

    Three things are checked, because any one of them alone lets a slider be silently inert:
      1. the child is a graph node;
      2. its SANITIZED name resolves -- the preview calls getObjectByName on
         sanitize_three_name(child), so a node that exists under a name three.js mangles
         differently is still unreachable, and a name shared with another node would drive the
         wrong part;
      3. the node's LOCAL matrix equals the joint's reported `origin`. The preview sets
         child.matrix = origin @ T(q), so if origin disagreed with where the part already sits,
         touching a slider would teleport the door before rotating it.
    """
    import numpy as np
    import kitchen_preview
    import kitchen_usd_load

    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints reconstructed -- this test would be vacuous"
    nodes = set(loaded.graph.nodes)

    missing = [j["child"] for j in joints if j["child"] not in nodes]
    assert not missing, (
        f"{len(missing)}/{len(joints)} joint children are not scene nodes (e.g. {missing[:3]}) -- "
        f"their sliders would move nothing; is load_kitchen_scene flattening the hierarchy again?"
    )

    sanitized = [kitchen_preview.sanitize_three_name(n) for n in nodes]
    counts = Counter(sanitized)
    unresolvable = [j["child"] for j in joints if counts[
        kitchen_preview.sanitize_three_name(j["child"])] != 1]
    assert not unresolvable, (
        f"{len(unresolvable)} joint children do not resolve to exactly one three.js node name "
        f"(e.g. {unresolvable[:3]}) -- getObjectByName would miss them or hit the wrong node"
    )

    for j in joints:
        parent = j["child"].rsplit("/", 1)[0] if "/" in j["child"] else loaded.graph.base_frame
        local = loaded.graph.get(frame_to=j["child"], frame_from=parent)[0]
        assert np.allclose(local, np.array(j["origin"], dtype=float), atol=1e-9), (
            f"{j['name']}: the scene node's local matrix is not the origin the preview would "
            f"drive it from, so the part would jump the moment the slider moves"
        )


def test_the_hierarchy_composes_to_the_same_world_bounds_as_the_flattened_loader(loaded):
    """The whole kitchen must end up exactly where baking world transforms into vertices put it.

    Reproducing USD's transform hierarchy means placement is now composed from a chain of local
    matrices instead of read straight off ComputeLocalToWorldTransform, and the classic failure is
    a transform applied twice (parent's own matrix folded into the child as well) or transposed
    wrongly somewhere in the chain -- both of which move or explode the scene while every count
    still looks right. These extents were measured on kitchen 1313 with the flattened loader,
    before the hierarchy existed, so they are an independent reference, not a snapshot of the
    current code's own output.
    """
    lo, hi = loaded.bounds
    extents = hi - lo
    expected = (2.5738, 3.0279, 2.2218)   # metres, flattened loader, kitchen 1313
    for axis, got, want in zip("xyz", extents, expected):
        assert abs(got - want) < 1e-3, (
            f"{axis} extent {got:.4f} m, expected {want:.4f} m -- the composed hierarchy does not "
            f"reproduce the flattened placement; a local or parent transform is wrong or doubled"
        )


def test_every_gprims_composed_world_transform_matches_the_usd(loaded):
    """Per-prim version of the bounds check, against USD's own answer.

    Bounds are an aggregate: a pair of prims could cancel each other's error and still span the
    right box. This compares each Gprim's graph-composed world matrix with what USD computes for
    the same prim (ComputeLocalToWorldTransform), which is the definitive statement that the local
    matrices and parent links are right. Note this is the routine the loader deliberately no longer
    calls -- it reads local transforms only -- so the two sides are genuinely independent.
    """
    import numpy as np
    from pxr import Usd, UsdGeom
    import kitchen_usd_load

    stage = Usd.Stage.Open(kitchen_usd_load.rotation_usd_path(KITCHEN))
    checked = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        name = kitchen_usd_load._node_name(prim)
        if name not in loaded.geometry:
            continue        # prims with no usable geometry; the count test covers losses
        usd_world = np.array(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            dtype=float,
        ).T                 # USD is row-major with translation in the last ROW
        assert np.allclose(usd_world, loaded.graph.get(name)[0], atol=1e-6), (
            f"{name}'s composed world transform disagrees with the USD's own"
        )
        checked += 1
    assert checked > 400, f"only compared {checked} prims -- fixture kitchen changed?"


def test_the_hierarchy_survives_the_glb_export_the_browser_actually_loads(loaded):
    """The joint children must still be nodes after export, not just in the trimesh scene.

    The composer renders through trimesh's scene_to_html, i.e. a GLB. Nothing in the browser sees
    scene.graph -- setJoint() resolves names against what GLTFLoader built. An exporter that pruned
    transform-only nodes (which is exactly what every joint child is) would leave the sliders inert
    again while every in-memory assertion above still passed, so this checks the artefact itself.
    """
    import json
    import struct
    import kitchen_preview
    import kitchen_usd_load

    glb = loaded.export(file_type="glb")
    assert glb[:4] == b"glTF", "not a GLB"
    json_len = struct.unpack("<I", glb[12:16])[0]
    gltf = json.loads(glb[20 : 20 + json_len].decode("utf-8"))
    exported = {
        kitchen_preview.sanitize_three_name(n["name"]) for n in gltf["nodes"] if n.get("name")
    }
    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints to check"
    missing = [
        j["child"] for j in joints
        if kitchen_preview.sanitize_three_name(j["child"]) not in exported
    ]
    assert not missing, (
        f"{len(missing)}/{len(joints)} joint children are absent from the exported GLB "
        f"(e.g. {missing[:3]}) -- the browser cannot find them, so the sliders are inert"
    )


def test_joint_names_are_unique_so_no_slider_drives_another_part():
    """Two joints sharing a name means one slider silently drives the other one's part.

    kitchen_preview registers joints into a JS Map keyed by name
    (`jointNodes.set(j.name, {node, origin})`) and looks them up the same way, and a Map OVERWRITES
    on a duplicate key -- so of two joints with one name, only the last registered is drivable and
    the other's slider moves it instead. The row label comes from j.child and the name appears only
    in the tooltip, which is why the labels all looked right while the wrong doors moved.

    USD leaf names repeat across furniture by design: kitchen 1313's 27 joints share just 14 leaf
    names ('corpus_to_door_0_0' occurs 7 times, once per wall cabinet), so 13 of 27 sliders were
    unusable. The loader therefore names a joint by its prim path minus '/world/', unique by
    construction. collect_joints has no such problem -- scene_synthesizer's own joint names are
    already scene-unique.

    The name must also identify its child one-to-one: a unique name pointing at a child that
    another joint also drives would be the same defect wearing a different hat.
    """
    import kitchen_usd_load

    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints reconstructed"

    names = [j["name"] for j in joints]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    assert not duplicated, (
        f"{len(names) - len(set(names))} of {len(names)} joints would be lost to the preview's "
        f"Map: {duplicated[:5]} are shared by more than one joint, so their sliders drive whichever "
        f"part was registered last"
    )

    by_name = {j["name"]: j["child"] for j in joints}
    assert len(set(by_name.values())) == len(by_name), (
        "two joint names resolve to the same child part, so one slider moves another's target"
    )


def test_the_axis_is_rotated_out_of_the_raw_token_vector():
    """The blunt statement that the joint-frame -> child-frame rotation happens at all.

    UsdPhysics stores `physics:axis` as a token (X/Y/Z) in the JOINT's frame; the preview drives
    T(q) in the CHILD's frame, so the token has to be rotated by physics:localRot1. Before that fix
    every one of kitchen 1313's 27 axes was exactly its raw token vector and this test failed;
    afterwards 17 of 27 differ. It is deliberately crude -- it proves the conversion is wired in,
    and the two tests below are the ones that prove it is the RIGHT conversion.
    """
    import numpy as np
    import kitchen_usd_load

    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints reconstructed"
    raw = list(kitchen_usd_load._AXIS_VECTORS.values())
    rotated = [
        j for j in joints
        if not any(np.allclose(j["axis"], v, atol=1e-6) for v in raw)
    ]
    assert rotated, (
        "every stored axis is still a raw token vector -- physics:localRot1 is not being applied, "
        "so joints whose joint frame is rotated relative to their child will drive the wrong axis"
    )


def test_axes_agree_with_collect_joints_on_a_freshly_built_kitchen():
    """Cross-check against the OTHER producer of joints, on a kitchen built from scratch.

    kitchen_build.collect_joints reads scene_synthesizer's own edge metadata, where the axis is
    already in the child's frame -- a completely separate route to the same quantity, which is why
    the freshly-generated composer path never had this bug. Rebuilding a kitchen of the same type
    in memory and matching joints by name is therefore a real check, unlike asserting that the
    stored axis equals localRot1 * token recomputed the same way, which would only restate the
    implementation.

    Axes live in each child's LOCAL frame, so they are invariant to the whole-kitchen rotation and
    to the furniture's placement, and should agree between the two routes. Agreement is required UP
    TO SIGN: the fresh build is a different random draw of the same kitchen type, so a cabinet that
    happens to share a name can have its door hinged on the other side, which flips the axis. That
    is a real difference between two kitchens, not a disagreement about the conversion. (Measured
    on seed 0: 22 of 27 joints match by name, and all 22 agree up to sign -- 15 with the sign too.
    Against the raw token vector only 15 of 22 agree even up to sign, which is the failure this
    catches.)
    """
    import numpy as np
    import kitchen_build
    import kitchen_usd_load

    kitchen_type = kitchen_usd_load.load_kitchen_data(KITCHEN)["kitchen_type"]
    builder = kitchen_build.KITCHEN_BUILDERS.get(kitchen_type)
    assert builder is not None, f"no builder for kitchen type {kitchen_type!r}"
    fresh = builder(seed=0, counter_height=0.95)   # the generator's own build call, minus objects
    reference = {j["name"]: j for j in kitchen_build.collect_joints(fresh)}
    assert reference, f"a fresh {kitchen_type} kitchen produced no joints to compare against"

    # Both sides name a joint '<group>/<joint name>' -- collect_joints from scene_synthesizer's own
    # metadata, the loader from the joint prim's path minus '/world/' -- so names match directly.
    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    matched = [(j, reference[j["name"]]) for j in joints if j["name"] in reference]
    assert len(matched) >= 15, (
        f"only {len(matched)} of {len(joints)} joints matched a freshly built {kitchen_type} "
        f"kitchen by name -- the comparison is too thin to mean anything; has the naming changed?"
    )
    disagreed = [
        (j["name"], j["axis"], ref["axis"])
        for j, ref in matched
        if not np.allclose(np.abs(j["axis"]), np.abs(np.asarray(ref["axis"], float)), atol=1e-6)
    ]
    assert not disagreed, (
        f"{len(disagreed)}/{len(matched)} axes disagree with collect_joints even up to sign, "
        f"e.g. {disagreed[:3]} -- the USD axis is not being brought into the child's frame"
    )


def test_drawers_slide_out_of_their_cabinet(loaded):
    """The physical check, and the one that pins the SIGN and the choice of rotation direction.

    A drawer must travel along the direction its cabinet opens. That direction is read off geometry
    alone -- from the corpus's own back panel, outward = normalize(corpus_centre - back_centre) --
    so it knows nothing about quaternions and cannot agree with a wrong convention by construction.
    Driving the joint to its `upper` limit the way the preview does (displacement
    origin[:3,:3] @ axis*q, composed through the parent's world transform) must move the drawer
    along that outward direction.

    This is what settled localRot1 * v against its inverse: both are horizontal and both look
    plausible in a still frame, but under the inverse all 7 drawers slide sideways through their
    cabinet's side panels (0/7 outward), and under the raw token they slide straight up (0/7).
    """
    import numpy as np
    from pxr import Usd, UsdPhysics
    import kitchen_usd_load

    stage = Usd.Stage.Open(kitchen_usd_load.rotation_usd_path(KITCHEN))
    corpus_of = {}
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            corpus_of[kitchen_usd_load._node_name(stage.GetPrimAtPath(body1[0]))] = (
                kitchen_usd_load._node_name(stage.GetPrimAtPath(body0[0]))
            )

    def world_centre(prefix):
        """Centre of the world-space bounds of every mesh under `prefix`, or None if there is none."""
        boxes = []
        for name in loaded.graph.nodes:
            if not (name == prefix or name.startswith(prefix + "/")) or name not in loaded.geometry:
                continue
            m = loaded.graph.get(name)[0]
            b = loaded.geometry[name].bounds
            corners = np.array([[x, y, z] for x in b[:, 0] for y in b[:, 1] for z in b[:, 2]])
            boxes.append((m[:3, :3] @ corners.T).T + m[:3, 3])
        if not boxes:
            return None
        pts = np.vstack(boxes)
        return (pts.min(axis=0) + pts.max(axis=0)) / 2.0

    checked, wrong = 0, []
    for j in kitchen_usd_load.load_kitchen_joints(KITCHEN):
        if j["type"] != "prismatic":
            continue
        corpus = corpus_of.get(j["child"])
        if corpus is None:
            continue
        centre, back = world_centre(corpus), world_centre(f"{corpus}/back")
        if centre is None or back is None:
            continue                      # no back panel to read an opening direction from
        outward = centre - back
        outward[2] = 0.0                  # a drawer opens horizontally
        if np.linalg.norm(outward) < 1e-6:
            continue
        outward /= np.linalg.norm(outward)

        parent = j["child"].rsplit("/", 1)[0] if "/" in j["child"] else loaded.graph.base_frame
        origin = np.asarray(j["origin"], dtype=float)
        travel = origin[:3, :3] @ (np.asarray(j["axis"], dtype=float) * j["upper"])
        travel = loaded.graph.get(parent)[0][:3, :3] @ travel
        if np.linalg.norm(travel) < 1e-9:
            continue
        travel /= np.linalg.norm(travel)

        checked += 1
        if float(np.dot(travel, outward)) < 0.9:
            wrong.append((j["name"], j["group"], travel.round(3).tolist(), outward.round(3).tolist()))

    assert checked >= 4, f"only {checked} drawers had a corpus back panel to check against"
    assert not wrong, (
        f"{len(wrong)}/{checked} drawers do not open outward -- (joint, group, travel, outward): "
        f"{wrong[:3]}"
    )


def test_revolute_limits_are_radians_not_degrees():
    """UsdPhysics stores revolute limits in DEGREES; collect_joints' consumers expect radians.
    A 90-degree door left unconverted would swing 90 radians."""
    import math
    import kitchen_usd_load
    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    revolute = [j for j in joints if j["type"] == "revolute"]
    assert revolute, "no revolute joints to check"
    for j in revolute:
        assert abs(j["upper"]) <= 2 * math.pi + 1e-6, (
            f"{j['name']} upper={j['upper']} looks like degrees, not radians"
        )


def test_joint_origin_is_the_child_rest_pose():
    """The loader takes each joint's origin from the child prim's own local transform, which is
    correct only because kitchens are exported with doors and drawers CLOSED (q=0). If a kitchen
    were ever exported with a joint open, this must fail instead of silently hinging a door in
    the wrong place.

    Checking this by recomputing GetLocalTransformation() on the same child prim (as an earlier
    version of this test did) is a tautology: both sides would be bit-identical whether or not
    the door was exported closed, because both are the same formula on the same data. Instead
    this derives each joint's q=0 pose through a route that never reads the child's own local
    transform at all -- the joint's own static physics frames, localPos0/localRot0 (the joint
    frame on the corpus/body0) and localPos1/localRot1 (the joint frame on the child/body1) --
    combined with the corpus's local transform. Because the corpus and the child share the same
    immediate USD parent (the furniture group, e.g. /world/refrigerator), that shared ancestor
    cancels out of the world-frame joint constraint, reducing the q=0 rest-pose relation to

        child_local = frame1^-1 @ frame0 @ corpus_local      (row-vector composition)

    This is independent of _local_matrix()'s formula, so it can genuinely disagree if the
    assumption is violated -- verified by deliberately rotating one door 45 degrees away from
    rest in a scratch copy of this USD and confirming exactly that joint's check fails while
    every other joint's still passes (see task-2-report.md for that run). On the real, unmodified
    kitchen 1313 the two routes agree to floating-point precision (previously measured max abs
    diff ~3e-8), which is this test passing for the right reason.
    """
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import kitchen_usd_load

    def frame_matrix(pos, quat):
        rot = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(quat))
        trans = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(pos))
        return np.array(rot * trans, dtype=float)  # raw USD row-major (untransposed)

    stage = Usd.Stage.Open(kitchen_usd_load.rotation_usd_path(KITCHEN))

    expected_by_child = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
        else:
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if not body0 or not body1:
            continue
        corpus_prim = stage.GetPrimAtPath(body0[0])
        child_prim = stage.GetPrimAtPath(body1[0])
        frame0 = frame_matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        frame1 = frame_matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        corpus_local = np.array(
            UsdGeom.Xformable(corpus_prim).GetLocalTransformation(), dtype=float
        )
        expected_by_child[str(child_prim.GetPath())] = np.linalg.inv(frame1) @ frame0 @ corpus_local

    joints = kitchen_usd_load.load_kitchen_joints(KITCHEN)
    assert joints, "no joints to check"
    checked = 0
    for j in joints:
        expected_raw = expected_by_child.get(f"/world/{j['child']}")
        assert expected_raw is not None, f"no UsdPhysics joint frames found for {j['child']}"
        expected_origin = expected_raw.T  # match _local_matrix()'s .T convention
        assert np.allclose(np.array(j["origin"], dtype=float), expected_origin, atol=1e-5), (
            f"{j['name']}'s origin does not match the q=0 pose implied by the joint's own "
            f"physics frames -- {j['child']} is not at rest in this USD"
        )
        checked += 1
    assert checked == len(joints), f"only verified {checked}/{len(joints)} joints"
