"""SimVLA: Kitchen scene generation.

The whole UI is a browser page served on 127.0.0.1 (SIMVLA_PREVIEW_PORT, default 8777) — setup,
3D preview, progress, prim picking and task composing. Nothing here uses Tk, so this runs on a node
with no DISPLAY: forward the port and the run drives your browser through every step.

This module owns the USD work and the linear director that sequences it. Everything that talks to
the browser lives in kitchen_wizard.py, which imports no Omniverse and is unit-tested.
"""

import glob
import json
import os
import random
import traceback

import kitchen_wizard

# The server comes up BEFORE Omniverse boots (~1 min), so the URL and the ssh -L line are on screen
# while it loads instead of after. kitchen_wizard imports no Omniverse, so this is safe here.
WIZARD = kitchen_wizard.WizardServer()

from kitchen_preview import auto_open_verdict     # noqa: E402  (after WIZARD on purpose)
# Pre-boot on purpose too: kitchen_gallery imports trimesh and stdlib only -- no scene_synthesizer,
# which must not be reached before Isaac Sim boots. It is imported AFTER WIZARD for the same reason
# auto_open_verdict is: the server has to be listening before anything else costs time.
import kitchen_gallery                            # noqa: E402  (after WIZARD on purpose)

_open_ok, _open_reason = auto_open_verdict()
_opened = False
if _open_ok:
    try:
        _opened = WIZARD.open()
    except Exception as _exc:   # a browser that refuses to launch must not kill the whole run
        print(f"[wizard] could not launch a browser: {_exc}")
    if not _opened and not _open_reason.startswith("SIMVLA"):
        _open_reason = "no browser here could render it (see below)"
print(WIZARD.banner(opened=_opened, reason=_open_reason, title="Kitchen generator"), flush=True)

# Before the boot on purpose: unset, this costs the whole session at the composer (see
# ensure_hf_user). Here it costs nothing and is said out loud, while the tab is still loading.
_defaulted_hf_user = kitchen_wizard.ensure_hf_user()
if _defaulted_hf_user:
    print(
        f"  HF_USER was not set — using {_defaulted_hf_user!r} so the composer can import\n"
        f"  isaaclab.simvla. Export your real HuggingFace username before pushing a dataset.",
        flush=True,
    )

print("  Booting Isaac Sim…", flush=True)

from isaaclab.app import AppLauncher              # noqa: E402
app_launcher = AppLauncher({"headless": True})
_ = app_launcher.app


from scene_synthesizer.usd_import import get_scene_paths
from scene_synthesizer.exchange.usd_export import add_mdl_material, bind_material_to_prims
# pxr + stdlib only -- see the module docstring for why this is NOT imported from goal_generator
# (module-scope killed the build at 10 s; lazy-scope hung it at commit_kitchen).
from object_materials import OBJECT_MATERIALS, create_random_object_material

#: See goal_generator._safe_texture_scale for the full account. scene_synthesizer's
#: add_mdl_material always writes texture_scale as Float2; Paint_Eggshell.mdl declares it a float,
#: so the binding is refused and every prim using it -- which is EVERY wall variant -- renders
#: black. This is the path build_furnished_mug_kitchen actually takes (ksg.commit_kitchen), so the
#: fix has to exist here too, not only in goal_generator.
_FLOAT_TEXTURE_SCALE_MDLS = ("Paint_Eggshell.mdl",)


def _safe_texture_scale(mtl_url, texture_scale):
    return None if any(m in mtl_url for m in _FLOAT_TEXTURE_SCALE_MDLS) else texture_scale

from scene_synthesizer import datasets

from kitchen_build import (
    ALL_TABLE_VARIANTS,
    CHAIR_VARIANTS,
    CHAIR_VARIANT_BY_UID,
    DEFAULT_FURNITURE_MATERIAL,
    FURNITURE_MATERIAL_GROUPS,
    FURNITURE_MATERIAL_SWATCHES,
    FURNITURE_SCALES,
    GEOMETRY2MATERIAL,
    KITCHEN_BUILDERS,
    MESH_TABLE_KEY_PREFIX,
    TABLE_VARIANT_BY_KEY,
    URL_MDL_MATERIAL,
    WALL_CABINET_DYNAMIC,
    # Private on purpose, and shared on purpose: see the `placed` filter in run_wizard. "What the
    # wizard placed as furniture" and "what is not a BODex object to rotate" must be one answer.
    _is_furniture,
    # Private on purpose too, and for the same class of reason: the chair edit page stands the
    # chosen table at the seat this very function would put the chair in, and re-deriving that
    # offset from CHAIR_TABLE_GAP_M and a footprint would be a second copy of the ring rule --
    # which is exactly how the page comes to show a placement the build does not make. See
    # _chair_seat_standoff_m.
    _seats_around_table,
    add_room_shell,
    apply_placements,
    build_kitchen,
    chair_facing_target,
    chair_footprint_m,
    chair_width_floor_m,
    chair_width_floor_scale,
    collect_joints,
    face_chair_toward,
    furniture_material_overrides,
    furniture_placement_problems,
    matching_meshes,
    material_display_names,
    pick_materials,
)
# Render-only entry points: this module never owns a server's lifetime. WIZARD (above) is the one
# server for the whole run, and it is handed each page as a string.
from kitchen_preview import DEFAULT_ROBOT as PREVIEW_DEFAULT_ROBOT
from kitchen_preview import ROBOTS as PREVIEW_ROBOTS
from kitchen_preview import render_page as preview_render_page
from task_composer import composer_page


# ========== GLOBAL OMNIVERSE APP ==========

from pxr import UsdGeom, Gf, Usd, Sdf, UsdPhysics, PhysxSchema
import omni.usd

# === simvla path resolution (auto-added) ===
import os as _simvla_os
from pathlib import Path as _SimvlaPath
def _simvla_find_repo_root():
    env = _simvla_os.environ.get("SIMVLA_REPO_ROOT")
    if env:
        return env
    for p in _SimvlaPath(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return str(p)
    raise RuntimeError("Cannot find repo root; set SIMVLA_REPO_ROOT env var")
SIMVLA_REPO_ROOT = _simvla_find_repo_root()
# === end simvla path resolution ===

# kitchen_wizard's /emit_goals handler builds the sbatch script from the environment (it cannot
# import this module -- that would boot Omniverse in a page handler), so the root it resolved here
# has to be visible there. setdefault, so an explicit SIMVLA_REPO_ROOT still wins.
os.environ.setdefault("SIMVLA_REPO_ROOT", SIMVLA_REPO_ROOT)



# ==================== COLLISION APPROX (convexDecomposition) ====================
# Your standalone script was doing:
#   UsdPhysics.MeshCollisionAPI.approximation = 'convexDecomposition'
# for a specific prim (e.g., bottle0) inside a kitchen USD.
#
# Here we integrate that into this pipeline so the *generated* kitchen USD,
# and therefore all 12 rotated variants, inherit the updated collision approximation.

# If you want to also force convexDecomposition on some kitchen fixtures (not just placed objects),
# add substrings here (case-insensitive), e.g. ["sink_cabinet", "dishwasher"].
CONVEX_DECOMP_EXTRA_SUBSTRINGS = []

#: WHICH PhysX approximation the placed object's collider gets. Unset = convexDecomposition, which
#: is what every kitchen before 2026-08-26 was built with and is byte-identical.
#:
#: WHY IT IS SELECTABLE. A convex decomposition cannot follow a thin concave shell. The bowl's
#: visual wall is 0.009-0.015 m thick (measured off the mesh at placement scale) while the jaws
#: come to rest at fingsep 0.0303 (kitchen 1208, job 2100585 env 13) -- i.e. the fingers stop on
#: hull that is roughly 3 cm thick and visibly pass through the rendered wall. `sdf` tracks thin
#: concave geometry accurately and is PhysX's answer for crockery.
#:
#: NOT PROVEN to be the cause of the bowl's grasp failures -- the wall/fingsep mismatch is
#: consistent with it, not evidence for it. This knob exists so the hypothesis can be TESTED by
#: building one kitchen each way, rather than argued.
OBJ_COLLIDER_APPROX = os.environ.get("SIMVLA_OBJ_COLLIDER", "convexDecomposition").strip()
_VALID_APPROX = {"convexDecomposition", "convexHull", "sdf", "boundingCube", "boundingSphere",
                 "meshSimplification", "none"}
if OBJ_COLLIDER_APPROX not in _VALID_APPROX:
    raise SystemExit(f"SIMVLA_OBJ_COLLIDER={OBJ_COLLIDER_APPROX!r}; expected one of "
                     f"{', '.join(sorted(_VALID_APPROX))}")


def ensure_convexDecomposition_approx_on_prim(prim, *, verbose: bool = False) -> bool:
    """Ensure MeshCollisionAPI.approximation is 'convexDecomposition' on the given prim.

    Returns True if it changed the authored value, False otherwise.
    """
    # Apply MeshCollisionAPI if missing (safe even if it ends up unused)
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(prim)

    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    if not mesh_api:
        return False

    approx_attr = mesh_api.GetApproximationAttr()
    if not approx_attr:
        approx_attr = mesh_api.CreateApproximationAttr()

    want = OBJ_COLLIDER_APPROX
    current = approx_attr.Get()
    if current != want:
        approx_attr.Set(want)
        if verbose:
            print(f"[ObjCollider][UPDATE] {prim.GetPath()}: {current!r} -> {want!r}")
        return True
    else:
        if verbose:
            print(f"[ObjCollider][OK]     {prim.GetPath()}: already {want!r}")
        return False


def apply_convex_decomposition_approx(
    stage: Usd.Stage,
    *,
    root_paths=None,
    match_substrings=None,
    meshes_only: bool = True,
    verbose: bool = False,
):
    """Apply convexDecomposition collision approximation to meshes that match.

    Matching rules:
      - If root_paths is provided: any prim under any root path matches.
      - If match_substrings is provided: any prim path/name containing a substring matches.

    Notes:
      - We default to meshes_only=True so we don't spam APIs onto Xforms.
      - Applying this on the *base* kitchen stage before exporting rotations means all rotated USDs inherit it.
    """
    root_paths = root_paths or []
    match_substrings = match_substrings or []

    # Normalize for string matching
    root_paths = [str(p) for p in root_paths]
    subs = [s.lower() for s in match_substrings]

    def _matches(prim) -> bool:
        p = str(prim.GetPath())
        n = prim.GetName().lower()

        if root_paths:
            if any(p == rp or p.startswith(rp + "/") for rp in root_paths):
                return True

        if subs:
            pl = p.lower()
            if any(s in pl or s in n for s in subs):
                return True

        return False

    checked = changed = 0
    for prim in stage.Traverse():
        if meshes_only and not prim.IsA(UsdGeom.Mesh):
            continue
        if not _matches(prim):
            continue

        checked += 1
        if ensure_convexDecomposition_approx_on_prim(prim, verbose=verbose):
            changed += 1

    print(f"[ConvexDecomp] checked {checked} mesh prim(s), updated {changed} to 'convexDecomposition'.")
    return {"checked": checked, "changed": changed}


def apply_materials(stage, picks, overrides=None):
    """Bind already-chosen MDL materials to the prims matching each geometry regex.

    Two passes, in order:

      1. GEOMETRY2MATERIAL -- the general one. Every prim in the kitchen gets the group its own
         path says it belongs to, table and chairs included (both map to 'floor').
      2. `overrides` -- {regex -> group} for the individual pieces of furniture the author gave a
         material of their own (kitchen_build.furniture_material_overrides). None or {} is the
         ordinary case and makes this pass a no-op.

    THE SECOND PASS WINS BECAUSE IT IS SECOND, and that is a property of the binding rather than an
    assumption about regex specificity: bind_material_to_prims calls
    UsdShade.MaterialBindingAPI(prim).Bind(), which writes the single `material:binding`
    relationship on that prim, so binding a prim again replaces its target. The narrower regex
    reaches a subset of the prims the first pass already bound and re-points exactly those.
    apply_materials has always depended on this WITHIN pass 1 -- GEOMETRY2MATERIAL's furniture
    entries are placed after ".*handle.*" so a mesh whose OBJ names a part "handle" still ends up
    wood -- and test_a_later_bind_replaces_an_earlier_one_on_the_same_prim holds it on a real
    exported stage rather than on that reasoning.
    """
    for geom_regex, group in list(GEOMETRY2MATERIAL.items()) + list((overrides or {}).items()):
        pick = picks.get(group)
        if pick is None:
            print(f"Warning: No material picked for group {group!r}")
            continue

        paths = get_scene_paths(
            stage=stage,
            prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
            scene_path_regex=geom_regex,
        )
        mtl_url, mtl_name, texture_scale = pick
        texture_scale = _safe_texture_scale(mtl_url, texture_scale)
        mtl = add_mdl_material(
            stage=stage,
            mtl_url=URL_MDL_MATERIAL + mtl_url,
            mtl_name=mtl_name,
            texture_scale=texture_scale,
        )
        bind_material_to_prims(stage=stage, material=mtl, prim_paths=paths)

    # PASS 3: the PLACED OBJECTS. GEOMETRY2MATERIAL covers fixtures only, so a mug or an apple
    # renders with whatever material its BODex mesh carries -- byte-identical in every kitchen the
    # scene generator has ever produced. For sim-to-real that is backwards: the one thing the
    # policy has to locate is the one thing that never varies.
    #
    # Runs LAST, so it re-points only the object prims and cannot disturb a fixture binding (see
    # the note above on why a later bind wins). A type with no OBJECT_MATERIALS entry gets None
    # back and is left completely alone, so every object except the apple is unchanged today.
    for obj_type in OBJECT_MATERIALS:
        obj_paths = get_scene_paths(
            stage=stage,
            prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
            scene_path_regex=f"/world/{obj_type}[0-9]+/.*",
        )
        if not obj_paths:
            continue
        mtl = create_random_object_material(
            stage, f"/world/Looks/{obj_type.capitalize()}_{random.randint(0, 10**9)}", obj_type)
        if mtl is None:
            continue
        bind_material_to_prims(stage=stage, material=mtl, prim_paths=obj_paths)
        print(f"[materials] {obj_type}: randomised colour bound to {len(obj_paths)} prim(s)",
              flush=True)


def commit_kitchen(kitchen, picks, kitchen_num, kitchen_data, material_overrides=None):
    """Export the accepted kitchen. The first thing in this pipeline that touches disk."""
    usd_filename = f"{KITCHEN_DIR}/kitchen_{kitchen_num:02d}.usd"

    # Walls go on HERE and not in build_kitchen: everything upstream of this point is what the
    # wizard previews, and a room closed on four sides previews as an opaque box. Seeded by the
    # kitchen number so regenerating kitchen 42 gives back the same room.
    walls = add_room_shell(kitchen, seed=kitchen_num)
    if walls:
        # Read by simvla_data_generator, in a different process, to decide whether to emit its own
        # make_walls_from_bounds slabs. The persisted fact rather than the flag: emission and
        # generation are separate runs and ROOM_SHELL may differ between them.
        kitchen_data["room_shell"] = True

    stage = kitchen.export(file_type="usd")
    # material_overrides is the author's per-piece choices, as {prim regex -> group}. It reaches
    # the stage only here, because a material is not geometry: nothing between build_kitchen and
    # this line can carry it, and the preview cannot show it either (the preview is a trimesh
    # scene; MDLs exist only in USD). See kitchen_build.furniture_material_overrides.
    apply_materials(stage, picks, material_overrides)
    stage.Export(usd_filename)

    bodex_dir = os.path.join(KITCHEN_DIR, "bodex")
    os.makedirs(bodex_dir, exist_ok=True)
    json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num:02d}.json")
    with open(json_path, "w") as f:
        json.dump(kitchen_data, f, indent=4)

    return usd_filename, json_path


def resolve_grasp_thumbs(objects, kitchen_data):
    """{object node_id -> *_segments_thumbnails folder} for every object with rendered BODex thumbnails.

    Uses the SAME derivation plan_arm_grasp does — load_grasp_file(mesh_path, "sim_parallel") +
    "_segments_thumbnails" — so the folder the composer writes selected_indices.json into is exactly
    the one the headless run later reads. kitchen_data maps an object's label to its BODex mesh path
    (the value later stamped onto the prim's BODex_path attr). An object whose grasp base or thumbnail
    folder can't be resolved (no BODex data, or no rendered thumbnails yet) is omitted, so it simply
    won't show a picker rather than showing a broken one.
    """
    from isaaclab.simvla.utils import load_grasp_file

    thumbs = {}
    for obj in objects:
        mesh_path = kitchen_data.get(obj["label"])
        if not mesh_path:
            continue
        try:
            base = load_grasp_file(mesh_path, "sim_parallel")
        except Exception:
            base = None
        if not base:
            continue
        folder = f"{base}_segments_thumbnails"
        if os.path.isdir(folder):
            thumbs[obj["node_id"]] = folder
    return thumbs


# =============== STEP 2: rotate & register (with a prim-picker callback) ===============
def rotate_and_register_envs(kitchen_num: int, progress=None, ask_prim=None):
    """Export the 12 rotation USDs and register an env for each. Returns the paths it EXPORTED.

    That return value is the only trustworthy record of what this run produced, which is why it
    exists: every one of the six give-up paths below returns normally, and the caller cannot tell
    them apart from a full run by looking at the directory afterwards. Nor can it look at
    timestamps -- fix_missing_fixed_joint_targets_and_wall_cabinets runs next, globs every
    kitchen_<N>_*.usd on disk (a previous run's included) and re-saves each one, so by the time
    anyone checks, twelve files exist and all twelve are newer than the base USD whether this run
    wrote them or not.
    """
    bodex_dir = os.path.join(KITCHEN_DIR, "bodex")
    json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num:02d}.json")

    if not os.path.exists(json_path):
        print(f"[Step2] JSON not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        kitchen_data = json.load(f)

    base_usd = f"{KITCHEN_DIR}/kitchen_{kitchen_num:02d}.usd"

    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(base_usd)
    init_file = f"{SIMVLA_REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/__init__.py"

    if not success:
        print(f"[Step2] Failed to open base USD: {base_usd}")
        return []

    print(f"[Step2] Opened base kitchen USD: {base_usd}")
    stage = usd_context.get_stage()

    for obj, fname in kitchen_data.items():
        if obj == "kitchen_type":
            continue
        old_path = f"/world/{obj}0"
        new_path = f"/world/{obj}"
        if stage.GetPrimAtPath(old_path):
            Sdf.CopySpec(stage.GetRootLayer(), old_path, stage.GetRootLayer(), new_path)
            stage.RemovePrim(old_path)
        obj_prim = stage.GetPrimAtPath(f"/world/{obj}")
        if obj_prim:
            attr = obj_prim.CreateAttribute("BODex_path", Sdf.ValueTypeNames.String)
            attr.Set(fname)

    candidate_prims = []
    for obj in kitchen_data:
        if obj == "kitchen_type":
            continue
        prim_path = f"/world/{obj}"
        if stage.GetPrimAtPath(prim_path):
            candidate_prims.append(prim_path)


    # ---- Step2.5: collision approximation (convexDecomposition) ----
    # Apply to all placed object prim hierarchies so all 12 rotated USDs inherit it.
    # Add extra kitchen substrings via CONVEX_DECOMP_EXTRA_SUBSTRINGS (top of file) if needed.
    apply_convex_decomposition_approx(
        stage,
        root_paths=candidate_prims,
        match_substrings=CONVEX_DECOMP_EXTRA_SUBSTRINGS,
        meshes_only=True,
        verbose=False,
    )

    if not candidate_prims:
        print("[Step2] No candidate prims found to rotate.")
        return []

    # ask_prim(candidate_paths) -> the chosen path, or None to cancel. A callback rather than a UI
    # call so this function never knows what the UI is; without one there is no way to ask, and we
    # stop exactly as a cancel does — nothing written.
    if ask_prim is None:
        print("[Step2] No prim picker available; skipping rotation.")
        return []
    rotate_obj = ask_prim(candidate_prims)
    if rotate_obj is None:
        print("[Step2] Rotation cancelled by user.")
        return []

    print(f"[Step2] Selected prim to rotate: {rotate_obj}")

    obj_prim = stage.GetPrimAtPath(rotate_obj)
    if not obj_prim:
        print(f"[Step2] Prim not found: {rotate_obj}")
        return []

    xformable = UsdGeom.Xformable(obj_prim)
    xform_ops = xformable.GetOrderedXformOps()

    translate_ops = [
        op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    translate_op = translate_ops[0] if translate_ops else None

    rotate_ops = [op for op in xform_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    rotate_op = rotate_ops[0] if rotate_ops else None

    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()

    if translate_op:
        xformable.SetXformOpOrder([translate_op, rotate_op])
    else:
        xformable.SetXformOpOrder([rotate_op])

    written = []
    for i in range(12):
        angle = i * 30.0
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, angle))

        usd_out = f"{KITCHEN_DIR}/kitchen_{kitchen_num:02d}_{i:02d}.usd"
        stage.GetRootLayer().Export(usd_out)
        written.append(usd_out)
        print(f"[Step2] Saved rotated kitchen USD: {usd_out}")
        if progress:
            progress("Building rotations", i, 12)

        with open(init_file, "r") as f:
            lines = f.readlines()
        lines.append(
            f"""
gym.register(
    id="Isaac-Kitchen-v{kitchen_num:02d}-{i:02d}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.kitchen_{kitchen_num:02d}_{i:02d}:AnubisKitchenEnvCfg",
    }},
)
"""
        )
        with open(init_file, "w") as f:
            f.writelines(lines)

    print("[Step2] All 12 rotated USDs created and envs registered.")
    return written


# =============== HELPERS FOR WALL_CABINET STATICIZATION ===============
# Whether this staticization pass (below) runs at all is controlled by
# kitchen_build.WALL_CABINET_DYNAMIC, imported in the kitchen_build import block above.
EXCEPT_COLLISION_KEYS = ("mug0", "sink_cabinet", "dishwasher", "countertop_dishwasher")
EXCEPT_DYNAMIC_KEYS   = ("mug0", "sink_cabinet")
TARGET_KEYS           = ("wall_cabinet",)  # wall_cabinet, wall_cabinet_0, etc.

def path_has_any(p: str, keys) -> bool:
    p = p.lower()
    return any(k in p for k in keys)

def is_collision_exception(p: str) -> bool:
    return path_has_any(p, EXCEPT_COLLISION_KEYS)

def is_dynamic_exception(p: str) -> bool:
    return path_has_any(p, EXCEPT_DYNAMIC_KEYS)

def is_wall_cabinet_prim(p: str) -> bool:
    return path_has_any(p, TARGET_KEYS)


def can_carry_collision(prim) -> bool:
    """Whether a collider authored on this prim would describe a SHAPE rather than a subtree.

    UsdPhysicsCollisionAPI is defined on geometry. Applied to an Xform (or to a typeless scope such
    as the `simplified_obj` group the OBJ importer wraps every mesh in), omni.physx still builds a
    collider for it -- out of the MERGED triangles of everything underneath -- and since such a prim
    carries no MeshCollisionAPI the approximation parses as None, which a dynamic body cannot use, so
    PhysX falls back to a CONVEX HULL of the whole subtree.

    That hull is the bug this guard exists to prevent. The mug's leaf mesh is authored
    convexDecomposition by apply_convex_decomposition_approx, which correctly opens the gap between
    body and handle -- but the ancestor hulls are additional shapes on the same rigid body, and the
    union of a decomposition with a hull of the same mesh IS the hull. The handle gap is filled back
    in, and the jaws have to span the full silhouette instead of the 0.0687 m cup body.

    Skipping non-geometry costs no coverage, and that is a property of the matcher rather than an
    assumption about these particular assets: the exception tests are SUBSTRING TESTS ON THE FULL
    PATH, so every descendant of a matching prim matches too (its path contains the parent's). Every
    leaf mesh under an exception is therefore reached on its own and gets its own collider.
    """
    return prim.IsA(UsdGeom.Gprim)


# =============== STEP 3+4: fixed joints + wall_cabinet cleanup ===============
def fix_missing_fixed_joint_targets_and_wall_cabinets(kitchen_num: int, progress=None):
    folder_path = KITCHEN_DIR
    pattern = f"{folder_path}/kitchen_{kitchen_num:02d}_*.usd"
    files = glob.glob(pattern)

    if not files:
        print(f"[Step3] No rotated USDs found for kitchen {kitchen_num:02d} at {pattern}")
        return

    for file_index, file in enumerate(files):
        # Reported before any of the validation `continue`s below so every file counts once,
        # even one that fails validation -- otherwise the progress bar in
        # _run_pipeline_with_progress can finish short of its 24 steps with no explanation.
        if progress:
            progress("Fixing joints", file_index, len(files))

        usd_context = omni.usd.get_context()
        success = usd_context.open_stage(file)

        parts = file.split("/")[-1].split("_")
        if len(parts) != 3:
            continue

        new_target_path = "/Root/" + file.split("/")[-1].removesuffix(".usd")

        if not success:
            print(f"[Step3] Failed to open {file} USD stage.")
            continue

        stage = usd_context.get_stage()
        if stage.GetPrimAtPath(new_target_path).GetPath().isEmpty:
            new_target_path = "/world"
            print(f"[Step3] Processing {file}")
        else:
            print(f"[Step3] {file} is not in an appropriate state.")
            continue

        # ---- Part A: original fixed-joint repair ----
        prims = [prim.GetPath() for prim in stage.Traverse()]
        fixed_joints = [prim for prim in prims if "fixed" in prim.name]

        for fixed_joint in fixed_joints:
            fixed_prim = stage.GetPrimAtPath(fixed_joint)
            rels = fixed_prim.GetRelationships()
            if not rels:
                continue
            if rels[0].GetTargets() == []:
                print(f"[Step3] {fixed_joint} missing target 0, converting parent to kinematic.")
                parent_prim = stage.GetPrimAtPath(fixed_joint.GetParentPath())
                rigid_api = UsdPhysics.RigidBodyAPI(parent_prim)
                rigid_api.CreateKinematicEnabledAttr(True)
                stage.RemovePrim(Sdf.Path(fixed_joint))

        # ---- Part B: wall_cabinet-only staticization (from final.py, restricted) ----
        if WALL_CABINET_DYNAMIC:
            print(f"[Step4] WALL_CABINET_DYNAMIC=True; leaving wall_cabinet physics intact in {file}.")
        else:

            # 1) Remove joints that involve wall_cabinet prims (but keep dynamic exceptions)
            removed_joints = 0
            for prim in list(stage.Traverse()):
                t = prim.GetTypeName() or ""
                if not t.endswith("Joint"):
                    continue

                joint_path = str(prim.GetPath())
                try:
                    j = UsdPhysics.Joint(prim)
                    body0 = j.GetBody0Rel().GetTargets()
                    body1 = j.GetBody1Rel().GetTargets()
                    related_paths = [joint_path] + [tgt.pathString for tgt in (body0 + body1)]

                    if not any(is_wall_cabinet_prim(p) for p in related_paths):
                        continue

                    keep = False
                    if any(is_dynamic_exception(tgt.pathString) for tgt in body0):
                        keep = True
                    if any(is_dynamic_exception(tgt.pathString) for tgt in body1):
                        keep = True
                    if is_dynamic_exception(joint_path):
                        keep = True

                except Exception:
                    continue

                if not keep:
                    stage.RemovePrim(prim.GetPath())
                    removed_joints += 1

            print(f"[Step4] Removed {removed_joints} joint(s) involving wall_cabinet prims in {file}.")

            # 2) Strip rigid bodies/articulations only from wall_cabinet prims
            removed_rb = removed_physx_rb = removed_artic = 0
            for prim in stage.Traverse():
                p = str(prim.GetPath())

                if is_wall_cabinet_prim(p) and not is_dynamic_exception(p):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                        removed_rb += 1
                    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                        removed_physx_rb += 1
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                        removed_artic += 1
                    if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                        prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                        removed_artic += 1

            print(f"[Step4] Removed RigidBodyAPI from {removed_rb} wall_cabinet prim(s) in {file}.")
            print(f"[Step4] Removed PhysxRigidBodyAPI from {removed_physx_rb} wall_cabinet prim(s) in {file}.")
            print(f"[Step4] Removed articulation APIs from {removed_artic} wall_cabinet prim(s) in {file}.")

            # 3) Remove collisions from wall_cabinet prims (except collision exceptions)
            removed_collision = removed_physx_collision = 0
            for prim in stage.Traverse():
                p = str(prim.GetPath())

                if is_collision_exception(p):
                    continue
                if not is_wall_cabinet_prim(p):
                    continue

                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                    removed_collision += 1
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
                    removed_physx_collision += 1

            print(f"[Step4] Removed CollisionAPI from {removed_collision} wall_cabinet prim(s) in {file}.")
            print(f"[Step4] Removed PhysxCollisionAPI from {removed_physx_collision} wall_cabinet prim(s) in {file}.")

        # 4) Ensure exceptions are correct (dynamic + collision)
        ensured_dynamic = ensured_collision = stripped_collision = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())

            if is_dynamic_exception(p):
                # The rigid body IS the Xform, so this half is deliberately outside the geometry
                # guard below -- kinematicEnabled belongs on the body, not on a shape.
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(False)
                    ensured_dynamic += 1
            elif not is_collision_exception(p):
                continue

            # Past here the prim is an exception of one kind or the other, and the only question
            # left is whether it is geometry.
            if not can_carry_collision(prim):
                # Not geometry, so it must not carry a collider -- see can_carry_collision for what
                # PhysX does with one. STRIPPED rather than merely not added, because this pass also
                # runs over kitchens an earlier version of it already wrote, and the subtree hull is
                # in those files: leaving it would mean the fix only reached kitchens built after it.
                stripped = False
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI); stripped = True
                if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI); stripped = True
                if stripped:
                    stripped_collision += 1
                continue

            need = False
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim); need = True
            if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                PhysxSchema.PhysxCollisionAPI.Apply(prim); need = True
            if need:
                ensured_collision += 1

        print(f"[Step4] Ensured kinematicEnabled=False on {ensured_dynamic} dynamic-exception bodies in {file}.")
        print(f"[Step4] Ensured collision present on {ensured_collision} exception prim(s) in {file}.")
        print(f"[Step4] Stripped subtree collision from {stripped_collision} non-geometry exception prim(s) in {file}.")

        omni.usd.get_context().save_as_stage(file)


# ======================= THE DIRECTOR =======================
# One linear pass, on the main thread, publishing each step to the browser and blocking on the
# answer. Omniverse's USD context is not thread-safe, so this thread must stay the only one that
# touches USD: the server's handler threads only read and write wizard state.

KITCHEN_DIR = str(
    _SimvlaPath(os.environ.get(
        "SIMVLA_ASSETS_DIR",
        f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data",
    )).expanduser().resolve() / "Kitchen"
)

#: "remove_table" and "remove_chairs" are filtered out by _preview_actions() below whenever the
#: kitchen being previewed has no table / no chairs -- there is nothing to remove, and a live
#: button that always 400s is worse than no button at all.
PREVIEW_ACTIONS = [
    {"id": "accept", "label": "Accept & Generate"},
    {"id": "reroll_scene", "label": "Re-roll Scene"},
    {"id": "reroll_materials", "label": "Re-roll Materials"},
    {"id": "remove_table", "label": "Remove Table"},
    {"id": "remove_chairs", "label": "Remove Chairs"},
    {"id": "back", "label": "Back to Setup"},
]


def _preview_actions(has_table: bool, has_chairs: bool) -> list[dict]:
    """PREVIEW_ACTIONS, with each Remove button left out when it has nothing to act on.

    Remove Chairs clears them ALL, exactly as Remove Table removes the one table. Per-chair
    removal belongs on the setup form, where each chair has its own Remove button beside the
    picker that added it -- the preview cannot offer that, because chairs are not selectable
    there (build_kitchen does not list them in `objects`).
    """
    dropped = set()
    if not has_table:
        dropped.add("remove_table")
    if not has_chairs:
        dropped.add("remove_chairs")
    return [a for a in PREVIEW_ACTIONS if a["id"] not in dropped]


#: The answer ids the preview's robot selector posts, and the prefix that identifies them. Not in
#: PREVIEW_ACTIONS because they are not buttons: the panel builds one <select> from
#: kitchen_preview.ROBOTS (see render_page's `scale_reference`), and this is only what `publish`
#: must accept back so the step does not 400 on a pick. "" is the No robot entry.
#:
#: The prefix is kitchen_wizard's, not a second copy of the same string: the edit step's selector
#: answers with it too (kitchen_wizard.edit_answer_ids), and _edit_furniture recognises a pick by
#: this very name. One grammar for "re-render this page for that robot", on both pages.
_ROBOT_PREFIX = kitchen_wizard.ROBOT_ANSWER_PREFIX
_ROBOT_OPTIONS = [_ROBOT_PREFIX + name for name in ["", *PREVIEW_ROBOTS]]


COMPOSE_ACTIONS = [
    {"id": "save", "label": "Save Task", "path": "/save_task"},
    {"id": "done", "label": "Done"},
]


def _progress_reporter(server):
    """24 steps: twelve rotations, then twelve joint fixups — the Tk progress bar's maximum."""
    done = {"n": 0}

    def progress(label, index, total):
        done["n"] += 1
        server.set_progress(f"{label} — {index + 1} of {total}", done["n"], 24)

    return progress


def _ask_prim(server, kitchen_num, notice=""):
    """The prim picker, as a step in the middle of the rotation pipeline.

    `notice` is the progress page's sub-line. It is republished with the page, because the picker
    replaces that page for as long as the question is up and re-showing it without the notice would
    drop "Committed … Building the 12 rotations next." for the rest of the run.
    """

    def ask(prim_paths):
        if not prim_paths:
            return None
        options = [{"id": p, "label": p} for p in prim_paths] + [{"id": "", "label": "Cancel"}]
        chosen = server.ask(
            "Choose the prim to rotate",
            "The twelve rotation USDs differ only in this prim's yaw, 30° apart.",
            options,
        )
        # back to the pipeline, with the same page the picker interrupted
        server.show_progress(f"Generating kitchen {kitchen_num:02d}", notice=notice)
        return chosen or None

    return ask


def _unusable_rotations(kitchen_num, written):
    """[(path, "not written" | "missing")] for every rotation USD this run cannot vouch for.

    `written` is what rotate_and_register_envs reports it exported. Judging by the DIRECTORY instead
    is the trap this signature exists to close, and neither half of a directory listing survives
    contact with the pipeline:

      * existence — re-generating a kitchen NUMBER leaves the previous run's twelve files in place,
        so a run that gave up (no kitchen_data JSON, base USD would not open, no candidate prims, no
        picker, picker cancelled, chosen prim not found — all of which return normally) still finds
        twelve files;
      * mtime — fix_missing_fixed_joint_targets_and_wall_cabinets runs in between, globs every
        kitchen_<N>_*.usd it can see and calls save_as_stage on each, so those same stale files come
        back newer than the base USD. It, not Export, is the last writer of every rotation.

    So the only honest question is "did THIS run export it", and then "is it still there" — a file
    exported and since deleted is a different problem from one never written, and says so.
    """
    exported = {os.path.abspath(path) for path in written or ()}
    out = []
    for i in range(12):
        path = f"{KITCHEN_DIR}/kitchen_{kitchen_num:02d}_{i:02d}.usd"
        if os.path.abspath(path) not in exported:
            out.append((path, "not written"))
        elif not os.path.exists(path):
            out.append((path, "missing"))
    return out


def _compose(server, scene_source, objects, materials, supports, joints, kitchen_data, kitchen_num):
    """Serve the composer on a kitchen and let the author save a task against it."""
    # Resolved once: the composer page marks which objects have grasps to pick, and /grasps/* serves
    # the very same folders — they must be the same map.
    thumbs = resolve_grasp_thumbs(objects, kitchen_data)
    server.set_grasp_thumbs(thumbs)
    server.set_save_context(kitchen_num=kitchen_num, kitchen_dir=KITCHEN_DIR)
    page = composer_page(
        scene_source, objects, materials, supports,
        joints=joints,
        grasp_thumbs=thumbs,
        actions=COMPOSE_ACTIONS,
    )
    server.publish("scene", page, options=["done"], phase="compose")
    server.wait_answer()


def _compose_on_existing(server, kitchen_num):
    """Author a task against a kitchen already on disk. The composer draws an in-memory trimesh
    scene, so a committed kitchen has to be read back from its USD.

    Returns True only when the composer actually opened, so the closing page cannot claim a run
    happened for a kitchen that was never loaded.
    """
    import kitchen_usd_load      # imports pxr; kept lazy exactly as the Tk version had it

    usd = kitchen_usd_load.rotation_usd_path(kitchen_num)
    if not os.path.exists(usd):
        server.notice(
            "Kitchen not on disk",
            f"No rotation USD for kitchen {kitchen_num:02d}: {usd}. Generate it first, or pick a "
            f"number that exists.",
        )
        return False
    try:
        loaded, objects, joints, kitchen_data = kitchen_usd_load.load_kitchen_for_composer(
            kitchen_num
        )
    except Exception as exc:
        traceback.print_exc()
        server.notice("Could not load the kitchen", f"Reading {usd} failed: {exc}")
        return False

    _compose(
        server, loaded.scene, objects,
        {},        # materials: USD keeps MDL references; mapping back is lossy
        [],        # supports: read-only kitchen, so drag-to-place stays OFF
        joints, kitchen_data, kitchen_num,
    )
    return True


#: The five built kitchens the gallery shows, cached for the run. Building all five costs ~27s and
#: the result never changes within a run, but caching to disk would need invalidating whenever
#: scene_synthesizer or kitchen_build changes -- and showing a room the user cannot actually get is
#: the same class of bug as the stale rotations.
_KITCHEN_GALLERY_CACHE = []


def _kitchen_gallery_scenes(server):
    """[(name, seed, trimesh.Scene)] for every kitchen type, built once per run.

    The seed is the type's index in sorted order, so the same five rooms appear on every run and a
    tile the user liked yesterday is the tile they get today.
    """
    if _KITCHEN_GALLERY_CACHE:
        return _KITCHEN_GALLERY_CACHE
    names = sorted(KITCHEN_BUILDERS)
    server.show_progress(
        "Building the kitchen gallery",
        notice="Each layout is built once, then kept for the rest of this run.",
    )
    built = []
    for index, name in enumerate(names):
        server.set_progress(f"Building {name.replace('_', ' ')}", index, len(names))
        kitchen, _data, _objects, _supports = build_kitchen(name, [], [], seed=index)
        built.append((name, index, kitchen.scene))
    server.set_progress("Ready", len(names), len(names))
    # Published only once every room is built. The guard above reads a NON-EMPTY cache as a complete
    # one, so filling it room by room would let a build that raised partway leave three behind and
    # have the next Browse… show them as if they were all of them -- a gallery quietly missing
    # kitchens the user can still generate from the form. Failing with the cache untouched means the
    # next attempt rebuilds instead.
    _KITCHEN_GALLERY_CACHE.extend(built)
    return _KITCHEN_GALLERY_CACHE


def _pick_robot(server, state):
    """Show the three robots and return the state with the picked one, or unchanged.

    THE FIRST STEP OF THE RUN, before the setup form: the robot is what every later size
    comparison is drawn against, so it is one decision for the whole session rather than a control
    on each page that draws one. The preview keeps its own selector, which is a per-page override
    of this choice and starts on it.

    Cancel returns `state` unmodified, exactly as _pick_kitchen's and _pick_table's do. Here that
    means the run keeps kitchen_preview.DEFAULT_ROBOT -- what run_wizard seeds the state with, and
    what every edit page opened on before this step existed -- so declining to choose leaves the
    wizard behaving as it did.

    ITS SCENE IS ROBOTS ONLY. This runs before any kitchen is built, so unlike every other 3D step
    there is nothing to copy from and nothing to contain: render_robot_page builds its own scene
    from the reference assets and only ever renders it.
    """
    server.publish(
        "scene",
        kitchen_wizard.render_robot_page(
            title="Pick your robot",
            text=("One robot for this run. Every size comparison after this — the furniture edit "
                  "step, the kitchen preview — is drawn against it."),
            chosen=state.get("robot") or "",
        ),
        options=kitchen_wizard.robot_answer_ids(),
        phase="setup",
    )
    picked = server.wait_answer().get("id", "")
    if not picked:
        return state                                    # Cancel: nothing changes
    return dict(state, robot=picked)


def _pick_kitchen(server, state):
    """Show the kitchen tiles and return the state with the picked type and seed, or unchanged."""
    scenes = _kitchen_gallery_scenes(server)
    scene, tiles = kitchen_gallery.kitchen_tiles(scenes)
    # One page: there are five kitchen types and there is no paging to do. Both the page and its
    # accepted ids read these two names rather than each carrying its own literal, so the panel
    # cannot draw a page turn whose id the step never published.
    page = pages = 1
    server.publish(
        "scene",
        kitchen_gallery.render_gallery_page(
            scene, tiles,
            title="Pick a kitchen",
            subtitle=(
                "Click a layout to highlight it, then Use it. Cancel keeps the current one."
            ),
            page=page, pages=pages,
            # Five tiles, and the name is the thing being chosen, so put it in the view too. The
            # mesh gallery does not: 24 BODex hashes would collide with each other.
            labels_in_view=True,
            per_row=3, units="m",       # rooms are metres; objects are centimetres
        ),
        options=kitchen_gallery.gallery_options(tiles, page, pages),
        phase="pick",
    )
    picked = server.wait_answer().get("id", "")
    if not picked:
        return state                                    # Cancel: nothing changes
    name, _, seed = picked.partition(":")
    return dict(state, kitchen_name=name, seed=int(seed))


def _pick_mesh(server, state, row, mesh_files):
    """Show the meshes for that row's object type and return the state with it chosen, or unchanged.

    Paged, because a type can have 148 variants and one page of them would be ~10 MB of geometry.
    """
    import grasp_manifest

    rows = list(state.get("objects") or [])
    obj_type = rows[row]["type"]
    paths = matching_meshes(obj_type, mesh_files)
    manifest = grasp_manifest.load()
    page = 1
    while True:
        shown, page, pages = kitchen_gallery.paginate(
            paths, page, per_page=kitchen_gallery.DEFAULT_PER_PAGE
        )
        # Only the shown page is resolved: the marker is per tile, and a 148-mesh type would
        # otherwise be looked up six times over for the five pages nobody opened.
        backed = (None if manifest is None else
                  {p for p in shown
                   if grasp_manifest.strategy_for_mesh(manifest, p) not in ("none", "unknown")})
        # obj_type, so a mesh the up table does not list is still stood the way THIS row will
        # place it -- the tiles are the only look at the mesh anyone gets before committing to it.
        scene, tiles = kitchen_gallery.mesh_tiles(shown, grasp_backed=backed, obj_type=obj_type)
        server.publish(
            "scene",
            kitchen_gallery.render_gallery_page(
                scene, tiles,
                title=f"Pick a {obj_type}",
                subtitle=(
                    f"{len(paths)} in the dataset. Click one to highlight it, then Use it. "
                    f"Cancel keeps the current choice."
                ),
                page=page, pages=pages, per_row=6, units="cm",
            ),
            options=kitchen_gallery.gallery_options(tiles, page, pages),
            phase="pick",
        )
        picked = server.wait_answer().get("id", "")
        if picked.startswith("page:"):
            page = int(picked.split(":", 1)[1])
            continue
        if not picked:
            return state                                # Cancel: nothing changes
        rows[row] = dict(rows[row], mesh=picked)
        return dict(state, objects=rows)


#: Every offered table's tile geometry, built once for the whole run. The twelve procedural ones
#: are cheap (a TableAsset, not a full kitchen); the fifty Objaverse ones are file I/O off
#: /lustre, so the set now has the same shape as _CHAIR_GALLERY_CACHE and is worth keeping for
#: the same reason -- the author reopens this picker every time they change their mind.
_TABLE_GALLERY_CACHE = []

#: Tiles per ROW in the table gallery. Five, matching the chair gallery, because the two now hold
#: comparable numbers of tiles and an author moving between them should not have the grid change
#: shape under them.
_TABLE_GALLERY_PER_ROW = 5


def _table_gallery_scenes(server):
    """[(label, key, trimesh.Scene)] for every offered table, built once per run.

    The variant KEY rides in the tile id, not the registry index the twelve-table gallery used.
    This gallery can page now, and an index into a PAGE is not an index into the registry --
    carrying the key is what makes a table picked on page 3 resolve to the table that was clicked.
    Same fix, same reason, as _chair_gallery_scenes carrying the uid.

    `variant.build().scene().scene` for BOTH halves rather than the chair gallery's
    trimesh.load(..., force="mesh") wrap: a TableVariant's build returns an Asset either way -- a
    TableAsset or a MeshAsset -- and Asset.scene() answers a trimesh.Scene, which is the type
    kitchen_tiles' .dump(concatenate=True) needs. The chair gallery has to wrap because
    CHAIR_VARIANTS carries a path rather than a builder.

    It keeps its progress reporting, which the twelve procedural tables did not need: fifty OBJs
    off /lustre is measurable work (~0.9 s for the 62, against ~0.1 s for the twelve alone).
    """
    if _TABLE_GALLERY_CACHE:
        return _TABLE_GALLERY_CACHE
    built = []
    for index, variant in enumerate(ALL_TABLE_VARIANTS):
        server.set_progress(f"Loading {variant.label}", index, len(ALL_TABLE_VARIANTS))
        built.append((variant.label, variant.key, variant.build().scene().scene))
    _TABLE_GALLERY_CACHE.extend(built)
    return _TABLE_GALLERY_CACHE


def _pick_table(server, state):
    """Show the offered tables and return the state with the picked one, or unchanged.

    Both halves of the registry in one gallery, procedural first: the author picks a table, and
    whether a TableAsset or an Objaverse mesh is behind it is an implementation detail
    (kitchen_build.ALL_TABLE_VARIANTS). Procedural first because they are the twelve this project
    shipped with and the ones whose dimensions are chosen rather than found.

    Cancel returns `state` unmodified -- for a kitchen with no table that means it stays without
    one, exactly as _pick_kitchen's Cancel keeps the current layout. There is no separate "no
    table" tile: declining a table IS cancelling this picker.
    """
    scenes = _table_gallery_scenes(server)
    if not scenes:
        server.notice(
            "No tables to pick from",
            "This checkout has no table registry at all, which should be impossible -- the twelve "
            "procedural variants live in kitchen_build.py. See scripts/simvla/objaverse_tables.md.",
            label="Back to setup",
        )
        return state

    # MEASURED for these tiles, not inherited from the chair gallery's answer and not
    # kitchen_gallery.DEFAULT_PER_PAGE. Both constraints are real and the tighter one wins:
    #
    #   * BYTES. _tiles_per_page over the 62 offered tables returns 43 -- they are far lighter
    #     than chairs (204,217 faces over 62 tiles, mean 3,293 and worst 65,286, against the chair
    #     library's 552,996 over 50, mean 11,060 and worst 112,145). So the 4 MB budget that
    #     binds the chair gallery to 8 tiles does not bind this one at all.
    #   * ROWS. 43 tiles five to a row is nine rows, and each tile is a shape the author is
    #     choosing by looking at it. The busiest page this wizard already ships is the object
    #     gallery at 24 tiles, six to a row -- four rows. That is the precedent, so four rows is
    #     the cap here too.
    #
    # Four rows of five is 20, which is what actually ships: 4 pages, and the heaviest RENDERED
    # page measures 2.49 MB -- inside the 4 MB budget and inside the 2.70 MB the unpaged
    # five-layout kitchen gallery already costs. At the byte answer of 43 it would have been
    # 3.83 MB. Computed once, outside the loop: the registry does not change between page turns,
    # so a page's size must not either.
    per_page = min(_tiles_per_page(scenes), _TABLE_GALLERY_PER_ROW * _GALLERY_MAX_ROWS)
    page = 1
    while True:
        shown, page, pages = kitchen_gallery.paginate(scenes, page, per_page=per_page)
        # The tile captions are "Table 1" .. "Table 62" now, which says where a table sits in the
        # registry and nothing about the table, so the panel gets a data line per tile as well:
        # the procedural twelve's old descriptive labels, and the short uid for the Objaverse
        # fifty. Built from the registry rather than carried in `scenes`, so the cached tuple
        # keeps the (label, key, scene) shape every stub of it expects.
        scene, tiles = kitchen_gallery.kitchen_tiles(
            shown, per_row=_TABLE_GALLERY_PER_ROW,
            details={v.key: (v.detail or "") for v in ALL_TABLE_VARIANTS},
        )
        server.publish(
            "scene",
            kitchen_gallery.render_gallery_page(
                scene, tiles,
                title="Pick a table",
                subtitle=(
                    f"{len(scenes)} offered. Click a table to highlight it, then Use it. "
                    f"Cancel keeps the current one."
                ),
                page=page, pages=pages,
                # The shape is the thing being chosen, so label the tiles in the view too -- same
                # reasoning _pick_kitchen gives for its five layouts.
                labels_in_view=True,
                per_row=_TABLE_GALLERY_PER_ROW, units="m",
            ),
            options=kitchen_gallery.gallery_options(tiles, page, pages),
            phase="pick",
        )
        picked = server.wait_answer().get("id", "")
        if picked.startswith("page:"):
            page = int(picked.split(":", 1)[1])
            continue
        if not picked:
            return state                                # Cancel: nothing changes
        # rpartition, not partition: the key is the tail and can never contain a colon, while the
        # label in front of it is display text nobody has promised will stay colon-free.
        _label, _, key = picked.rpartition(":")
        # A newly picked table starts UNEDITED. The scale and material belong to the piece, not to
        # the slot: leaving a previous table's 1.15x behind would silently resize a table the
        # author has only just chosen, and they would have no way to tell it had happened.
        #
        # Then straight into the edit step for it, because picking is not the end of it -- which is
        # the whole point of this step existing. Cancelling there keeps these defaults.
        return _edit_table(server, dict(state, table=key, table_scale=1.0, table_material=None))


def _edit_scene_for_chair(uid):
    """The chair `uid` as a trimesh.Scene at its exported size, or None if it cannot be loaded.

    ONE chair off disk rather than _chair_gallery_scenes' whole library: Edit… on the setup form
    reaches this with a cold cache, and loading 31 OBJs to draw one chair would put ~0.7 s in front
    of a page that needs one file. None rather than an exception -- a checkout that has lost
    /lustre mid-run should still get the edit panel, with the figure alone in the view.

    Wrapped in a Scene for the same reason _chair_gallery_scenes wraps: trimesh.load returns a bare
    Trimesh for a single-geometry OBJ, and every consumer here wants Scene.dump().
    """
    import trimesh

    variant = CHAIR_VARIANT_BY_UID.get(uid)
    if variant is None:
        return None
    try:
        return trimesh.Scene(trimesh.load(variant.obj_path, force="mesh"))
    except Exception:
        traceback.print_exc()
        return None


def _edit_scene_for_table(key, scale: float = 1.0):
    """The table `key` as a trimesh.Scene, or None if it cannot be built.

    variant.build().scene().scene for both halves of the registry, exactly as _table_gallery_scenes
    does -- and built FRESH rather than taken from that function's run-long cache, because the edit
    page adds a reference figure to the scene it is given. A cached scene would come back from the
    table gallery with a mannequin standing in the tile.

    `scale` goes to the VARIANT'S OWN BUILDER, which is what add_table does with it too, so the
    geometry that comes back is the geometry the commit will build: a procedural table is REBUILT
    at that size with its top and legs at their designed thickness, a mesh table is scaled. The
    table's own edit page leaves this at 1.0 and lets the browser stretch the mesh live (there is
    no round trip per slider step to rebuild on); the chair's edit page, which draws the table only
    to size a chair against it, asks for the real thing.
    """
    variant = TABLE_VARIANT_BY_KEY.get(key)
    if variant is None:
        return None
    try:
        return variant.build(float(scale)).scene().scene
    except Exception:
        traceback.print_exc()
        return None


def _chair_seat_standoff_m(bounds, footprint_m: float) -> float:
    """How far in +y a table with these world `bounds` stands from a chair seated at its near edge.

    THE RING'S OWN ANSWER, not a rule restated here: kitchen_build._seats_around_table is asked for
    a ONE-seat arrangement, which it puts at the middle of the -y side -- the side facing the room,
    which is the side the seating ring always uses (its +y side is the walkway to the counter). The
    distance back to the table's plan centre is what the edit page needs, because that page draws
    the chair at the origin and moves the table.

    `footprint_m` is the chair's plan-view footprint AT THE SCALE BEING DRAWN
    (kitchen_build.chair_footprint_m takes the scale), so a chair dragged bigger is seated further
    out -- exactly as the build would seat it.

    The seat's YAW is dropped, deliberately: the ring turns each seat on the assumption that the
    chair's own +x is its front, and face_chair_toward then corrects that for any chair whose
    facing was measured confidently, so the placed orientation is not knowable from the uid alone.
    The edit page draws the chair in its exported orientation and says so.
    """
    centre_y = (float(bounds[0][1]) + float(bounds[1][1])) / 2.0
    seat = _seats_around_table(bounds, 1, [float(footprint_m)])[0]
    return centre_y - float(seat[1][3])


def _edit_furniture(server, *, title, text, scale, material, scales=None, piece=None,
                    measured_floor=None, floor_note="", note="", robot=None, table=None,
                    table_prompt=""):
    """Publish the edit step for one piece and block. (scale, material), or None if cancelled.

    The galleries' protocol exactly: publish, wait for an answer, "" means Cancel and the caller
    returns its state untouched. Same phase, too -- editing what you just picked is part of
    picking, not a step of its own in the strip.

    Every answer this page can give is in `options`, so the browser cannot produce one the director
    would have to refuse; see kitchen_wizard.edit_answer_ids.

    `piece` is the geometry the page draws and resizes live. None is a supported state (the page
    then shows the reference figure alone) rather than a reason not to offer the edit: the values
    the author picks are still valid, and a chair whose OBJ has gone missing is not a reason to
    refuse to set its material.

    The DEFAULT group comes back as None rather than as its own name. "The author chose nothing"
    and "the author chose floor" draw the same picture, but only the first should leave the state
    saying the piece is unedited -- and only the second would put an override binding on the stage
    that re-binds a prim to the group it already had.

    A LOOP, because the page's scale reference is chosen server-side -- the same shape the preview
    loop has, for the same reason (kitchen_preview.render_page's `scale_reference`): only the robot
    the author picked is embedded, so switching robots means re-rendering this page. A pick is not
    an answer to the step; it is a request for a different picture of it. The step ends on Accept
    or Cancel, exactly as before.
    """
    scales = FURNITURE_SCALES if scales is None else tuple(scales)
    material = material or DEFAULT_FURNITURE_MATERIAL
    # This page IS the comparison, so it opens with a robot standing there rather than with none
    # the way the preview does -- and with THE ROBOT THE RUN WAS STARTED WITH (_pick_robot), not
    # with DEFAULT_ROBOT: sizing a chair against a 1.19 m Anubis and then generating a kitchen for
    # a 1.61 m AI Worker is the mistake the session-wide choice exists to stop. None means the
    # caller had no session state to read it from, which is the pre-_pick_robot behaviour.
    robot = PREVIEW_DEFAULT_ROBOT if robot is None else robot
    # The page's own menu, so every id it can post is one this step published -- which is what
    # makes a click here impossible to refuse. Built once: it does not depend on the robot.
    options = kitchen_wizard.edit_answer_ids(scales, FURNITURE_MATERIAL_GROUPS)
    while True:
        server.publish(
            "choice",
            kitchen_wizard.render_edit_page(
                title=title, text=text,
                scales=scales, scale=scale,
                materials=FURNITURE_MATERIAL_GROUPS,
                material=material,
                piece=piece,
                swatches=FURNITURE_MATERIAL_SWATCHES,
                measured_floor=measured_floor,
                floor_note=floor_note,
                note=note,
                robot=robot,
                table=table,
                table_prompt=table_prompt,
            ),
            options=options,
            phase="pick",
        )
        answered = server.wait_answer().get("id", "")
        if not answered:
            return None                                 # Cancel: nothing changes
        if answered.startswith(_ROBOT_PREFIX):
            # "robot:<name>:<scale>:<group>". The scale and the material ride along because they
            # are CLIENT state -- a slider and a menu, neither posted until Accept -- and a
            # re-render from `scale` and `material` as this call was entered would silently undo
            # whatever the author had dialled in. Re-publishing from the pending pair instead is
            # what makes picking a robot cost nothing. (The preview has no such pair to carry:
            # its client state is dragged placements, which ride their own /placements POST and
            # are already on the server by the time the pick arrives.)
            robot, _, pending = answered[len(_ROBOT_PREFIX):].partition(":")
            raw_scale, _, material = pending.partition(":")
            scale = float(raw_scale)
            continue
        # partition, not rpartition: the scale leads and can never contain a colon, while a group
        # name is only promised not to (they are MATERIALS' own keys, and two contain a space).
        raw_scale, _, group = answered.partition(":")
        return float(raw_scale), (None if group == DEFAULT_FURNITURE_MATERIAL else group)


def _edit_table(server, state):
    """The table's size scale and material, or the state unchanged. Reachable twice over:
    straight after _pick_table, and again from Edit… on the setup form, so a mistake costs one
    click rather than removing and re-picking the table.

    With no table this is a no-op rather than an error. validate_setup already refuses the action
    and the form disables the button, so this is the third layer -- and the one that decides what
    happens if the other two are ever wrong is better off doing nothing than raising inside the
    form loop.
    """
    key = state.get("table")
    if key is None:
        return state
    variant = next((v for v in ALL_TABLE_VARIANTS if v.key == key), None)
    label = variant.label if variant else "the table"
    detail = (variant.detail if variant else "") or ""
    # A PROCEDURAL table is REBUILT at width*s / depth*s / height*s, keeping its top and leg
    # thickness (see kitchen_build._procedural_table_asset); the browser's live resize stretches
    # the whole mesh, legs included. The difference is a few millimetres of leg at the extremes
    # and the shape is otherwise exact, but the page must not silently show something the commit
    # will not build. Mesh tables carry [s, s, s] baked into the geometry, which IS what the
    # browser draws.
    procedural = variant is not None and not variant.key.startswith(MESH_TABLE_KEY_PREFIX)
    return _apply_edit(
        server, state,
        title=f"Edit {label}",
        text=(f"{detail}. " if detail else "")
             + "Size and material, on all three axes. The work surface moves with the slider, "
               "so a resized table no longer pairs with an unresized chair.",
        scale=state.get("table_scale") or 1.0,
        material=state.get("table_material"),
        apply=lambda scale, material: dict(state, table_scale=scale, table_material=material),
        piece=_edit_scene_for_table(key),
        note=("This table is built from primitives, so it is rebuilt at the size you choose with "
              "its legs at their original thickness — the live preview stretches them instead."
              if procedural else ""),
    )


def _edit_chair(server, state, indices):
    """The size scale and material of the chairs at `indices`, or the state unchanged.

    `indices` are positions in the flat chairs list. One of them straight after _pick_chair -- the
    chair just added, and only it, so editing a chair the author has two of already does not
    silently change the other one. All of a row's when Edit… is pressed on the setup form, which is
    every chair that is identical to it and therefore every chair that row is showing one count for.

    An edited chair falls out of its row into one of its own, because kitchen_wizard._chair_rows
    groups on the whole spec. That is how two chairs of one design come to differ at all.
    """
    chairs = list(state.get("chairs") or [])
    indices = [i for i in indices if 0 <= i < len(chairs)]
    if not indices:
        return state
    first = chairs[indices[0]]
    variant = next((v for v in CHAIR_VARIANTS if v.uid == first["uid"]), None)
    label = variant.label if variant else "the chair"
    detail = (variant.detail if variant else "") or ""

    def apply(scale, material):
        edited = list(chairs)
        for index in indices:
            edited[index] = dict(edited[index], scale=scale, material=material)
        return dict(state, chairs=edited)

    # The whole range for every chair, with the point where THIS chair stops being as wide as the
    # library requires of a chair MARKED on the slider rather than withheld from it. That floor is
    # a realism judgement (a narrower chair is strictly easier to place, so no gate ever objected),
    # and the author is looking at the chair while they decide -- see
    # kitchen_build.chair_width_floor_scale.
    floor_scale = chair_width_floor_scale(first["uid"])
    floor_note = "" if floor_scale is None else (
        f"Below {floor_scale:.2f}× this chair is narrower than the {chair_width_floor_m():g} m the "
        f"chair library requires of a chair at all — it is offered, but it stops reading as one."
    )
    return _apply_edit(
        server, state,
        title=f"Edit {label}" + (f" (×{len(indices)})" if len(indices) > 1 else ""),
        text=(f"{detail}. " if detail else "")
             + "Size and material, on all three axes. The seat and back move with it, so a "
               "resized chair no longer tucks under an unresized table.",
        scale=float(first.get("scale") or 1.0),
        material=first.get("material"),
        apply=apply,
        piece=_edit_scene_for_chair(first["uid"]),
        measured_floor=floor_scale,
        floor_note=floor_note,
        table=_chair_scale_reference_table(state, first["uid"]),
        # Only the CHAIR page has a table section, so only it says what an empty one means. The
        # table's own edit page passes neither and the block is not drawn there at all -- "the
        # table it goes with" is not a question about a table.
        table_prompt=("This kitchen has no table yet, so there is nothing to size this chair "
                      "against. Pick one with Table… on the setup form, then open this step "
                      "again."),
    )


def _chair_scale_reference_table(state, uid):
    """The chosen table, ready for the chair edit page to draw beside the chair. None if there is
    none, or if it cannot be built.

    THE EDITED TABLE, not a default-sized one: the variant the author picked, built at the scale
    they set (_edit_scene_for_table hands that to the variant's own builder, so a procedural table
    is rebuilt rather than stretched) and tinted with the material they chose. A 1.00x table shown
    beside a 1.30x chair would answer a question nobody asked.

    `offsets_m` is one standoff per offered scale, in FURNITURE_SCALES order, so the browser can
    re-seat the table on every slider step without owning a copy of the seating rule -- see
    _chair_seat_standoff_m, and kitchen_wizard.render_edit_page's `table` for what the page does
    with it.
    """
    key = state.get("table")
    if not key:
        return None
    scene = _edit_scene_for_table(key, state.get("table_scale") or 1.0)
    if scene is None:
        return None
    variant = TABLE_VARIANT_BY_KEY.get(key)
    bounds = scene.bounds
    return {
        "scene": scene,
        "label": variant.label if variant else "the table",
        "detail": (variant.detail if variant else "") or "",
        "material": state.get("table_material") or DEFAULT_FURNITURE_MATERIAL,
        "offsets_m": [_chair_seat_standoff_m(bounds, chair_footprint_m(uid, s))
                      for s in FURNITURE_SCALES],
    }


def _apply_edit(server, state, *, title, text, scale, material, apply, scales=None, piece=None,
                measured_floor=None, floor_note="", note="", table=None, table_prompt=""):
    """_edit_furniture, then `apply` -- or `state` untouched on Cancel. The half _edit_table and
    _edit_chair share, so "Cancel changes nothing" is written once instead of twice.

    The session's robot is read off `state` HERE rather than by each caller, because this is the
    one place both of them pass through: a caller that forgot it would silently fall back to
    DEFAULT_ROBOT and draw the wrong comparison, which looks like nothing at all.
    """
    edited = _edit_furniture(server, title=title, text=text, scale=scale, material=material,
                             scales=scales, piece=piece, measured_floor=measured_floor,
                             floor_note=floor_note, note=note, table=table,
                             table_prompt=table_prompt,
                             robot=state.get("robot", PREVIEW_DEFAULT_ROBOT))
    if edited is None:
        return state
    return apply(*edited)


#: The chair library's tile geometry, loaded once for the whole run. Unlike the twelve tables this
#: is file I/O, not procedural geometry -- twenty OBJs off /lustre, measured at ~0.7 s for the set
#: -- and the add-another flow reopens the gallery once per chair, so it is worth keeping.
_CHAIR_GALLERY_CACHE = []


def _chair_gallery_scenes(server):
    """[(label, uid, trimesh.Scene)] for every offered chair, loaded once per run.

    The uid rides in the tile id (kitchen_tiles builds "<name>:<seed>"), NOT the registry index the
    table gallery uses. This gallery can page, and an index into a PAGE is not an index into
    CHAIR_VARIANTS -- carrying the uid is what makes a chair picked on page 2 resolve to the chair
    that was clicked.

    Measured, not assumed: every one of the twenty offered chairs loads as a bare `trimesh.Trimesh`
    (`trimesh.load` returns a Scene only for a multi-geometry file, and none of these are), while
    kitchen_tiles flattens each tile with `.dump(concatenate=True)` -- a Scene method. Hence the
    explicit wrap; without it the gallery raises AttributeError on the first chair.
    """
    if _CHAIR_GALLERY_CACHE:
        return _CHAIR_GALLERY_CACHE
    import trimesh                      # stdlib-adjacent; kitchen_gallery already imports it

    built = []
    for index, variant in enumerate(CHAIR_VARIANTS):
        server.set_progress(f"Loading {variant.label}", index, len(CHAIR_VARIANTS))
        mesh = trimesh.load(variant.obj_path, force="mesh")
        built.append((variant.label, variant.uid, trimesh.Scene(mesh)))
    _CHAIR_GALLERY_CACHE.extend(built)
    return _CHAIR_GALLERY_CACHE


#: What ONE gallery page may cost, all in: the HTML render_gallery_page returns, its inlined
#: three.js bundle and its base64 GLB included. Not a round number -- the smallest budget that is
#: satisfiable at all, given what the pages this wizard already serves cost.
#:
#: Measured 2026-08-13 against the fifty-chair registry (552,996 faces over 50 tiles; the smallest
#: chair is 120 faces and the largest 112,145 -- a 900x span), by rendering real pages:
#:
#:   * Any gallery page costs ~0.74 MB before it holds a single tile: scene_to_html inlines the
#:     whole three.js bundle. That is a floor nothing here can move.
#:   * The heaviest page this wizard already serves UNPAGED is the five-layout kitchen gallery,
#:     at 2.70 MB. The kitchen preview -- the page the author waits for on every step -- is
#:     1.07 MB, and the twelve-table gallery is 0.76 MB. So 2.70 MB is the precedent.
#:   * The registry's single heaviest chair is 2.53 MB of geometry on its own, so ANY page holding
#:     it costs at least 3.27 MB. A budget below that cannot be met at any page size: the rule
#:     below would return 1 tile per page and still blow it.
#:
#: 4 MB is the precedent (2.70 MB) raised only as far as the registry's own worst tile (3.27 MB)
#: forces, plus rounding. Raising it further is the wrong lever -- the way to get fewer pages is
#: to decimate the handful of 100k-face chairs, not to ship a heavier page.
_GALLERY_PAGE_BUDGET_B = 4 * 1024 * 1024

#: What a gallery page costs before it holds any tile: three.js, the panel, the CSS. Measured by
#: regressing rendered page bytes on (tile count, total faces) over 40 random chair subsets --
#: 0.74 MB, and the per-TILE term came out at -1.1 KB, i.e. indistinguishable from zero. A tile
#: costs what its geometry costs and nothing more, which is why the rule below counts faces and
#: not tiles.
_GALLERY_PAGE_FIXED_B = 760_000

#: Marginal page bytes per mesh face, from the same regression: 23.7 B, rounded up to 24. It is
#: not a coincidence -- a glTF face carries 3 uint32 indices and its share of two float32x3
#: attributes, and scene_to_html base64s the GLB, which is the 4/3.
#:
#: Predicts a rendered page to 1.1% median / 6.4% worst error over those 40 subsets, which is
#: what makes _tiles_per_page an estimate worth trusting instead of a page-render loop.
_GALLERY_BYTES_PER_FACE = 24

#: The most tile ROWS a gallery page shows, when the byte budget is not what binds it.
#:
#: A byte budget is the only constraint _tiles_per_page can see, and for a light library it is not
#: the one that matters: the 62 offered tables total 204,217 faces, so the 4 MB budget allows 43
#: tiles on a page and five to a row that is NINE rows of shapes the author is choosing by looking
#: at them. The busiest page this wizard already ships is the object gallery -- 24 tiles, six to a
#: row, four rows -- so four rows is the precedent rather than a preference.
#:
#: Applied only where it binds: the chair gallery's byte answer is 8 tiles, well inside 4 rows of
#: five, so nothing about it changes.
_GALLERY_MAX_ROWS = 4


def _tiles_per_page(scenes, budget_b=_GALLERY_PAGE_BUDGET_B):
    """The largest uniform page size whose HEAVIEST page still fits `budget_b`.

    A number derived from what is actually on offer, not a constant, because tile weight is not a
    property this codebase gets to assume. `DEFAULT_PER_PAGE = 24` was calibrated against 71 KB
    BODex meshes and says so in its own comment; a chair tile is ~260 KB on average and up to
    2.5 MB, so the same 24 puts the chair gallery's heaviest page at 7.12 MB and the whole
    fifty-chair library on one page at 13.08 MB. The table gallery now being built will land
    somewhere else again. Whatever the tiles weigh, this answers for them.

    `scenes` is kitchen_tiles' input shape, (label, id, trimesh.Scene) -- so the meshes are already
    loaded (_chair_gallery_scenes caches them) and counting faces costs nothing.

    THE HEAVIEST ACTUAL PAGE, not the heaviest conceivable one. The items are paged in registry
    order, so which tiles share a page is known here: page k is scenes[k*n:(k+1)*n]. Sizing against
    a hypothetical page of the n heaviest chairs would be sizing for an arrangement that cannot
    occur (that page is 10.35 MB at n=12) and would cut the page to a handful of tiles for nothing.

    That also makes the answer non-monotonic in n -- the heavy chairs cluster, so n=6 has a worse
    heaviest page (4.52 MB) than n=8 (3.83 MB). Hence walking DOWN from the whole library and
    taking the first n that fits, rather than up from 1 and stopping at the first that does not.

    Floors at 1: a page has to hold something. If even one tile is over budget the caller ships an
    over-budget page and the budget's docstring is where to look -- see _GALLERY_PAGE_BUDGET_B on
    why the fix for that is the mesh, not this number.
    """
    faces = [
        sum(len(g.faces) for g in scene.geometry.values() if hasattr(g, "faces"))
        for _label, _id, scene in scenes
    ]
    for per_page in range(len(faces), 0, -1):
        worst = max(sum(faces[start:start + per_page])
                    for start in range(0, len(faces), per_page))
        if _GALLERY_PAGE_FIXED_B + worst * _GALLERY_BYTES_PER_FACE <= budget_b:
            return per_page
    return 1


def _pick_chair(server, state):
    """Show the chair library and return the state with one MORE chair, or unchanged.

    APPENDS rather than replaces -- that is the whole difference from _pick_table, and it is what
    makes the flow add-another: six chairs are six trips through this picker. Cancel returns
    `state` untouched, exactly as the other galleries do, so backing out of the picker is how you
    decide against another chair.

    Two refusals happen here rather than in the gallery, because the gallery cannot express either:

      * At kitchen_wizard.MAX_CHAIRS there is nothing to add. The form already hides the button at
        the cap, but the form is not the only thing that can POST, and validate_setup would refuse
        the seventh chair only on the NEXT submit -- after the author had picked it.
      * With no chairs on offer there is nothing to show, and an empty gallery is not a page that
        can be drawn: render_gallery_page exports the tile scene to GLB, and trimesh refuses an
        empty scene outright ("Can't export empty scenes!"). A checkout without /lustre has exactly
        that registry, so this is the ordinary case there, not a corner. It gets a notice saying
        where chairs come from instead of a traceback.
    """
    chairs = list(state.get("chairs") or [])
    if len(chairs) >= kitchen_wizard.MAX_CHAIRS:
        server.notice(
            "That is already the most chairs",
            f"A kitchen carries at most {kitchen_wizard.MAX_CHAIRS} chairs. Remove one on the "
            f"setup form before adding another.",
            label="Back to setup",
        )
        return state

    scenes = _chair_gallery_scenes(server)
    if not scenes:
        server.notice(
            "No chairs to pick from",
            "This checkout has no chair library, so there is nothing to place. The chairs live "
            "on objaverse_chairs (override the root with CHAIR_OBJ_DIR); see "
            "scripts/simvla/objaverse_chairs.md. Everything else about this kitchen still works.",
            label="Back to setup",
        )
        return state

    # Sized from what these tiles actually weigh, NOT kitchen_gallery.DEFAULT_PER_PAGE -- that 24
    # is calibrated for 71 KB BODex meshes and is 3.4x too generous for a chair. See
    # _tiles_per_page. Computed once, outside the loop: the library is cached and does not change
    # between page turns, so a page's size must not either.
    per_page = _tiles_per_page(scenes)
    page = 1
    while True:
        shown, page, pages = kitchen_gallery.paginate(scenes, page, per_page=per_page)
        # "Chair 7" is a position in the registry, not the chair; the panel's data line carries
        # the short uid beside the measurements so the thing the state and the exported scene are
        # actually keyed by is on screen. See kitchen_build.FURNITURE_NUMBERING.
        scene, tiles = kitchen_gallery.kitchen_tiles(
            shown, per_row=5, details={v.uid: (v.detail or "") for v in CHAIR_VARIANTS},
        )
        server.publish(
            "scene",
            kitchen_gallery.render_gallery_page(
                scene, tiles,
                title="Pick a chair",
                subtitle=(
                    f"{len(scenes)} in the library, {len(chairs)} of "
                    f"{kitchen_wizard.MAX_CHAIRS} placed. Click a chair to highlight it, then Use "
                    f"it. Cancel adds none."
                ),
                page=page, pages=pages,
                # Labelled in the view like the table and kitchen galleries: the shape is the thing
                # being chosen, and a chair's label is short.
                labels_in_view=True,
                per_row=5, units="m",
            ),
            options=kitchen_gallery.gallery_options(tiles, page, pages),
            phase="pick",
        )
        picked = server.wait_answer().get("id", "")
        if picked.startswith("page:"):
            page = int(picked.split(":", 1)[1])
            continue
        if not picked:
            return state                                # Cancel: nothing changes
        # rpartition, not partition: the uid is the tail and can never contain a colon, while the
        # label in front of it is display text nobody has promised will stay colon-free.
        _label, _, uid = picked.rpartition(":")
        # ONE DICT PER INSTANCE, at the defaults, then the edit step for THAT ONE chair -- the last
        # position in the list, which is the one this pick just created. Editing only it is what
        # lets an author who already has two of this chair add a third at a different size without
        # the other two changing under them.
        added = dict(state, chairs=chairs + [{"uid": uid, "scale": 1.0, "material": None}])
        return _edit_chair(server, added, [len(chairs)])


def _add_chair(server, state, row):
    """One more of the chair on row `row`, or a notice saying why not. Returns the new state.

    This is the + on a chair row, and it exists so that a second copy of a chair costs one click
    instead of another trip through a fifty-tile gallery. It APPENDS, exactly as _pick_chair does:
    the state is a flat list of chairs with repeats, build_kitchen numbers them chair_0, chair_1,
    ... in that order, and the grouping the author sees is drawn from it
    (kitchen_wizard._chair_rows) rather than stored.

    A COPY OF THE ROW'S OWN SPEC, not of its uid: a row is one distinct (uid, scale, material), so
    "one more of this chair" means one more chair identical to the ones in it. Adding a bare uid at
    the defaults would put an unedited chair on a row of edited ones and the two would be shown as
    one count of one design.

    `row` is an index into that grouped list rather than a uid, because once one chair of a design
    is edited a uid names two rows. validate_setup has already checked it against the same
    grouping of the same list -- _chair_rows is pure, and the form posts back the very list it was
    drawn from.

    At the cap it publishes a NOTICE rather than returning the state unchanged and letting the
    click read as a no-op. The form already disables + at the cap, so this is the same defence in
    depth _pick_chair's own cap check is -- the form is not the only thing that can POST, and a
    button that appears to work and does not is worse than one that explains itself.
    """
    chairs = list(state.get("chairs") or [])
    if len(chairs) >= kitchen_wizard.MAX_CHAIRS:
        server.notice(
            "That is already the most chairs",
            f"A kitchen carries at most {kitchen_wizard.MAX_CHAIRS} chairs, and this one already "
            f"has {len(chairs)}. Take one out with − before adding another.",
            label="Back to setup",
        )
        return state
    indices = kitchen_wizard.chair_row_indices(chairs, row)
    if not indices:
        return state
    return dict(state, chairs=chairs + [dict(chairs[indices[0]])])


def _remove_chair(state, row):
    """One fewer of the chair on row `row` -- its LAST copy. Returns the new state.

    The last rather than the first because the list is positional: the nth chair becomes chair_<n>
    and gets the nth seat. Dropping the last copy leaves every earlier chair on the seat it was
    already shown in, while dropping the first shifts everything after it round the table for no
    reason the author asked for.

    `row` is an index into the grouped rows, for the same reason _add_chair's is.

    No `server`: unlike +, this cannot be refused. A row only exists because there is at least one
    of that chair to take away, and the count going to zero simply removes the row (the form
    redraws from the grouped list, and a spec with no copies has no row).
    """
    chairs = list(state.get("chairs") or [])
    indices = kitchen_wizard.chair_row_indices(chairs, row)
    if indices:
        chairs.pop(indices[-1])
    return dict(state, chairs=chairs)


def _face_chairs(kitchen, chairs) -> None:
    """Turn every chair in `kitchen` toward the table (or, with no table, the counter run).

    Called at every point the answer could change -- right after a build, and again after
    apply_placements, because the target is the table's CURRENT centre and a table dragged since
    the build would leave the chairs facing where it used to be.

    Facing is absolute rather than incremental: face_chair_toward rotates a chair so that it points
    AT the target, whatever it pointed at before. So calling it twice converges instead of
    accumulating, which is what lets the previewed kitchen (faced after its build, then again after
    its drag) and the throwaway check kitchen (faced once, after the same drag is applied to it)
    end up in the same pose. Chairs whose backrest signal is below FACING_CONFIDENCE_CUT are left
    as exported -- by design, and identically in both kitchens, since it is a property of the mesh.

    The ids are derived from the chair LIST rather than read off the scene graph because that is
    how build_kitchen assigns them: the nth chair it is given becomes chair_<n>. Only the length
    is read -- facing is a property of the mesh and its seat, not of the scale or the material the
    author chose for it -- so this takes the same list build_kitchen was handed rather than a
    second, parallel one that could fall out of step with it.
    """
    if not chairs:
        return
    target = chair_facing_target(kitchen)
    for index in range(len(chairs)):
        face_chair_toward(kitchen, f"chair_{index}", target)


def run_wizard(server, mesh_files):
    """The whole run, top to bottom, on the main thread. Returns the kitchen number, or None."""
    server.set_setup_context(mesh_files=mesh_files)

    # THE ROBOT FIRST, before the form: it is what every later size comparison is drawn against,
    # and it is one choice for the run rather than one per page that draws one. Seeded with
    # DEFAULT_ROBOT so that Cancel on that step means "keep what the wizard always used" -- the
    # galleries' own Cancel convention, applied to the one step that has nothing behind it.
    #
    # It rides on `state` from here on, which is why nothing below has to pass it: every step that
    # draws a comparison is reached through that state (validate_setup carries `robot` on it, and
    # the form echoes it on every submit), so _edit_table and _edit_chair read it where they
    # already read the table's scale.
    #
    # What the form was showing when it last left for a gallery, so it comes back pre-filled --
    # on the first pass it holds the robot and nothing else, which the form reads as a fresh form.
    state = _pick_robot(server, {"robot": PREVIEW_DEFAULT_ROBOT})
    while True:
        server.publish("form", kitchen_wizard.render_form_page(
            kitchen_wizard.form_payload(KITCHEN_DIR, state)
        ), phase="setup")
        setup = server.wait_answer()

        if setup["action"] in ("add_chair", "remove_chair", "edit_table", "edit_chair"):
            # The +/-/Edit… on a chair row, and Edit… on the table. Deliberately NOT inside the
            # gallery guard below: none of these draws a tile, a GLB or any trimesh geometry -- the
            # edit step is a rendered string like the notice pages are -- so there is no gallery
            # for them to fail to show, and wrapping them would only turn a real bug in here into a
            # "Could not show the gallery" notice about a gallery that was never opened.
            if setup["action"] == "add_chair":
                state = _add_chair(server, setup["state"], setup["chair_row"])
            elif setup["action"] == "remove_chair":
                state = _remove_chair(setup["state"], setup["chair_row"])
            elif setup["action"] == "edit_table":
                state = _edit_table(server, setup["state"])
            else:
                state = _edit_chair(
                    server, setup["state"],
                    kitchen_wizard.chair_row_indices(
                        setup["state"].get("chairs") or [], setup["chair_row"]
                    ),
                )
            continue                            # back to the form, with the new count

        if setup["action"] in ("pick_kitchen", "pick_mesh", "pick_table", "pick_chair"):
            # Guarded with the SAME recovery the build gets, for the same reason: a gallery that
            # will not draw is a recoverable authoring problem, and ending the run over one costs
            # another ~1 minute Isaac boot to retry. Not hypothetical -- kitchen_tiles flattens each
            # room with Scene.dump(concatenate=True), which the installed trimesh already warns is
            # deprecated for removal, and the day it goes every Browse… click (and every Table…
            # click, which flattens the same way) raises here.
            try:
                if setup["action"] == "pick_kitchen":
                    state = _pick_kitchen(server, setup["state"])
                elif setup["action"] == "pick_table":
                    state = _pick_table(server, setup["state"])
                elif setup["action"] == "pick_chair":
                    state = _pick_chair(server, setup["state"])
                else:
                    state = _pick_mesh(server, setup["state"], setup["row"], mesh_files)
            except Exception as exc:
                traceback.print_exc()
                # setup["state"] is the form the user left to come here, so this returns them to it
                # rather than to a blank one: they lost the pick, not the setup.
                state = setup["state"]
                server.notice(
                    "Could not show the gallery",
                    f"{exc}\n\nNothing changed. The console has the full traceback.",
                    label="Back to setup",
                )
            continue                            # back to the form, pre-filled

        if setup["action"] == "compose_existing":
            if _compose_on_existing(server, setup["kitchen_num"]):
                return setup["kitchen_num"]
            # Nothing was loaded; back to setup to correct it. `state` is deliberately NOT refreshed
            # from this setup: validate_setup returns before parsing compose_existing's kitchen name
            # and rows, so its payload carries nothing to refresh from, and overwriting a real state
            # with those blanks would clear the form.
            continue

        # Everything below is `generate`, and it is the last thing the user submitted -- so it is
        # what the form must show if the run comes back here. Refreshed HERE rather than only after
        # a gallery because "back here" is mostly not an error: `back` is an ordinary preview
        # button, and a failed build or commit returns too. A form re-rendered from the gallery-era
        # state looks prefilled -- the type and the picked layout are still on it -- while the
        # number and the rows typed since have silently vanished, which is worse than a blank form
        # because nothing on screen says anything was lost. This is the shape the setup page reads
        # back (kitchen_num as text, rows without their generated obj_n), the same one
        # validate_setup hands to a gallery.
        state = {
            "kitchen_num": str(setup["kitchen_num"]),
            "kitchen_name": setup["kitchen_name"],
            "seed": setup.get("seed"),
            "objects": [
                {"type": o["type"], "loc": o["loc"], "mesh": o["mesh"]} for o in setup["objects"]
            ],
            "table": setup.get("table"),
            "table_scale": setup.get("table_scale") or 1.0,
            "table_material": setup.get("table_material"),
            # A copy per chair, not just of the list: these dicts go on to be edited by the edit
            # step, and sharing them with the list handed to build_kitchen would let one mutate
            # the other.
            "chairs": [dict(c) for c in (setup.get("chairs") or [])],
            # The session's robot, carried back onto the form's state like everything else here:
            # `back` from the preview re-renders this form, and a robot dropped on the way would
            # silently reset every later comparison to DEFAULT_ROBOT. Taken from `setup` -- what
            # the form posted -- and never from the preview loop's own `robot` below, which is a
            # per-page override of this choice rather than a change to it.
            "robot": setup.get("robot"),
        }

        kitchen_num = setup["kitchen_num"]
        random.seed(None)
        kitchen = kitchen_data = objects = supports = picks = None
        # The table variant key, or None -- carried outside `state` because it changes mid-preview
        # (Remove Table) without a trip back through the setup form, unlike everything in `state`.
        # Its scale and material travel with it, for the same reason and so that Remove Table can
        # clear all three together: a material override aimed at /world/table/... after the table
        # is gone would be a binding for prims that are not there.
        table_key = setup.get("table")
        table_scale = setup.get("table_scale") or 1.0
        table_material = setup.get("table_material")
        # The chairs, one dict per instance, carried outside `state` for exactly the same reason:
        # Remove Chairs changes them mid-preview. A copy per chair, so clearing or rebuilding from
        # them here cannot edit the form's own list.
        chair_specs = [dict(c) for c in (setup.get("chairs") or [])]
        # Which robot stands beside the counter, "" for none. It starts on THE SESSION'S ROBOT
        # (_pick_robot, carried on the setup state): this page is one of the size comparisons that
        # choice was made for, and opening it on none would make the author pick a robot again on
        # the one page that has a selector. The selector stays, as a per-page override -- picking
        # from it changes this variable and nothing else, which is why it is a local here and not
        # written back to `state`.
        robot = setup.get("robot") or ""

        # Building and committing are the recoverable half of the run: an object that will not fit,
        # a kitchen type that fails to build, a directory that cannot be written. The Tk version
        # showed the error and returned to its setup window; ending the whole run instead would cost
        # another ~1 minute Isaac boot to retry a fixable mistake. The console gets the traceback,
        # the browser gets the message. Everything past the commit is deliberately NOT in here: a
        # half-written rotation set is not something to hand back to the form.
        try:
            while True:
                if kitchen is None:
                    kitchen, kitchen_data, objects, supports = build_kitchen(
                        setup["kitchen_name"], setup["objects"], mesh_files,
                        seed=setup.get("seed"), table=table_key, table_scale=table_scale,
                        chairs=chair_specs,
                    )
                    # Faced before the preview is drawn, not only before the commit: the preview is
                    # the only look the author gets at the scene, and chairs shown facing one way
                    # and committed facing another would make the picture a lie.
                    _face_chairs(kitchen, chair_specs)
                if picks is None:
                    picks = pick_materials()

                actions = _preview_actions(bool(table_key), bool(chair_specs))
                server.publish("scene", preview_render_page(
                    kitchen.scene, objects, material_display_names(picks), supports,
                    joints=collect_joints(kitchen), actions=actions,
                    # The scale reference: a robot on the clearest patch of floor beside a dimension
                    # line, the cheapest way to judge whether a room is the size a room should be.
                    # `robot` starts on the SESSION'S robot rather than empty -- the run has
                    # already been asked which one, and this is one of the comparisons it was
                    # asked for. That costs this page 5-13 MB of GLB it used not to carry until
                    # the author opened the selector; the selector is still there to switch robots
                    # or to take it off. render_page draws it into a COPY of this scene; `kitchen`
                    # is the object this loop goes on to commit.
                    scale_reference=robot,
                ), options=[a["id"] for a in actions] + _ROBOT_OPTIONS, phase="preview")

                action = server.wait_answer()["id"]
                placements = server.placements

                if action == "reroll_scene":
                    # Same seed, so the room the user picked from the gallery stays: seed reaches
                    # only the scene_synthesizer builder, and mesh choice and placement are
                    # randomised per build regardless. table_key is untouched -- the CHOICE
                    # survives a re-roll, only its (now stale) dragged position does not.
                    kitchen = None              # fresh meshes, positions, support choice
                    continue
                if action == "remove_table":
                    # The spec promises the choice stays reversible without restarting setup.
                    # Nothing removes the table from THIS scene object -- the loop rebuilds from
                    # table_key on the next pass, and table_key=None builds a kitchen with none,
                    # exactly as reroll_scene already rebuilds for a fresh layout.
                    table_key = None
                    # The scale and the material belong to the table, so they go with it. Left
                    # behind, table_material would still put an override binding on
                    # /world/table/... at commit -- a binding for prims that no longer exist.
                    table_scale = 1.0
                    table_material = None
                    kitchen = None
                    continue
                if action == "remove_chairs":
                    # The mirror of Remove Table, and rebuilt the same way rather than deleting
                    # nodes from this scene: chair poses are laid out for the set as a whole
                    # (kitchen_build._seats_around_table), so removing one from a built scene would
                    # leave the rest in seats chosen for a party that is no longer coming.
                    chair_specs = []
                    kitchen = None
                    continue
                if action == "back":
                    break                       # nothing written; back to the setup form
                # Keep whatever was dragged, so composing or a material re-roll does not undo it.
                apply_placements(kitchen, placements)
                # And re-face the chairs against wherever the table has just been dragged to. Both
                # this kitchen and the throwaway check kitchen below are faced immediately after
                # their own apply_placements, which is what keeps the scene that gets validated and
                # the scene that gets committed the same scene. See _face_chairs on why running it
                # twice on this kitchen converges rather than accumulating.
                _face_chairs(kitchen, chair_specs)
                if action == "reroll_materials":
                    picks = None                # same geometry, new materials
                    continue
                if action.startswith(_ROBOT_PREFIX):
                    # The scale reference is chosen on the server, so switching robots means
                    # re-rendering this page. Below apply_placements deliberately: the author has
                    # arranged the room and is now standing a robot in it, and a re-render that
                    # dropped their drags would make the picture they were judging go away.
                    robot = action[len(_ROBOT_PREFIX):]
                    continue

                if action == "accept" and (table_key or chair_specs):
                    # furniture_placement_problems mutates the joint configuration (open_every_joint)
                    # to check the OPEN pose as well as the closed one -- see its docstring. Run
                    # it on a throwaway rebuild, not on `kitchen`: that is the scene about to be
                    # committed (and re-previewed if rejected), and it must not come back with
                    # every door swung open.
                    #
                    # The position to check is read off the LIVE kitchen's scene graph, not
                    # replayed from `placements` again: publish("scene", ...) clears
                    # server.placements on every fresh preview it shows (see its own docstring --
                    # "one long-lived server must do it explicitly, or a re-roll inherits the
                    # previous round's drags"), including the very republish this rejection branch
                    # causes below. A second Accept click with no further drag on that republished
                    # page would see placements == {}, and re-applying an empty dict to a fresh
                    # rebuild would silently check the table's DEFAULT position -- always valid,
                    # per test_the_default_position_is_itself_valid -- while the kitchen actually
                    # about to be committed still carries the rejected one from the FIRST click's
                    # apply_placements call a few lines up. Reading graph.get("table") sidesteps
                    # that reset entirely: it is the position this run will really export,
                    # independent of which page publish last cleared the delta dict.
                    #
                    # Chairs are read the same way and for the same reason, one node per chair,
                    # even though nothing can drag one today (they are not in `objects`, so the
                    # preview cannot select them): replaying their poses from anywhere but the live
                    # graph would be the same bug waiting for the day they become draggable.
                    live = {}
                    if table_key:
                        live["table"] = kitchen.scene.graph.get("table")[0]
                    for index in range(len(chair_specs)):
                        node = f"chair_{index}"
                        live[node] = kitchen.scene.graph.get(node)[0]
                    check_kitchen, _check_data, _check_objects, _check_supports = build_kitchen(
                        setup["kitchen_name"], setup["objects"], mesh_files,
                        seed=setup.get("seed"), table=table_key, table_scale=table_scale,
                        chairs=chair_specs,
                    )
                    apply_placements(check_kitchen, live)
                    # Faced AFTER the placements, at the same point in the sequence the previewed
                    # kitchen was faced above, so both are turned toward the same target.
                    _face_chairs(check_kitchen, chair_specs)
                    problems = furniture_placement_problems(check_kitchen)
                    if problems:
                        # Every problem, not just the first: one bad position can produce several
                        # distinct messages (a clash plus more than one blocked door), and showing
                        # one at a time would make the author drag, fix it, and get rejected again
                        # for a clash that was there all along.
                        #
                        # The messages name their own offender ("chair 0 overlaps the table by
                        # 39 mm", "the table overlaps base_cabinet by 175 mm"), so the title stays
                        # general and only the advice branches -- with no table there is nothing on
                        # this page to drag, and telling the author to drag it would be nonsense.
                        server.notice(
                            "This furniture does not fit",
                            "The kitchen cannot be generated like this:\n"
                            + "\n".join(f"- {p}" for p in problems)
                            + "\n\n" + (
                                "Drag the table somewhere else, then Accept & Generate again."
                                if table_key else
                                "Remove Chairs here, or take some out on the setup form, then "
                                "Accept & Generate again."
                            ),
                        )
                        continue                # back to the SAME preview -- kitchen is untouched
                break                           # ACCEPT
        except Exception as exc:
            traceback.print_exc()
            server.notice(
                "Could not build this kitchen",
                f"{exc}\n\nNothing was written. The console has the full traceback. Change the "
                f"setup and try again.",
                label="Back to setup",
            )
            continue
        if action == "back":
            continue

        try:
            usd_filename, json_path = commit_kitchen(
                kitchen, picks, kitchen_num, kitchen_data,
                # The author's per-piece material choices, as {prim regex -> group}. Built HERE,
                # from the same two variables the build above used, so the stage that is exported
                # and the scene that was previewed and validated describe one kitchen.
                material_overrides=furniture_material_overrides(
                    table=table_material, chairs=chair_specs
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            server.notice(
                "Could not write the kitchen",
                f"{exc}\n\nKitchen {kitchen_num:02d} was not generated; anything already written "
                f"for it may be incomplete. The console has the full traceback.",
                label="Back to setup",
            )
            continue

        # FURNITURE DOES NOT COUNT -- not the table, and not a single one of the six chairs.
        #
        # build_kitchen puts an `objects` entry per piece of furniture so the preview can select
        # and drag it, and that entry once made the bare `if not objects` guard below unreachable
        # the moment a table was chosen: with a table and zero placed BODex objects, `objects` is
        # [table] -- truthy -- so the run fell through to a rotation step that finds no candidate
        # prims and returns [], while the joint fixup still ran and re-saved a previous run's
        # kitchen_<N>_*.usd (see the note below on why that half is load-bearing). The run then
        # reported a kitchen committed with rotations it never made, and the "none of them could
        # be placed" diagnostic could never fire again either.
        #
        # Chairs are entries of exactly that kind, so they reintroduce that bug six times over
        # unless they are excluded here too -- which is why this is a call to kitchen_build's own
        # _is_furniture rather than a second list of names beside it. That predicate is what the
        # placement gate already means by "furniture" (fullmatch on `table` or `chair_<n>`, so a
        # `side_table` or a `chairlift` is not swept in), and sharing it makes "the wizard placed
        # this as furniture" and "this is not a BODex object to rotate" one decision instead of
        # two that can drift apart the next time a piece of furniture is added.
        #
        # `objects` itself is left intact on purpose -- the preview and the composer both want the
        # furniture in it.
        placed = [o for o in objects if not _is_furniture(o["node_id"])]

        if not placed:
            # Nothing landed in the scene, so there is nothing to rotate: the twelve rotations
            # differ only in one placed object's yaw, and on an object-free kitchen they would be
            # twelve identical copies of the base scene. Skipping them is not a failure -- the
            # scene is committed and complete -- so this returns the kitchen number rather than
            # reporting it ungenerated.
            #
            # The joint fixup is skipped WITH them, and that is the load-bearing half: it globs
            # kitchen_<N>_*.usd, so on a re-used number it would re-save a PREVIOUS run's
            # rotations, leaving files that describe a different kitchen looking current.
            skipped = (
                f"Kitchen {kitchen_num:02d} is committed:\n\n{usd_filename}\n{json_path}\n\n"
                f"It has no placed objects, so there was nothing to rotate: the 12 rotation USDs "
                f"were not generated and no task envs were registered. Composing a task needs "
                f"objects to bind skills to, so it is not offered."
            )
            if setup["objects"]:
                skipped += (
                    f"\n\nYou asked for {len(setup['objects'])} object(s), and none of them could "
                    f"be placed on this kitchen — the preview listed none. Check that the "
                    f"placement locations you chose exist on a "
                    f"{setup['kitchen_name'].replace('_', ' ')} kitchen."
                )
            server.notice("Scene generated", skipped, label="Finish")
            return kitchen_num

        committed = f"Committed {usd_filename} and {json_path}. Building the 12 rotations next."
        server.set_phase("build")   # committed; the rest of this run is the pipeline
        server.show_progress(f"Generating kitchen {kitchen_num:02d}", notice=committed)

        progress = _progress_reporter(server)
        # What the rotation step reports it exported — the only record the joint fixup below cannot
        # forge by re-saving whatever it finds on disk.
        written = rotate_and_register_envs(
            kitchen_num, progress=progress, ask_prim=_ask_prim(server, kitchen_num, committed)
        )
        fix_missing_fixed_joint_targets_and_wall_cabinets(kitchen_num, progress=progress)

        unusable = _unusable_rotations(kitchen_num, written)
        if unusable:
            path, why = unusable[0]
            server.notice(
                "Rotations are incomplete",
                f"Kitchen {kitchen_num:02d} was committed, but {len(unusable)} of its 12 rotation "
                f"USDs are unusable (first: {path} was {why}). Task composition is not offered — "
                f"goal generation needs all 12, exported for THIS kitchen; any files of that name "
                f"already on disk belong to an earlier run. The console says which step gave up, "
                f"so kitchen {kitchen_num:02d} was not completed.",
            )
            return None                     # committed, but not generated: do not claim otherwise

        if server.ask(
            "Scene ready",
            f"Kitchen {kitchen_num:02d} is generated and its 12 rotations are registered.",
            [{"id": "yes", "label": "Compose a task"}, {"id": "no", "label": "Finish"}],
        ) == "yes":
            _compose(
                server, kitchen.scene, objects, material_display_names(picks), supports,
                collect_joints(kitchen), kitchen_data, kitchen_num,
            )
        return kitchen_num


# ======================= MAIN PIPELINE =======================
if __name__ == "__main__":
    generated = None
    try:
        try:
            bodex_data = datasets.load_dataset("BODex")
            mesh_files = bodex_data.get_filenames()
        except Exception as exc:
            # Also on the console: this can fail on an unattended headless run where nobody is
            # watching the browser, and SystemExit skips the summary print at the bottom, so
            # without these two lines the process would exit 1 in complete silence.
            traceback.print_exc()
            print(f"[Main] Could not load the BODex dataset: {exc}", flush=True)
            WIZARD.notice(
                "Could not load the BODex dataset",
                f"{exc}\n\nCheck ~/.config/scene-synth.datasets/config.yaml.",
                label="Close",
            )
            WIZARD.finish("Stopped", "The BODex dataset could not be loaded, so nothing ran.")
            raise SystemExit(1)

        generated = run_wizard(WIZARD, mesh_files)
        WIZARD.finish(
            "Done",
            f"Pipeline complete for kitchen {generated:02d}." if generated is not None
            else "No kitchen was generated.",
        )
    # Every exit below publishes a closing page too. shutdown() in the finally kills the server the
    # moment it returns, so a path that ends on a notice leaves the tab holding a dead button and a
    # poller that fails silently — the browser has no other way to learn the run is over.
    #
    # With ONE measured exception, which the branch below cannot fix from here: Ctrl-C. SimulationApp
    # installs its own SIGINT handler at boot (after AppLauncher, signal.getsignal(SIGINT) is
    # `SimulationApp.__init__.<locals>.signal_handler`, not default_int_handler), so Omniverse tears
    # the process down and no KeyboardInterrupt is ever raised here. An HTTP-driven smoke run
    # confirmed it: SIGINT while a step was waiting killed the server within 0.2s, published no
    # closing page — the browser is left exactly as described above — and even this function's own
    # print never reached the log. The process still exits 0 with no traceback, so it is a clean
    # exit that simply does not tell the browser.
    #
    # The branch is kept, not deleted: it is still the correct behaviour, it costs nothing, and it
    # runs if this is ever driven from a python whose SIGINT handler Omniverse has not replaced.
    # Restoring signal.default_int_handler after the boot would make it live again, but that changes
    # how Omniverse shuts down and wants its own verification.
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
        WIZARD.finish("Stopped", "Interrupted at the generator's console.")
    except Exception as exc:
        traceback.print_exc()
        WIZARD.notice(
            "Something went wrong",
            f"{exc}\n\nThe generator's console has the full traceback.",
            label="Close",
        )
        WIZARD.finish("Stopped", f"The run ended early: {exc}")
    finally:
        WIZARD.shutdown()

    if generated is None:
        print("[Main] No kitchen was generated.")
    else:
        print(f"[Main] Pipeline complete for kitchen {generated:02d}.")
