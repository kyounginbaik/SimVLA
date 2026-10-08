"""Headless: generate ONE NEW kitchen holding a single mug, then author its goal template.

Drives the WIZARD's own build path -- kitchen_build.build_kitchen -> kitchen_scene_generator.
commit_kitchen -> rotate_and_register_envs -- with every browser step replaced by a fixed choice.
That path rather than goal_generator.generate_kitchen_scene (which every build_*_kitchen*.py script
uses) for two reasons that matter here:

  * build_kitchen honours an explicit "mesh" on an object spec; generate_kitchen_scene picks one at
    random from everything matching the type. Pinning the mesh is what lets the run state which
    mug it grasped and check that mesh's grasp count beforehand.
  * build_kitchen derives the placement frame per MESH (_object_pose -> mesh_orientation.
    resolved_up), so a core_* mug is stood on its base with its canonical +Y vertical -- the frame
    the BODex grasp poses were authored in. generate_kitchen_scene decides that per TYPE.

commit_kitchen also adds the room shell, which is the feature this branch exists for.

Second half writes the task template. It is authored HERE, against the committed stage, because the
success condition needs the mug's resting height: "lifted" is only meaningful relative to the
support it was resting on, and that number is a property of the kitchen this script just built.

Importing kitchen_scene_generator boots Isaac Sim headless (and starts the wizard's HTTP server,
which nothing here answers -- harmless on a batch node). Run it with isaaclab.sh -p.
"""

import json
import hashlib
import os
import random
import sys

#: A new number, verified free of USDs, kitchen_data, task .py files, goal files and gym.register
#: entries before this script was written.
KNUM = int(os.environ.get("MUGK_NUM", "1200"))
SEED = int(os.environ.get("MUGK_SEED", str(KNUM % (2**32))))
if not 0 <= SEED < 2**32:
    raise ValueError("MUGK_SEED must be between 0 and 2**32 - 1")

#: l_shaped: the type every proven mug/grasp kitchen in this repo uses (kitchen_data_07,
#: _813, _910, _993 are all l_shaped), so the support labels the placer asks for exist.
KTYPE = os.environ.get("MUGK_TYPE", "l_shaped")

#: loc 1 = kitchen_build.PLACEMENTS "Above cabinet" -> countertop_base_cabinet, the main counter
#: run. Chosen over a fridge/wall shelf on purpose: the brief's "somewhere Anubis can actually
#: reach". At the default counter_height=0.95 this is the height band the recorded successful
#: Anubis mug lift worked at (handover7 measured the mug resting at z=0.903).
LOC = int(os.environ.get("MUGK_LOC", "1"))

#: Public acquisition example. This object is present in both archives at the pinned revision and
#: has left/right scale010 arrays plus a cached selected_indices.json, so scene and goal generation
#: do not depend on an unpublished mesh. Its minimum projected body span at scale 0.1 is about
#: 0.0867 m, wider than Anubis's nominal 0.080 m opening. It is therefore a reproducible authoring
#: input, not a claim of physical grasp success; use a robot-compatible mesh for collection.
MESH = os.environ.get(
    "MUGK_MESH",
    os.environ.get("BODEX_OBJ_DIR", "BODex_obj")
    + "/use_data/core_mug_39361b14ba19303ee42cfae782879837"
    "/mesh/simplified.obj",
)

#: Where the authored template lands. Deliberately NOT scripts/simvla/templates/: that directory is
#: shared, and this template is one run's artefact.
TEMPLATE_OUT = os.environ.get(
    "MUGK_TEMPLATE", "campaign_out/mugchain/mugchain_grasp.json"
)

#: How far above its resting height the mug must be to count as lifted, in metres. grasp_gate.
#: DELTA_LIFT_M -- the campaign's own lift threshold -- is 0.05; this is that with margin, so a
#: mug merely jostled on the counter cannot satisfy the emitted success term.
LIFT_M = float(os.environ.get("MUGK_LIFT_M", "0.08"))


def _fail(msg):
    print(f"[mugchain] FAILED: {msg}", flush=True)
    sys.exit(1)


# Boots Isaac Sim at import; everything below needs it.
import kitchen_scene_generator as ksg                              # noqa: E402
from kitchen_build import build_kitchen, pick_materials            # noqa: E402
from pxr import Usd, UsdGeom                                       # noqa: E402
import omni.usd                                                    # noqa: E402
import numpy as np                                                 # noqa: E402

# The scene builder's local generator is seeded separately below. Material
# choices and authored material IDs also use process RNGs; seed those explicitly
# so identical inputs do not silently reroll appearance on every headless run.
random.seed(SEED)
np.random.seed(SEED)

print(f"[mugchain] kitchen={KNUM} type={KTYPE} loc={LOC}\n[mugchain] mesh={MESH}", flush=True)
if not os.path.exists(MESH):
    _fail(f"mesh does not exist: {MESH}")

# ---------------------------------------------------------------- 1. build
spec = [{"obj_n": "mug0", "type": "mug", "loc": LOC, "mesh": MESH}]
# mesh_files=[] is safe and deliberate: it is only consulted by resolve_mesh, which build_kitchen
# calls only for a spec with no explicit "mesh". Passing [] skips loading the whole BODex dataset.
kitchen, kitchen_data, objects, supports = build_kitchen(KTYPE, spec, [], seed=SEED)
print(f"[mugchain] supports={len(supports)} objects={objects}", flush=True)
if not any(o["label"] == "mug0" for o in objects):
    _fail(
        f"the mug did not land on support for loc {LOC} on a {KTYPE} kitchen -- build_kitchen "
        f"placed {[o['label'] for o in objects]}. Nothing was written."
    )

# ---------------------------------------------------------------- 2. commit
picks = pick_materials()
usd_filename, json_path = ksg.commit_kitchen(kitchen, picks, KNUM, kitchen_data)
print(f"[mugchain] committed {usd_filename}\n[mugchain] committed {json_path}", flush=True)

# ---------------------------------------------------------------- 3. rotate + register
# ask_prim is the wizard's browser prim-picker, as a callback. Headless, the choice is fixed: the
# mug is the only rigid object in this kitchen, so the rotation set varies the thing the task is
# about. Returning None here is how the wizard's CANCEL is spelled, so an unexpected candidate list
# must raise rather than return None -- a cancel would write nothing and still exit 0.
def _pick_mug(candidates):
    print(f"[mugchain] rotation candidates: {candidates}", flush=True)
    mugs = [p for p in candidates if p.rsplit("/", 1)[-1] == "mug0"]
    if len(mugs) != 1:
        _fail(f"expected exactly one /world/mug0 among {candidates}")
    return mugs[0]


written = ksg.rotate_and_register_envs(KNUM, ask_prim=_pick_mug)
print(f"[mugchain] rotations written: {len(written)}", flush=True)
if len(written) != 12:
    _fail(f"rotate_and_register_envs exported {len(written)}/12 rotation USDs")

ksg.fix_missing_fixed_joint_targets_and_wall_cabinets(KNUM)

unusable = ksg._unusable_rotations(KNUM, written)
if unusable:
    _fail(f"{len(unusable)} of 12 rotations unusable; first: {unusable[0]}")
print("[mugchain] all 12 rotations usable", flush=True)

# ---------------------------------------------------------------- 4. measure the mug
# The resting height, off the committed stage rather than off the trimesh scene, so it is the same
# number the runtime will see.
#
# THE PRIM'S WORLD TRANSLATION, NOT ITS AABB CENTRE. These differ by half the mug's height (5.8 cm
# here) and picking the wrong one silently mis-calibrates the success term by that much. The
# predicate that consumes this is predicates_math.obj_z, which reads ctx.obj_pos_w[role][:, 2] --
# the RIGID BODY's world position, i.e. the prim origin. The AABB centre is what plan_arm_grasp
# uses for its grasp reference frame (simvla_data_generator: reference_frame[-1] = (min+max)/2),
# which is a different quantity for a different purpose; measured against a first run, anchoring
# `lo` to it demanded a 13.8 cm lift where 8 cm was meant, and a genuinely held mug (peak 1.021)
# fell short of it.
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/world/mug0")
if not prim:
    _fail("/world/mug0 is not on the committed stage")
world_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
rest_z = float(world_xf.ExtractTranslation()[2])
rng = (
    UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    .ComputeWorldBound(prim)
    .ComputeAlignedRange()
)
lo_z, hi_z = rng.GetMin()[2], rng.GetMax()[2]
print(
    f"[mugchain] mug0 body-origin z={rest_z:.4f} (the obj_z anchor); "
    f"world AABB z=[{lo_z:.4f}, {hi_z:.4f}] centre={(lo_z + hi_z) / 2.0:.4f} "
    f"height={hi_z - lo_z:.4f}",
    flush=True,
)

# ---------------------------------------------------------------- 5. author the goal template
# grasp_templates.build_grasp_template is the campaign's own generator: nav -> arm.grasp (BODex
# grasp data, because mug is in scene_spec.GRASPABLE_TYPES) -> gripper close -> arm.reset (lift).
# Two edits, both forced by the code as it stands today:
#
#   roles   -- it emits `container` and `container_handle` alongside `target`, inherited from the
#              vase archetypes. NOTHING in these four steps references them (@target only), but
#              task_bind.bind_roles raises BindingError when a kitchen has no handled drawer, or
#              more than one. This task never opens a drawer; carrying the roles only adds a way
#              to fail.
#   success -- task_template.validate_template REQUIRES a success condition, and
#              build_grasp_template declares none (nor does any file in scripts/simvla/templates/).
#              So task_emit refuses its output as shipped. Composed here from the predicate
#              registry as the honest reading of "held": the mug is above where it was resting,
#              AND the right gripper is on it. obj_z alone would pass on a mug knocked off the
#              counter; obj_near_eef alone is the false-positive predicate_contract.spec_warnings
#              explicitly warns about.
from grasp_templates import build_grasp_template                   # noqa: E402

t = build_grasp_template("mug", "single")
t["name"] = f"mugchain_grasp_k{KNUM}"
t["roles"] = [r for r in t["roles"] if r["name"] == "target"]
# task_emit reads the SCENE FROM THE STAGE (_scene_from_stage); this field is only consulted when a
# template is used to synthesise a new scene, which is not what is happening here.
t["scene"] = []
t["success"] = {
    "all": [
        {"obj_z": {"role": "@target", "lo": round(rest_z + LIFT_M, 4)}},
        {"obj_near_eef": {"role": "@target", "arm": "right", "radius": 0.15}},
    ]
}

os.makedirs(os.path.dirname(TEMPLATE_OUT), exist_ok=True)
with open(TEMPLATE_OUT, "w") as f:
    json.dump(t, f, indent=2)
print(f"[mugchain] wrote template {TEMPLATE_OUT}", flush=True)
print(json.dumps(t["success"], indent=2), flush=True)

with open(MESH, "rb") as mesh_stream:
    mesh_sha256 = hashlib.sha256(mesh_stream.read()).hexdigest()
facts = {
    "kitchen_num": KNUM,
    "seed": SEED,
    "room_shell_seed": KNUM,
    "material_picks": picks,
    "mesh_sha256": mesh_sha256,
    "kitchen_type": KTYPE,
    "loc": LOC,
    "mesh": MESH,
    "usd": usd_filename,
    "kitchen_data": json_path,
    "rotations": written,
    "mug_rest_z": rest_z,
    "mug_aabb_z": [lo_z, hi_z],
    "mug_aabb_centre_z": (lo_z + hi_z) / 2.0,
    "success_lo": round(rest_z + LIFT_M, 4),
    "template": TEMPLATE_OUT,
    "task_id_00": f"Isaac-Kitchen-v{KNUM:02d}-00",
}
facts_path = os.path.join(os.path.dirname(TEMPLATE_OUT), f"kitchen_{KNUM}_facts.json")
with open(facts_path, "w") as f:
    json.dump(facts, f, indent=2)
print(f"[mugchain] wrote {facts_path}", flush=True)
print("[mugchain] DONE", flush=True)
