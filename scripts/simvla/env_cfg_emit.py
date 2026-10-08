"""SimVLA: which top-level kitchen prims get an env-cfg block, and what that block says.

Split out of simvla_data_generator for the same reason skills.py was: that module imports pxr,
omni.usd, tkinter and PIL at module scope, so nothing living in it can be unit-tested, and both
loops below decide something no source-shape test can see -- what the EMITTED task file contains.

The stage read stays behind, injected as `probe`: a prim's world transform and joint names, or
its world-space AABB. What is here is the composition, which is where two shipped defects lived:

  * The room shell's walls carry no RigidBodyAPI. scene_synthesizer's Scene.add_walls defaults to
    joint_type=None; add_object attaches joint-edge metadata only when joint_type is not None; and
    the USD exporter sets rigid_body_api=True only inside the branch that needs that metadata.
    Every other top-level kitchen object takes add_object's joint_type="fixed" default, which is
    why the pre-shell corpus works. A RigidObjectCfg pointed at a wall makes RigidObject's
    _initialize_impl raise "Failed to find a rigid body when resolving ...", so every task file
    generated from a room-shell kitchen was dead at load.

  * The same prim walk feeds the free-space search. Left in, each wall is an obstacle AND drags
    world_bounds out to the walls' outer AABB, so the corridor between the furniture and the wall
    reads as free floor and gets promoted into initial_pos_ranges. That one does not crash: it
    just spawns the robot behind the counter run, in a random share of every collection run.

Both skips come from kitchen_build.room_shell_prim_names(), never from a "wall_" prefix --
wall_cabinet* is wall-MOUNTED furniture and must keep its rigid body and its footprint.

Imports kitchen_build (pure: scene_synthesizer / trimesh / numpy) and nothing else, so this is
testable on a laptop.
"""

from __future__ import annotations

import math
import os

import random
import re
from typing import Callable, Iterable, Optional, Sequence

from kitchen_build import room_shell_prim_names

#: FURNITURE DOORS AND DRAWERS MUST HOLD THEIR POSITION.
#:
#: These joints were emitted with stiffness 0.0 and damping 1.0 -- FREE SWINGING. init_state sets
#: every one to 0.0 (closed), and they then fall open under gravity within a few frames of the
#: reset, because nothing holds them. By the time the robot arrives, cabinet doors and dishwasher
#: hatches stand open across its path: obstacles that block the base approach and hand cuRobo a
#: scene it cannot plan through. Seen directly on the bowl-to-sink run (job 2093919) -- 10
#: plan_fail_r and 10 A_r across 48 episodes, with the doors open in shot.
#:
#: STIFFNESS RATHER THAN FRICTION. Friction only stops motion, so a door that starts ajar stays
#: ajar; stiffness actively returns it to the commanded 0.0, and "closed at the start" is the
#: requirement. Damping rises with it because at the shipped 1.0 a 50 N/rad spring would set the
#: door oscillating instead of settling.
#:
#: TASKS THAT OPEN FURNITURE MUST TURN THIS OFF. Every *_to_drawer template pulls a handle and
#: expects the drawer to STAY pulled; a spring would drag it shut behind the robot. Setting
#: SIMVLA_FURNITURE_STIFFNESS=0 restores the previous emission exactly.
#:
#: THE VALUES ARE UNVERIFIED IN SIM: chosen to be clearly enough to hold a door against gravity
#: while staying well inside the 87 N.m effort limit. First thing to measure if doors are seen
#: drifting open, or slamming.
#: RAISED 50 -> 300 ON OBSERVATION: at 50 the doors were seen OPEN mid-episode in some envs of the
#: bowl run (env2/env3) while closed in others (env0). So the spring held against GRAVITY but not
#: against the ROBOT touching them -- the base pushes with 2000 N and the arm with 300 N.m, against
#: which 50 N/rad was never going to survive. Per-env variation was the tell: a scene-wide failure
#: would have opened every env identically.
FURNITURE_STIFFNESS = float(os.environ.get("SIMVLA_FURNITURE_STIFFNESS", "300.0") or 0)
FURNITURE_DAMPING = (float(os.environ.get("SIMVLA_FURNITURE_DAMPING", "30.0") or 0)
                     if FURNITURE_STIFFNESS > 0 else 1.0)



#: probe(prim_path) -> (pos, quat, joint_names), already rounded the way the config wants them.
ObjectProbe = Callable[[str], tuple]

#: bbox_probe(prim_path) -> (min_x, min_y, max_x, max_y), or None when the prim has no extent.
BBoxProbe = Callable[[str], Optional[Sequence[float]]]


def object_blocks(
    first_level_list: Iterable[str],
    probe: ObjectProbe,
    log: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """One RigidObjectCfg / ArticulationCfg source block per top-level kitchen prim.

    Skips, in order: Looks (a material scope, not an object), the room shell's walls (no rigid
    body -- see the module docstring) and wall cabinets, which the env config does not drive.
    Everything else gets exactly one block of its own.

    wall_cabinet* is still PROBED before it is dropped. It is furniture, not room shell -- the
    two are told apart by room_shell_prim_names, never by a "wall_" prefix -- so a wall cabinet
    the stage cannot answer for is worth the same warning as any other prim.
    """
    shell = room_shell_prim_names()
    blocks: list[str] = []
    for path in first_level_list:
        try:
            name = path.split("/")[-1]
            if name == "Looks":
                continue
            if name in shell:
                continue
            pos, quat, joint_names = probe(path)
            joint_pos, actuator_block = {j: 0.0 for j in joint_names}, ""
            if not "wall_cabinet" in name:
                if joint_names:
                    actuator_block = f''',\n\t\t\t\t\tactuators={{\n\t\t\t\t\t\t"default": ImplicitActuatorCfg(joint_names_expr=[{", ".join(f'"{j}"' for j in joint_names)}],effort_limit=87.0,velocity_limit=100.0,stiffness={FURNITURE_STIFFNESS},damping={FURNITURE_DAMPING})\n\t\t\t\t\t\t}}'''
                block = f"""{name} = ArticulationCfg(prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",spawn=None,init_state=ArticulationCfg.InitialStateCfg(pos={pos}, rot={quat}, joint_pos={joint_pos}){actuator_block})""" if joint_names else f"""{name} = RigidObjectCfg(prim_path="{{ENV_REGEX_NS}}/Kitchen/{name}",spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos={pos}, rot={quat}))"""
                # Inside the `if`, which is the whole fix. The append used to sit outside it,
                # so a wall_cabinet iteration re-appended the PREVIOUS prim's `block` -- which
                # is why kitchen_12_00.py repeats sink_cabinet six times (its own block, plus
                # one per wall cabinet that followed it). Two consequences, one cosmetic and
                # one not: every task file carried duplicate assignments of identical
                # statements, and `block` could be read before it was ever bound. The room
                # shell made that second path reachable -- a scene whose only prims sorting
                # before the first wall cabinet are Looks and shell walls now leaves `block`
                # unbound, and the loop's own `except` reports the UnboundLocalError as
                # "Could not generate config for ...", which reads like a stage failure and is
                # not one. A wall cabinet contributes no block, which is what it always meant.
                blocks.append(block)
        except Exception as e:
            if log is not None:
                log(f"Warning: Could not generate config for {path}: {e}")
    return blocks


def furniture_bounds(
    first_level_list: Iterable[str],
    bbox_probe: BBoxProbe,
    log: Optional[Callable[[str], None]] = None,
) -> list[tuple[float, float, float, float]]:
    """The XY footprints the free-space search treats as obstacles, and whose union is
    world_bounds.

    The room shell is excluded: a wall is not furniture to walk around, it is the edge of the
    room, and including it turns the whole perimeter corridor into a robot spawn range.

    A prim with no extent contributes nothing rather than a degenerate box at the origin.
    """
    shell = room_shell_prim_names()
    bounds: list[tuple[float, float, float, float]] = []
    for path in first_level_list:
        try:
            if path.split("/")[-1] in shell:
                continue
            aabb = bbox_probe(path)
            if aabb is None:
                continue
            min_x, min_y, max_x, max_y = (float(v) for v in aabb)
            bounds.append((min_x, min_y, max_x, max_y))
        except Exception as e:
            if log is not None:
                log(f"Warning: Could not process prim {path}: {e}")
    return bounds


# --------------------------------------------------------------------------------------
# Lighting
# --------------------------------------------------------------------------------------

#: The marker block the template wraps its lighting defaults in, replaced wholesale below.
#: Same idiom (and same file) as "# -------Change-------" ... "# -------Stop-------"; the
#: names differ so neither non-greedy pattern can end on the other's terminator.
_LIGHTING_BLOCK = re.compile(
    r"# -------Lighting-------.*?# -------StopLighting-------", re.DOTALL
)

#: name -> (low, high), and colours are per CHANNEL. Sphere 1500-5000 is half to ~1.7x the
#: pre-randomization 3000; distant 500-2000 brackets the old 1000; colour stays a near-white
#: tint around the old 0.75. Narrow on purpose: the head camera is a 200 degree fisheye, and a
#: scene it cannot expose is a wasted episode. The template carries the same four ranges as
#: random.uniform calls, which is what a reader of the template sees.
LIGHTING_RANGES: dict[str, tuple[float, float]] = {
    "sphere_light_intensity": (1500.0, 5000.0),
    "distant_light_intensity": (500.0, 2000.0),
    "sphere_light_color": (0.65, 0.85),
    "distant_light_color": (0.65, 0.85),
}


def lighting_seed(kitchen_num):
    """The seed for `kitchen_num`, normalized.

    kitchen_num arrives from _write_env_config_file as format_num's zero-padded STRING
    ("07"), so seeding on the raw value would light kitchen "07" and kitchen 7 differently
    -- and format_num falls back to str() for anything non-numeric, which is the only way
    a caller can produce the un-padded spelling. Normalizing to the integer makes the two
    the same kitchen. A genuinely non-numeric id is handed to Random() as a string, which
    seeds from a hash of it that is stable across runs and platforms.
    """
    try:
        return int(kitchen_num)
    except (TypeError, ValueError):
        return str(kitchen_num)


def lighting_values(kitchen_num) -> dict:
    """This kitchen's four lighting values: two floats and two (r, g, b) tuples.

    Seeded from the kitchen number ALONE. The sub-variant is deliberately not part of the
    seed: kitchen_12_00 .. kitchen_12_11 are twelve camera rotations of one built room, and
    a room that changes colour when the camera turns is a worse artifact than one that does
    not vary at all.

    A private random.Random, never the module-level `random`: the global is shared with
    whatever else the generator process is doing, so seeding it would be both fragile and
    rude.
    """
    rng = random.Random(lighting_seed(kitchen_num))

    def channel(name):
        low, high = LIGHTING_RANGES[name]
        return round(rng.uniform(low, high), 4)

    # Draw order is part of the contract: changing it re-lights every kitchen.
    return {
        "sphere_light_intensity": channel("sphere_light_intensity"),
        "distant_light_intensity": channel("distant_light_intensity"),
        # Three independent draws, not one value broadcast to three channels: a broadcast
        # value only changes grey level, while per-channel gives the warm/cool cast that
        # actually differs between a real kitchen at noon and one at dusk.
        "sphere_light_color": tuple(channel("sphere_light_color") for _ in range(3)),
        "distant_light_color": tuple(channel("distant_light_color") for _ in range(3)),
    }


def lighting_block(kitchen_num) -> str:
    """The literal source that replaces the template's lighting block, markers included."""
    values = lighting_values(kitchen_num)
    return "\n".join(
        [
            "# -------Lighting-------",
            f"# Baked in at emission, seeded from kitchen {lighting_seed(kitchen_num)!r}",
            "# -- see env_cfg_emit.lighting_values. Literals, never a sampling call: this",
            "# module is imported once per training/eval process, so a draw made here would",
            "# re-light the scene on every launch, share one value across every parallel env,",
            "# and light a replay differently from the episode it replays. The sub-variant",
            "# number is not part of the seed, so all 12 rotations of this kitchen match.",
            f"sphere_light_intensity = {values['sphere_light_intensity']!r}",
            f"distant_light_intensity = {values['distant_light_intensity']!r}",
            f"sphere_light_color = {values['sphere_light_color']!r}",
            f"distant_light_color = {values['distant_light_color']!r}",
            "# -------StopLighting-------",
        ]
    )


def substitute_lighting(config_text: str, kitchen_num) -> str:
    """Bake this kitchen's lighting into the template text.

    Raises ValueError when the markers are gone, rather than returning the text unchanged:
    a silent no-op reinstates the whole defect (a task file that re-randomizes its lighting
    at import) in a file nobody re-reads afterwards.

    The replacement is a lambda, not a string, for the same reason the Change block's is --
    re.sub reinterprets a string replacement as a template, so a backslash in it becomes a
    group reference or a bad escape. Float repr never emits one today; passing a function
    means it cannot start to.
    """
    if not _LIGHTING_BLOCK.search(config_text):
        raise ValueError(
            "env-cfg template has no '# -------Lighting-------' block; the emitted task "
            "file would keep the template's random.uniform() calls and re-randomize its "
            "lighting at import in every training process"
        )
    block = lighting_block(kitchen_num)
    return _LIGHTING_BLOCK.sub(lambda _m: block, config_text)


# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------

#: The marker block the template wraps its floor/wall material choice in, replaced wholesale
#: below. Same idiom, and same file, as "# -------Lighting-------" / "# -------StopLighting----
#: ---" and "# -------Change-------" / "# -------Stop-------". Named distinctly from both: none
#: of the three "Stop*" terminators is a substring of another (StopMaterials and StopLighting
#: both continue past "Stop" with a letter, never the seven dashes "# -------Stop-------" needs
#: next), so no non-greedy pattern here can be fooled into ending on a different block's marker.
_MATERIALS_BLOCK = re.compile(
    r"# -------Materials-------.*?# -------StopMaterials-------", re.DOTALL
)

#: Candidate finishes, kept in exact sync with kitchen_env_cfg_source.py's floor_list /
#: wall_list. Duplicated here rather than parsed out of the template, for the same reason
#: LIGHTING_RANGES duplicates the template's random.uniform bounds instead of reading them back:
#: the template imports isaaclab at module scope, so neither this module nor the tests that
#: exercise it can import it to read the lists off the live objects.
FLOOR_MATERIALS: tuple[str, ...] = (
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak_Planks.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Parquet_Floor.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks.mdl",
)

WALL_MATERIALS: tuple[str, ...] = (
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Adobe_Brick.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Brick_Pavers.mdl",
    "{NVIDIA_NUCLEUS_DIR}/Materials/Base/Masonry/Concrete_Block.mdl",
)


def materials_values(kitchen_num) -> dict:
    """This kitchen's floor and wall material path, one random.choice from each list.

    Drawn off a random.Random INSTANCE OF ITS OWN -- never lighting_values' -- and seeded
    from its OWN derivation of lighting_seed(kitchen_num), not the identical value
    lighting_values seeds from. Two random.Random objects never share mutable state, so a
    second instance can never shift a draw lighting_values already made or will make later,
    regardless of call order -- but that is a weaker guarantee than it sounds: two
    independent Random objects seeded with the SAME value still walk the same MT19937
    stream from the same starting point, so their early outputs are correlated with each
    other even though neither object's state is shared. That correlation is not
    theoretical here: seeding this function with the bare lighting_seed(kitchen_num)
    measured corr(sphere_light_intensity, floor_material_index) ~ 0.28 and
    corr(distant_light_intensity, wall_material_index) ~ 0.20 over 3000 kitchens --
    baked-in enough for a vision policy trained on this corpus to pick "brighter kitchen"
    up as a proxy for "which floor", which is exactly the spurious cue per-kitchen
    randomization exists to avoid. See test_the_materials_are_not_correlated_with_this_
    kitchens_lighting in test_kitchen_env_cfg_lighting.py for the direct measurement and
    a threshold that fails on that regression.

    The seed is mixed into a string, not a bare int or a tuple: random.Random falls back
    to hash() for any seed type other than None/int/float/str/bytes/bytearray (see
    random.Random.seed's source), and hash() of a str -- or of a tuple containing one --
    is salted per-process by PYTHONHASHSEED, which would make materials_values
    nondeterministic across runs and break "same kitchen_num -> same materials, always".
    An f-string seed instead takes random.seed's str branch, which hashes the bytes with
    SHA-512 and never calls hash() at all, so it is stable across interpreter invocations
    regardless of hash randomization -- confirmed empirically, not just from reading the
    source.
    """
    rng = random.Random(f"materials:{lighting_seed(kitchen_num)!r}")
    return {
        "floor_material": rng.choice(FLOOR_MATERIALS),
        "wall_material": rng.choice(WALL_MATERIALS),
    }


def materials_block(kitchen_num) -> str:
    """The literal source that replaces the template's materials block, markers included."""
    values = materials_values(kitchen_num)
    return "\n".join(
        [
            "# -------Materials-------",
            f"# Baked in at emission, seeded from kitchen {lighting_seed(kitchen_num)!r}",
            "# -- see env_cfg_emit.materials_values. Literals, never a sampling call: this",
            "# module is imported once per training/eval process, so a draw made here would",
            "# share one floor/wall across every parallel env, repaint the scene on every",
            "# process launch, and paint a replay differently from the episode it replays.",
            f"floor_material = {values['floor_material']!r}",
            f"wall_material = {values['wall_material']!r}",
            "# -------StopMaterials-------",
        ]
    )


def substitute_materials(config_text: str, kitchen_num) -> str:
    """Bake this kitchen's floor/wall material into the template text.

    Raises ValueError when the markers are gone, rather than returning the text unchanged --
    same reasoning as substitute_lighting: a silent no-op reinstates the whole defect (a task
    file that re-picks its floor and wall material at import) in a file nobody re-reads
    afterwards.

    The replacement is a lambda, not a string, for the same reason substitute_lighting's is --
    re.sub reinterprets a string replacement as a template, so a backslash in it becomes a group
    reference or a bad escape. Passing a function means the splice is inserted verbatim.
    """
    if not _MATERIALS_BLOCK.search(config_text):
        raise ValueError(
            "env-cfg template has no '# -------Materials-------' block; the emitted task "
            "file would keep the template's random.choice() calls and re-pick its floor and "
            "wall material at import in every training process"
        )
    block = materials_block(kitchen_num)
    return _MATERIALS_BLOCK.sub(lambda _m: block, config_text)


# --------------------------------------------------------------------------------------
# Scene camera
# --------------------------------------------------------------------------------------

#: The marker block the template wraps scene_cam's pos/rot in, replaced wholesale below. Same
#: idiom as "# -------Lighting-------" and "# -------Materials-------", and named so that no
#: terminator is a prefix-with-dashes of another: "# -------StopSceneCam-------" continues past
#: "Stop" with a letter, so the Change block's non-greedy ".*?# -------Stop-------" cannot end on
#: it (and this block sits after the Change block in the template regardless).
_SCENE_CAM_BLOCK = re.compile(
    r"# -------SceneCam-------.*?# -------StopSceneCam-------", re.DOTALL
)

#: The pos/rot lines live three tabs deep, inside TiledCameraCfg.OffsetCfg(, inside the
#: TiledCameraCfg(, inside the scene class. The Lighting and Materials blocks are at MODULE level
#: and so emit unindented lines; this one cannot, and an unindented splice is an IndentationError
#: at import of the emitted task file rather than a bad picture. test_scene_cam_pose asserts the
#: emitted file still parses.
_SCENE_CAM_INDENT = "\t\t\t"

#: scene_cam's shipped pose: eye, and the (w, x, y, z) rotation the eye looks along.
#:
#: THIS IS KITCHEN 1400'S CAMERA AND ONLY KITCHEN 1400'S. It is a look-at from (-1.25,-2.55,2.25)
#: to (0.25,-0.55,0.95) -- the floor in front of 1400's fridge -- and it is what every kitchen in
#: the corpus is emitted with, because nothing between the template and the task file ever looked
#: at the room. It is kept EXACTLY, to the digit, as the default: the mug/fridge tasks that this
#: framing was tuned for are scored off videos shot with it, and re-aiming every kitchen to fix
#: one would silently re-frame all of them.
SCENE_CAM_DEFAULT_POS = (-1.25, -2.55, 2.25)
SCENE_CAM_DEFAULT_ROT = (0.8688422, -0.1061994, 0.2123989, 0.4344211)

#: Kitchens whose scene_cam is aimed at something other than 1400's fridge floor, as
#: (eye, look_at) in the KITCHEN's own frame -- the same frame the emitted object poses and the
#: room shell are measured in, because scene_cam is parented to {ENV_REGEX_NS} and not to the
#: robot.
#:
#: A TABLE AND NOT A NEW DEFAULT, on purpose. The default pose above is wrong for most of the
#: corpus in the same way it is wrong here (it is outside kitchen 1215's and kitchen 1218's walls
#: entirely), but it is the framing the shipped mug-on-countertop demos were judged against, and
#: a computed-per-kitchen default would re-frame every one of them in a single re-emit with no
#: way to tell which videos came from which rule. An entry here re-aims exactly one kitchen.
#:
#: 1218 -- push-chair. The task is a table at (1.25,-2.99) with two chairs north of it at
#: (0.20,-1.85) and (1.25,-1.70) that a robot working the corridor above them pushes back in;
#: none of that is anywhere near 1400's aim point, which is on the counter run at y = -0.41, so
#: every frame shot with the default is of the fridge and the worktop with a wall across the left
#: of it -- the table projects entirely OUTSIDE the frame (|u| up to 2.32) and both chairs are
#: cut by an edge.
#:
#: The eye is high in the room's north-east quarter, above the wall cabinets on the north run
#: (their tops are at z 1.967, the wall top is 2.375) looking down and south-west across the
#: chairs to the table. It is not a corner picked by eye. It is the outcome of a search over
#: every camera position in the measured room that
#:   * is inside the shell with 0.10 m clearance and not inside any prim's box;
#:   * frames the table and BOTH chairs -- at their pulled-out AND their pushed-in poses -- with
#:     every one of their 8 bounding-box corners inside 92% of the half-frame at the template's
#:     own 14 mm lens (so the lens does NOT change, and no other kitchen's optics move);
#:   * frames all six of the task's nav poses, each as a 0.60 m x 0.60 m x 1.19 m box (Anubis's
#:     measured height, kitchen_preview.ROBOT_HEIGHTS_M);
#:   * keeps the sightline to those prims clear of the counter run, the wall cabinets, the range
#:     and the range hood.
#: Measured, not chosen: check_scene_cam.py --num 1218 --kitchen <usd> re-derives all of it off
#: the USD, and test_scene_cam_pose.py asserts it on CPU against the same measurements.
#:
#: WHAT IT DOES NOT SEE, stated because a check that hid it would be worse than no check: the
#: robot's SPAWN band (x[2.378,2.596] y[-3.061,-2.071]) is behind the range hood from here, so
#: the robot enters frame from the lower right a second or two into the episode rather than
#: starting in it. No position in this room both frames the whole task at 14 mm and sees that
#: band -- the room is 3.8 x 4.3 m and the task fills it.
SCENE_CAM_AIMS: dict = {
    1218: ((2.30, 0.20, 2.30), (0.80, -1.50, 0.45)),
    #: Kitchen 1219 -- the one-chair bimanual push. A SIDE view, not the overhead-ish one 1218 got,
    #: and the difference is the whole point of the run: the question this video has to answer is
    #: HOW HIGH THE HANDS ARE, and height read down a camera looking along the push axis is
    #: foreshortened into nothing. The push runs along -y, so the eye sits out on +x, roughly
    #: level with the robot's shoulder, and the arm's posture is across the frame.
    #: Checked with check_scene_cam.py --num 1219 against this kitchen's own USD.
    #: Kitchen 1219 -- the one-chair bimanual push. FROM BEYOND THE TABLE, looking back up the
    #: push axis, and both halves of that are the shot: the robot FACES the camera (a pose behind
    #: it films its back, which is what the first attempt did) and the table sits in the near
    #: foreground, so "the chair goes in to the table" is legible instead of happening off-screen.
    #: Slightly off-axis in x so the arm's HEIGHT is still read across the frame rather than
    #: foreshortened, which is the question this whole run exists to answer.
    #: Verified with check_scene_cam.py --num 1219: eye clear of every prim, chair and robot 1.00
    #: in frame at both the approach and the push. The table reads 0.38 in frame because its far
    #: corners fall outside -- it is the foreground, not a target.
    1219: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    #: Kitchen 1225 -- the same room, seed and chair as 1219; only the TASK differs (the base
    #: parks 0.10 m further back so the arms reach further forward). So it takes 1219's camera
    #: verbatim rather than a fresh one: an identical scene filmed from a different angle would
    #: make the two runs look like different experiments when the only thing that changed is the
    #: arm. Re-checked against kitchen_1220.usd, not assumed.
    1225: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    #: Kitchen 1226 -- same room, seed and chair again; the base parks 0.05 m further back still.
    #: Same camera for the same reason: the only thing that differs between 1219, 1225 and 1226 is
    #: the arm, and filming them from three angles would make one change look like three
    #: experiments.
    1226: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    #: Kitchen 1233 -- same room, seed and chair; the push now aims at the TABLE rather than at the
    #: success radius, so the chair ends 0.150 m from the table edge instead of 0.499. Same camera
    #: again: four rounds filmed identically is what makes them comparable.
    1233: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    #: Kitchen 1234 -- the same room again; the success radius now sits just outside the SEATED gap
    #: rather than at the midpoint, so the chair comes to rest 0.249 m from the table edge instead
    #: of 0.499. Same camera for the same reason as 1225/1226/1233.
    1234: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    #: Kitchen 1235 -- same room again; the success condition now also requires the arms home, so
    #: the episode runs through the retreat and the reset instead of ending mid-push. Same camera.
    1235: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
    1236: ((2.60, -3.60, 1.70), (1.20, -1.90, 0.85)),
}


def look_at_world_quat(eye, target) -> tuple:
    """The (w, x, y, z) an Isaac Lab convention="world" camera at `eye` needs to look at `target`.

    convention="world" means the camera looks along its own +X, with +Y left and +Z up, so the
    rotation matrix's COLUMNS are [forward, left, up] expressed in world. Getting that wrong is
    one of the two ways this camera has already been pointed at nothing (the other is putting the
    eye outside the room); a hand-written quaternion is the other. Nothing here is hand-written.

    Raises ValueError for a degenerate aim -- eye == target, or an aim straight up or straight
    down, where "left" is not defined by the world up vector. Both would otherwise produce a NaN
    quaternion, which renders a black frame and reads as a lighting bug.
    """
    ex, ey, ez = (float(v) for v in eye)
    tx, ty, tz = (float(v) for v in target)
    fx, fy, fz = tx - ex, ty - ey, tz - ez
    fn = math.sqrt(fx * fx + fy * fy + fz * fz)
    if fn < 1e-9:
        raise ValueError(f"scene_cam look-at is degenerate: eye {eye} == target {target}")
    fx, fy, fz = fx / fn, fy / fn, fz / fn
    # left = world_up x forward, which for world_up = (0, 0, 1) is (-fy, fx, 0).
    lx, ly = -fy, fx
    ln = math.sqrt(lx * lx + ly * ly)
    if ln < 1e-6:
        raise ValueError(
            f"scene_cam look-at from {eye} to {target} is vertical; 'left' is undefined against "
            f"the world up vector, so the roll would be arbitrary"
        )
    lx, ly, lz = lx / ln, ly / ln, 0.0
    # up = forward x left, already unit because forward and left are unit and perpendicular.
    ux = fy * lz - fz * ly
    uy = fz * lx - fx * lz
    uz = fx * ly - fy * lx
    # Columns [forward, left, up] -> m[row][col].
    m = ((fx, lx, ux), (fy, ly, uy), (fz, lz, uz))
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > -0.99:
        w = math.sqrt(1.0 + trace) / 2.0
        s = 4.0 * w
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    else:
        # trace near -1 makes the branch above divide by ~0. Pivot on the largest diagonal term.
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i][i] - m[j][j] - m[k][k]) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (m[j][i] + m[i][j]) / s
        q[k] = (m[k][i] + m[i][k]) / s
        w = (m[k][j] - m[j][k]) / s
        x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return tuple(round(v / n, 7) for v in (w, x, y, z))


def scene_cam_values(kitchen_num) -> dict:
    """This kitchen's scene_cam pose: {"pos": (x, y, z), "rot": (w, x, y, z), "aim": ... }.

    Un-overridden kitchens get SCENE_CAM_DEFAULT_* back unchanged, to the digit, so every kitchen
    already emitted re-emits byte-for-byte identical. `aim` is None for those -- the default's
    look-at point is recorded in the template's comment, not recomputed here, precisely so that
    re-deriving it can never round one of the shipped literals.
    """
    entry = SCENE_CAM_AIMS.get(lighting_seed(kitchen_num))
    if entry is None:
        return {"pos": SCENE_CAM_DEFAULT_POS, "rot": SCENE_CAM_DEFAULT_ROT, "aim": None}
    eye, aim = entry
    return {
        "pos": tuple(float(v) for v in eye),
        "rot": look_at_world_quat(eye, aim),
        "aim": tuple(float(v) for v in aim),
    }


def scene_cam_block(kitchen_num) -> str:
    """The literal source that replaces the template's scene_cam pos/rot, markers included."""
    values = scene_cam_values(kitchen_num)
    if values["aim"] is None:
        note = [
            "# scene_cam's default pose -- kitchen 1400's, and the corpus's. This kitchen has no",
            "# entry in env_cfg_emit.SCENE_CAM_AIMS, so the shipped literals are emitted",
            "# unchanged. If this kitchen's task is not on the counter run, it is not in shot:",
            "# check with check_scene_cam.py --num <n> --kitchen <usd> before filming it.",
        ]
    else:
        note = [
            f"# scene_cam re-aimed for kitchen {lighting_seed(kitchen_num)}: a look-at from",
            f"# {values['pos']} to {values['aim']}, computed at emission by",
            "# env_cfg_emit.look_at_world_quat -- see SCENE_CAM_AIMS for why this kitchen has its",
            "# own camera and what the pose was measured against. Do not hand-edit the",
            "# quaternion here; it is overwritten on the next emit.",
        ]
    lines = (
        ["# -------SceneCam-------"]
        + note
        + [
            f"pos={values['pos']!r},",
            f"rot={values['rot']!r},",
            "# -------StopSceneCam-------",
        ]
    )
    return ("\n" + _SCENE_CAM_INDENT).join(lines)


def substitute_scene_cam(config_text: str, kitchen_num) -> str:
    """Bake this kitchen's scene_cam pose into the template text.

    Raises ValueError when the markers are gone rather than returning the text unchanged -- same
    reasoning as substitute_lighting: a silent no-op here re-emits kitchen 1400's camera into a
    kitchen that overrode it, and the only symptom is a video of the wrong half of the room,
    discovered after a GPU run rather than before one.
    """
    if not _SCENE_CAM_BLOCK.search(config_text):
        raise ValueError(
            "env-cfg template has no '# -------SceneCam-------' block; the emitted task file "
            "would keep kitchen 1400's hard-coded scene_cam pose even for a kitchen with an "
            "entry in SCENE_CAM_AIMS, and film the wrong part of the room"
        )
    block = scene_cam_block(kitchen_num)
    return _SCENE_CAM_BLOCK.sub(lambda _m: block, config_text)
