"""
SimVLA: Motion planning sub-goal generator
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
import re
import tkinter as tk
import traceback
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
_ = app_launcher.app

import os as _simvla_os
from pathlib import Path as _SimvlaPath

import numpy as np
import omni.usd
import scene_synthesizer as synth
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scene_synthesizer import datasets, utils
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer.exchange.usd_export import (
    add_mdl_material,
    bind_material_to_prims,
)
from scene_synthesizer.usd_import import get_scene_paths


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

import simvla_paths
import scene_starters
import task_template
from kitchen_build import OBJECT_TYPES, add_fixtures, real_placement_dims
from scene_spec import (
    Extent,
    default_object,
    kitchen_surfaces,
    placement_decision,
    sampled_lift,
)


# -----------------------
# Active scene
# -----------------------
# The task's scene (list of scene_spec.SceneObject) is the single source of what to place and
# where. Replaces the old per-task TASK_PROFILES: `--scene <template.json>` loads a TaskTemplate
# and uses its `.scene`; with no flag we default to the profile-3 starter for continuity.
_ACTIVE_SCENE = scene_starters.PROFILE_SCENES["3"]

# Door-handle lift height when no scene object declares a lift. Equals the old profile target_z
# range (0.75, 0.80); Extent(0.775, 0.025) reproduces that band.
_DOOR_HANDLE_DEFAULT_LIFT = Extent(0.775, 0.025)


# -----------------------
# Paths / constants
# -----------------------
KITCHEN_DIR = str(simvla_paths.assets_dir() / "Kitchen")
BODEX_DIR = os.path.join(KITCHEN_DIR, "bodex")
GOAL_DIR = str(simvla_paths.goals_dir())
ISAAC_KITCHEN_INIT = f"{SIMVLA_REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/__init__.py"


# -----------------------
# Materials
# -----------------------
DEFAULT_TEXTURE_SCALE = 0.25
URL_MDL_MATERIAL = (
    "http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/"
)

MATERIALS: dict[str, list[tuple[str, str | None, float | None]]] = {
    "sink": [
        ("vMaterials_2/Metal/Aluminum.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("vMaterials_2/Metal/Aluminum_Brushed.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("vMaterials_2/Metal/Aluminum_Foil.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("vMaterials_2/Metal/Aluminum.mdl", "Aluminum_Polished", DEFAULT_TEXTURE_SCALE),
        ("vMaterials_2/Metal/Aluminum_Sheet.mdl", None, DEFAULT_TEXTURE_SCALE),
    ],
    "countertop": [
        ("vMaterials_2/Metal/Copper_Hammered.mdl", "Copper_Hammered_Shiny", 0.5),
        ("Base/Stone/Marble.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Granite_Dark.mdl", "Granite_Dark", DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Granite_Light.mdl", None, DEFAULT_TEXTURE_SCALE),
        (
            "vMaterials_2/Metal/Stainless_Steel_Milled.mdl",
            "Stainless_Steel_Milled_Worn",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("Base/Stone/Terrazzo.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Slate.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Porcelain_Tile_4.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Porcelain_Tile_4_Linen.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Ceramic_Tile_12.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Porcelain_Smooth.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Bamboo.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Cherry.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Oak.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Oak_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Ash.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Ash_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Walnut.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Walnut_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("vMaterials_2/Stone/Terrazzo.mdl", None, DEFAULT_TEXTURE_SCALE),
        (
            "vMaterials_2/Stone/Stone_Natural_Black.mdl",
            "Stone_Natural_Black_Shiny",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Stone/Steel_Grey.mdl",
            "Steel_Grey_Bright",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("vMaterials_2/Stone/Basaltite.mdl", "Basaltite_Worn", DEFAULT_TEXTURE_SCALE),
    ],
    "glass": [("Base/Glass/Tinted_Glass_R85.mdl", None, DEFAULT_TEXTURE_SCALE)],
    "tinted glass": [("Base/Glass/Tinted_Glass_R75.mdl", None, DEFAULT_TEXTURE_SCALE)],
    "cabinet": [
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_White",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Vanilla",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Cashmere",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Peach",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Taupe",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Leaf",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Ash",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Denim",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Light_Denim",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("Base/Wood/Bamboo.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Cherry.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Oak.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Oak_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Birch_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Ash.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Ash_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Walnut.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Walnut_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
    ],
    "rusted metal": [("Base/Metals/RustedMetal.mdl", None, DEFAULT_TEXTURE_SCALE)],
    "glossy black": [
        (
            "vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl",
            "Black",
            DEFAULT_TEXTURE_SCALE,
        )
    ],
    "handle": [("vMaterials_2/Metal/Silver_Foil.mdl", None, DEFAULT_TEXTURE_SCALE)],
    "appliances": [
        ("vMaterials_2/Metal/Aluminum_Brushed.mdl", None, DEFAULT_TEXTURE_SCALE)
    ],
    "wall": [
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Pale_Rose",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Lime",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_White",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Vanilla",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Cashmere",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Peach",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Taupe",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Paint/Paint_Eggshell.mdl",
            "Paint_Eggshell_Leaf",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
            "Wood_Tiles_Pine_Brickbond",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
            "Ceramic_Tiles_Diamond_Mint",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
            "Ceramic_Tiles_Diamond_Red_Varied",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
            "Ceramic_Tiles_Diamond_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl",
            "Ceramic_Tiles_Offset_Diamond_Graphite_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
            "Ceramic_Tiles_Diamond_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Glazed_Subway",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl",
            "Facade_Brick_Red_Clinker_Painted_White",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl",
            "Facade_Brick_Red_Clinker_Painted_Yellow",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl",
            "Facade_Brick_Red_Clinker_Sloppy_Paint_Job",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Plaster/Plaster_Wall.mdl",
            "Plaster_Wall",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Plaster/Plaster_Wall.mdl",
            "Plaster_Wall_Cracked",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_Cappucino",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_White",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_White_Worn_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_Gray",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_Dark_Gray_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl",
            "Ceramic_Tiles_Subway_Dark_Gray",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl",
            "Ceramic_Tiles_Penny_Antique_White",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl",
            "Ceramic_Tiles_Penny_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl",
            "Ceramic_Tiles_Penny_Lime_Green_Varied",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl",
            "Ceramic_Tiles_Penny_Graphite_Varied",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl",
            "Ceramic_Tiles_Penny_Mint_Varied",
            DEFAULT_TEXTURE_SCALE,
        ),
    ],
    "floor": [
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl",
            "Ceramic_Tiles_Diamond_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("Base/Wood/Parquet_Floor.mdl", "Parquet_Floor", DEFAULT_TEXTURE_SCALE),
        (
            "vMaterials_2/Concrete/Concrete_Floor_Damage.mdl",
            "Concrete_Floor_Damage",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Wood/Wood_Tiles_Beech.mdl",
            "Wood_Tiles_Beech_Herringbone",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("Base/Stone/Adobe_Octagon_Dots.mdl", "Adobe_Octagon_Dots", None),
        (
            "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
            "Wood_Tiles_Pine_Brickbond",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
            "Wood_Tiles_Pine_Herringbone",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Wood/Wood_Tiles_Pine.mdl",
            "Wood_Tiles_Pine_Mosaic",
            DEFAULT_TEXTURE_SCALE,
        ),
        ("Base/Wood/Oak.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Wood/Oak_Planks.mdl", None, DEFAULT_TEXTURE_SCALE),
        ("Base/Stone/Terracotta.mdl", "Terracotta", None),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl",
            "Ceramic_Tiles_Versailles_Antique_White_Dirty",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl",
            "Ceramic_Tiles_Versailles_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Square.mdl",
            "Ceramic_Tiles_Square_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl",
            "Ceramic_Tiles_Pinwheel_White_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl",
            "Ceramic_Tiles_Pinwheel_Antique_White_Dirty",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Paseo.mdl",
            "Ceramic_Tiles_Paseo_White_Worn_Matte",
            DEFAULT_TEXTURE_SCALE,
        ),
        (
            "vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl",
            "Ceramic_Tiles_Offset_Diamond_Antique_White_Dirty",
            DEFAULT_TEXTURE_SCALE,
        ),
    ],
    "white plastic": [
        (
            "vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl",
            "Plastic_Thick_Translucent",
            DEFAULT_TEXTURE_SCALE,
        ),
    ],
}


#: MDLs whose `texture_scale` is a FLOAT, not a float2.
#:
#: scene_synthesizer's add_mdl_material (exchange/usd_export.py) ALWAYS creates the input as
#: Sdf.ValueTypeNames.Float2 and sets a 2-tuple. For an MDL that declares the parameter as a
#: plain float, Hydra then refuses the binding:
#:     [UsdToMdl] '.../Paint_Eggshell_*/Shader' parameter 'texture_scale':
#:     Reached invalid assignment. Tried to assign a 'GfVec2f'(USD) to a 'float'(MDL).
#: and every prim bound to that material renders untextured -- black. Paint_Eggshell is the WALL
#: material for every variant in MATERIALS["wall"], which is why generated kitchens looked like
#: they had no walls at all; it is also half of MATERIALS["cabinet"].
#:
#: 48 of these errors per run went unnoticed because the run scripts filtered "UsdToMdl" out of
#: the logs. Passing texture_scale=None makes add_mdl_material skip the input entirely, so the
#: material binds and uses its own default scale.
_FLOAT_TEXTURE_SCALE_MDLS = ("Paint_Eggshell.mdl",)


def _safe_texture_scale(mtl_url, texture_scale):
    """None for MDLs that declare texture_scale as a float; unchanged otherwise."""
    return None if any(m in mtl_url for m in _FLOAT_TEXTURE_SCALE_MDLS) else texture_scale


GEOMETRY2MATERIAL: dict[str, str] = {
    "(.*cabinet.*corpus.*|.*cabinet.*door|.*drawer.*board.*|.*cabinet.*closed.*|/world/kitchen_island/.*)|/world/corner.*": "cabinet",
    "(.*refrigerator.*|.*range_hood.*|.*range.*|.*dishwasher.*)": "sink",
    ".*countertop.*": "countertop",
    ".*/corpus/sink|.*/corpus/sink_countertop": "sink",
    "(/world/wall/geometry.*|/world/wall_(x|y|_y|_x)/geometry_0)": "wall",
    "/world/floor/geometry.*": "floor",
    ".*handle.*": "handle",
}


# -----------------------
# Template helpers
# -----------------------
def create_preview_surface_material(
    stage,
    mat_path,
    base_color=(0.72, 0.72, 0.72),
    metallic=0.8,
    roughness=0.45,
):
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")

    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*base_color)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_random_silver_material(stage, mat_path, seed=None):
    rng = random.Random(seed)

    # neutral silver: keep RGB close to each other
    gray = rng.uniform(0.62, 0.82)

    # tiny channel variation to avoid every sink looking identical
    eps_r = rng.uniform(-0.015, 0.015)
    eps_g = rng.uniform(-0.015, 0.015)
    eps_b = rng.uniform(-0.015, 0.015)

    base_color = (
        max(0.0, min(1.0, gray + eps_r)),
        max(0.0, min(1.0, gray + eps_g)),
        max(0.0, min(1.0, gray + eps_b)),
    )

    metallic = rng.uniform(0.72, 0.92)
    roughness = rng.uniform(0.25, 0.3)

    return create_preview_surface_material(
        stage=stage,
        mat_path=mat_path,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
    )


# Object-material randomisation lives in object_materials.py (pxr + stdlib only) so that
# kitchen_scene_generator can import it without dragging Isaac in. See that module's
# docstring for the two build failures that forced the split.

def is_generated_goal_file(path: str) -> bool:
    base = os.path.basename(path)
    return bool(re.match(r"Isaac-Kitchen-v\d{1,}-\d{2}\.reloadable\.json$", base))


def list_reloadable_templates(goal_dir: str) -> tuple[list[str], bool]:
    """
    Returns (paths, only_generated).
    If no "template-like" files exist, it returns generated ones too but flags only_generated=True.
    """
    os.makedirs(goal_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(goal_dir, "*.reloadable.json")))
    templ = [f for f in files if is_generated_goal_file(f)]
    if templ:
        return templ, False
    return files, True


def load_reloadable_template(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "goals" in data and isinstance(data["goals"], list) and data["goals"]:
        return data
    raise ValueError(f"Template {path} does not have top-level 'goals' list.")


def _collect_prim_paths(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "prim_path" and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_collect_prim_paths(v))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_collect_prim_paths(x))
    return out


def _base_obj_type_from_prim_path(p: str) -> str | None:
    name = p.split("/")[-1].lower()
    base = re.sub(r"\d+$", "", name)
    if base in OBJECT_TYPES:
        return base
    return None


def infer_object_list_from_template(scene: list, seed: int) -> list[dict]:
    """Per-kitchen placement list derived from the task's scene. A thin consumer of
    scene_spec.placement_decision, which owns the whole of the (pure, seedable) sampling."""
    return placement_decision(scene, seed)


def _objects_to_place_from_decision(decision: list[dict]) -> list[dict]:
    """Adapt a placement_decision list into the {obj_n, type, surface, dims, lift} rows the
    USD placement path consumes. The decision is already one entry per object, so its `name`
    is the obj_n and no per-type counting is needed."""
    return [
        {
            "obj_n": e["name"],
            "type": e["object_type"],
            "surface": e["surface"],
            "dims": e["dims"],
            "lift": e["lift"],
            "mesh": e.get("mesh"),
        }
        for e in decision
    ]


def _target_z_from_objects(objects_to_place: list[dict], seed: int, log=print) -> float:
    """Door-handle lift height. Historically a per-task range sampled on the GPU; now the lift of
    the manipulated scene object if one declares it, else the historical (0.75, 0.80) band. The
    fallback is logged so an implicit default is never silent."""
    lifts = [o["lift"] for o in objects_to_place if o.get("lift") is not None]
    if lifts:
        return float(lifts[0])
    z = sampled_lift(_DOOR_HANDLE_DEFAULT_LIFT, seed)
    log(
        f"[target_z] no scene object declares a lift; using default door-handle height "
        f"{z:.3f} (sampled from {_DOOR_HANDLE_DEFAULT_LIFT.center}+/-"
        f"{_DOOR_HANDLE_DEFAULT_LIFT.spread})"
    )
    return float(z)


def _scene_from_roles(template) -> list:
    """Empty-scene fallback: one default object per role that references an object_type, taken
    from scene_spec.SCENE_DEFAULTS. No clutter — only the objects the roles actually need."""
    scene = []
    counts: dict[str, int] = {}
    for r in template.roles:
        if r.object_type is None:
            continue
        i = counts.get(r.object_type, 0)
        counts[r.object_type] = i + 1
        scene.append(default_object(f"{r.object_type}{i}", r.object_type))
    return scene


# -----------------------
# Kitchen ID allocation
# -----------------------
def next_kitchen_id() -> int:
    os.makedirs(BODEX_DIR, exist_ok=True)
    max_id = -1
    for path in glob.glob(os.path.join(BODEX_DIR, "kitchen_data_*.json")):
        base = os.path.basename(path)
        try:
            num_str = os.path.splitext(base)[0].split("_")[-1]
            max_id = max(max_id, int(num_str))
        except Exception:
            continue
    return max_id + 1


# -----------------------
# Kitchen generation (procedural)
# -----------------------
KITCHEN_FUNCS = {
    "island": ps.kitchen_island,
    "l_shaped": ps.kitchen_l_shaped,
    "peninsula": ps.kitchen_peninsula,
    "u_shaped": ps.kitchen_u_shaped,
    "single_wall": ps.kitchen_single_wall,
}


def _label_supports(kitchen, kitchen_name: str) -> None:
    # Registers exactly the labels scene_spec.kitchen_surfaces reports for this layout. Both that
    # function and this map now derive from kitchen_build.PLACEMENTS, so the guard in
    # generate_kitchen_scene, the labels created here, and the generator's menu share one source.
    from kitchen_build import support_geom_ids
    geom_ids = support_geom_ids()
    for label in kitchen_surfaces(kitchen_name):
        kitchen.label_support(label=label, geom_ids=geom_ids[label])


def _choose_place_support(kitchen_name: str, loc: int) -> str | None:
    from kitchen_build import PLACEMENT_BY_LOC, LOCATION_KITCHENS
    placement = PLACEMENT_BY_LOC.get(loc)
    if placement is None or kitchen_name not in LOCATION_KITCHENS.get(loc, frozenset()):
        return None
    return placement.key


#: The two orientation frames the placer supports, expressed as (up, front, origin-per-axis).
#: Z_UP stands an object on its base with its canonical +Z vertical; Y_UP with its canonical +Y
#: vertical. The BODex grasp poses are authored in the mesh's canonical frame, so the object MUST be
#: placed with its canonical up vertical or every grasp candidate is rotated ~90 deg off (the arm
#: then reaches empty air and the sub-grasp check collects 0/N — the 'cup lies on its side' bug).
_Z_UP = ((0, 0, 1), (0, 1, 0), ("com", "com", "bottom"))
_Y_UP = ((0, 1, 0), (0, 0, -1), ("com", "bottom", "com"))
_X_UP = ((1, 0, 0), (0, 0, -1), ("bottom", "com", "com"))


def _clutter_up_axis(obj_type: str):
    """The canonical up (symmetry) axis of a clutter object, read off its mesh extents in scene_spec:
    a surface-of-revolution object has a circular cross-section, so two of its three canonical extents
    are ~equal and the third — the height/symmetry axis — is the up axis. Returns 'x'/'y'/'z', or None
    for a box-like object with no two ~equal extents (keep the caller's default). Only clutter carries
    an extents tuple; the original graspables (bottle/bowl/mug/...) are not in CLUTTER and return None,
    so their proven orientation is untouched."""
    from scene_spec import CLUTTER
    entry = CLUTTER.get(obj_type)
    dims = entry.get("dims") if entry else None
    if not dims or len(dims) != 3:
        return None
    dx, dy, dz = dims

    def close(a, b):
        return a > 0 and b > 0 and abs(a - b) / max(a, b) < 0.12

    if close(dx, dy):   # circular in the XY plane -> Z is the symmetry axis
        return "z"
    if close(dx, dz):   # circular in the XZ plane -> Y is the symmetry axis
        return "y"
    if close(dy, dz):   # circular in the YZ plane -> X is the symmetry axis
        return "x"
    return None


def _orient_for_object(obj_type: str):
    if obj_type in ["apple", "sodacan"]:
        return _Z_UP
    axis = _clutter_up_axis(obj_type)
    if axis == "z":
        return _Z_UP
    if axis == "x":
        return _X_UP
    # 'y' and box-like/None (and every original graspable) keep the historical Y-up placement.
    return _Y_UP


class PositionIteratorGridStartXCenter(utils.PositionIteratorGrid):
    """Start scan at x=center (i index), but y starts at min (j=0)."""

    def __call__(self, support):
        super().__call__(support)  # sets polygon, start/end, i=j=0, etc.

        minx, miny, maxx, maxy = self.polygon.bounds
        cx = (minx + maxx) / 2.0

        # start x-index near center; keep y-index at 0 (miny)
        i0 = int(round((cx - minx) / self.step[0]))
        self.i = max(0, i0)
        self.j = 2

        return self


def generate_kitchen_scene(
    kitchen_num: int,
    kitchen_type: str,
    objects_to_place: list[dict],
    mesh_files: list[str],
) -> str:
    os.makedirs(BODEX_DIR, exist_ok=True)
    os.makedirs(KITCHEN_DIR, exist_ok=True)

    if kitchen_type not in KITCHEN_FUNCS:
        raise ValueError(f"Unknown kitchen_type: {kitchen_type}")

    usd_filename = os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num:02d}.usd")

    kitchen_data: dict[str, str] = {"kitchen_type": kitchen_type}
    height = torch.empty((), device="cuda").uniform_(0.8, 0.95).item()
    print(height)

    low, high = 0.02, 0.05
    handle_h = torch.empty(()).uniform_(low, high)
    low, high = 0.1, 0.15
    handle_w = torch.empty(()).uniform_(low, high)
    low, high = 0.01, 0.03
    handle_d = torch.empty(()).uniform_(low, high)

    kitchen = KITCHEN_FUNCS[kitchen_type](
        seed=None,
        counter_height=height,
        base_cabinet_args={
            "num_drawers_horizontal": 1,
            "num_drawers_vertical": 1,
            "handle_height": handle_h.item(),
            "handle_width": handle_w.item(),
            "handle_depth": handle_d.item(),
        },
    )

    kitchen.unwrap_geometries(
        r"(sink_cabinet/sink_countertop|countertop_.*|.*countertop)"
    )
    # This batch path and kitchen_build.build_kitchen (the wizard/GUI path) must agree on what
    # a kitchen contains: support_geom_ids()/kitchen_surfaces() -- which _label_supports below
    # reads from -- advertise fixture placements (e.g. loc 40 "microwave_interior") to both
    # paths, derived from the same kitchen_build.PLACEMENTS/LOCATION_KITCHENS table. Without
    # this call, a kitchen built here has no microwave/table node, so _label_supports finds no
    # supports for those labels, and a later support_generator(support_ids=...) call KeyErrors
    # even though kitchen_surfaces(kitchen_type) says the location exists. Fixtures before
    # supports, same reason as build_kitchen: _label_supports walks the scene graph, so
    # anything added after it would carry no support labels. seed=None here (this path is
    # unseeded), so an unseeded default_rng() matches existing behaviour.
    add_fixtures(kitchen, kitchen_type, np.random.default_rng(), height)
    _label_supports(kitchen, kitchen_type)

    for obj_info in objects_to_place:
        obj_n = obj_info["obj_n"]
        obj_type = obj_info["type"]
        if obj_type not in OBJECT_TYPES:
            continue

        # Global Constraint: an object placed on a surface this kitchen layout lacks is loudly
        # skipped for this kitchen with a named reason, never silently dropped. The valid set is
        # scene_spec.kitchen_surfaces(kitchen_type) — the same source _label_supports registers
        # from — so we never hand support_generator a label the kitchen never created.
        surface = obj_info["surface"]
        if surface and surface not in kitchen_surfaces(kitchen_type):
            print(
                f"[scene] skipping {obj_n!r} ({obj_type}): surface {surface!r} not on "
                f"kitchen type {kitchen_type!r}"
            )
            continue

        candidates = [elmt for elmt in mesh_files if obj_type in elmt.lower()]
        if not candidates:
            raise RuntimeError(f"No mesh files found for object type: {obj_type}")
        fname = random.choice(candidates)

        kitchen_data[obj_n] = fname

        # surface + dims now come from the scene's placement_decision, not from loc/_ACTIVE_PROFILE.
        place = obj_info["surface"]
        if not place:
            continue

        up, front, origin = _orient_for_object(obj_type)

        # Scale each object to its real-world size (kitchen_build.REAL_HEIGHT), preserving aspect,
        # instead of the arbitrary BODex mesh scale. No-op for types without a real height.
        scale = real_placement_dims(obj_type, obj_info["dims"])
        kitchen.place_objects(
            obj_id_iterator=utils.object_id_generator(obj_n),
            obj_asset_iterator=synth.assets.asset_generator(
                itertools.repeat(fname, 1),
                scale=scale,
                up=up,
                front=front,
                origin=origin,
                align=True,
            ),
            obj_support_id_iterator=kitchen.support_generator(support_ids=place),
            #            obj_position_iterator=utils.PositionIteratorGrid(step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04),
            obj_position_iterator=PositionIteratorGridStartXCenter(
                step_x=0.03, step_y=0.03, noise_std_x=0.0001, noise_std_y=0.0001
            ),
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(
                lower=0.0, upper=0.0
            ),
        )
    stage = kitchen.export(file_type="usd")

    # materials
    for geom_regex, material_group in GEOMETRY2MATERIAL.items():
        paths = get_scene_paths(
            stage=stage,
            prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
            scene_path_regex=geom_regex,
        )
        if not MATERIALS.get(material_group):
            continue
        mtl_url, mtl_name, texture_scale = random.choice(MATERIALS[material_group])
        texture_scale = _safe_texture_scale(mtl_url, texture_scale)
        if material_group == "sink":
            mtl = create_random_silver_material(
                stage,
                f"/World/Looks/SinkMetal_{random.randint(0, 10**9)}",
            )
        else:
            mtl = add_mdl_material(
                stage=stage,
                mtl_url=URL_MDL_MATERIAL + mtl_url,
                mtl_name=mtl_name,
                texture_scale=texture_scale,
            )
        bind_material_to_prims(stage=stage, material=mtl, prim_paths=paths)

    stage.Export(usd_filename)

    json_path = os.path.join(BODEX_DIR, f"kitchen_data_{kitchen_num:02d}.json")
    with open(json_path, "w") as f:
        json.dump(kitchen_data, f, indent=4)

    return usd_filename


def _dedup_append_register(init_file: str, block: str, unique_key: str) -> None:
    try:
        txt = open(init_file).read()
    except Exception:
        txt = ""
    if unique_key in txt:
        return
    with open(init_file, "a") as f:
        f.write("\n" + block + "\n")


def _apply_convex_decomposition_under(
    stage, root_path: str, verbose: bool = False
) -> int:
    """
    Set MeshCollisionAPI.approximation = convexDecomposition for all Mesh prims
    under root_path (including root itself if it is a Mesh).
    Returns number of prims changed.
    """
    root_path = root_path.rstrip("/")
    changed = 0

    for prim in stage.Traverse():
        p = prim.GetPath().pathString

        # only under root_path (root itself or descendants)
        if p != root_path and not p.startswith(root_path + "/"):
            continue

        # Only apply to Mesh prims (this will catch /World/mug0/simplified_obj/simplified_obj)
        if prim.GetTypeName() != "Mesh":
            continue

        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)

        mesh_api = UsdPhysics.MeshCollisionAPI(prim)
        approx_attr = mesh_api.GetApproximationAttr()
        if not approx_attr:
            approx_attr = mesh_api.CreateApproximationAttr()

        cur = approx_attr.Get()
        if cur != UsdPhysics.Tokens.convexDecomposition:
            approx_attr.Set(UsdPhysics.Tokens.convexDecomposition)
            changed += 1
            if verbose:
                print(f"[convexDecomposition] {p}: {cur!r} -> 'convexDecomposition'")

    return changed


def rotate_and_register_envs_auto(
    kitchen_num: int,
    rotate_obj: str | None,
    target_z: float,
    num_rotations: int = 12,
    register_envs: bool = True,
    init_file: str = ISAAC_KITCHEN_INIT,
) -> list[str]:
    json_path = os.path.join(BODEX_DIR, f"kitchen_data_{kitchen_num:02d}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[Step2] JSON not found: {json_path}")

    kitchen_data = json.load(open(json_path))

    base_usd = os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num:02d}.usd")
    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(base_usd):
        raise RuntimeError(f"[Step2] Failed to open base USD: {base_usd}")
    stage = usd_context.get_stage()

    # Move door handle a bit up
    tc = Usd.TimeCode.Default()

    handle_path = "/world/base_cabinet/drawer_0_0/door_handle"
    prim = stage.GetPrimAtPath(handle_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"prim not found or invalid: {handle_path}")

    stage.SetEditTarget(stage.GetRootLayer())

    xformable = UsdGeom.Xformable(prim)
    cache = UsdGeom.XformCache(tc)

    world_M = cache.GetLocalToWorldTransform(prim)
    cur_world_t = world_M.ExtractTranslation()
    cur_world_z = float(cur_world_t[2])

    # target_z now comes from the scene's placement decision (the manipulated object's lift),
    # supplied by the caller, instead of being sampled from _ACTIVE_PROFILE["target_z"] here.
    dz_world = target_z - cur_world_z  # how far to lift, in world coords

    if dz_world > 0.2:
        raise RuntimeError(
            f"door_handle target_z ({target_z:.3f}) is {dz_world:.3f}m above current "
            f"world z ({cur_world_z:.3f}); refusing to generate goal."
        )

    parent = prim.GetParent()
    parent_M = cache.GetLocalToWorldTransform(parent)
    parent_inv = parent_M.GetInverse()

    delta_local = parent_inv.TransformDir(Gf.Vec3d(0.0, 0.0, dz_world))

    t_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
            break
    if t_op is None:
        t_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        cur_local_t = Gf.Vec3d(0.0, 0.0, 0.0)
    else:
        cur_local_t = t_op.Get(tc) or Gf.Vec3d(0.0, 0.0, 0.0)

    t_op.Set(cur_local_t + delta_local, tc)

    print(
        f"[Step2] door_handle world z {cur_world_z:.3f} -> {target_z:.3f} (dz_world={dz_world:.3f})"
    )

    for obj, fname in kitchen_data.items():
        if obj == "kitchen_type":
            continue
        old_path = f"/world/{obj}0"
        new_path = f"/world/{obj}"
        if stage.GetPrimAtPath(old_path):
            Sdf.CopySpec(stage.GetRootLayer(), old_path, stage.GetRootLayer(), new_path)
            stage.RemovePrim(old_path)
        obj_prim = stage.GetPrimAtPath(new_path)
        if obj_prim:
            attr = obj_prim.CreateAttribute("BODex_path", Sdf.ValueTypeNames.String)
            attr.Set(fname)
    candidates = [
        f"/world/{obj}"
        for obj in kitchen_data.keys()
        if obj != "kitchen_type" and stage.GetPrimAtPath(f"/world/{obj}")
    ]
    if not candidates:
        raise RuntimeError("[Step2] No candidate prims found to rotate.")

    if not rotate_obj or rotate_obj not in candidates:
        rotate_obj = candidates[0]
    # Every placed object, not just the rotation target. Task profiles place two objects and
    # `rotate_obj` is merely the first of them, so the other one used to be skipped entirely.
    # This helper is the only place that applies MeshCollisionAPI at all, so skipping it did not
    # leave the object with a coarse collider — it left it with NO collider. A bowl the robot is
    # supposed to grasp and put things into had nothing to collide with. Verified against a real
    # USD export: with the old single-object call, `bowl0` came out with no collision meshes.
    n_changed = sum(
        _apply_convex_decomposition_under(stage, prim_path, verbose=False)
        for prim_path in candidates
    )
    print(
        f"[Step2] convexDecomposition applied under {len(candidates)} object(s) "
        f"{candidates}: changed={n_changed}"
    )

    prim = stage.GetPrimAtPath(rotate_obj)
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    translate_ops = [
        op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    translate_op = translate_ops[0] if translate_ops else None
    rotate_ops = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    rotate_op = rotate_ops[0] if rotate_ops else None
    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()
    xformable.SetXformOpOrder(
        [translate_op, rotate_op] if translate_op else [rotate_op]
    )

    out_paths: list[str] = []

    if not rotate_obj or rotate_obj not in candidates:
        rotate_obj = candidates[0]

    for i in range(num_rotations):
        angle = i * (360.0 / float(num_rotations))
        rotate_op.Set(Gf.Vec3f(0.0, 0.0, float(angle)))
        usd_out = os.path.join(KITCHEN_DIR, f"kitchen_{kitchen_num:02d}_{i:02d}.usd")
        stage.GetRootLayer().Export(usd_out)
        out_paths.append(usd_out)

        if register_envs:
            unique_key = f'id="Isaac-Kitchen-v{kitchen_num:02d}-{i:02d}"'
            block = f"""
gym.register(
    id="Isaac-Kitchen-v{kitchen_num:02d}-{i:02d}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.kitchen_{kitchen_num:02d}_{i:02d}:AnubisKitchenEnvCfg",
    }},
)
""".rstrip()
            _dedup_append_register(init_file, block, unique_key)

    return out_paths

def make_hidden_goal_app():
    import simvla_data_generator as simvla

    root = tk.Tk()
    root.withdraw()
    app = simvla.GoalGeneratorApp(root)
    return root, app


def guess_rotate_obj_from_required(objects_to_place: list[dict]) -> str | None:
    return f"/world/{objects_to_place[0]['obj_n']}" if objects_to_place else None


def expected_goal_paths(kitchen_num: int, sub: int) -> tuple[str, str]:
    # same naming as _batch_generate_single
    k = f"{int(kitchen_num):02d}"
    s = f"{int(sub):02d}"
    base = os.path.join(GOAL_DIR, f"Isaac-Kitchen-v{k}-{s}.json")
    reloadable = os.path.join(GOAL_DIR, f"Isaac-Kitchen-v{k}-{s}.reloadable.json")
    return base, reloadable

def distribute_equally(total: int, keys: list[str]) -> dict[str, int]:
    base = total // len(keys)
    rem = total - base * len(keys)
    out = {k: base for k in keys}
    for i in range(rem):
        out[keys[i % len(keys)]] += 1
    return out


@dataclass
class BatchConfig:
    task_name: str
    template_path: str
    total_houses: int
    counts_by_type: dict[str, int]
    register_envs: bool
    run_cleanup: bool
    prompt_rotate: bool

class BatchHouseTaskGUI(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        self.master.title("Batch House+Goal Generator (Option 1, v2)")
        self.pack(fill=tk.BOTH, expand=True)

        os.makedirs(GOAL_DIR, exist_ok=True)
        os.makedirs(BODEX_DIR, exist_ok=True)
        os.makedirs(KITCHEN_DIR, exist_ok=True)

        self.mesh_files = datasets.load_dataset("BODex").get_filenames()
        self._goal_root = None
        self._goal_app = None
        self._goal_only_generated = False

        self._build_ui()
        self._refresh_templates()

    def _build_ui(self):
        tpl_frame = ttk.LabelFrame(
            self, text="1) Predefined tasks (reloadable templates)", padding=10
        )
        tpl_frame.pack(fill=tk.X, pady=(0, 10))

        self.tpl_list = tk.Listbox(tpl_frame, height=7)
        self.tpl_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tpl_btns = ttk.Frame(tpl_frame)
        tpl_btns.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        ttk.Button(tpl_btns, text="Refresh", command=self._refresh_templates).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(tpl_btns, text="Preview JSON", command=self._preview_template).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            tpl_btns, text="Script new task (editor)", command=self._open_task_editor
        ).pack(fill=tk.X, pady=2)

        cfg = ttk.LabelFrame(self, text="2) Batch settings", padding=10)
        cfg.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(cfg, text="Task name:").grid(row=0, column=0, sticky="w")
        self.task_name_var = tk.StringVar(value="test")
        ttk.Entry(cfg, textvariable=self.task_name_var, width=30).grid(
            row=0, column=1, sticky="w", padx=5
        )

        ttk.Label(cfg, text="Total houses:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.total_var = tk.IntVar(value=2)
        ttk.Entry(cfg, textvariable=self.total_var, width=10).grid(
            row=1, column=1, sticky="w", padx=5, pady=(8, 0)
        )

        dist = ttk.LabelFrame(
            cfg, text="Kitchen type distribution (sum must match total)", padding=10
        )
        dist.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.type_vars: dict[str, tk.IntVar] = {
            "island": tk.IntVar(value=0),
            "l_shaped": tk.IntVar(value=1),
            "peninsula": tk.IntVar(value=1),
            "u_shaped": tk.IntVar(value=0),
            "single_wall": tk.IntVar(value=0),
        }
        rr = 0
        for k in ["island", "l_shaped", "peninsula", "u_shaped", "single_wall"]:
            ttk.Label(dist, text=k).grid(row=rr, column=0, sticky="w")
            ttk.Entry(dist, textvariable=self.type_vars[k], width=8).grid(
                row=rr, column=1, sticky="w", padx=5
            )
            rr += 1
        ttk.Button(dist, text="Divide equally", command=self._on_divide_equally).grid(
            row=0, column=2, rowspan=rr, padx=10, sticky="ns"
        )

        opt = ttk.LabelFrame(cfg, text="Options", padding=10)
        opt.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.register_envs_var = tk.BooleanVar(value=True)
        self.cleanup_var = tk.BooleanVar(value=True)
        self.prompt_rotate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opt,
            text="Register envs (append gym.register)",
            variable=self.register_envs_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            opt, text="Run cleanup (Step3+4) if available", variable=self.cleanup_var
        ).pack(anchor="w")
        ttk.Checkbutton(
            opt,
            text="Prompt for which prim to rotate (used for 12 rotations)",
            variable=self.prompt_rotate_var,
        ).pack(anchor="w")
        ttk.Label(
            opt, text="Rotations: always 12 (00..11) | Kitchens: always brand-new"
        ).pack(anchor="w", pady=(6, 0))

        runf = ttk.Frame(self)
        runf.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            runf,
            text="Generate batch (kitchens + 12 rotations + goal jsons)",
            command=self._run_batch,
        ).pack(side=tk.LEFT)

        logf = ttk.LabelFrame(self, text="Log", padding=10)
        logf.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(logf, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, msg: str):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.master.update_idletasks()

    def _choose_rotate_prim_dialog(self, prim_paths, default=None):
        """Modal dialog to choose which prim (e.g., /world/bottle0) to rotate."""
        if not prim_paths:
            return None
        win = tk.Toplevel(self.master)
        win.title("Choose prim to rotate")
        win.transient(self.master)
        win.grab_set()

        ttk.Label(
            win, text="Select the prim to rotate (used to export 12 rotated USDs):"
        ).pack(padx=10, pady=(10, 5), anchor="w")

        listbox = tk.Listbox(win, height=min(12, len(prim_paths)), width=70)
        for p in prim_paths:
            listbox.insert(tk.END, p)
        listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Preselect default if provided
        if default in prim_paths:
            idx = prim_paths.index(default)
            listbox.selection_set(idx)
            listbox.see(idx)

        result = {"value": None}

        def on_ok(event=None):
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning(
                    "No selection", "Please select a prim to rotate.", parent=win
                )
                return
            result["value"] = prim_paths[int(sel[0])]
            win.destroy()

        def on_cancel():
            win.destroy()

        btn = ttk.Frame(win)
        btn.pack(pady=(5, 10))
        ttk.Button(btn, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)

        listbox.bind("<Double-Button-1>", on_ok)

        self.master.wait_window(win)
        return result["value"]

    def _refresh_templates(self):
        self.tpl_list.delete(0, tk.END)
        self._templates, self._goal_only_generated = list_reloadable_templates(GOAL_DIR)

        for p in self._templates:
            name = os.path.basename(p)
            if self._goal_only_generated and is_generated_goal_file(p):
                name = "[GENERATED] " + name
            self.tpl_list.insert(tk.END, name)

        if self._templates:
            self.tpl_list.selection_set(0)

        if self._goal_only_generated:
            self.log(
                "[WARN] No reusable templates found. Showing generated outputs too."
            )
            self.log(
                "       Use 'Script new task (editor)' and save a template file name (not Isaac-Kitchen-v...)."
            )

    def _selected_template_index(self) -> int | None:
        sel = self.tpl_list.curselection()
        if not sel:
            messagebox.showwarning(
                "No template selected", "Select a predefined task template first."
            )
            return None
        return int(sel[0])

    def _preview_template(self):
        idx = self._selected_template_index()
        if idx is None:
            return
        path = self._templates[idx]
        try:
            data = load_reloadable_template(path)
            s = json.dumps(data, indent=2)[:9000]
        except Exception as e:
            s = f"Failed to load: {e}"
        win = tk.Toplevel(self.master)
        win.title(f"Preview: {os.path.basename(path)}")
        t = tk.Text(win, width=120, height=35)
        t.pack(fill=tk.BOTH, expand=True)
        t.insert(tk.END, s)

    def _open_task_editor(self):
        import simvla_data_generator as simvla

        win = tk.Toplevel(self.master)
        win.title("Task editor (GoalGeneratorApp)")
        win.geometry("1250x750")
        simvla.GoalGeneratorApp(win)

    def _on_divide_equally(self):
        total = int(self.total_var.get())
        dist = distribute_equally(total, list(self.type_vars.keys()))
        for k, v in dist.items():
            self.type_vars[k].set(v)

    def _gather_config(self) -> BatchConfig:
        idx = self._selected_template_index()
        if idx is None:
            raise RuntimeError("No template selected.")
        template_path = self._templates[idx]
        _ = load_reloadable_template(template_path)  # validate

        task_name = self.task_name_var.get().strip()
        if not task_name:
            raise ValueError("Task name cannot be empty.")

        total = int(self.total_var.get())
        if total <= 0:
            raise ValueError("Total houses must be > 0.")

        counts = {k: int(v.get()) for k, v in self.type_vars.items()}
        s = sum(counts.values())
        if s != total:
            self.log(f"[WARN] distribution sum={s} != total={total}. Auto-adjusting...")
            if s <= 0:
                counts = distribute_equally(total, list(counts.keys()))
            else:
                scaled = {k: int(round(total * (c / s))) for k, c in counts.items()}
                diff = total - sum(scaled.values())
                keys = list(scaled.keys())
                for i in range(abs(diff)):
                    kk = keys[i % len(keys)]
                    scaled[kk] += 1 if diff > 0 else -1
                counts = scaled
            for k, v in counts.items():
                self.type_vars[k].set(v)

        return BatchConfig(
            task_name=task_name,
            template_path=template_path,
            total_houses=total,
            counts_by_type=counts,
            register_envs=bool(self.register_envs_var.get()),
            run_cleanup=bool(self.cleanup_var.get()),
            prompt_rotate=bool(self.prompt_rotate_var.get()),
        )

    def _ensure_goal_app(self):
        if self._goal_app is None:
            self._goal_root, self._goal_app = make_hidden_goal_app()

            # Forward internal GoalGeneratorApp logs into this GUI log:
            original_log = self._goal_app.log

            def forwarded_log(msg: str):
                try:
                    original_log(msg)
                except Exception:
                    pass
                self.log("[GOAL] " + str(msg))

            self._goal_app.log = forwarded_log

    def _run_cleanup_if_available(self, kitchen_num: int):
        if not self.cleanup_var.get():
            return
        try:
            import kitchen_scene_generator_for_batch as ksg

            if hasattr(ksg, "fix_missing_fixed_joint_targets_and_wall_cabinets"):
                ksg.fix_missing_fixed_joint_targets_and_wall_cabinets(kitchen_num)
                self.log(f"[CLEANUP] done for kitchen {kitchen_num:02d}")
            else:
                self.log(
                    "[CLEANUP] kitchen_scene_generator missing fix function; skipped."
                )
        except Exception as e:
            self.log(f"[CLEANUP] failed: {e}")
            self.log(traceback.format_exc())

    def _run_batch(self):
        try:
            cfg = self._gather_config()
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            return

        template_data = load_reloadable_template(cfg.template_path)
        # The population (names / types / surfaces) is seed-independent; use seed=0 for the preview.
        population = _objects_to_place_from_decision(
            infer_object_list_from_template(_ACTIVE_SCENE, seed=0)
        )

        self.log(f"[TASK] Template: {os.path.basename(cfg.template_path)}")
        if self._goal_only_generated and is_generated_goal_file(cfg.template_path):
            self.log(
                "[WARN] You selected a GENERATED file as a template. It may contain kitchen-specific prim_paths."
            )
        self.log(f"[TASK] Scene objects: {[o['obj_n'] for o in population]}")

        rotate_choice = None
        if cfg.prompt_rotate:
            preview_objects = population
            candidates = [f"/world/{o['obj_n']}" for o in preview_objects]
            default_name = guess_rotate_obj_from_required(preview_objects)
            default_path = (
                f"/world/{default_name}"
                if default_name
                else (candidates[0] if candidates else None)
            )
            rotate_choice = self._choose_rotate_prim_dialog(
                candidates, default=default_path
            )
            if rotate_choice is None:
                self.log("[CANCEL] Rotation selection cancelled.")
                return
            self.log(f"[STEP2] Using rotate prim for ALL kitchens: {rotate_choice}")

        schedule: list[str] = []
        for k, c in cfg.counts_by_type.items():
            schedule += [k] * int(c)
        schedule = schedule[: cfg.total_houses]
        random.shuffle(schedule)

        start_id = next_kitchen_id()
        self.log(f"[BATCH] Starting kitchen_num={start_id} | total={cfg.total_houses}")

        self._ensure_goal_app()
        goal_app = self._goal_app

        ok_kitchens = 0
        ok_goals = 0
        fail_kitchen = 0

        for idx in range(cfg.total_houses):
            kitchen_num = start_id + idx
            kitchen_type = schedule[idx]

            try:
                # build objects: one placement decision per kitchen, seeded by kitchen_num so
                # dims/lift vary per house while names/types/surfaces stay fixed.
                objects_to_place = _objects_to_place_from_decision(
                    infer_object_list_from_template(_ACTIVE_SCENE, seed=kitchen_num)
                )

                self.log(
                    f"\n[STEP1] kitchen {kitchen_num} | type={kitchen_type} | objects={len(objects_to_place)}"
                )
                _ = generate_kitchen_scene(
                    kitchen_num, kitchen_type, objects_to_place, self.mesh_files
                )
                ok_kitchens += 1

                rotate_obj_name = guess_rotate_obj_from_required(objects_to_place)
                rotate_obj = (
                    rotate_choice
                    if rotate_choice
                    else (f"/world/{rotate_obj_name}" if rotate_obj_name else None)
                )
                target_z = _target_z_from_objects(
                    objects_to_place, kitchen_num, log=self.log
                )
                self.log(f"[STEP2] rotate prim: {rotate_obj}")
                rotate_and_register_envs_auto(
                    kitchen_num=kitchen_num,
                    rotate_obj=rotate_obj,
                    target_z=target_z,
                    num_rotations=12,
                    register_envs=cfg.register_envs,
                )

                self._run_cleanup_if_available(kitchen_num)

                self.log(
                    f"[STEP5] generating goal jsons for kitchen {kitchen_num} sub 00..11"
                )
                success_this_kitchen = 0
                for sub in range(12):
                    ok = goal_app._batch_generate_single(
                        kitchen_num, sub, template_data, cfg.task_name
                    )
                    if ok:
                        success_this_kitchen += 1
                        ok_goals += 1
                    else:
                        self.log(
                            f"[FAIL] goal gen failed for v{kitchen_num}-{sub:02d} (see [GOAL] logs above)"
                        )
                        base, reloadable = expected_goal_paths(kitchen_num, sub)
                        # These won't exist on failure; but leaving this check here makes it obvious if naming differs.
                        if os.path.exists(base) or os.path.exists(reloadable):
                            self.log(
                                f"[NOTE] Some output exists anyway: {base} or {reloadable}"
                            )

                self.log(
                    f"[DONE] kitchen {kitchen_num} finished. goals_ok={success_this_kitchen}/12"
                )

            except Exception as e:
                fail_kitchen += 1
                self.log(f"[ERROR] kitchen {kitchen_num} failed: {e}")
                self.log(traceback.format_exc())

        self.log("\n================ SUMMARY ================")
        self.log(f"Created kitchens: {ok_kitchens}/{cfg.total_houses}")
        self.log(
            f"Goal generations succeeded (sub-level): {ok_goals}/{cfg.total_houses * 12}"
        )
        self.log(f"Failures (kitchen-level): {fail_kitchen}")
        self.log(f"Goals saved under: {GOAL_DIR}")
        self.log("=========================================")


def main():
    global _ACTIVE_SCENE
    parser = argparse.ArgumentParser(
        description="Goal generator (consolidated) — load a task's scene and launch the batch GUI."
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Path to a TaskTemplate JSON; its `scene` is used as the active scene. A template "
            "with an empty scene falls back to a logged default built from its roles. Without "
            "this flag the profile-3 starter scene is used, for continuity with the old --task 3."
        ),
    )
    args = parser.parse_args()
    if args.scene:
        with open(args.scene) as f:
            template = task_template.from_json(f.read())
        scene = template.scene
        if not scene:
            scene = _scene_from_roles(template)
            print(
                f"[goal_generator] template {template.name!r} has an empty scene; falling back "
                f"to a default scene from its roles at SCENE_DEFAULTS: {[o.name for o in scene]}"
            )
        _ACTIVE_SCENE = scene
        print(
            f"[goal_generator] active scene from {args.scene}: {[o.name for o in _ACTIVE_SCENE]}"
        )
    else:
        print(
            f"[goal_generator] active scene: default profile-3 starter "
            f"({[o.name for o in _ACTIVE_SCENE]})"
        )

    root = tk.Tk()
    root.geometry("980x820")
    BatchHouseTaskGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
