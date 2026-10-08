"""SimVLA: pure scene construction for the kitchen generator.

Imports only scene_synthesizer / trimesh / numpy / stdlib — deliberately no isaaclab,
omni or pxr, so this module (and therefore build_kitchen and apply_placements) can be
tested without booting Omniverse.
"""

from __future__ import annotations

import functools
import itertools
import math
import random
import re
from pathlib import Path

import numpy as np
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets
from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import utils

DEFAULT_TEXTURE_SCALE = 0.25
URL_MDL_MATERIAL = 'http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/'

MATERIALS = {
    'sink': [
        ('Base/Stone/Ceramic_Smooth_Fired.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'countertop': [
        ('vMaterials_2/Metal/Copper_Hammered.mdl', 'Copper_Hammered_Shiny', 0.5),
        ('Base/Stone/Marble.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Dark.mdl', 'Granite_Dark', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Granite_Light.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Metal/Stainless_Steel_Milled.mdl', 'Stainless_Steel_Milled_Worn', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Slate.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Tile_4_Linen.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Ceramic_Tile_12.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Porcelain_Smooth.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Terrazzo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Stone_Natural_Black.mdl', 'Stone_Natural_Black_Shiny', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Steel_Grey.mdl', 'Steel_Grey_Bright', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Stone/Basaltite.mdl', 'Basaltite_Worn', DEFAULT_TEXTURE_SCALE),
    ],
    'glass': [
        ('Base/Glass/Tinted_Glass_R85.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'tinted glass': [
        ('Base/Glass/Tinted_Glass_R75.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'cabinet': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Ash', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Denim', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Light_Denim', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Bamboo.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Cherry.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Birch_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Ash_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Walnut_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
    ],
    'rusted metal': [
        ('Base/Metals/RustedMetal.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'glossy black': [
        ('vMaterials_2/Paint/Carpaint/Carpaint_Solid.mdl', 'Black', DEFAULT_TEXTURE_SCALE)
    ],
    'handle': [
        ('vMaterials_2/Metal/Silver_Foil.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'appliances': [
        ('vMaterials_2/Metal/Aluminum_Brushed.mdl', None, DEFAULT_TEXTURE_SCALE)
    ],
    'wall': [
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Pale_Rose', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Lime', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Vanilla', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Cashmere', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Peach', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Taupe', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Paint/Paint_Eggshell.mdl', 'Paint_Eggshell_Leaf', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Mint', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_Red_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Graphite_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Glazed_Subway', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Painted_Yellow', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Masonry/Facade_Brick_Red_Clinker.mdl', 'Facade_Brick_Red_Clinker_Sloppy_Paint_Job', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Plaster/Plaster_Wall.mdl', 'Plaster_Wall_Cracked', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Cappucino', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Subway.mdl', 'Ceramic_Tiles_Subway_Dark_Gray', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Antique_White', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Lime_Green_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Graphite_Varied', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Penny.mdl', 'Ceramic_Tiles_Penny_Mint_Varied', DEFAULT_TEXTURE_SCALE),
    ],
    'floor': [
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond.mdl', 'Ceramic_Tiles_Diamond_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Parquet_Floor.mdl', 'Parquet_Floor', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Concrete/Concrete_Floor_Damage.mdl', 'Concrete_Floor_Damage', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Beech.mdl', 'Wood_Tiles_Beech_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Adobe_Octagon_Dots.mdl', 'Adobe_Octagon_Dots', None),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Brickbond', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Herringbone', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Wood/Wood_Tiles_Pine.mdl', 'Wood_Tiles_Pine_Mosaic', DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Wood/Oak_Planks.mdl', None, DEFAULT_TEXTURE_SCALE),
        ('Base/Stone/Terracotta.mdl', 'Terracotta', None),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Versailles.mdl', 'Ceramic_Tiles_Versailles_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Square.mdl', 'Ceramic_Tiles_Square_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_White_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Pinwheel.mdl', 'Ceramic_Tiles_Pinwheel_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Paseo.mdl', 'Ceramic_Tiles_Paseo_White_Worn_Matte', DEFAULT_TEXTURE_SCALE),
        ('vMaterials_2/Ceramic/Ceramic_Tiles_Glazed_Diamond_Offset.mdl', 'Ceramic_Tiles_Offset_Diamond_Antique_White_Dirty', DEFAULT_TEXTURE_SCALE),
    ],
    'white plastic': [
        ('vMaterials_2/Plastic/Plastic_Thick_Translucent.mdl', 'Plastic_Thick_Translucent', DEFAULT_TEXTURE_SCALE),
    ],
}

GEOMETRY2MATERIAL = {
    "(.*cabinet.*corpus.*|.*cabinet.*door|.*drawer.*board.*|.*cabinet.*closed.*|/world/kitchen_island/.*)|/world/corner.*": "cabinet",
    "(.*refrigerator.*|.*range_hood.*|.*range.*|.*dishwasher.*|.*microwave.*)": "appliances",
    "/world/range/corpus/heater.*": 'rusted metal',
    "/world/range/corpus/top": 'glossy black',
    ".*/corpus/sink": 'sink',
    ".*countertop.*": "countertop",
    ".*handle.*": "handle",
    "/world/range/.*window": 'tinted glass',
    "(.*glass|.*_window)": 'glass',
    "/world/plate.*": 'sink',
    "(/world/wall/geometry.*|/world/wall_(x|y|_y|_x)(/geometry.*)?)": 'wall',
    "/world/floor/geometry.*": 'floor',
    # The table gets the floor's wood, not its own group: same MATERIALS['floor'] list, a
    # second GEOMETRY2MATERIAL entry pointing at it (nodes_to_primpaths gives table its own
    # /world/table/... subtree, so this never collides with the room's actual floor tiles).
    # "/world/table/.*" (not ".*table.*") on purpose: the narrower anchor can never accidentally
    # swallow an unrelated fixture whose path merely contains the substring "table".
    #
    # ".*" for the CHILD is what makes this cover an Objaverse table as well as a procedural one,
    # and it is not a spare wildcard. A TableAsset contributes exactly five children it names
    # itself (leg_0..leg_3, top; see procedural_assets.TableAsset). A MeshAsset contributes
    # whatever its source OBJ's material group was called, which nothing in this repo chooses:
    # measured over the fifty offered mesh tables (2026-08-13) they arrive under 23 DISTINCT names
    # -- 22 "geometry_0", 7 "Object_0", 13 bare "<uid>.obj" and eight one-offs including
    # "Table 1_Poles_0", "Lay001_Material #106_0" and "Box001_02 - Default_0". Every one of the
    # fifty happens to contribute exactly ONE child, which _name_the_table_top then renames to
    # "top" -- so what this wildcard covers TODAY is a name this file does choose. It is written
    # as ".*" anyway because the child COUNT is the asset's to decide: a mesh arriving with its
    # top and its legs as separate material groups keeps those groups' names on every child but
    # the top, and a pattern written against one asset's shape is the defect the wall entry
    # shipped, where "/world/wall/geometry.*" matched geometry_0 and left the other three children
    # of every wall untextured.
    # test_every_table_geometry_node_ends_up_wood reads these names out of a real built scene
    # graph, one per offered table, rather than out of a constant.
    #
    # This entry needed no change to cover the mesh tables: they enter the graph under the same
    # object id "table" that add_table has always used, so the wildcard already reached them. It
    # is placed after ".*handle.*" and "(.*glass|.*_window)" so that a table whose OBJ happens to
    # name a mesh "handle" or "glass" still ends up wood -- apply_materials rebinds in insertion
    # order, so the last matching entry wins.
    "/world/table/.*": 'floor',
    # Chairs take the floor's wood too -- chair_manifest records material_group "floor" for every
    # accepted chair, so they match the table rather than getting a group of their own.
    #
    # ".*" for the child, NOT a guess at its name. A chair's geometry children are named by the
    # source OBJ's own material groups, so the leaf differs per asset: measured across the 50
    # offered variants (2026-08-13) they carry 24 DISTINCT names -- 16 "geometry_0", 10
    # "Object_0", 3 "defaultMaterial", and 21 one-offs including "Chair_02 - Default_0",
    # "SM_Stool_2_Low_Steel_MAT_0", "Rocking_chair_Material__25_0" and eleven bare
    # "<uid>_obj". Nothing in this repo picks those names.
    # That is why this cannot be written in the shape of the table's entry, whose five children
    # (leg_0..leg_3, top) TableAsset does control -- and it is the same defect the wall entry
    # shipped, where "/world/wall/geometry.*" matched geometry_0 and left the other three
    # children of every wall untextured.
    #
    # Anchored on "/world/chair_<digits>/" so it cannot reach a fixture whose name merely
    # contains "chair", and placed after ".*handle.*" and "(.*glass|.*_window)" so that a chair
    # whose OBJ happens to name a mesh "handle" or "glass" still ends up wood: apply_materials
    # rebinds in insertion order, so the last matching entry wins.
    "/world/chair_[0-9]+/.*": 'floor',
    ".*dishwasher.*basket": 'white plastic',
}

#: The material group a piece of furniture gets when the author has not chosen one. Read out of
#: GEOMETRY2MATERIAL by an EXACT key lookup rather than written as the string "floor" a third time:
#: if that entry is ever re-anchored or re-pointed, this raises KeyError at import instead of
#: quietly disagreeing with the pass it is the default for.
DEFAULT_FURNITURE_MATERIAL: str = GEOMETRY2MATERIAL["/world/chair_[0-9]+/.*"]

#: The material groups the wizard offers per piece of furniture, in a stable order with the default
#: first. MATERIALS' own keys -- curated lists of MDLs that are known to load -- and never a raw
#: MDL path: a free-text path reaches add_mdl_material unchecked and fails at export, after the
#: kitchen is otherwise finished.
#:
#: ALL of them, deliberately, rather than a furniture-shaped subset. Every group here is a real
#: surface a table or a chair can plausibly be ('glass' for a glass-topped table, 'rusted metal'
#: for a metal stool, 'appliances' for brushed steel), the author is looking at a preview before
#: anything is written, and a hand-picked subset would be one more list to keep in step with
#: MATERIALS. sorted() with the default lifted out, so the order is the dict's content and not its
#: insertion history.
FURNITURE_MATERIAL_GROUPS: tuple[str, ...] = (DEFAULT_FURNITURE_MATERIAL,) + tuple(
    sorted(group for group in MATERIALS if group != DEFAULT_FURNITURE_MATERIAL)
)

#: What a material NAME suggests its colour is, as sRGB. Read by _material_swatch below to give
#: each group in MATERIALS one representative colour for the edit page's live preview.
#:
#: CHOSEN, NOT MEASURED, and the distinction matters enough to say twice. Nothing here sampled an
#: MDL: the .mdl files live on an Omniverse content server, they are only ever resolved on a USD
#: stage inside Isaac, and nothing in this repo has ever opened one. What IS derived from the data
#: is which tokens apply to which group -- _material_swatch reads MATERIALS' own MDL paths and
#: variant names -- so re-pointing a group at different MDLs moves its swatch without anyone
#: editing this table. The token-to-colour step is a judgement about what the word "walnut" looks
#: like, and it is wrong to present it as anything else; the edit page says so on the page itself.
#:
#: Ordered longest-token-first at use, so 'granite_dark' wins over 'granite' and
#: 'paint_eggshell_white' over 'paint_eggshell'.
_MATERIAL_NAME_COLOURS: dict[str, tuple[int, int, int]] = {
    # woods
    "bamboo": (206, 173, 116), "birch": (222, 200, 165), "cherry": (150, 84, 58),
    "oak": (183, 143, 96), "ash": (200, 180, 152), "walnut": (95, 65, 47),
    "pine": (206, 168, 116), "beech": (206, 175, 135), "parquet": (172, 128, 79),
    "wood": (176, 136, 92),
    # stone and ceramic
    "marble": (226, 224, 219), "granite_dark": (74, 76, 79), "granite_light": (168, 166, 160),
    "granite": (120, 120, 120), "terrazzo": (198, 192, 182), "slate": (78, 84, 88),
    "basaltite": (86, 84, 80), "steel_grey": (146, 150, 154), "terracotta": (174, 96, 66),
    "adobe": (196, 150, 112), "porcelain": (232, 230, 225), "ceramic": (224, 222, 216),
    "concrete": (158, 156, 150), "stone_natural_black": (44, 44, 46), "stone": (150, 148, 144),
    "brick": (150, 82, 62), "plaster": (214, 208, 198),
    # metals
    "copper": (184, 115, 70), "stainless_steel": (176, 178, 180), "aluminum": (176, 178, 182),
    "silver": (192, 194, 196), "rustedmetal": (140, 82, 52), "metal": (168, 170, 172),
    # paints and plastics, by their own colour word
    "white": (238, 238, 236), "vanilla": (238, 228, 202), "cashmere": (222, 208, 186),
    "peach": (238, 196, 168), "taupe": (170, 156, 140), "leaf": (128, 152, 108),
    "denim": (92, 112, 144), "graphite": (72, 74, 78), "mint": (168, 208, 192),
    "lime": (176, 200, 110), "pale_rose": (232, 202, 198), "cappucino": (176, 146, 116),
    "gray": (150, 152, 154), "grey": (150, 152, 154), "black": (36, 36, 38),
    "red": (162, 66, 56), "yellow": (216, 190, 118), "green": (120, 158, 108),
    "eggshell": (232, 228, 218), "plastic": (226, 226, 224), "glass": (168, 190, 200),
}

#: The colour a group falls back to when nothing in its MDL names is a word this file knows.
#: Deliberately a flat neutral rather than a guess: an unrecognised group should look like "no
#: information", not like beige.
_MATERIAL_SWATCH_FALLBACK: tuple[int, int, int] = (168, 168, 168)


def _material_swatch(group: str) -> str:
    """One representative '#rrggbb' for a MATERIALS group, for the edit page's live tint.

    The MEAN over the group's entries, not its first member. Which MDL inside a group a run gets is
    rolled at build time (pick_materials), so no single member is the answer, and the mean is the
    honest summary of "what this group tends to look like" -- MATERIALS['cabinet'] is ten eggshell
    paints and eleven woods, and showing only the first would promise white.

    Each entry contributes one colour: the longest token from _MATERIAL_NAME_COLOURS found in its
    MDL path or its variant name. Longest wins so 'Granite_Dark' does not read as 'granite', and
    entries matching nothing are skipped rather than pulled toward the fallback.
    """
    hits = []
    for entry in MATERIALS.get(group, ()):
        mdl = str(entry[0]).lower()
        variant = str(entry[1]).lower() if len(entry) > 1 and entry[1] else ""
        text = f"{mdl} {variant}".replace("/", " ").replace("-", "_")
        matched = [token for token in _MATERIAL_NAME_COLOURS if token in text]
        if matched:
            hits.append(_MATERIAL_NAME_COLOURS[max(matched, key=len)])
    if not hits:
        return "#%02x%02x%02x" % _MATERIAL_SWATCH_FALLBACK
    return "#%02x%02x%02x" % tuple(
        int(round(sum(colour[channel] for colour in hits) / len(hits))) for channel in range(3)
    )


#: {group -> '#rrggbb'} for every group the wizard offers. An APPROXIMATION of the group, and the
#: edit page has to say so where the author can read it: the real material is an Omniverse MDL,
#: which exists only on the USD stage at commit and cannot be rendered in a trimesh scene at all.
FURNITURE_MATERIAL_SWATCHES: dict[str, str] = {
    group: _material_swatch(group) for group in FURNITURE_MATERIAL_GROUPS
}


class PositionIteratorGridStartXCenter(utils.PositionIteratorGrid):
    """Start scan at x=center (i index), but y starts at min (j=0)."""
    def __call__(self, support):
        super().__call__(support)  # sets polygon, start/end, i=j=0, etc.

        minx, miny, maxx, maxy = self.polygon.bounds
        cx = (minx + maxx) / 2.0

        # start x-index near center; keep y-index at 0 (miny)
        i0 = int(round((cx - minx) / self.step[0]))
        self.i = max(0, i0)
        self.j = 0

        return self


# ps.kitchen_galley is deliberately not wired here: the library's galley layout routes the
# range hood through the upper wall cabinet -- observed clipping/overlap on a GPU render of a
# generated galley kitchen. That is a geometry defect in scene_synthesizer.procedural_scenes
# itself, not something this module's placement/menu logic can correct, so galley is withdrawn
# rather than shipped into the training corpus. See WALL_CABINET_DYNAMIC below for the other
# register that gates what ships.
KITCHEN_BUILDERS = {
    "island": ps.kitchen_island,
    "l_shaped": ps.kitchen_l_shaped,
    "peninsula": ps.kitchen_peninsula,
    "u_shaped": ps.kitchen_u_shaped,
    "single_wall": ps.kitchen_single_wall,
}

import fnmatch
import trimesh          # already permitted by this module's docstring; used by _world_bounds
import trimesh.transformations as tra   # used by default_table_transform
from dataclasses import dataclass
from .placements import KITCHEN_TYPES, LOCATION_KITCHENS, LOCATION_LABELS, PLACEMENTS, PLACEMENT_BY_KEY, PLACEMENT_BY_LOC, Placement, WALL_CABINET_DYNAMIC, _ALL, _WALL_CABINET_PLACEMENTS, available_locations




# Wall cabinets ship static: kitchen_scene_generator's export pass strips their joints,
# rigid bodies and articulation. That pass arrived in the initial commit with no recorded
# rationale (`git log -S EXCEPT_DYNAMIC_KEYS`), so re-enabling it is treated as unknown-risk
# and gated here. Setting this False restores the previous behaviour exactly, in both the
# physics and the menu — there is deliberately no state where the menu offers a placement
# the export cannot support.

# The exported kitchen gets four walls around it (scene_synthesizer's Scene.add_walls), added at
# COMMIT rather than at build: build_kitchen's scene is what the wizard previews and what the
# gallery tiles render, and a room closed on four sides renders as an opaque box in both.
#
# Setting this False restores the previous behaviour exactly — no walls in the exported USD, and
# the env cfg's own make_walls_from_bounds slabs left in charge, exactly as before. There is
# deliberately no state where both sets of walls exist: see commit_kitchen, which records
# room_shell in the kitchen JSON precisely so the emitter can stand down.
ROOM_SHELL = True

#: Which faces get a wall. Deliberately not the ceiling ('z'): the wizard's preview camera and
#: every rendered thumbnail look down into the room from above.
ROOM_SHELL_DIMENSIONS: tuple[str, ...] = ("x", "-x", "y", "-y")







def _pattern_to_regex(pattern: str) -> str:
    r"""One fnmatch pattern from the registry as a regex label_support() can consume.

    label_support(geom_ids=...) does `re.compile(arg).search(name)` over the scene's
    geometry-bearing node names (scene_synthesizer/scene.py::_get_support_polygons), so a
    raw fnmatch pattern handed to it is silently REINTERPRETED as a regex. That is not a
    theoretical mismatch: "base_cabinet/shelf_*" becomes "base_cabinet/shel" + "f*", and
    "wall_cabinet/top" (a literal that exists on no kitchen -- the nodes are
    "wall_cabinet_1/top", "wall_cabinet_2/top") matches nothing at all, so label_support
    logs "No supports found", never registers the label, and the eventual
    support_generator(support_ids=...) dies with a KeyError mid-batch.

    fnmatch.translate() gives the exact fnmatch semantics as a regex. On CPython it returns
    '(?s:...)\Z' -- end-anchored but NOT start-anchored, which matters because label_support
    uses .search(): without the leading \A, "countertop_base_cabinet" would also match a
    hypothetical "extra_countertop_base_cabinet". Asserting the \Z suffix rather than assuming
    it keeps this honest if a future CPython changes the wrapper.

    The trailing "(?:/.*)?" is what makes this agree with the GUI path rather than merely
    differ from it in a new way. A pattern names a scene-graph NODE, and a matched node is
    not always the node that carries geometry: unwrap_geometries() splits
    "countertop_base_cabinet" into a group node plus "countertop_base_cabinet/geometry_0",
    and only the latter is in graph.nodes_geometry -- the list label_support filters. The GUI
    path gets the child for free because it passes the bare node name and .search()
    substring-matches it; here the descendant has to be spelled out.
    """
    translated = fnmatch.translate(pattern)
    if not translated.endswith(r"\Z"):
        raise AssertionError(
            f"fnmatch.translate({pattern!r}) returned {translated!r}, which is not "
            r"\Z-anchored; _pattern_to_regex's descendant suffix assumes that form"
        )
    return r"\A" + translated[: -len(r"\Z")] + r"(?:/.*)?\Z"


def support_geom_regex(placement: Placement) -> str:
    """A single regex matching every node this placement's fnmatch patterns select, or any
    descendant of one. Equivalent to match_support_nodes() expanded to geometry nodes.

    Alternation composes safely here: each branch is independently anchored \\A...\\Z and
    '|' binds loosest, so "\\A(?s:a)\\Z|\\A(?s:b)\\Z" is (a)|(b), not "\\A(?s:a)(\\Z|\\A)(?s:b)\\Z".
    """
    return "|".join(_pattern_to_regex(p) for p in placement.patterns)


def support_geom_ids() -> dict[str, str]:
    """Persisted placement key -> the geometry regex it resolves to.

    goal_generator labels supports by these keys because task templates persist them as each
    object's "placement" field; the GUI path labels by scene-graph node name. Both now read the
    same registry, so a location cannot exist in one path and not the other.

    Every one of the row's patterns is included, not just patterns[0]: rows like
    "wall_cabinet_top" and "base_cabinet" carry both the bare and the numbered form, and
    dropping the tail silently loses whole cabinets (or, on single_wall, the entire location).
    Locked down by test_support_geom_ids_label_every_offered_location.
    """
    return {p.key: support_geom_regex(p) for p in PLACEMENTS}





@dataclass(frozen=True)
class Fixture:
    """One extra object added on top of a base layout, and where it attaches.

    Applied in build_kitchen, NOT at commit like the room shell -- the opposite placement,
    for the opposite reason. A wall is scenery and must stay out of the preview, which would
    render a closed room as an opaque box. A fixture exists to carry a support surface the
    user drags objects onto, so it has to be in the scene _label_supports walks.

    `kitchens` is per-TYPE and never sampled per kitchen. The wizard builds its placement
    menu from LOCATION_KITCHENS before any kitchen exists, because a build costs 5-7s, so a
    location's availability has to be a property of the type. A fixture present on only some
    island kitchens would make the menu offer somewhere the room does not have.

    `parent`/`parent_anchor`/`obj_anchor` feed Scene.add_object's relative placement rather
    than an absolute transform. A microwave anchored to the base cabinet's top follows it
    when counter_height changes; one at a hardcoded z silently drifts. kitchen_island already
    attaches its island this way.
    """

    key: str
    kitchens: frozenset[str]
    build: "Callable"
    parent: str
    parent_anchor: tuple[str, str, str]
    obj_anchor: tuple[str, str, str]


FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        key="microwave",
        kitchens=_ALL,
        # height=0.26, not the 0.35 m default: the layout builder samples wall_cabinet_z
        # uniform(1.25, 1.35), so the counter-to-cabinet gap is only 0.30-0.40 m (counter
        # top at 0.95 m). At 0.35 m the microwave's own top lands at 1.30 m, which clips
        # into the cabinet whenever that sample falls below 1.30 -- roughly half of all
        # seeds. 0.26 m puts the top at 1.21 m, ~4 cm clear of the lowest cabinet observed
        # (1.25 m). Width/depth are unaffected -- the height was the only wrong dimension.
        build=lambda rng, counter_height: procedural_assets.MicrowaveAsset(
            width=0.58, depth=0.39, height=0.26,
            handle_left=bool(rng.integers(0, 2)),
        ),
        parent="countertop_base_cabinet",
        parent_anchor=("center", "back", "top"),
        obj_anchor=("center", "back", "bottom"),
    ),
    # bin and shelf were tried here and withdrawn -- not a revert of the task, just of these
    # two entries. Both anchor to an appliance embedded in the counter run (sink_cabinet,
    # refrigerator), and no anchor on such a parent works: measured against every corner and
    # center combination, on both the closed pose and every joint opened, across all five
    # layouts and 6 seeds --
    #
    #   * a FRONT-facing anchor sits inside the parent's own door swing. The sink cabinet is
    #     directly adjacent to a forward-opening appliance (dishwasher, range, or
    #     base_cabinet, depending on layout/seed) on every layout, so the bin's front anchor
    #     clipped that swing at up to 266 mm; the refrigerator's own door swings into its
    #     front face, so the shelf's front anchor clipped it at up to 201 mm. "Front" of an
    #     appliance is definitionally where its door goes, so there is no front-facing corner
    #     that dodges it.
    #   * the only anchor that measures 0.0 mm against BOTH poses on every layout and seed is
    #     the parent's own BACK face -- but that lands the fixture behind the counter run, in
    #     the wall. Measured on single_wall seed 0: the counter run's roomward face is at
    #     y = 0.36, the shelf (back-anchored to the refrigerator) occupied y[0.36, 0.72] and
    #     the bin (back-anchored to the sink cabinet) occupied y[0.36, 0.66] -- both entirely
    #     on the far side of y = 0.36, i.e. inside the wall. The robot's spawn band is
    #     y in [-1.5, -1] (simvla_data_generator's single_wall special case), so a fixture
    #     back there is unreachable by the arm and outside every camera in the room.
    #   * a SIDE anchor (left/right) clipped a neighbouring appliance in 3 of 5 layouts each,
    #     because which side of sink_cabinet/refrigerator continues the counter run (versus
    #     standing free) flips between layouts -- no single left/right choice clears all five.
    #
    # So every anchor on these two parents is either in a door's swing or in the wall; there
    # is no fourth option to try. Contrast microwave (sits ON the counter, inside the run's
    # own footprint: measured x[0.58, 1.16] y[-0.11, 0.36] on single_wall) and table (stands
    # in open floor at the island's far corner) -- both anchor to a parent that is not itself
    # embedded in the counter run, which is exactly what bin and shelf lacked.
    #
    # The table is NOT here. It used to be, anchored to kitchen_island's far corner -- the only
    # corner of the only layout that had an island where it cleared every door on every seed
    # (every other corner clashed, worst case 306 mm, several against the island's own drawers).
    # That search produced exactly one hardcoded pose, which is the thing least useful to someone
    # authoring a task. It is now user-chosen and user-placed: see TABLE_VARIANTS and add_table.
)


@dataclass(frozen=True)
class TableVariant:
    """One table the wizard offers, as pure data.

    Unlike Fixture, there is no `kitchens` field and no anchor: a table is chosen and placed
    by the user, on any layout, so nothing here decides where it goes. That is the whole
    difference between this registry and FIXTURES -- see the design doc for why the table
    stopped being an automatic fixture.

    `build(scale=1.0)` takes ONE argument, and it is not a sample: a variant is a fixed shape the
    user picked from a gallery, and two kitchens built from the same variant and the same scale
    must show the same table or the tile lied about what you were choosing. The gallery calls it
    bare, so a tile is always the unedited shape.

    `scale` MULTIPLIES ALL THREE AXES -- see FURNITURE_SCALE_IS_UNIFORM. The two halves of the
    registry reach that differently, which is why the scale is the BUILDER's argument rather than
    something add_table applies to whatever comes back:

      * an Objaverse table is a MeshAsset, and scene_synthesizer scales its geometry for us --
        MeshAsset(path, scale=[s, s, s]);
      * a procedural one is a TableAsset, which is PARAMETERISED by width/depth/height, and
        scene_synthesizer refuses to scale it geometrically at all ("Can't scale primitives
        non-uniformly", measured against 1.15 -- its top and legs are trimesh primitives, and a
        non-uniform transform would stop them being primitives, hence stop them exporting as USD
        Cube/Cylinder prims). So it is rebuilt at width*s, depth*s, height*s.

    Those two are not identical operations and the difference is deliberate: rebuilding leaves the
    surface thickness and the leg thickness as designed, so a table 15% bigger does not get 15%
    fatter legs. Both multiply the table's overall extents by s, which is what the scale means.

    ONE SHAPE FOR BOTH SOURCES. Twelve of these are procedural TableAssets written out below;
    the rest are Objaverse meshes read off /lustre (see _load_mesh_table_variants). They differ
    only in what `build` returns -- a procedural_assets.TableAsset or a synth.assets.MeshAsset,
    both of which add_table and the gallery consume through the same `.scene().scene` /
    `kitchen.add_object` surface. Deliberately NOT two dataclasses: the wizard picks a table and
    where it came from is an implementation detail, so a caller that had to ask which registry a
    key belonged to would be the bug this shape prevents.

    The last three fields are the mesh half's provenance and are None on all twelve procedural
    variants, which have no uid, no file, and dimensions that are already written above in the
    call that builds them:

      uid          -- the Objaverse uid, i.e. the table_manifest entry this came from.
      obj_path     -- the exported OBJ. The generator consumes OBJs via MeshAsset, never the
                      converted USDs; the USDs are the standalone library (see objaverse_tables.md).
      footprint_m  -- (x, y) horizontal extents of the exported mesh, in metres, straight off the
                      manifest's export_footprint_m. Recorded so the ring sweep and the gallery
                      can size themselves without loading fifty meshes to measure what the export
                      stage already measured.

    `label` IS A NUMBER now -- "Table 7" -- and `detail` is what the label used to say. See
    FURNITURE_NUMBERING for what the number means and what it does not.
    """

    key: str
    label: str
    build: "Callable"
    uid: str = None
    obj_path: str = None
    footprint_m: tuple = None
    #: The card's data line: what this table IS, since the label is only a number. "Long dining"
    #: for a procedural variant (that string was its display label before the renumbering, and it
    #: is real information about a shape nobody can otherwise name), "uid e7cc55" for an Objaverse
    #: one. Never empty.
    detail: str = None


#: What every furniture label promises, and what it does not.
#:
#: "Table 1 ... Table 62" and "Chair 1 ... Chair 31" are positions in an explicitly ordered
#: registry, not identities:
#:
#:   * the twelve procedural tables take 1-12 in the order they are written in this file, which
#:     is a literal and moves only when somebody edits it;
#:   * the Objaverse tables take 13-62 and the chairs take 1-31, both ordered by the manifest's
#:     own `selection_rank` -- the 0-based order build_chair_manifest.select()'s greedy max-min
#:     shape-IoU pass chose them in -- with the uid as a tie-break and as the whole order for a
#:     manifest too old to carry the field. Never the manifest's file order, which is the order
#:     the fetch stage happened to append in, and never dict iteration.
#:
#: THE NUMBER IS NOT A NAME. Rebuild the library -- re-run curate, re-run select, accept one more
#: candidate -- and every rank after the change shifts, so "Chair 7" is then a different chair.
#: The uid is the identity: it is what CHAIR_VARIANT_BY_UID is keyed by, what the wizard's state
#: carries, what the gallery answers with, and what a scene is reproducible from. That is why the
#: short uid is on every card's data line beside the measurements, and why anyone writing down
#: which chair they used should write down the uid rather than the number.
FURNITURE_NUMBERING = "selection_rank, then uid; procedural tables first, in registry order"


#: What a piece of furniture may be multiplied by, on all three axes -- see
#: FURNITURE_SCALE_IS_UNIFORM.
#:
#: +/-30%, WIDENED FROM +/-15% ON REQUEST, and re-swept at the new extremes rather than argued
#: from the old ones. The 2026-08-16 sweep, at 0.70 and 1.30 (scripts/simvla/kitchen_build.py's
#: own gate, on real built scenes):
#:
#:   * ring, geometry only: 31 chairs x {0.70,1.30} x 62 tables x {0.70,1.30} x counts 2..8 =
#:     53,816 arrangements, worst slack 0.000000 m, none overlapping.
#:   * gate: every chair at 1.30x, eight of it, island/dining_long at 1.30x -- 31 cases, 0 rejected.
#:   * gate: five layouts x {widest, narrowest} chair, eight of it, at 1.30 and at 0.70 -- 20
#:     cases, 0 rejected.
#:   * gate: all 62 tables at 1.30x with the widest chair at 1.30x, eight of it -- 62 cases,
#:     0 rejected.
#:
#: RE-SWEPT AGAIN on 2026-08-17, when the scale stopped being plan-view only
#: (FURNITURE_SCALE_IS_UNIFORM). Height reaches things a footprint never did -- a chair back into a
#: countertop's underside, a table top into a wall cabinet -- so the top of the range was measured
#: again rather than carried over:
#:
#:   * gate: every chair at 1.30x, eight of it, table at 1.30x, on all five layouts, both joint
#:     configurations -- 155 cases, 0 rejected.
#:   * gate: all 62 tables at 1.30x at their own default pose, on all five layouts, both joint
#:     configurations -- 310 cases, 0 rejected, and no table-versus-countertop pair at all.
#:
#: The ring is untouched by any of it: a seat reserves the plan diagonal x scale
#: (chair_footprint_m) and default_table_transform's walkway is max(1.2, 0.7 + depth/2), so both
#: read plan-view lengths that scale exactly as they did before.
#:
#: MAX_PENETRATION_M was not touched for any of it.
#:
#: THE TOP is still what the placement gate can carry. The ring reserves diagonal x scale
#: (chair_footprint_m) and grows itself until no pair overlaps, so chair-versus-chair is safe at
#: any scale by construction; what actually binds is a chair swinging out far enough to stand in
#: an opened appliance door, and the sweep above says 1.30 clears that on every layout.
#:
#: THE BOTTOM IS NO LONGER A HARD STOP, and that is the deliberate change. The old floor was a
#: REALISM judgement borrowed from curation: build_chair_manifest rejects a candidate mesh whose
#: LARGER horizontal extent is under export_width_min_m (0.35 m), so a shrink that takes an
#: accepted chair under that bar produces, by scaling, what the curation stage would have refused.
#: That is still true and still worth saying -- but it is a judgement about whether a chair reads
#: as a chair, not a constraint the geometry or the gate imposes (a narrower chair is strictly
#: easier to place), and the author is entitled to overrule it for a scene they are looking at.
#: So it is SHOWN and not enforced: chair_width_floor_scale reports where a given chair crosses
#: the bar, the edit page marks that point on the slider, and every step down to 0.70 is offered.
#: Over the 31 offered chairs the larger extent runs 0.383..0.750 m, so 25 of them clear the bar
#: at 0.70 and 6 cross it somewhere between 0.75 and 0.95.
#:
#: SYMMETRIC because a range that shrank less than it grew would need explaining every time.
FURNITURE_SCALE_MIN: float = 0.70
FURNITURE_SCALE_MAX: float = 1.30

#: The step the wizard offers between the two. 5% is a visible change on a 0.5 m chair (25 mm) and
#: leaves thirteen choices over the widened range.
#:
#: The edit page's slider moves in exactly these steps and nothing between them. That is not a
#: cosmetic choice: the answer the browser posts must be one the step published in its own option
#: list (kitchen_wizard.edit_answer_ids), and a CONTINUOUS slider could post 1.0734, which the
#: director would have to refuse -- "the POST returned 200 and nothing happened" is the failure
#: this codebase keeps out of every other step. The slider's own `value` is an INDEX into
#: FURNITURE_SCALES, so an off-by-a-float cannot happen in the browser either.
FURNITURE_SCALE_STEP: float = 0.05

#: Every scale the wizard offers, smallest first, as exact multiples of the step (integers scaled
#: down, never a float accumulation -- 0.85 + 0.05 * 3 is 0.9999999999999999, which is not 1.0 and
#: would not compare equal to an unedited piece's default).
FURNITURE_SCALES: tuple[float, ...] = tuple(
    round(FURNITURE_SCALE_MIN + i * FURNITURE_SCALE_STEP, 10)
    for i in range(int(round((FURNITURE_SCALE_MAX - FURNITURE_SCALE_MIN)
                             / FURNITURE_SCALE_STEP)) + 1)
)

#: Why a furniture scale touches all three axes, in one place both add_table and add_chair point at.
#:
#: UNIFORM, ON REQUEST (2026-08-17), REVERSING the plan-view-only rule this constant used to carry.
#: The slider is one number and it now means "this piece, bigger" -- x, y and z together.
#:
#: WHAT THAT COSTS, kept here rather than deleted with the rule it justified, because the
#: measurement behind it is still true and is what an author is trading away:
#:
#:   * a table's WORK SURFACE is 0.74 m and the chairs are drawn to match it, so a chair tucks
#:     under a table. Nine of the twelve procedural TableAsset variants are written at
#:     height=0.74, every Objaverse table is exported scaled so its work surface lands there
#:     (build_table_manifest's TableType.target_height_m), and the chair library's export heights
#:     are a uid-seeded draw in [0.75, 1.00] to the backrest top.
#:   * scaling breaks that pairing, and the range is wide enough for it to be obvious: a 0.74 m
#:     table at 1.30x stands at 0.96 m -- above the counter run's own 0.95 m -- and a chair at
#:     0.70x is 0.53-0.70 m to the backrest, which is child furniture. A table at 1.30 and a chair
#:     at 0.70 in the same kitchen do not go together at all.
#:
#: NOBODY ENFORCES THE PAIRING. The two sliders are independent by design (a table scale and a
#: per-chair scale), so keeping a chair tucked under a table is the AUTHOR's business now, not the
#: builder's. The one thing that is still enforced is that the result is physically placeable: the
#: placement gate measures the built geometry, so a table that has grown into a wall cabinet and a
#: chair whose back has grown into a countertop are both REPORTED -- see _is_reportable_clash,
#: whose countertop paragraph records what this change made reachable.
FURNITURE_SCALE_IS_UNIFORM = "scale multiplies x, y and z alike; the 0.74 m pairing is the author's"


def check_furniture_scale(scale) -> float:
    """`scale` as a float, or ValueError. The bound, enforced where the geometry is built.

    REFUSES rather than clamps. A scale outside the range is a caller with a bug -- the wizard
    offers exactly FURNITURE_SCALES and refuses anything else before it gets here -- and silently
    building a 2x chair that the range was never swept for is the failure this bound exists to
    prevent. Clamping would also make the returned scene disagree with the number the author was
    shown, which is the one thing a preview must not do.

    bool is excluded explicitly: it is an int subclass, so True would otherwise pass as 1.0 and
    build an unedited piece for a caller that meant something else entirely. Same guard, same
    reason, as validate_setup's on `seed`.
    """
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"Furniture scale must be a number, not {type(scale).__name__}.")
    scale = float(scale)
    if not FURNITURE_SCALE_MIN <= scale <= FURNITURE_SCALE_MAX:
        raise ValueError(
            f"Furniture scale {scale:g} is outside "
            f"{FURNITURE_SCALE_MIN:g}..{FURNITURE_SCALE_MAX:g}."
        )
    return scale


@functools.lru_cache(maxsize=1)
def chair_width_floor_m() -> float:
    """The narrowest a chair may be and still be a chair, in metres, as the CURATION stage
    defines it -- build_chair_manifest's export_width_min_m.

    Read out of that module rather than written here, for the same reason DEFAULT_FURNITURE_MATERIAL
    is read out of GEOMETRY2MATERIAL: it is a measured number with a findings document behind it
    (2026-08-13-chair-export-width-findings.md), and a copy of it in this file would go on being
    enforced after somebody re-measured it. Cached because chair_width_floor_scale is called per
    chair per page render and the import walks sys.path.
    """
    return float(_import_build_chair_manifest().DEFAULT_THRESHOLDS["export_width_min_m"])


def chair_width_floor_scale(uid):
    """The smallest OFFERED scale at which this chair is still as wide as the chair library
    requires of a chair -- or None when nothing is known about its width.

    ADVISORY, not a bound. Every scale in FURNITURE_SCALES is offered for every chair; this is the
    number the edit page marks on its slider, so that shrinking a chair past the point where it
    stops reading as a chair is a decision the author takes on purpose rather than one the wizard
    makes for them. See FURNITURE_SCALE_MIN for why the old hard stop was dropped.

    The criterion is on the LARGER horizontal extent, matching build_chair_manifest._export_width
    exactly -- it fires only when a chair is under the bar on BOTH axes -- which is why
    export_footprint_m is kept on the variant beside its diagonal.

    None for an unknown uid or a chair the manifest never measured, which is the same fail-open
    choice _export_width itself makes for an entry with no footprint: nothing is known about such a
    chair's width, so marking a point on the slider would be a verdict on a measurement that was
    never taken. None also when the chair clears the bar at every offered scale -- 25 of the 31 do
    -- because there is then no point to mark.
    """
    variant = CHAIR_VARIANT_BY_UID.get(uid)
    extents = getattr(variant, "export_footprint_m", None) if variant is not None else None
    if not extents:
        return None
    floor = chair_width_floor_m()
    widest = max(float(v) for v in extents)
    # 1e-12 so a chair sitting exactly on the bar is not marked by a float representation.
    clearing = [s for s in FURNITURE_SCALES if widest * s >= floor - 1e-12]
    if not clearing or clearing[0] <= FURNITURE_SCALES[0]:
        return None
    return clearing[0]


def check_chair_scale(uid, scale) -> float:
    """check_furniture_scale. The bound add_chair enforces, which is the RANGE and nothing else.

    Kept as its own function, with the uid it no longer reads, because the per-chair question is a
    real one that this is the natural place to ask -- and because every call site that means "this
    chair's scale" should keep saying so if the width floor ever becomes a bound again. What
    changed is only that the floor is advisory now (chair_width_floor_scale, and see
    FURNITURE_SCALE_MIN): a narrow chair is strictly easier to place, so nothing the gate measures
    ever objected to one, and the objection that remains is about realism -- the author's call.
    """
    return check_furniture_scale(scale)


def _by_selection_rank(entries: list[dict]) -> list[dict]:
    """Manifest entries in the one order the numbering is allowed to use. See FURNITURE_NUMBERING.

    `selection_rank is None` sorts first so the key is total on a manifest that has the field and
    still well-defined on one that does not (then it is uid order, for every entry, rather than
    file order for some and rank order for others).
    """
    return sorted(entries, key=lambda e: (e.get("selection_rank") is None,
                                          e.get("selection_rank", 0), e["uid"]))


def _procedural_table_asset(scale: float = 1.0, **params):
    """One procedural table, multiplied by `scale` on all three axes.

    width, depth AND height are multiplied -- height because the slider is uniform now, see
    FURNITURE_SCALE_IS_UNIFORM. thickness, leg_thickness and leg_margin are still left alone: a
    bigger table is not a table with fatter legs, and leg_margin is already a fraction of
    width/depth, so the legs move outward with the top for free.

    That leaves the top slab and the leg stock at their designed thickness while the table itself
    grows, which is the deliberate difference from the mesh half (_mesh_asset_scale scales every
    vertex, thickness included). It is a few millimetres at either extreme on a 0.74 m table and
    the alternative is either fatter legs or dropping the rebuild entirely -- see below for why
    the rebuild is not optional.

    Rebuilt rather than scaled because scene_synthesizer will not scale a TableAsset: its top and
    legs are trimesh primitives and it raises NotImplementedError ("Can't scale primitives
    non-uniformly") for any non-uniform factor, [1.15, 1.15, 1.0] included. A UNIFORM factor is
    not obviously refused by that message, but rebuilding is kept anyway: it is the path this
    library is already swept at, and it keeps the primitives exporting as USD Cube/Cylinder prims.
    See TableVariant.
    """
    params = dict(params)
    params["width"] = float(params["width"]) * float(scale)
    params["depth"] = float(params["depth"]) * float(scale)
    params["height"] = float(params["height"]) * float(scale)
    return procedural_assets.TableAsset(**params)


#: The twelve procedural tables as (key, what it is, the TableAsset parameters). The middle column
#: used to be the display label; it is the data line now -- see TableVariant.detail.
#:
#: Parameters rather than a zero-argument lambda, because `build` now takes a scale and a
#: procedural table reaches it by being REBUILT at width*s / depth*s (see _procedural_table_asset).
#: A lambda would have to be written twelve times over to thread that through.
_PROCEDURAL_TABLES: tuple = (
    ("square_small",  "Small square",     dict(width=0.70, depth=0.70, height=0.74)),
    ("square_med",    "Square",           dict(width=0.90, depth=0.90, height=0.74)),
    ("dining_small",  "Small dining",     dict(width=1.10, depth=0.70, height=0.74)),
    ("dining_long",   "Long dining",      dict(width=1.80, depth=0.85, height=0.74)),
    ("dining_wide",   "Wide dining",      dict(width=1.40, depth=1.00, height=0.74)),
    ("bar_tall",      "Bar height",       dict(width=1.20, depth=0.60, height=1.05)),
    ("bar_small",     "Small bar",        dict(width=0.75, depth=0.55, height=1.05)),
    ("low_coffee",    "Low table",        dict(width=1.00, depth=0.55, height=0.45)),
    ("boxy_square",   "Square, box legs", dict(width=0.90, depth=0.90, height=0.74, leg_as_box=True, leg_thickness=0.07)),
    ("boxy_long",     "Long, box legs",   dict(width=1.60, depth=0.80, height=0.74, leg_as_box=True, leg_thickness=0.08)),
    ("thin_legs",     "Slim legs",        dict(width=1.20, depth=0.75, height=0.74, leg_thickness=0.025, leg_margin=0.08)),
    ("inset_legs",    "Inset legs",       dict(width=1.30, depth=0.80, height=0.74, leg_margin=0.35)),
)

TABLE_VARIANTS: tuple[TableVariant, ...] = tuple(
    TableVariant(key=key, label=f"Table {number}",
                 build=functools.partial(_procedural_table_asset, **params), detail=detail)
    for number, (key, detail, params) in enumerate(_PROCEDURAL_TABLES, start=1)
)

#: What every Objaverse table's key starts with, so a mesh key can never collide with one of the
#: twelve procedural keys above. Those are hand-written words ("square_small", "boxy_long"); these
#: are this prefix plus a 32-hex-digit Objaverse uid, so the two sets are disjoint by construction
#: rather than by inspection -- and test_a_mesh_table_key_cannot_collide_with_a_procedural_one
#: holds it that way against a rebuilt library.
MESH_TABLE_KEY_PREFIX = "mesh_"


def _mesh_table_asset(obj_path: str, scale: float = 1.0):
    """The MeshAsset for one Objaverse table, its footprint multiplied by `scale`.

    A named function, not a lambda in the loop below: a lambda would close over the loop variable
    and every variant would build the LAST table.

    See _mesh_asset_scale for the [s, s, s] and why z scales with the rest.
    """
    return synth.assets.MeshAsset(obj_path, **_mesh_asset_scale(scale))


def _mesh_asset_scale(scale: float) -> dict:
    """The MeshAsset kwargs that multiply a mesh by `scale` on all three axes.

    `scale=[s, s, s]`, which scene_synthesizer BAKES INTO THE GEOMETRY (Asset._get_scale accepts
    a 3-vector and applies it to the loaded mesh). Baked is the requirement, not an implementation
    detail: the placement gate measures fixture_overlaps through trimesh's fcl collision manager,
    which is handed each node's transform as a rigid pose, so a scale carried in the SCENE-GRAPH
    transform instead would be silently dropped there and six scaled-up chairs would overlap while
    the gate reported clean.

    At exactly 1.0 the kwarg is omitted rather than passed as [1, 1, 1]: an unedited piece must
    take the identical code path it took before this argument existed, so nothing about a
    default kitchen can move by a rounding error.
    """
    if float(scale) == 1.0:
        return {}
    return {"scale": [float(scale), float(scale), float(scale)]}


def _load_mesh_table_variants() -> tuple:
    """The Objaverse tables the wizard offers, read from the asset manifest rather than written
    here -- the same derivation _load_chair_variants does, against table_manifest instead.

    usd_path is the filter for the same reason it is there: curate accepts 84 candidates and
    build_table_manifest.select() picks 50 by greedy max-min shape-IoU diversity, and only those
    50 are converted, because conversion boots Isaac Sim per mesh. obj_path is what the generator
    actually consumes, so both must exist.

    Returns () when there is no manifest -- a checkout without /lustre is a supported state, so
    this must not raise at import time. table_manifest.load() already answers an empty skeleton
    for a missing or unreadable file, and accepted() reads [] out of it.

    NO height in the variant, unlike ChairVariant. Every one of these is scaled so its WORK
    SURFACE lands at 0.74 m (build_table_manifest's TableType.target_height_m), which is the same
    height nine of the twelve procedural variants are written at, so there is no per-table number
    for a caller to branch on. The overall height differs from 0.74 on four of the fifty, by 2 to
    49 mm, where something sits proud of the work surface -- see objaverse_tables.md.

    These continue the procedural twelve's numbering rather than starting again at 1, because the
    wizard offers ONE table gallery over both halves and two tables called "Table 1" in it would
    be the whole point of numbering them thrown away. Hence `start=len(TABLE_VARIANTS) + 1`, read
    off that tuple rather than written as 13.
    """
    from . import table_manifest

    manifest = table_manifest.load()
    offered = [entry for entry in table_manifest.accepted(manifest)
               if entry.get("usd_path") and entry.get("obj_path")]

    out = []
    for number, entry in enumerate(_by_selection_rank(offered), start=len(TABLE_VARIANTS) + 1):
        footprint = entry.get("export_footprint_m") or None
        out.append(TableVariant(
            key=f"{MESH_TABLE_KEY_PREFIX}{entry['uid']}",
            label=f"Table {number}",
            build=functools.partial(_mesh_table_asset, entry["obj_path"]),
            uid=entry["uid"],
            obj_path=entry["obj_path"],
            footprint_m=tuple(float(v) for v in footprint) if footprint else None,
            # The identity, on the data line, because the label above is only a position -- see
            # FURNITURE_NUMBERING. Six hex digits is enough to tell fifty uids apart and short
            # enough to sit beside the measurements without wrapping.
            detail=f"uid {entry['uid'][:6]}",
        ))
    return tuple(out)


MESH_TABLE_VARIANTS: tuple = _load_mesh_table_variants()

#: Every table the wizard can offer, procedural first. This is the list a gallery pages through
#: and the list add_table resolves against; TABLE_VARIANTS stays the twelve procedural ones so
#: that a caller wanting exactly those (the sanity tests on TableAsset's parameters, say) still
#: has them by name.
ALL_TABLE_VARIANTS: tuple[TableVariant, ...] = TABLE_VARIANTS + MESH_TABLE_VARIANTS

#: Keyed over BOTH sets, because this is the map add_table looks a wizard's choice up in and the
#: wizard's choice may be either. Empty of mesh entries on a checkout without /lustre.
TABLE_VARIANT_BY_KEY: dict[str, TableVariant] = {v.key: v for v in ALL_TABLE_VARIANTS}


@dataclass(frozen=True)
class ChairVariant:
    """One chair the wizard offers, read from the asset manifest rather than written here.

    This is the structural difference from TableVariant: a table is twelve parameter sets in
    this file, but a chair is a file on /lustre produced by scripts/tools/build_chair_manifest.py.
    So the registry is derived, can be empty, and must not raise when the library is absent --
    a checkout without /lustre is a supported state, not an error.

    `label` IS A NUMBER -- "Chair 7" -- and `detail` carries the short uid, because the number is
    a position in this registry and the uid is the identity. See FURNITURE_NUMBERING.
    """

    uid: str
    label: str
    obj_path: str
    height_m: float
    facing_axis: str
    #: The card's data line: "uid 304253". Shown beside the measurements wherever this chair is
    #: named, so the thing a scene is reproducible from is never further away than the label.
    detail: str = None
    #: This chair's own plan-view footprint in metres -- the DIAGONAL of its exported horizontal
    #: extents, i.e. the widest it can be at any yaw. Derived from the manifest's
    #: export_footprint_m; see chair_footprint_m for why the diagonal and not the larger extent.
    #: None when the manifest carries no measurement for it (an older manifest, or an entry
    #: written before build_chair_manifest recorded the field); chair_footprint_m falls back to
    #: CHAIR_FOOTPRINT_M for those. Optional with a default so a manifest without the field
    #: still loads rather than raising at import time.
    footprint_m: float = None
    #: The same measurement BEFORE the diagonal is taken: (smaller, larger) horizontal extent, as
    #: the manifest records it. Kept as well as the diagonal because the two answer different
    #: questions -- the diagonal is what a seat reserves at any yaw (chair_footprint_m), and the
    #: LARGER extent is the quantity build_chair_manifest's export_width_min_m criterion is
    #: written against, which is what chair_width_floor_scale has to compare a shrink against.
    #: None on a manifest that recorded neither.
    export_footprint_m: tuple = None


def _load_chair_variants() -> tuple:
    """Chairs with a usd_path only -- the set that went through conversion.

    curate accepts 161 candidates; build_chair_manifest.select() picks 50 of them by greedy
    max-min shape-IoU diversity, and only those 50 are converted -- conversion boots Isaac Sim per
    mesh, so it deliberately happens after the selection rather than before it. usd_path is the
    marker for "this chair is in the offered library", which is why this filter reads it.

    Note what has changed about the PROMISE here. When this was twenty chairs, every one had been
    looked at by a human, and that inspection was the guard against the geometric criteria's known
    false-accept (a pedestal office chair passes every threshold). The fifty have not been: they
    were selected by measurement and converted. One known bad case is in them -- uid
    8a4a3a90bc104f11b82cedd9b4e5ab6b, which reads as an abstract stand rather than something a
    person sits in. The contact sheet that would let a human skim the set is Task 5 of the
    fifty-chairs plan and does not exist yet.
    """
    from . import chair_manifest

    manifest = chair_manifest.load()
    offered = [entry for entry in chair_manifest.accepted(manifest)
               if entry.get("usd_path") and entry.get("obj_path")]

    out = []
    for number, entry in enumerate(_by_selection_rank(offered), start=1):
        # export_footprint_m is [smaller, larger] of the two horizontal extents AFTER the export
        # scale (build_chair_manifest._export_footprint, verified against every exported OBJ read
        # back off disk to 9.5e-09 m -- and re-verified here against all fifty, max error 9.5e-09).
        # Its DIAGONAL is what a seat has to reserve; see chair_footprint_m.
        footprint = entry.get("export_footprint_m") or None
        out.append(ChairVariant(
            uid=entry["uid"],
            label=f"Chair {number}",
            obj_path=entry["obj_path"],
            height_m=float(entry.get("height_m", 0.85)),
            facing_axis=entry.get("facing_axis", "X"),
            # The identity, on the data line, because the label above is only a position in this
            # registry and shifts when the library is rebuilt -- see FURNITURE_NUMBERING.
            detail=f"uid {entry['uid'][:6]}",
            footprint_m=float(math.hypot(*footprint)) if footprint else None,
            export_footprint_m=tuple(float(v) for v in footprint) if footprint else None,
        ))
    return tuple(out)


CHAIR_VARIANTS: tuple = _load_chair_variants()
CHAIR_VARIANT_BY_UID: dict = {v.uid: v for v in CHAIR_VARIANTS}

# Attribute name under which each kitchen carries its own chair_<n> -> uid mapping. See
# _chair_uids below for why it belongs to the kitchen and not to this module.
_CHAIR_UID_ATTR = "_simvla_chair_uid_by_id"


def _chair_uids(kitchen) -> dict[str, str]:
    """This kitchen's chair_<n> -> uid mapping, created on first use.

    add_chair writes it; _chair_mesh_for reads it. The scene graph carries only the
    already-transformed geometry, never which ChairVariant produced it, so this is the one place
    facing_direction_for (and the weak-signal check beside it) can recover the source mesh for a
    given chair_<n> node.

    It hangs off the KITCHEN rather than off this module because chair ids are only unique within
    one kitchen: every kitchen's first chair is "chair_0". As a module-level dict, two kitchens
    built in one process both wrote "chair_0" and the second silently won -- after which
    auto-facing would load the wrong mesh for the first kitchen's chair and rotate it by the
    wrong angle, with no error anywhere. It just faces the wrong way in one preview out of ten.
    """
    uids = getattr(kitchen, _CHAIR_UID_ATTR, None)
    if uids is None:
        uids = {}
        setattr(kitchen, _CHAIR_UID_ATTR, uids)
    return uids


# Which kitchen types actually contain each location's geometry. Hand-maintained because
# the GUI needs it before any kitchen is built (a build costs 5-7s); kept honest by
# test_available_locations_matches_what_the_kitchen_actually_has.




# Graspable types + clutter, both from scene_spec's canonical source, so the type list and dims here
# cannot drift from the composer's OBJECT_TYPES / the clutter table. (scene_spec imports kitchen_build
# only lazily, inside a function, so this module-level import does not cycle.)
from .scene_spec import GRASPABLE_TYPES, CLUTTER  # noqa: F401
from .object_dims import (
    OBJECT_TYPES, OBJECT_BASE_DIMS, REAL_HEIGHT, _up_axis_index, real_placement_dims,
)


def object_scale(obj_type: str, size: float, mug_height: float | None = None) -> list[float]:
    """[width, depth, height] scaled by the per-instance size multiplier.

    mug_height is retired: it defaults to None and the base height is used. The argument is
    kept optional so any legacy caller still works, but new code passes only (type, size).
    """
    try:
        width, depth, height = OBJECT_BASE_DIMS[obj_type]
    except KeyError:
        raise ValueError(
            f"No dimensions defined for object type {obj_type!r}. "
            f"Add it to OBJECT_BASE_DIMS. Known types: {sorted(OBJECT_BASE_DIMS)}"
        ) from None
    if mug_height is not None:          # legacy override; new code omits it
        height = mug_height
    return [width * size, depth * size, height * size]


def matching_meshes(obj_type: str, mesh_files) -> list[str]:
    """Every BODex mesh this object type could be, in dataset order.

    The one definition of what "mug" means: resolve_mesh picks from this list at random, and the
    gallery shows exactly this list. When those two disagreed, a user could pick a mesh the
    generator would never have chosen.

    Matched on the mesh directory's CATEGORY prefix, not on the type name appearing anywhere in the
    path. The old substring rule handed `can` all eight `sem_Candle_*` meshes and the `sem_SodaCan`
    one as well, because "can" sits inside both words -- so a `can` row could spawn a candle, and
    the manifest attributed those meshes to a different type than the one that placed them.
    """
    from .scene_spec import CATEGORIES, MESH_EXCLUDE

    prefixes = CATEGORIES.get(obj_type, ())
    out = []
    for f in mesh_files:
        name = f.split("/")[-3] if len(f.split("/")) >= 3 else f
        if name not in MESH_EXCLUDE and name.startswith(prefixes):
            out.append(f)
    return out


def resolve_mesh(obj_type: str, mesh_files) -> str:
    """Pick a random BODex mesh whose path mentions obj_type."""
    matches = matching_meshes(obj_type, mesh_files)
    if not matches:
        raise ValueError(
            f"No BODex mesh matched object type {obj_type!r} "
            f"({len(mesh_files)} file(s) loaded)."
        )
    return random.choice(matches)


def _object_pose(obj_type: str, mesh_path: str | None = None):
    """(up, front, origin) for an asset. All three are derived from `up`, so they cannot disagree.

    `up` comes from mesh_orientation.resolved_up, which the mesh PICKER also calls so its tiles
    are drawn the way this places them. The per-mesh table behind it exists because the BODex
    bundle mixes two datasets with different conventions: ShapeNetCore meshes (core_*) store +Y
    up, ShapeNetSem ones (sem_*) mostly store +Z, and applying the single +Y rule to both laid 19
    of 29 object types on their side -- a vase rested on 0.1% of its surface and would topple the
    moment physics started.

    Deriving front and origin here rather than tabling them is what keeps a fixed `up` from
    breaking placement instead: measured, a corrected up with the old origin sinks the object
    6.1 cm into the countertop, and a Z-down mesh hangs 12.6 cm underneath it.
    """
    from .mesh_orientation import front_for, origin_for, resolved_up

    up = resolved_up(obj_type, mesh_path)
    return up, front_for(up), origin_for(up)


def match_support_nodes(kitchen, placement: Placement) -> list[str]:
    """Scene-graph nodes this placement resolves to on THIS kitchen, in pattern order.

    Returns [] when the kitchen type has none of them (e.g. "Above island" on an
    l_shaped kitchen) — the caller treats that as "this location is impossible here".
    """
    nodes = kitchen.scene.graph.nodes
    found: list[str] = []
    for pattern in placement.patterns:
        for node in sorted(nodes):
            if fnmatch.fnmatchcase(node, pattern) and node not in found:
                found.append(node)
    return found


def _world_bounds(kitchen, node: str):
    """World-frame [[min], [max]] corners of a node's geometry, or None.

    Two traps, both verified against scene_synthesizer 1.15:
      - `scene.subscene(node).bounds` returns None for these nodes. It cannot be used.
      - `scene.geometry[name].bounds` is in the LOCAL frame. Every refrigerator shelf
        reports the same ~0.01-tall local box; only the graph transform tells them apart.
    `scene.graph[node]` returns (4x4 world transform, geometry name), which is the pair
    needed to place the local bounding box into the world.
    """
    try:
        transform, geom_name = kitchen.scene.graph[node]
    except Exception:
        return None
    geom = kitchen.scene.geometry.get(geom_name or node)
    if geom is None:
        return None
    corners = trimesh.transform_points(geom.bounding_box.vertices, transform)
    return np.array([corners.min(axis=0), corners.max(axis=0)])


def _support_area(kitchen, node: str) -> float:
    """Horizontal (XY) extent of a support node, used by pick="largest"."""
    bounds = _world_bounds(kitchen, node)
    if bounds is None:
        return 0.0
    return float((bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1]))


def fixture_overlaps(kitchen) -> dict[tuple[str, str], float]:
    """Worst interpenetration depth, in metres, for each pair of distinct fixtures that touch.

    Keys are ordered pairs of TOP-LEVEL object names ("range", "wall_cabinet_0"), never
    geometry nodes. That grouping is the whole point: run over raw geometry, the range's
    oven rack alone -- 88 near-identical sub-meshes, the same ones loc 12's comment calls
    out -- reports about 1,200 self-colliding pairs at up to 352 mm on every layout, and
    the signal is buried. Collapsed to fixtures, a layout has 6 to 16 pairs.

    A depth of 0.0 is a genuine entry, not padding: cabinets meet flush, and the caller
    needs to see that they touch. Real clipping is what a threshold separates -- see
    MAX_PENETRATION_M below for the measured tiers.

    Forces a full collision-manager rebuild first (reset=True) rather than calling
    kitchen.collision_manager() directly. Scene.collision_manager() only rebuilds from
    scratch when its cached hash disagrees with the current scene; otherwise it reuses
    Scene.synchronize_collision_manager()'s INCREMENTAL path, which re-applies every
    node's transform but silently loses the true penetration depth for pairs that
    already existed before the incremental update -- verified directly: after
    add_fixtures calls add_object (Task 2), the deliberate ~30.8 mm countertop-corner
    seam that predates any fixture reads back as ~0.0 mm on all five layouts through the
    incremental path, and exactly 30.8 mm again through a forced reset. Every kitchen
    build_kitchen returns has had add_object called on it since fixtures were added, so
    without this the gate would silently stop measuring real overlaps for any pair that
    isn't the fixture just inserted.
    """
    kitchen.synchronize_collision_manager(reset=True)
    _hit, _names, contacts = kitchen.collision_manager().in_collision_internal(
        return_names=True, return_data=True
    )

    worst: dict[tuple[str, str], float] = {}
    for contact in contacts:
        left, right = sorted(contact.names)
        owner_left, owner_right = left.split("/")[0], right.split("/")[0]
        if owner_left == owner_right:
            continue                        # one fixture's own sub-meshes; see docstring
        pair = (owner_left, owner_right) if owner_left < owner_right else (owner_right, owner_left)
        depth = abs(float(contact.depth))
        if depth > worst.get(pair, -1.0):
            worst[pair] = depth
    return worst


#: Fixture-vs-fixture and fixture-vs-appliance overlaps have no by-design abutment, so this
#: bar is tight. Countertop pairs are excluded from the gate entirely (see
#: _is_countertop_pair) rather than accommodated by this number -- see its docstring for why
#: no fixed threshold can be correct for them.
#:
#: History: 0.050 (Task 1) -> 0.040 (Task 2 fix round 1, after a real 48.3 mm microwave clip
#: slid under 0.050) -> 0.020 (Task 2 fix round 2). The 0.040 step was itself unsound: the
#: "30.8 mm countertop seam" it was calibrated against is not a constant. It is that
#: kitchen's counter_thickness, sampled uniform(0.03, 0.05) at procedural_scenes.py:99, and
#: fcl reports the full slab thickness as penetration depth for two countertop segments that
#: merely abut -- measured directly on island: seed 0 gives 30.8 mm, seed 2 gives 46.3 mm,
#: seed 4 gives 49.5 mm, seed 5 gives 40.3 mm. A fixed threshold chasing that distribution is
#: chasing a moving target; excluding the pairs is the only fix that holds across seeds.
MAX_PENETRATION_M = 0.020


def _is_countertop_pair(pair: tuple[str, str]) -> bool:
    """True if either object in `pair` is a countertop segment.

    Countertop segments are authored to abut their neighbours flush (and each other, at the
    corner seam, by a deliberate overlap) -- see MAX_PENETRATION_M's docstring for the
    counter_thickness measurement showing that "depth" for these pairs is a per-seed slab
    dimension, not a clipping signal. fixture_overlaps keeps reporting them (unfiltered --
    see its own docstring); the gate excludes them here, at the point that decides pass/fail,
    so the raw data stays available to anything else that reads fixture_overlaps directly.
    """
    a, b = pair
    return a.startswith("countertop") or b.startswith("countertop")


def open_every_joint(kitchen) -> list[str]:
    """Mutate `kitchen` to the most-open configuration every articulated joint can reach.

    fixture_overlaps (and every gate built on it before this function existed) only ever
    measured the CLOSED configuration. That is a real blind spot: a door swings through
    space no closed-pose check ever visits, so a fixture placed in a door's swing path
    reads as a clean 0.0 mm right up until a task tries to open that door. Confirmed
    directly: a table anchored to kitchen_island's front face measured 0.0 mm against the
    closed refrigerator and 199.2 mm against it with the door open (fix round 3).

    Each joint's target is min(pi/2, its own limit_upper). pi/2 is the standard "door
    swung open" reference this project's door-open task phase uses, but
    Scene.update_configuration does NOT itself enforce joint limits (verified directly: it
    accepts an out-of-range value without raising), and this kitchen's prismatic drawer
    joints have limit_upper well under pi/2 (0.2754-0.63 m observed, vs pi/2 = 1.571) --
    a flat pi/2 would push those drawers to ~1.6 m of unrealistic travel, distorting
    whatever it is a caller is trying to measure. Revolute door joints range up to a
    limit_upper of pi (a full 180 degrees), so min() also caps those at the pi/2 reference
    rather than swinging them all the way open.

    Returns the joint names whose target was actually clamped below pi/2 (limit_upper <
    pi/2), so a caller can report which ones needed it.
    """
    joint_names = kitchen.get_joint_names()
    if not joint_names:
        return []

    props = kitchen.get_joint_properties()
    uppers = np.array([props[name]["limit_upper"] for name in joint_names])
    target = np.minimum(np.pi / 2.0, uppers)
    kitchen.update_configuration(list(target))

    clamped = uppers < (np.pi / 2.0 - 1e-9)
    return [name for name, was_clamped in zip(joint_names, clamped) if was_clamped]


def _node_top_z(kitchen, node: str) -> float:
    """World-frame height of a node's top face."""
    bounds = _world_bounds(kitchen, node)
    return float(bounds[1][2]) if bounds is not None else float("-inf")


def fridge_shelf_order(kitchen) -> list[str]:
    """Refrigerator shelf nodes, highest first.

    Mesh index MOSTLY tracks height here but not entirely: shelf_3/2/1/0 descend in
    order and then shelf_4 sits below all of them, at the bottom of the compartment.
    Measuring rather than sorting by name is what keeps "Fridge shelf 5" pointing at
    the bottom shelf instead of a middle one. Used to verify the registry's hardcoded
    per-shelf patterns (Task 4 Step 5).

    All five are in the main compartment behind `refrigerator/door`: the freezer is a
    separate top compartment (freezer_separator at z~0.99, its own door above that) and
    contains no shelf geometry, so there is no freezer placement to register.
    """
    shelves = [
        n for n in kitchen.scene.graph.nodes
        if fnmatch.fnmatchcase(n, "refrigerator/shelf_[0-9]")
    ]
    return sorted(shelves, key=lambda n: -_node_top_z(kitchen, n))


def _support_for(kitchen_name: str, loc: int, kitchen=None) -> str:
    """Support label for a placement location, or "" when the pair is impossible.

    The support label IS the scene-graph node name: unique by construction, which is
    what label_support() requires (same label twice overwrites).
    """
    placement = PLACEMENT_BY_LOC.get(loc)
    if placement is None or kitchen is None:
        return ""

    nodes = match_support_nodes(kitchen, placement)
    # A matched node is not the same thing as a labelled one. label_support finds the horizontal
    # facets of the geometry it is pointed at, and when it finds none it logs "No supports found
    # for label '<x>'" and registers NOTHING -- after which place_objects(support_ids=[node])
    # raises KeyError mid-batch. Measured on the Objaverse table library: 49 of the 50 offer a
    # tabletop facet, and `db9344bf45ca` offers none (its top is 1664 up-facing triangles at
    # z = 0.721 that trimesh never merges into a facet above label_support's 0.01 m^2 floor), so
    # `build_kitchen("island", <a bowl at loc 42>, table=<that one>)` raised KeyError: 'table/top'.
    # Filtering on what was actually registered turns that crash into the same outcome an
    # impossible (kitchen, loc) pair already gets: the object is skipped and left out of `objects`.
    # _label_supports is build_kitchen's only caller and runs before this, so the metadata is
    # populated by the time this reads it; a kitchen that was never labelled has no key at all and
    # falls through to the pattern match, which is what it means to ask before labelling.
    labelled = kitchen.scene.metadata.get("support_polygons")
    if labelled:
        nodes = [n for n in nodes if n in labelled]
    if not nodes:
        return ""
    if placement.pick == "largest":
        return max(nodes, key=lambda n: _support_area(kitchen, n))
    return random.choice(nodes)


def _label_supports(kitchen, kitchen_name: str) -> list[str]:
    """Label every placement support and return the geometry node names it labeled.

    Both the labeling and the menu now come from PLACEMENTS, so a support that no menu
    entry can select cannot exist. The returned list is exactly the set of support
    geometry the preview's pickSurface() raycasts against, so a drag can only ever land
    on a real support surface.

    "labeled" is checked, not assumed. This appended every matched node unconditionally, so the
    returned list -- which the preview restricts its raycast to -- could name a label that does
    not exist, and a drag aimed at it would resolve to nothing. label_support returns the support
    data it registered and an empty list when it registered none; see _support_for for the
    measured case (one Objaverse table in fifty).
    """
    supports: list[str] = []
    for placement in PLACEMENTS:
        for node in match_support_nodes(kitchen, placement):
            if node in supports:
                continue
            if not kitchen.label_support(label=node, geom_ids=node):
                continue
            supports.append(node)
    return supports


def fixtures_for(kitchen_name: str) -> list[Fixture]:
    """The fixtures this layout gets, in registry order."""
    return [f for f in FIXTURES if kitchen_name in f.kitchens]


def add_fixtures(kitchen, kitchen_name: str, rng, counter_height: float) -> list[str]:
    """Attach this layout's fixtures. Mutates `kitchen`; returns the object ids added.

    Public (not `_add_fixtures`) because it has two callers: `build_kitchen` below (the
    wizard/GUI path) and `goal_generator.generate_kitchen_scene` (the batch path). Both must
    agree on what a kitchen contains -- `support_geom_ids()`/`kitchen_surfaces()` advertise
    fixture placements (e.g. loc 40 "microwave_interior") to both, so a kitchen built without
    this call has locations LOCATION_KITCHENS claims but the scene graph does not, and
    support_generator(support_ids=...) KeyErrors. See
    test_support_geom_ids_label_every_offered_location in test_kitchen_build.py.

    joint_type is left at add_object's default of "fixed" ON PURPOSE. That default is what
    makes the USD exporter apply RigidBodyAPI (exchange/usd.py:746-752). add_walls defaults
    to joint_type=None instead, which is exactly why the room shell's walls exported with no
    rigid body and made every emitted task file raise "Failed to find a rigid body" until it
    was caught. Do not pass joint_type here without re-reading that history.
    """
    added: list[str] = []
    for fixture in fixtures_for(kitchen_name):
        asset = fixture.build(rng, counter_height)
        kitchen.add_object(
            asset,
            fixture.key,
            connect_parent_id=fixture.parent,
            connect_parent_anchor=fixture.parent_anchor,
            connect_obj_anchor=fixture.obj_anchor,
        )
        added.append(fixture.key)
    return added


#: How far the walkway between the counter run and the table's ORIGIN is, in metres, for a table
#: no deeper than the deepest procedural variant. See default_table_transform for the measurement.
TABLE_WALKWAY_M = 1.2

#: The depth default_table_transform's TABLE_WALKWAY_M was calibrated against: "dining_wide", the
#: deepest of the twelve procedural variants at 1.00 m, and the deepest table
#: test_the_default_position_is_itself_valid certifies. A table deeper than this keeps its NEAR
#: EDGE where this one's sits rather than its origin where this one's sits -- see
#: default_table_transform.
TABLE_CALIBRATED_DEPTH_M = 1.00


def default_table_transform(kitchen, depth: float = None) -> np.ndarray:
    """Where a table starts before the user drags it: clear of the furniture, on the floor.

    Placed beyond the furniture's -y face, centred on x. That side is chosen because it is the
    room side on every layout this generator builds -- simvla_data_generator's single_wall case
    puts the robot's spawn band at y in [-1.5, -1], i.e. negative y, and the appliance runs back
    onto positive y. A default on the +y side would start the table inside the wall, which is
    exactly how the withdrawn shelf fixture failed.

    1.2 m, not the 0.9 m first tried: `bounds` is the CLOSED-pose bounding box (add_table calls
    this before any joint is opened), so it does not account for a door swinging further into
    -y than the closed appliance run. Measured directly: at 0.9 m, single_wall's dishwasher
    (the appliance whose door swings furthest into the room) clips the "dining_long" table by
    up to 174.1 mm once every joint is opened (seed 0; 154.2 mm seed 1, 138.1 mm seed 2 -- 0 on
    seeds 3-5, because the dishwasher's swept door position also varies by seed). 1.1 m already
    clears all six seeds with 0 mm to spare on the worst one; 1.2 m keeps a real margin for
    table variants deeper than "dining_long" (0.85 m) that this default has to serve too, e.g.
    "dining_wide" at 1.00 m. Every other layout cleared at 0.9 m already and clears a fortiori
    at 1.2 m, since increasing this offset only moves the table further from every appliance.

    1.2 m IS AN ORIGIN OFFSET, AND THE THING THAT HAS TO CLEAR THE DOOR IS THE NEAR EDGE. Every
    number above was measured on tables at most 1.00 m deep, where the two are within 0.10 m of
    each other. The Objaverse library reaches 1.93 m deep, and at that depth a fixed origin offset
    puts the near edge 0.235 m from the counter instead of 0.700 m -- inside the swept door.
    Measured over all fifty offered mesh tables at their default pose, five layouts, seed 0: 21 of
    50 were rejected, every one of them on `single_wall` and every one of them the table against an
    opened range / base_cabinet / dishwasher door, 27 to 678 mm. Not one was a chair, and the
    closed-pose depth was 0.0 mm on all 300 combinations.

    So `depth` (the table's own y-extent, which only add_table knows) shifts the origin far enough
    to keep the NEAR EDGE where TABLE_CALIBRATED_DEPTH_M's near edge sits. The max() is what makes
    this a strict extension rather than a re-calibration: for every table at or under the
    calibrated depth -- which is all twelve procedural variants -- the offset is exactly the
    TABLE_WALKWAY_M this was certified at, so nothing that passed before moves by a micron. Only a
    table deeper than anything previously offered is pushed out, and it is pushed exactly far
    enough to stand where "dining_wide" stands.

    depth=None keeps the old behaviour for a caller that has no asset to measure.

    Returns a world-frame 4x4 that positions the table's ORIGIN, and z is left at 0 here
    deliberately: a TableAsset's origin is near its bbox centre, not its underside, so this
    matrix on its own buries half the table in the floor. add_table() bottom-aligns it, since
    only add_table has the built asset whose bounds say how far down that origin sits.

    The caller may replace it wholesale with a dragged matrix.
    """
    bounds = kitchen.scene.bounds
    centre_x = float(bounds[0][0] + bounds[1][0]) / 2.0
    walkway = TABLE_WALKWAY_M
    if depth is not None:
        walkway = max(walkway, (TABLE_WALKWAY_M - TABLE_CALIBRATED_DEPTH_M / 2.0)
                      + float(depth) / 2.0)
    clear_y = float(bounds[0][1]) - walkway
    return tra.translation_matrix([centre_x, clear_y, 0.0])


def add_table(kitchen, variant_key: str, transform=None, scale: float = 1.0) -> str:
    """Put the chosen table into the scene at `transform`, or at the default pose.

    Returns the object id, always "table" -- one table per kitchen, and PLACEMENTS' table_top
    row matches the literal node name "table/top".

    `scale` multiplies the table on all three axes -- see FURNITURE_SCALE_IS_UNIFORM --
    and it is handed to the variant's own builder rather than applied to what comes back, because
    the two halves of the registry reach it differently (TableVariant says how). Everything
    downstream of the build measures the asset rather than assuming its size, so nothing else here
    has to know: default_table_transform is passed the BUILT depth, so a scaled-up table is pushed
    out far enough to keep its near edge clear of a swung door; the bottom-alignment below reads
    the built min_z; and _seats_around_table lays the chair ring out from _table_bounds, which is
    the scaled table's real world extent.

    The default pose is BOTTOM-ALIGNED here: default_table_transform positions the table's
    origin, and a TableAsset's origin is neither its underside nor exactly its centre -- a
    0.74 m table measures z in [-0.355, +0.385] locally, about 15 mm off centre. Dropping that
    origin at z=0 left the underside 355 mm BELOW the floor: an object at loc 42 landed at
    ~0.39 m instead of ~0.72 m, add_room_shell measured a scene with a negative z-min, and in
    Isaac the table intersected the ground plane. Offsetting by -min_z (never by height/2,
    which the 15 mm asymmetry makes wrong) puts the legs on z=0. Nothing caught this because
    the build scene has no floor geometry, so fixture_overlaps sees no contact -- and the
    preview hid it, since dropOnto() bottom-aligns on any drag, so only the accepted DEFAULT
    ever shipped buried.

    An explicitly passed `transform` is used as given, deliberately: it is a wholesale world
    pose, and the one that reaches production is a dragged matrix that dropOnto() has ALREADY
    bottom-aligned against the z=0 ground plane. Lifting it again here would float the table.

    joint_type is left at add_object's default of "fixed", for the same reason add_fixtures
    leaves it: that default is what makes the USD exporter apply RigidBodyAPI. add_walls
    defaults to None instead, which is why the room shell's walls exported with no rigid body
    and broke every emitted task file until a review caught it.
    """
    variant = TABLE_VARIANT_BY_KEY.get(variant_key)
    if variant is None:
        raise ValueError(f"Unknown table variant: {variant_key!r}")

    asset = variant.build(check_furniture_scale(scale))

    if transform is None:
        built = asset.scene().scene.bounds
        min_z = float(built[0][2])
        # Left-multiplied: the lift is along WORLD z, so it must not be rotated by whatever
        # the default pose is (a pure translation today, but this survives it gaining a yaw).
        transform = tra.translation_matrix([0.0, 0.0, -min_z]) @ default_table_transform(
            kitchen, depth=float(built[1][1] - built[0][1]))

    kitchen.add_object(asset, "table", transform=transform)
    # Both of these are no-ops for the twelve procedural variants, whose five children TableAsset
    # names itself (leg_0..leg_3, top), and both are load-bearing for an Objaverse one, whose one
    # child is named by the source OBJ's own material group. See each function.
    _normalise_geometry_names(kitchen, "table")
    _name_the_table_top(kitchen)
    return "table"


def _name_the_table_top(kitchen) -> str:
    """Rename the table's work-surface child to "top", so `table/top` names it whatever built it.
    Returns that node, or None when the kitchen has no table geometry.

    PLACEMENTS' table_top row is the fnmatch pattern ("table/top",) and loc 42 -- "On the table" --
    is that pattern resolving. A TableAsset supplies the name for free; an Objaverse mesh supplies
    whatever its OBJ declared, which is "geometry_0" on 16 of the fifty, "Object_0" on 7,
    "<uid>_obj" on 17 and one-offs like "Table_1_Poles_0" on the rest. Against those, "table/top"
    matches nothing, match_support_nodes returns [], _label_supports never labels a surface, and
    the wizard offers "On the table" as a menu entry that silently places nothing -- the exact
    failure mode _pattern_to_regex's docstring records for "wall_cabinet/top". Measured before this
    existed: 0 of the fifty resolved table_top; after it, 50 of 50 do.

    The work surface is the child reaching HIGHEST, ties broken by plan area. Every mesh in the
    library today contributes exactly one child (50/50, measured 2026-08-13), so on the library as
    it stands this is simply "rename the only child"; the rule is written for more than one because
    the child count comes from the asset, not from this repo, and a table whose top and legs
    arrived as separate material groups must still put its TOP under the placement pattern rather
    than a leg. Highest rather than largest: on a desk with a modesty panel the panel can out-area
    the top in plan, and an object placed on a panel would hang in the air beside the desk.

    Already-correct is left alone rather than re-done: a procedural table already has table/top and
    renaming it to itself would be a no-op through two scene-graph rewrites.

    Renames through rename_geometries / rename_nodes for the same reason _normalise_geometry_names
    does -- they carry the object_nodes metadata across and invalidate the scene-graph cache.
    """
    nodes = [n for n in kitchen.scene.graph.nodes_geometry if n.split("/")[0] == "table"]
    if not nodes:
        return None
    if "table/top" in nodes:
        return "table/top"

    def _reach(node):
        bounds = _world_bounds(kitchen, node)
        if bounds is None:
            return (-math.inf, -math.inf)
        return (float(bounds[1][2]), _support_area(kitchen, node))

    top = max(nodes, key=_reach)
    if top in kitchen.scene.geometry:
        kitchen.rename_geometries([(top, "table/top")])
    kitchen.rename_nodes([(top, "table/top")])
    return "table/top"


def _furniture_bounds(kitchen) -> np.ndarray:
    """kitchen.scene.bounds, but excluding any chairs already placed.

    add_chair calls default_chair_transforms(kitchen, index + 1) fresh on every invocation, and
    plain kitchen.scene.bounds includes whatever is already in the scene -- so once one chair
    sits at its default pose (deliberately outside the room's original footprint, to clear the
    furniture), the NEXT call's bounds already include that chair, and the spread keeps drifting
    outward call after call. Measured on six sequential add_chair calls before this fix: y went
    -3.021, -4.140, -5.259, -6.377, -7.496, -8.615 -- 5.6 m of drift by the sixth chair, with x
    converging on x_hi the same way. Excluding chair_* top-level nodes anchors every call to the
    same furniture footprint, which is what default_chair_transforms' docstring promises and only
    a single one-shot call happened to satisfy.
    """
    mins, maxs = [], []
    for node in kitchen.scene.graph.nodes_geometry:
        if node.split("/")[0].startswith("chair_"):
            continue
        bounds = _world_bounds(kitchen, node)
        if bounds is None:
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    return np.array([np.min(mins, axis=0), np.max(maxs, axis=0)])


#: How many seats add_chair lays its default arrangement out for, independent of how many chairs
#: have actually been placed. EIGHT, because that is the wizard's own cap on chairs per kitchen.
#:
#: It has to be the cap and not a number below it. add_chair draws chair i's seat from a
#: max(DEFAULT_CHAIR_ROW, i + 1)-seat ring, so with this at 6 and eight chairs placed, chairs 0-5
#: come from a six-seat ring, chair 6 from a seven-seat one and chair 7 from an eight-seat one --
#: three rings of different radii, and the last two chairs land beside neighbours they were never
#: laid out against. Measured on island / square_small, eight chairs at their own defaults, over
#: all fifty offered chairs:
#:
#:   DEFAULT_CHAIR_ROW = 6   16 of 50 clean, 34 REJECTED
#:   DEFAULT_CHAIR_ROW = 8   50 of 50 clean, worst depth 0.0 mm
#:
#: Every failure was "chair 5 overlaps chair 6" or "chair 6 overlaps chair 7", at 0.58 m and
#: 0.63 m apart against the 1.118 m their own slot rule reserves. Note the shape of that result:
#: a third of the library passed, so it LOOKED like it worked.
#:
#: Raising it re-lays the six-chair ring too -- six chairs now take the first six seats of an
#: eight-seat arrangement -- so the whole placement sweep was re-run against it rather than
#: assumed unaffected. Geometrically: 62 tables (the twelve procedural and the fifty Objaverse) x
#: all 50 chair widths x counts 2..8 = 86,800 adjacent seat pairs, worst slack 0.0 m, none
#: overlapping.
#:
#: WHAT IT COSTS, recorded because it is a real change to what a default kitchen looks like: at
#: eight seats the ring stands 1.18-2.80 m off square_small's 0.70 m edge, against 0.61-1.47 m at
#: six. The eight-chair findings call 1.35 m "no longer reads as seating at a table". That is
#: geometrically valid and visually a ring of chairs in a room; it is the user's call and they
#: made it.
#:
#: add_chair asked for an arrangement of `index + 1` seats and took the LAST one, which is a
#: harmonic series converging on the furniture's x_hi rather than a layout: measured on island /
#: seed 0 with a 2.979 m span, the six chairs landed at x = 1.076, 1.573, 1.821, 1.970, 2.069,
#: 2.140 -- gaps of 497, 248, 149, 99 and 71 mm, so the last four interpenetrated and the gate
#: reported "chair 4 overlaps chair 5 by 169 mm". Six chairs at their OWN defaults were rejected
#: before the author had touched anything.
#:
#: Asking for a fixed six every time is what makes each chair land in the seat it would have had
#: if all six were placed at once, so adding them one at a time gives the same arrangement as
#: adding them together -- and _furniture_bounds / _table_bounds already guarantee every call
#: measures the same footprint.
DEFAULT_CHAIR_ROW = 8


#: The plan-view footprint a seat reserves for a chair whose own footprint is NOT known, in
#: metres. Every offered chair has a measurement (chair_footprint_m), so this is the fallback for
#: a manifest written before build_chair_manifest recorded export_footprint_m.
#:
#: 1.12 m is the registry's worst case UNDER THE MEASURE chair_footprint_m uses -- the plan-view
#: diagonal, not the axis-aligned width. Measured over the fifty offered chairs (2026-08-13,
#: after the re-curation that restored _aspect_ratio and hand-rejected 21 non-chairs): widest
#: diagonal 1.112 m (uid a51e4acfdbb3, 0.826 m across its box), median 0.818 m, narrowest
#: 0.542 m. test_the_reserved_chair_footprint_covers_every_offered_chair fails if a wider chair is
#: ever added to the registry.
#:
#: It read 0.91 and was the number EVERY seat was sized by, which was wrong twice over, and the
#: two errors hid each other:
#:
#:   * 0.91 was max(x, y) of the widest chair's exported box. The ring yaws every chair, so the
#:     width a chair presents to its neighbour is not its axis-aligned extent -- 14 of the fifty
#:     have a plan diagonal over 0.91 m, up to 1.112 m. The uniform ring survived that only
#:     because 0.91 is comfortably more than the DIAGONAL of a typical chair (median 0.818 m), so
#:     the slack it carried for narrow chairs paid for the rotation swell of wide ones.
#:   * That slack is exactly what made every kitchen's chairs sit 1.01 m apart and 0.53 m off the
#:     table edge, narrow sets included.
#:
#: Taking the footprint per chair without also making it rotation-invariant removes the slack and
#: leaves the swell: measured on the previous fifty, six of `3108e5f9977b` (0.635 x 0.625
#: exported, so 0.890 m across a 45-degree yaw) went from clean to "chair 2 overlaps chair 3 by
#: 309 mm" on island/square_small. The two changes have to land together, which is why this
#: fallback is a diagonal even though it is larger than any chair's width.
CHAIR_FOOTPRINT_M = 1.12

#: The clear gap left between two neighbouring chairs, on top of their own footprints, so that
#: they clear each other rather than merely touching at exactly the bar.
CHAIR_SEAT_GAP_M = 0.10

#: Centre-to-centre spacing between two chairs of unknown width sharing a side. _seat_slots
#: produces the per-chair version: two neighbours are spaced (w_i + w_j) / 2 + CHAIR_SEAT_GAP_M
#: apart, which is this number exactly when both fall back. The uniform pitch this layout used
#: before the footprint went per-chair was 1.01 m (0.91 + 0.10) on EVERY seat; the measured chairs
#: now pitch from 0.64 m to 1.21 m, with a median of 0.92 m.
CHAIR_SEAT_PITCH_M = CHAIR_FOOTPRINT_M + CHAIR_SEAT_GAP_M

#: The clear gap between the table's edge and a chair's nearest face. Half that chair's own
#: footprint is the point of first contact, so its centre sits that plus this out from the edge
#: (see _table_clearance). Kept deliberately tight: the chairs are meant to read as seating AT the
#: table, and Task 3 measured that a chair does not report against the table at all until it is
#: pushed 20 cm PAST the edge.
CHAIR_TABLE_GAP_M = 0.075

#: How far a default chair's CENTRE sits outside the table's edge when nothing about the chair is
#: known -- half the fallback footprint plus the gap. The ring itself uses _table_clearance, which
#: is this rule applied to the WIDEST chair actually on the ring.
CHAIR_TABLE_CLEARANCE_M = CHAIR_FOOTPRINT_M / 2.0 + CHAIR_TABLE_GAP_M

#: How far the no-table chair row stands off the furniture's closed bounding box, centre to box.
#:
#: 0.75 m was not enough and the placement gate says so: on single_wall, six chairs at 0.75 m
#: blocked the refrigerator, base cabinet and range doors by up to 136 mm. The number that is
#: actually calibrated against a swung door is default_table_transform's 1.2 m -- its docstring
#: records single_wall's dishwasher, the deepest-swinging appliance, clipping a table by 174 mm
#: when that offset was 0.9 m. A chair's centre at 1.2 m puts its near edge at 0.825 m, further
#: out than the deepest table's own near edge (1.2 - 1.00/2 = 0.7 m), which is the position
#: test_the_default_position_is_itself_valid already holds.
CHAIR_COUNTER_CLEARANCE_M = 1.2


def chair_footprint_m(uid, scale: float = 1.0) -> float:
    """The plan-view width one seat has to reserve for the chair `uid`, in metres.

    `scale` is that chair's own scale (FURNITURE_SCALE_MIN..MAX), and multiplying it in HERE is
    what makes a scaled chair reach the ring: the diagonal is a plan-view length, the scale
    multiplies both plan-view extents, so the scaled diagonal is diagonal x scale exactly. Every
    seat-sizing path -- _seat_widths, _seat_slots, _table_clearance, _grow_past_the_corners -- takes
    its numbers from this one function, so there is no second place a scale could be forgotten.

    The scale is uniform now (FURNITURE_SCALE_IS_UNIFORM) and this number is unchanged by that: a
    seat reserves PLAN AREA, and the piece's height has never entered it. What the ring cannot see
    is the one thing the height reaches -- a tall chair back standing in a countertop's underside
    -- which is the placement gate's job and is measured there (_is_reportable_clash).

    The DIAGONAL of that chair's own measured export_footprint_m -- hypot(w, d) -- not the larger
    of the two extents. The ring yaws every chair it seats (see _seats_around_table), and a yawed
    box presents its diagonal at 45 degrees; reserving max(w, d) reserves the width the chair has
    in the one orientation it is never placed in. Measured on `3108e5f9977b`, 0.635 x 0.625
    exported: its placed world AABB is 0.89 m across, and six of them reserved 0.635 m each
    collide by 309 mm on island/square_small. The diagonal is the smallest bound that holds at
    every yaw, and it is rotation-invariant, so the ring does not have to know the angle in
    advance.

    Read straight off the manifest, never off the mesh: this is called once per chair per build
    (and add_chair is called per chair per build), and a trimesh.load here would put a mesh load
    on the layout path for the sake of a number the export stage already measured.

    CHAIR_FOOTPRINT_M when the manifest has no measurement. Unknown uids (and None) get the
    fallback rather than raising: this is asked while LAYING OUT a ring, where a uid that
    add_chair would reject is not this function's error to report, and reserving the worst case
    for it is the safe reading.
    """
    variant = CHAIR_VARIANT_BY_UID.get(uid)
    if variant is None or variant.footprint_m is None:
        return CHAIR_FOOTPRINT_M * float(scale)
    return float(variant.footprint_m) * float(scale)


def _seat_widths(count: int, footprints=None) -> list:
    """The per-seat footprints a `count`-seat arrangement reserves.

    `footprints` is what the caller knows: one width per chair it is actually placing, in seat
    order. Shorter than `count` is the normal case -- the ring is laid out for DEFAULT_CHAIR_ROW
    seats however many chairs exist (see that constant), so the tail seats are empty.

    An empty seat reserves as much as the WIDEST chair already on the ring, not the registry's
    worst case. The point of the per-chair footprint is that a set of narrow chairs stops reading
    as if it held the widest one; padding the empty seats with CHAIR_FOOTPRINT_M would put that
    chair back on the ring in spirit and leave two narrow chairs spread as far apart as before.
    Reserving the widest chair the author has actually chosen is the smallest reservation that
    still holds another chair like the ones they picked. With nothing known at all -- a bare
    default_chair_transforms call with no footprints -- the fallback stands.
    """
    widths = [float(w) for w in (footprints or [])][:count]
    pad = max(widths) if widths else CHAIR_FOOTPRINT_M
    return widths + [pad] * (count - len(widths))


def _seat_slots(widths) -> list:
    """The arc length each seat occupies: its own footprint plus one CHAIR_SEAT_GAP_M.

    This is the reflow rule, and it is the one rule that makes a mixed-width ring right. Two
    neighbours end up (w_i + w_j) / 2 + CHAIR_SEAT_GAP_M apart centre to centre -- the mean of the
    two footprints, so a narrow chair beside the armchair gets the room the ARMCHAIR needs and no
    more, and two narrow chairs get neither more nor less than they need. Handing every seat the
    same slot cannot express that: the pitch between two specific neighbours depends on both of
    them, not on the widest chair anywhere on the ring.

    Reduces exactly to the old uniform CHAIR_SEAT_PITCH_M when every width is CHAIR_FOOTPRINT_M.
    """
    return [w + CHAIR_SEAT_GAP_M for w in widths]


def _table_clearance(widths) -> float:
    """How far the ring's seat CENTRES stand off the table's edge, before any growth.

    One radius for the whole ring, so it takes the widest chair on it: the ring is a single offset
    curve and a per-seat radius would put neighbours at different distances from the table, which
    reads as a mistake rather than as tighter packing. Half that chair's footprint is its point of
    first contact with the table, plus CHAIR_TABLE_GAP_M of clear air.
    """
    return max(widths) / 2.0 + CHAIR_TABLE_GAP_M


def _table_bounds(kitchen):
    """World bounds of the table, or None if this kitchen has none.

    Reads only nodes whose top-level object is "table", so -- unlike _furniture_bounds -- it is
    unaffected by the counter run as well as by chairs. That is what keeps the seating ring fixed
    while chairs are added to it one at a time.
    """
    mins, maxs = [], []
    for node in kitchen.scene.graph.nodes_geometry:
        if node.split("/")[0] != "table":
            continue
        bounds = _world_bounds(kitchen, node)
        if bounds is None:
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        return None
    return np.array([np.min(mins, axis=0), np.max(maxs, axis=0)])


def _walk_the_ring(cx, cy, width, depth, slots, clearance) -> list:
    """The (x, y) seat points for these slots on a ring at this clearance.

    The open path: down the -x end, along the -y side, up the +x end. Both ends start at the
    table's own centre line rather than at the +y corner, so no seat reaches the walkway -- see
    _seats_around_table for why the +y side is not seating.

    Each seat sits at the centre of its own slot, then everything is STRETCHED to fill whatever
    path the ring actually has. The clearance only ever grows the ring, so the path is >= what the
    slots need; scaling by path / needed spreads the surplus in proportion to each chair's own slot
    rather than bunching every chair at the -x end and leaving the surplus at the +x end.
    """
    half_x = width / 2.0 + clearance
    half_y = depth / 2.0 + clearance
    legs = (
        (half_y, lambda t: (cx - half_x, cy - t)),          # -x end, from cy toward -y
        (2.0 * half_x, lambda t: (cx - half_x + t, cy - half_y)),   # the -y side
        (half_y, lambda t: (cx + half_x, cy - half_y + t)),  # +x end, back toward cy
    )
    path = sum(length for length, _ in legs)
    needed = sum(slots)

    seats = []
    walked = 0.0
    for slot in slots:
        walk = path * (walked + slot / 2.0) / needed
        walked += slot
        for length, point in legs:
            if walk <= length:
                seats.append(point(walk))
                break
            walk -= length
        else:                                # floating-point overshoot on the last seat
            seats.append(legs[-1][1](legs[-1][0]))
    return seats


#: How many bisection steps _grow_past_the_corners takes. Each one halves a bracket that starts
#: no wider than the base clearance itself (a metre or so), so 60 steps put the answer within
#: ~1e-18 m -- far below double precision on these magnitudes, i.e. exact. Bisection rather than
#: "grow by the shortfall": the shortfall shrinks by well under half per round near the solution
#: (growing the clearance by d lengthens the path by 4d AND stretches every gap, so the two
#: effects partly cancel), and a 40-round shortfall walk still left 1.2e-06 m on the table --
#: enough for a test asserting the bound at 1e-06 to fail.
_CORNER_GROWTH_STEPS = 60


def _grow_past_the_corners(cx, cy, width, depth, widths, slots, clearance) -> float:
    """`clearance`, grown until no two adjacent chairs can overlap. Never shrinks it.

    THE ARC IS NOT THE DISTANCE. Seats are spread by arc length along the ring, which spaces two
    neighbours on the same straight leg exactly right -- but the ring turns two right angles, and
    a pair straddling a corner is separated by the CHORD, which is shorter than the arc between
    them. _seats_around_table's own docstring used to claim the opposite ("going around a corner
    costs arc length, so corner neighbours are handled by the same rule as the rest"); measured on
    island / boxy_long with six of `c14060be9257`, the four straight pairs sit at 0.839 m to the
    millimetre -- exactly their slot rule -- and the two corner pairs sit at 0.670 m, 20% short,
    which the placement gate reports as "chair 4 overlaps chair 5 by 21 mm".
    That defect is older than this function: the uniform CHAIR_FOOTPRINT_M ring had it too and hid
    it, because reserving 0.91 m for a 0.74 m chair leaves 20% of slack to spend on a corner. Take
    the slack away by reserving per chair and the corner is what is left.

    The bar here is (w_i + w_j) / 2 -- the two chairs' bounding discs just touching -- and NOT
    their slot spacing (w_i + w_j) / 2 + CHAIR_SEAT_GAP_M. The gap buys clear air between two
    chairs standing SIDE BY SIDE along one edge of the table; a corner pair does not share an
    edge, it faces two different ones, so the thing that has to hold there is the geometric one:
    they must not intersect. Since chair_footprint_m is a bound at every yaw, a disc of that
    diameter contains the chair however the seat turns it, and centres that far apart cannot
    overlap whatever the two chairs are.

    Costs nothing on 8 of the 12 table variants at six chairs: on a square-ish table the seats
    land exactly ON the corners, so no pair straddles one and this returns its argument unchanged.
    Where it does bite it is small -- boxy_long's median-width ring goes 0.795 -> 0.909 m, against
    0.915 m for the uniform ring it replaces -- and a ring of the narrowest offered chair still
    stands 0.526 m off square_small's edge against the uniform ring's 1.165 m.
    """
    def short_by(c):
        seats = _walk_the_ring(cx, cy, width, depth, slots, c)
        return max(
            [(widths[i] + widths[i + 1]) / 2.0 - math.dist(seats[i], seats[i + 1])
             for i in range(len(seats) - 1)] + [0.0]
        )

    if short_by(clearance) <= 0.0:
        return clearance                     # already clear: the common case, and free

    # Bracket, then bisect. The chords grow without bound with the clearance, so a big enough
    # clearance always clears and the doubling terminates.
    lo, hi = clearance, clearance + max(widths)
    for _ in range(_CORNER_GROWTH_STEPS):
        if short_by(hi) <= 0.0:
            break
        lo, hi = hi, hi * 2.0
    for _ in range(_CORNER_GROWTH_STEPS):
        mid = (lo + hi) / 2.0
        if short_by(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi                                # the clear side of the bracket, never the short one


def _seats_around_table(bounds, count: int, footprints=None) -> list:
    """`count` seating POSES around a table with these world `bounds`.

    Poses, not positions: each seat carries the yaw its own place on the ring implies (see "The
    orientation is the seat's, not the chair's" below), so these are full 4x4s.

    Seats go on the THREE sides away from the counter run -- the -y long side and both ends --
    and never on the +y side, which is the one facing the kitchen. That is a measurement, not a
    preference, and it overrides the tidier "two per long side, one per short side" reading:

    default_table_transform puts the table 1.2 m from the furniture's closed bounding box, and
    that 1.2 m was itself calibrated so the TABLE clears an opened appliance door (its docstring
    records single_wall's dishwasher clipping a table by 174 mm at 0.9 m). A chair seated on the
    +y side of the table sits roughly 0.5 m nearer the counter than the table's own near edge,
    which is inside the swept door. Measured with six of the widest chair, seed 0: l_shaped's
    refrigerator door clipped by 385 mm, peninsula's range by 191 mm, u_shaped's sink cabinet by
    150 mm and dining_wide by 316 mm. Only the island layout had room. So the +y side is not
    seating; it is the walkway between the table and the counter.

    Seats are then spread by ARC LENGTH along that open path, rather than assigned per side.
    Per-side assignment cannot serve twelve table variants from 0.70 x 0.70 to 1.80 x 0.85: two
    chairs on square_small's 0.70 m side land 350 mm apart, narrower than one chair. Arc length
    gives the same spacing whatever the proportions -- but it does NOT handle a corner by itself,
    because a pair straddling one is separated by the chord and not by the arc. See
    _grow_past_the_corners, which is what makes the corner pairs true.

    Seats sit at the CENTRE OF THEIR OWN SLOT along the path (_seat_slots), which is half a step
    in from each end -- so the outermost chairs are half a gap from the walkway rather than
    standing in it -- and, for equal footprints, is exactly the old (i + 0.5) / count.

    THE ORIENTATION IS THE SEAT'S, NOT THE CHAIR'S. Each seat is yawed to look at the table's
    centre, and that yaw belongs here rather than to face_chair_toward because at a ring corner
    the angle is fixed by WHERE the seat is, not by anything about the chair standing in it. Left
    to face_chair_toward alone it was skipped entirely for any chair below FACING_CONFIDENCE_CUT
    (rightly -- see that constant: a near-symmetric mesh has no backrest to read), so those chairs
    sat square-on while their neighbours splayed, and a corner pair collided. Measured on the
    twenty-chair registry: the four below the cut failed from THREE chairs upward on 8 of the 12
    table variants, and the narrowest of them produced the deepest penetration -- width was not
    the driver, orientation was (measured directly, sweep 3).

    The yaw assumes the chair's own +x is its front. For a chair above the cut that assumption is
    then CORRECTED by face_chair_toward, which re-aims it by the angle its measured facing needs
    and so lands it in exactly the pose it landed in before this yaw existed. For a chair below
    the cut nothing corrects it, and nothing should: its front is arbitrary either way, so the
    ring's tangent is no worse a direction than the export frame's and is the one that packs.
    """
    centre = (bounds[0] + bounds[1]) / 2.0
    cx, cy = float(centre[0]), float(centre[1])
    width = float(bounds[1][0] - bounds[0][0])
    depth = float(bounds[1][1] - bounds[0][1])

    # The open path is (width + depth + 4 * clearance) long, so the clearance is what decides
    # whether `count` chairs fit along it. Grow it when they otherwise would not: at the base
    # clearance, six of the widest chair overlap by 135 mm around "square_med" and "boxy_square"
    # (0.90 x 0.90) and by 20-40 mm around "dining_small", "bar_small", "low_coffee" and
    # "inset_legs". This is one rule applied to every table, not a special case for the small
    # ones -- the tables that already have room are unaffected, since the max() below never
    # shrinks the ring.
    #
    # It does mean a small table gets a visibly wide ring (0.70 x 0.70 pushes the chairs about
    # 0.93 m off the table edge). That is honest: six people do not fit at a 0.70 m table, and a
    # default that reads as "a ring of chairs" is better than one the placement gate rejects
    # before the author has touched anything.
    widths = _seat_widths(count, footprints)
    slots = _seat_slots(widths)
    needed = sum(slots)
    clearance = max(_table_clearance(widths), (needed - width - depth) / 4.0)
    clearance = _grow_past_the_corners(cx, cy, width, depth, widths, slots, clearance)

    seats = _walk_the_ring(cx, cy, width, depth, slots, clearance)

    poses = []
    for x, y in seats:
        # Rotate about the seat itself, never the world origin: tra.translation_matrix(p) @ Rz is
        # exactly the yaw-about-p that face_chair_toward composes on top of (its `yaw @ pose` is
        # T(p) R T(-p) T(p) = T(p) R), so the two agree by construction.
        angle = float(np.arctan2(cy - y, cx - x))
        poses.append(tra.translation_matrix([x, y, 0.0])
                     @ tra.rotation_matrix(angle, [0.0, 0.0, 1.0]))
    return poses


def default_chair_transforms(kitchen, count: int, footprints=None) -> list:
    """`count` distinct starting poses, spread so newly added chairs do not stack.

    `footprints` is one plan-view width per chair being placed, in seat order -- the caller's own
    chair_footprint_m readings. It may be shorter than `count` (the tail seats are empty) or None
    (nothing is known, every seat reserves CHAIR_FOOTPRINT_M). See _seat_widths and _seat_slots
    for how a mixed-width ring reflows.

    Two layouts, which is what the design doc asked for ("placed in turn around the table if one
    exists, or along the open floor beside the counter run if not"):

    AROUND THE TABLE when the kitchen has one -- see _seats_around_table. A single row was used
    for both cases originally, and it cannot work: six chairs across the furniture's 2.979 m
    x-span is a 426 mm pitch, while the chairs are 441 mm wide at the median and 728 mm at the
    widest, so six chairs at their OWN defaults were rejected by the placement gate before the
    author had touched anything.

    ALONG THE OPEN FLOOR when it has none, on the room side of the furniture -- negative y on
    every layout this generator builds: simvla_data_generator's single_wall case puts the robot's
    spawn band at y in [-1.5, -1], and the appliance runs back onto positive y. A default on +y
    would start a chair inside the wall, which is how the withdrawn shelf fixture failed. That
    row is now pitched at CHAIR_SEAT_PITCH_M and centred on the furniture, rather than dividing
    the span into `count + 1` -- dividing the span is what produced the 426 mm pitch.

    Measured against _furniture_bounds / _table_bounds, never kitchen.scene.bounds directly:
    add_chair calls this fresh on every invocation, and plain scene.bounds would include chairs
    already placed, which drifts the layout outward call after call. See _furniture_bounds.
    """
    table = _table_bounds(kitchen)
    if table is not None:
        return _seats_around_table(table, count, footprints)

    bounds = _furniture_bounds(kitchen)
    centre_x = float(bounds[0][0] + bounds[1][0]) / 2.0
    y = float(bounds[0][1]) - CHAIR_COUNTER_CLEARANCE_M

    # Slot centres along the row, centred on the furniture -- the same _seat_slots rule the ring
    # uses, in one dimension, so a row of narrow chairs closes up the way a ring of them does.
    # Reduces to (i - (count - 1) / 2) * CHAIR_SEAT_PITCH_M when every width is the fallback.
    #
    # No yaw here, unlike the ring. On a straight row the position does NOT determine the angle:
    # every seat faces the same way, so there is no positional component to separate out and
    # nothing for a below-cut chair to be square-on to. Facing stays face_chair_toward's business.
    slots = _seat_slots(_seat_widths(count, footprints))
    span = sum(slots)
    walked = 0.0
    poses = []
    for slot in slots:
        poses.append(tra.translation_matrix(
            [centre_x + walked + slot / 2.0 - span / 2.0, y, 0.0]))
        walked += slot
    return poses


def add_chair(kitchen, uid: str, transform=None, index: int = 0, footprints=None,
              scale: float = 1.0, dynamic: bool = False) -> str:
    """Put a chair into the scene as chair_<index>. Returns that id.

    `scale` multiplies this chair on all three axes -- see FURNITURE_SCALE_IS_UNIFORM.
    It is baked into the mesh (_mesh_asset_scale), not carried in the node transform, because the
    placement gate's collision manager only sees geometry. It also reaches the SEAT the chair
    stands in, via the default `footprints` below: a chair 15% wider needs 15% more of the ring,
    and a scale that widened the mesh without widening its reservation is exactly how six chairs
    come to overlap while the gate that is supposed to catch it passes.

    `footprints` is the whole arrangement's per-seat widths (see default_chair_transforms), which
    only a caller that knows every chair can supply -- build_kitchen does, and passes it. A lone
    add_chair knows one chair, so its default is that chair's own footprint for every seat: this
    call cannot see what will sit in seat 4, and assuming more chairs like this one is both the
    common case (the wizard's + button adds copies of one chair) and self-consistent, which is
    what matters. Two DIFFERENT chairs added by two bare add_chair calls would each lay out a ring
    of their own width and land on rings of different radii -- the defect class DEFAULT_CHAIR_ROW
    exists to prevent -- so a caller placing a mixed set must pass footprints, as build_kitchen
    does.

    Distinct ids per instance are load-bearing: fixture_overlaps groups contacts by top-level
    object name and discards same-object pairs, so two chairs sharing an id would have a real
    collision between them silently dropped by both gates.

    Records obj_id -> uid on THIS KITCHEN before returning -- the scene graph carries only the
    already-transformed geometry, never which ChairVariant produced it, and facing_direction_for
    needs exactly this mapping to load the right mesh back for a given chair_<n> node. See
    _chair_uids for why the mapping belongs to the kitchen rather than to this module.

    `dynamic` decides whether this chair is a PROP OR AN OBJECT, and it is the whole difference
    between a chair the robot can push and one welded to the floor. It defaults to False, so
    every kitchen built before it existed -- 1202, 1216, 1217 and every wizard kitchen -- is
    bitwise what it was.

    False attaches the chair with add_object's default joint_type="fixed", for the same reason
    add_fixtures and add_table leave it: that default is what makes the USD exporter apply
    RigidBodyAPI. add_walls defaults to None instead, which is why the room shell's walls
    exported with no rigid body and broke every emitted task file until a review caught it.

    True attaches it with joint_type="floating" -- THE SAME ATTACHMENT A PLACED OBJECT GETS.
    apply_placements says why that is the one to reach for: a placed object hangs off its
    support geometry on an edge carrying "the metadata the USD exporter's joint gate uses to
    recognize a free-floating rigid body (a 'floating' joint)", and the exporter applies
    RigidBodyAPI/MassAPI inside that same branch. Concretely, in scene_synthesizer:

      * add_object writes {"joint": {"name": "<obj>/<parent>_<type>_joint", "type": <type>}}
        onto the parent->child edge, for any joint_type that is not None (scene.py:1429).
      * The exporter reads exactly that edge (exchange/usd.py:657-667). For BOTH "fixed" and
        "floating" it applies rigid_body_api=True, mass_api=True (usd.py:746-754) -- which is
        why the RigidBodyAPI check the build driver already ran could never see this defect.
      * The types part company one level down, in add_joint_to_usd: "floating" writes nothing
        (usd.py:493), "fixed" writes a PhysicsFixedJoint prim at /world/<obj>/world_fixed_joint
        (usd.py:495-506) with body_0_path="" -- the world edge passes "" (usd.py:738) -- so its
        physics:body0 relationship has no targets.
      * kitchen_scene_generator.fix_missing_fixed_joint_targets_and_wall_cabinets then finds
        that prim BY NAME ("fixed" in prim.name), sets kinematicEnabled=True on the chair and
        deletes the joint. Kitchen 1217's build log, job 2106962:
        "/world/chair_0/world_fixed_joint missing target 0, converting parent to kinematic."

    A kinematic body ignores contact, which is what smoke job 2107003 measured: the base held
    against chair_1 for 300+ frames, d(chair_1) 0.449 -> 0.427 against a 0.468 m contact
    threshold, the achieved velocity collapsing from 0.150 to 0.003 m/s while the chair reported
    net +0.000 the whole time. joint_type=None is NOT the alternative -- see above; it removes
    the rigid body along with the weld.
    """
    variant = CHAIR_VARIANT_BY_UID.get(uid)
    if variant is None:
        raise ValueError(f"Unknown chair: {uid!r}")
    scale = check_chair_scale(uid, scale)

    obj_id = f"chair_{index}"
    kitchen.add_object(
        synth.assets.MeshAsset(variant.obj_path, **_mesh_asset_scale(scale)),
        obj_id,
        # "fixed" is add_object's own default and is passed explicitly so the two branches read
        # as one choice rather than as a default and an override -- see the docstring.
        joint_type="floating" if dynamic else "fixed",
        # A fixed DEFAULT_CHAIR_ROW-seat arrangement, NOT one of `index + 1` seats -- see that
        # constant for the measured convergence the old form produced. max() so an index past the
        # wizard's cap still resolves to a seat instead of raising IndexError; past the cap the
        # arrangement widens per chair again, which is unreachable from the wizard.
        transform=default_chair_transforms(
            kitchen, max(DEFAULT_CHAIR_ROW, index + 1),
            footprints if footprints is not None else [chair_footprint_m(uid, scale)],
        )[index]
                  if transform is None else transform,
    )
    # Before anything else touches this chair: its geometry children carry the source OBJ's own
    # material-group names, which are arbitrary text. One of the twenty offered chairs is called
    # "Chair_02 - Default_0", and the spaces in it kill kitchen.export(file_type="usd") -- inside
    # commit_kitchen, at the first byte written. See _normalise_geometry_names.
    _normalise_geometry_names(kitchen, obj_id)
    _chair_uids(kitchen)[obj_id] = uid
    return obj_id


@functools.lru_cache(maxsize=1)
def _tf_module():
    """pxr.Tf if this process has it, else None.

    Imported lazily and never at module scope: kitchen_build must stay importable without Isaac
    (the whole test suite does exactly that), and kitchen_wizard's pre-boot invariant forbids
    reaching anything Omniverse-shaped before AppLauncher has run.

    Caching a None is safe precisely because _usd_safe_identifier's fallback is pinned to Tf's
    behaviour by test -- a process that answers None here produces the same names as one that
    answers Tf.
    """
    try:
        from pxr import Tf
    except Exception:
        return None
    return Tf


def _usd_safe_identifier(name: str) -> str:
    """`name` reduced to a legal USD prim name.

    Tf.MakeValidIdentifier is the authority and is used whenever pxr is importable. Guessing
    which characters matter is the entire bug this exists to fix: scene_synthesizer's exporter
    (exchange/usd.py:191) replaces only "-", ":" and "." and therefore lets a SPACE through, and
    a chair whose OBJ names its material group "Chair_02 - Default_0" kills
    kitchen.export(file_type="usd") outright.

    The fallback is a byte-for-byte reproduction of Tf's rule, not a hand-picked character list:
    every byte outside [0-9A-Za-z_] becomes "_", a leading digit becomes "_", and an empty result
    becomes "_". It operates on UTF-8 BYTES because Tf does -- Tf.MakeValidIdentifier("unicode"
    with two 2-byte characters) returns "__n__code", one underscore per byte, not per character.
    test_the_identifier_fallback_agrees_with_tf pins this against the real Tf for every offered
    chair's own name plus the awkward cases.
    """
    tf = _tf_module()
    if tf is not None:
        return name if tf.IsValidIdentifier(name) else tf.MakeValidIdentifier(name)

    out = bytearray()
    for byte in name.encode("utf-8"):
        legal = (48 <= byte <= 57) or (65 <= byte <= 90) or (97 <= byte <= 122) or byte == 95
        out.append(byte if legal else 0x5F)
    safe = out.decode("ascii")
    if not safe:
        return "_"
    if safe[0].isdigit():
        safe = "_" + safe[1:]
    return safe


def _normalise_geometry_names(kitchen, obj_id: str) -> dict:
    """Rename obj_id's geometry children to legal USD prim names. Returns {old node: new node}.

    A chair's geometry children are named by the source OBJ's own material groups, so their names
    are arbitrary text that no code in this repo chooses. MeshAsset guards only the one case where
    a name STARTS WITH A DIGIT (assets.py, "This creates problems when exporting as USD"); every
    other illegal character reaches the exporter, which then only fixes three of them.

    Names are kept DISTINCT after normalisation. Two children differing only in an illegal
    character ("a b" and "a-b") both reduce to "a_b" and would collide into a single prim -- one
    mesh silently replacing another, which is worse than the crash this function prevents,
    because nothing raises. Already-legal names are reserved first so a collision renames the
    offending child rather than the innocent one.

    Renames through Scene.rename_nodes / rename_geometries rather than by touching the graph
    directly: those carry the object_nodes / object_geometry_nodes metadata across and invalidate
    the scene-graph cache, both of which a hand-rolled rename would silently skip.
    """
    prefix = f"{obj_id}/"
    leaves = {
        node: node[len(prefix):]
        for node in sorted(kitchen.scene.graph.nodes_geometry)
        if node.startswith(prefix)
    }
    taken = {leaf for leaf in leaves.values() if _usd_safe_identifier(leaf) == leaf}

    renames: dict[str, str] = {}
    for node, leaf in leaves.items():
        safe = _usd_safe_identifier(leaf)
        if safe == leaf:
            continue
        base, suffix = safe, 2
        while safe in taken:
            safe = f"{base}_{suffix}"
            suffix += 1
        taken.add(safe)
        renames[node] = prefix + safe

    if not renames:
        return {}

    geometries = kitchen.scene.geometry
    geometry_map = [(old, new) for old, new in renames.items() if old in geometries]
    if geometry_map:
        kitchen.rename_geometries(geometry_map)
    kitchen.rename_nodes(list(renames.items()))
    return renames


def _import_build_chair_manifest():
    """Lazy import of scripts/tools/build_chair_manifest.py -- the Task 1 module that owns
    facing_direction. Resolved relative to THIS file, not the caller's cwd, the same way that
    module resolves its own sibling scripts/simvla import (see its export()/convert()
    docstrings): `Path(__file__).resolve().parent.parent / "tools"` is scripts/tools regardless
    of where the process was launched from.

    The sys.path insert is guarded (only added once) so that placing many chairs -- one lookup
    per chair -- does not grow sys.path by one duplicate entry per call.
    """
    import sys

    tools_dir = str(Path(__file__).resolve().parent.parent / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import build_chair_manifest

    return build_chair_manifest


def _chair_mesh_for(kitchen, chair_id: str):
    """The trimesh.Trimesh add_chair actually placed at chair_id, read back through the uid
    add_chair recorded on `kitchen`. The mesh comes from disk, not from the scene graph, which
    carries no uid -- but WHICH mesh is a question only this kitchen can answer, since every
    kitchen has a chair_0 of its own.
    """
    uid = _chair_uids(kitchen).get(chair_id)
    variant = CHAIR_VARIANT_BY_UID.get(uid) if uid is not None else None
    if variant is None:
        raise KeyError(
            f"{chair_id!r} has no chair variant on record -- add_chair must place it before "
            "its facing can be measured."
        )
    return trimesh.load(variant.obj_path, force="mesh")


def facing_direction_for(kitchen, chair_id: str) -> tuple:
    """The base facing (build_chair_manifest.facing_direction) of whichever chair variant
    add_chair placed at chair_id, in that chair's own exported frame -- before any placement yaw.
    """
    return _import_build_chair_manifest().facing_direction(_chair_mesh_for(kitchen, chair_id))


#: Task 1's own measurement (chair-placement Task 1 findings): splitting 38 real, exported
#: chairs by this same relative-offset magnitude at 0.02 puts 8/38 below it -- near-symmetric
#: geometry, no real seat/back split, one of them the exact chair whose sign disagreed with its
#: recorded facing_axis -- and 30/38 at or above it; restricted to those 30, agreement between
#: the offset's sign and the recorded axis is 30/30 (100%). Below the cut, facing_direction
#: still returns A unit vector (the sign of a near-zero offset, not "unknown"), but rotating a
#: chair on that sign is turning it on noise -- worse than leaving it as exported, because a
#: rotated chair reads as a deliberate placement and an un-rotated one reads as merely not yet
#: handled.
FACING_CONFIDENCE_CUT = 0.02


def _facing_confidence_for(kitchen, chair_id: str) -> float:
    """|offset on the dominant horizontal axis| / that axis's own extent -- Task 1's own
    "relative offset" metric (chair-placement Task 1 findings), the measurement
    FACING_CONFIDENCE_CUT's 0.02 was derived from. This is NOT the same number
    facing_direction (scripts/tools/build_chair_manifest.py) computes internally and discards
    on its way to a unit vector -- that internal value is `hypot(*offset)`, the raw offset
    magnitude in metres (e.g. ~0.0815 m for CHAIR_VARIANTS[0]); this one divides the
    dominant-axis COMPONENT of that same offset by the mesh's own extent along that axis,
    giving a unitless relative figure (~0.0852 for that same chair) that is comparable across
    chairs of different sizes. Recomputed here from the same mesh with the same vertex-split
    method rather than widening facing_direction's own return type: that signature is locked by
    Task 1's own test suite (test_chair_manifest.py unpacks it as `fx, fy =
    facing_direction(mesh)`, a strict 2-tuple), so there is no seam to extend it through without
    breaking that contract.

    See FACING_CONFIDENCE_CUT for what a caller does with the number this returns.
    """
    mesh = _chair_mesh_for(kitchen, chair_id)
    v = np.asarray(mesh.vertices, dtype=float)
    z_lo, z_hi = float(v[:, 2].min()), float(v[:, 2].max())
    upper = v[v[:, 2] > z_lo + 0.5 * (z_hi - z_lo)]
    if len(upper) == 0:
        return 0.0
    offset = upper[:, :2].mean(axis=0) - v[:, :2].mean(axis=0)
    dominant = int(np.argmax(np.abs(offset)))
    extent = float(v[:, dominant].max() - v[:, dominant].min())
    return float(abs(offset[dominant]) / extent) if extent > 1e-9 else 0.0


def chair_facing_target(kitchen) -> tuple:
    """The XY point chairs should face: the table's centre if there is one, else the counter
    run's.

    Most kitchens will have no table -- it is a user choice -- so the fallback is not an edge
    case, it is the common path. Countertops exist on every layout this generator builds, which
    is what makes them a safe second choice.
    """
    tops = [n for n in kitchen.scene.graph.nodes_geometry if n.split("/")[0] == "table"]
    if not tops:
        tops = [n for n in kitchen.scene.graph.nodes_geometry
                if n.split("/")[0].startswith("countertop")]
    pts = [kitchen.scene.graph.get(n)[0][:2, 3] for n in tops]
    centre = np.mean(np.asarray(pts, dtype=float), axis=0)
    return (float(centre[0]), float(centre[1]))


def face_chair_toward(kitchen, chair_id: str, target_xy) -> None:
    """Yaw a chair in place so it faces target_xy.

    Rotates about the chair's OWN centre, not the world origin -- a rotation about the origin
    would sweep the chair across the room as well as turning it, which is the classic version of
    this bug and looks like a placement failure rather than a rotation one.

    Skips the rotation entirely when this chair's backrest signal is below FACING_CONFIDENCE_CUT
    (see its docstring): a near-symmetric mesh has no reliable "away from the backrest"
    direction, and rotating on that noise is worse than leaving the chair as exported -- it
    looks deliberate and is arbitrary.

    Keeps the object on its existing parent edge and carries that edge's data forward, exactly
    as apply_placements does: re-parenting under "world" would leave the pose right but strip
    the joint metadata the USD exporter reads to decide this is a real body. (In practice a
    chair's parent IS already "world" -- add_chair never anchors one to another fixture -- so
    this mostly guards a future caller that changes that.)
    """
    graph = kitchen.scene.graph
    pose = np.asarray(graph.get(chair_id)[0], dtype=float)
    here = pose[:2, 3]

    to_target = np.asarray(target_xy, dtype=float) - here
    n = float(np.linalg.norm(to_target))
    if n < 1e-9:
        return                                   # already on top of the target; nothing to face
    to_target /= n

    if _facing_confidence_for(kitchen, chair_id) < FACING_CONFIDENCE_CUT:
        return                                   # no reliable backrest signal; leave as exported

    base = np.asarray(facing_direction_for(kitchen, chair_id), dtype=float)
    current = pose[:2, :2] @ base
    current /= float(np.linalg.norm(current))

    angle = float(np.arctan2(to_target[1], to_target[0]) - np.arctan2(current[1], current[0]))
    about = np.array([here[0], here[1], 0.0])
    yaw = trimesh.transformations.rotation_matrix(angle, [0, 0, 1], about)

    parent = graph.transforms.parents.get(chair_id)
    if parent is None:
        graph.update(frame_to=chair_id, frame_from="world", matrix=yaw @ pose)
        return
    edge = graph.transforms.edge_data[(parent, chair_id)]
    preserved = {k: v for k, v in edge.items() if k != "matrix"}
    graph.update(frame_to=chair_id, frame_from=parent, matrix=yaw @ pose, **preserved)


#: add_chair names its objects chair_0, chair_1, ... -- fullmatch, never a substring test, so a
#: fixture whose name merely contains "chair" is not mistaken for one.
_CHAIR_ID_RE = re.compile(r"chair_\d+")


def _is_furniture(name: str) -> bool:
    """True for the user-placed furniture the placement gate answers about: the table and every
    chair.

    `name` is a TOP-LEVEL object name, the only thing fixture_overlaps ever puts in a pair key --
    so these are exactly the names add_table and add_chair give their objects.
    """
    return name == "table" or _CHAIR_ID_RE.fullmatch(name) is not None


def _furniture_label(name: str) -> str:
    """How a placement problem names an object.

    Furniture gets prose -- "the table", "chair 0" -- because the author put it there and has to
    find it again, and with six chairs in the room "the chair" would not say which. Anything else
    keeps the exact name fixture_overlaps reports, which is both what the table-only messages
    already said and what the preview labels the fixture.
    """
    if name == "table":
        return "the table"
    if _CHAIR_ID_RE.fullmatch(name):
        return name.replace("_", " ")
    return name


def _is_reportable_clash(pair: tuple[str, str], depth: float) -> bool:
    """Whether one fixture_overlaps entry is a placement problem worth telling the author about.

    Three filters, in the order they are cheapest to reason about:

    1. At least one side must be furniture. Two fixtures clipping each other is a generator
       defect that the fixture suite owns; surfacing it here would tell the author to fix
       something they cannot move.

    2. The depth must clear MAX_PENETRATION_M -- see its docstring for the measured tiers.

    3. The countertop exclusion applies to the TABLE ONLY, deliberately.

       _is_countertop_pair exists for an fcl artefact measured on countertop-versus-countertop:
       segments authored to abut flush report the whole slab thickness as penetration depth
       (30.8 / 46.3 / 49.5 / 40.3 mm on island seeds 0 / 2 / 4 / 5, because "depth" there is that
       seed's counter_thickness, not a clipping signal).

       Chairs have no such authored abutment with a countertop -- nothing in this module places a
       chair relative to one -- and the depths fcl reports for a chair are real geometry, not a
       slab constant. So excluding chair pairs here could only ever hide a real clash, and is not
       done.

       IT IS LOAD-BEARING NOW, and the note that used to stand here ("unreachable; the tallest
       chair reaches 7.3 mm against a 0.934 m back") is retired as stale on two counts. Re-measured
       on island / seed 0 (2026-08-17), driving the tallest OFFERED chair 50 mm into the counter
       run, whose slab spans 0.919..0.950 m:

         * at 1.00x that chair is 0.960 m to the back, not 0.934 -- the library has been
           re-curated since -- and it reports 129.0 mm against countertop_base_cabinet. So the
           branch was ALREADY reachable before the scale became uniform; the old note was
           describing a chair the registry no longer offers.
         * at FURNITURE_SCALE_MAX it is 1.249 m and reports 20.1 mm. Lower, not higher, because
           fcl's depth is the particular sub-mesh pair that touches, not how far through the slab
           the chair reaches -- which is exactly why no fixed threshold can be read as a distance.

       Both are over the 20 mm bar and both are REPORTED, which is the behaviour this exclusion's
       shape was chosen for. Nothing here changed; what changed is that the paragraph can no
       longer claim the branch is decorative.

       WHAT IS NOT COVERED, recorded because the same measurement found it. The table half of this
       exclusion has a hole, and a uniform scale widens it. A table driven 50 mm into the counter
       run at 1.00x is caught -- "the table overlaps base_cabinet by 50 mm" -- because a 0.740 m
       top is below the slab and meets the cabinet body. At FURNITURE_SCALE_MAX that top is
       0.962 m, above the slab's own 0.950 m: the only pair it makes is
       (countertop_base_cabinet, table) at 50.0 mm, this exclusion drops it, and the gate reports
       the table against the refrigerator alone. The hole predates the uniform scale -- "bar_tall"
       is 1.05 m unscaled and already vanishes the same way (100.8 mm, excluded) -- but it used to
       need one of the two bar-height variants and now any table dragged there at the top of the
       range falls in it.

       LEFT AS IT IS, deliberately, because removing the exclusion is a gate change and not a
       scaling change. The argument for removing it is that its premise expired: the table was an
       automatic fixture anchored to kitchen_island's far corner when this was written, and it is
       user-placed now (see FIXTURES), so it has no more authored abutment with a countertop than
       a chair does. Measured before leaving it alone: over 62 tables x FURNITURE_SCALE_MAX x five
       layouts at their DEFAULT poses, both joint configurations, not one table-versus-countertop
       pair occurs at all -- so dropping the exclusion would reject nothing that passes today. It
       is the author's call to make, not this commit's.

       Note `"table" in pair` is tuple membership -- exact equality against one of the two
       elements -- NOT a substring test on a name. It cannot disagree with an `== "table"` test
       on either element, and it cannot be fooled by a "side_table": fixture_overlaps builds
       every key as a 2-tuple of top-level names and drops same-owner pairs, so "table" appears
       at most once in a pair.
    """
    if not (_is_furniture(pair[0]) or _is_furniture(pair[1])):
        return False
    if depth <= MAX_PENETRATION_M:
        return False
    if _is_countertop_pair(pair) and "table" in pair:
        return False
    return True


def furniture_placement_problems(kitchen) -> list[str]:
    """Why the user-placed furniture -- the table and every chair -- is not acceptable where it
    currently stands. Empty means every piece is fine.

    Runs the same two gates the automated fixture tests run, on the scene the author actually
    dragged: once at the closed pose, once with every joint opened. The open pass is not
    optional -- the table's own original anchor measured 0.0 mm closed and 199.2 mm against
    the open refrigerator door, and a closed-pose check would have accepted it.

    Covers chair-versus-chair as well as furniture-versus-fixture. That pair is not hypothetical:
    two chairs at the same spot measure 40.1 mm, and 24.6 mm at 0.4 m apart (island / seed 0),
    both over the bar, while chairs 0.6 m apart produce no pair at all.

    MUTATES the kitchen's joint configuration (open_every_joint). Callers that need the scene
    afterwards should re-build or re-close it; the wizard discards this copy either way.

    Messages name the pair and the depth, because "invalid position" tells the author nothing
    they can act on.
    """
    problems: list[str] = []

    def _clashes() -> dict:
        return {
            pair: depth
            for pair, depth in fixture_overlaps(kitchen).items()
            if _is_reportable_clash(pair, depth)
        }

    def _sides(pair: tuple[str, str]) -> tuple[str, str]:
        """(subject, other) -- the furniture side leads the sentence, because it is the thing the
        author can move.

        When BOTH sides are furniture (chair-versus-chair, or a chair against the table) the
        first element leads and the second is named as the other side, so the message still names
        both. fixture_overlaps sorts each pair, so which one leads is stable across runs rather
        than depending on collision order.
        """
        return pair if _is_furniture(pair[0]) else (pair[1], pair[0])

    for pair, depth in sorted(_clashes().items(), key=lambda kv: -kv[1]):
        subject, other = _sides(pair)
        problems.append(
            f"{_furniture_label(subject)} overlaps {_furniture_label(other)} by "
            f"{depth * 1000:.0f} mm"
        )

    open_every_joint(kitchen)
    for pair, depth in sorted(_clashes().items(), key=lambda kv: -kv[1]):
        subject, other = _sides(pair)
        if _is_furniture(other):
            # Furniture has no door to block, so the open pass has nothing new to say about a
            # furniture-versus-furniture pair -- opening the joints does not move either piece.
            # Repeating the closed wording lets the dedupe below drop it instead of emitting
            # "chair 0 blocks chair 1's door when open", which is simply not a thing.
            message = (
                f"{_furniture_label(subject)} overlaps {_furniture_label(other)} by "
                f"{depth * 1000:.0f} mm"
            )
        else:
            message = (
                f"{_furniture_label(subject)} blocks {other}'s door when open, by "
                f"{depth * 1000:.0f} mm"
            )
        if message not in problems:
            problems.append(message)

    return problems


def table_placement_problems(kitchen) -> list[str]:
    """The old name for furniture_placement_problems, kept for its live callers.

    Not table-only: it is the same function, so a kitchen that has chairs in it gets its chairs
    checked too. That is the intended behaviour for every caller -- there is no caller that wants
    a table checked while a chair is left clipping the refrigerator.
    """
    return furniture_placement_problems(kitchen)


#: What one entry of build_kitchen's `chairs` may say. Only `uid` is required.
CHAIR_SPEC_KEYS: frozenset = frozenset({"uid", "transform", "scale", "material", "dynamic"})


def chair_spec(spec) -> dict:
    """One `chairs` entry, normalised to all five keys. ValueError on anything else.
    See build_kitchen for what each key means.

    ONE SHAPE, A DICT PER INSTANCE, and this function is where that is enforced. The shape used to
    be a flat list of uids in the wizard and a list of (uid, transform) pairs at build_kitchen, and
    both worked only while every chair of a given uid was interchangeable. Per-piece scale and
    material end that: two chairs can now be the same uid and different objects, so the instance
    has to carry its own properties.

    A tuple -- the OLD pair shape -- is refused by name rather than by a generic type error, and
    that is the point of writing this out. Every call site had to change, and a missed one silently
    drops the author's edit; a `(uid, transform)` pair would otherwise iterate perfectly well as a
    two-element sequence and produce a chair with no scale and no material, which is exactly
    today's behaviour and therefore invisible.

    `material` is carried here and IGNORED by build_kitchen, deliberately. A material is not
    geometry: it is an MDL bound to prim paths at export (see furniture_material_overrides and
    apply_materials), long after this scene is built. Splitting it into a second, parallel list
    keyed by position is how the two would drift apart, so the instance keeps all of its own
    properties in one place and each consumer reads the ones it needs.
    """
    if isinstance(spec, (tuple, list)):
        raise ValueError(
            "A chair is one dict per instance -- {'uid': ..., 'scale': ..., 'material': ...} -- "
            f"not a {type(spec).__name__}. The (uid, transform) pair shape is gone: a chair now "
            "carries its own scale and material, so two chairs of the same uid can differ."
        )
    if not isinstance(spec, dict):
        raise ValueError(f"A chair must be a dict, not {type(spec).__name__}.")
    unknown = set(spec) - CHAIR_SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown chair fields: {', '.join(sorted(unknown))}.")
    uid = spec.get("uid")
    if not isinstance(uid, str):
        raise ValueError(f"A chair needs a uid, not {type(uid).__name__}.")
    material = spec.get("material")
    if material is not None and material not in MATERIALS:
        raise ValueError(f"Unknown material group {material!r}.")
    return {
        "uid": uid,
        "transform": spec.get("transform"),
        # check_chair_scale rather than check_furniture_scale, so that this and add_chair go on
        # asking the same question about the same chair if the per-chair floor ever becomes a
        # bound again (it is advisory today -- see chair_width_floor_scale).
        "scale": check_chair_scale(uid, spec.get("scale", 1.0)),
        "material": material,
        # bool(), not the value as given: this ends up choosing add_object's joint_type, and a
        # truthy string or a numpy bool sailing through to `"floating" if dynamic else "fixed"`
        # would work by accident here and be a different type in the spec a caller reads back.
        # DEFAULT FALSE, which is what keeps every kitchen built before this key bitwise itself.
        "dynamic": bool(spec.get("dynamic", False)),
    }


def furniture_material_overrides(table=None, chairs=None) -> dict:
    """{prim-path regex -> MATERIALS group} for the pieces the author gave a material of their own.

    Applied by apply_materials as a SECOND pass, after the GEOMETRY2MATERIAL group pass, so that
    the narrower regex here rebinds exactly the prims of exactly one piece and the earlier, broader
    binding stands everywhere else. UsdShade.MaterialBindingAPI.Bind() writes one relationship per
    prim, so re-binding a prim replaces its target and the later call wins -- which is the same
    property apply_materials' own insertion order already depends on (see the notes on the
    "/world/table/.*" and "/world/chair_.../.*" entries in GEOMETRY2MATERIAL), verified on a real
    exported stage by test_a_later_bind_replaces_an_earlier_one_on_the_same_prim.

    `table` is the table's chosen group, or None. `chairs` is build_kitchen's own `chairs` list, in
    the same order -- the nth entry is chair_<n>, exactly as add_chair numbers them -- so a chair
    with no choice of its own contributes no entry at all rather than an entry that re-binds it to
    what it already had.

    The regexes are anchored on the object id and end in "/.*" for the same reason
    GEOMETRY2MATERIAL's own furniture entries do: a chair's geometry CHILDREN are named by the
    source OBJ's material groups, which is arbitrary text this repo does not choose. get_scene_paths
    matches with re.match, so "/world/chair_0/.*" is anchored at the front and the trailing slash
    keeps it off /world/chair_01.
    """
    overrides: dict[str, str] = {}
    if table is not None:
        overrides["/world/table/.*"] = table
    for index, spec in enumerate(chairs or []):
        material = chair_spec(spec)["material"]
        if material is not None:
            overrides[f"/world/chair_{index}/.*"] = material
    return overrides


def build_kitchen(kitchen_name, object_specs, mesh_files, *, seed=None, counter_height=0.95,
                   table=None, table_scale=1.0, chairs=None):
    """Build a kitchen and place objects on it. Writes nothing to disk.

    object_specs: [{"obj_n": "bowl0", "type": "bowl", "loc": 1, "mesh": None}, ...]
                  "mesh" is optional: an explicit BODex path, or None/absent to pick at random.

    table: a TABLE_VARIANT_BY_KEY key to add before supports are labelled, or None for no table.
           Added between add_fixtures and _label_supports on purpose -- see the comment at the
           call site below.

    table_scale: what to multiply that table by, on all three axes, in
           FURNITURE_SCALE_MIN..MAX -- see FURNITURE_SCALE_IS_UNIFORM. A separate argument rather
           than a dict alongside `table`, because a kitchen has exactly one table: the shape
           change `chairs` needed is about telling two INSTANCES of the same design apart, and
           there is no second table to tell this one from. The table's chosen MATERIAL is not
           here at all, for the same reason a chair's is ignored -- see chair_spec.

    chairs: a list of dicts -- ONE SHAPE ONLY, see chair_spec, which is where the fields are
           checked -- added in order as chair_0, chair_1, ... via add_chair, or None/[] for no
           chairs. Per instance:
             uid       -- a CHAIR_VARIANT_BY_UID key. Required.
             transform -- a world-frame 4x4, or absent/None for add_chair's own default seat.
             scale     -- uniform size multiplier, or absent for 1.0.
             material  -- a MATERIALS group bound at EXPORT (furniture_material_overrides), or
                          absent/None to keep the group GEOMETRY2MATERIAL already gives it.
                          Carried here and ignored by this function; see chair_spec.
             dynamic   -- True to attach this chair with a FLOATING joint, i.e. as a movable
                          rigid body rather than a prop welded to the world. Absent/False keeps
                          the fixed joint every chair has always had. See add_chair, which owns
                          the mechanism and the evidence for it.
           Also added before supports are labelled, for the same reason the table is -- see the
           comment at the call site below. This does NOT auto-face the chairs toward anything; a
           caller that wants that calls chair_facing_target and face_chair_toward itself, per
           chair, after build_kitchen returns.

    Returns (kitchen, kitchen_data, objects, supports):
      kitchen      -- scene_synthesizer.Scene; kitchen.scene is a trimesh.Scene
      kitchen_data -- {"kitchen_type": name, "<obj_n>": "<mesh path>", ...}
                      An "<obj_n>" entry is recorded only for objects that were
                      actually placed, i.e. whose (kitchen_name, loc) pairing has a
                      valid support per _support_for(). Objects skipped because the
                      location is impossible for this kitchen type (e.g. "Above
                      island" on a non-island kitchen) have no entry here. Objects
                      that had a valid support but failed to fit still get an entry
                      (only the `objects` list below excludes them).
      objects      -- [{"label": "bowl0", "node_id": "bowl00", "location": "Above cabinet",
                        "support": "countertop_base_cabinet"}]
                      Only objects that actually landed in the scene graph. "support" is the
                      scene-graph node the object was placed on, which the preview uses to
                      disambiguate a drag ray that crosses several supports.
                      When `table` is given, this also carries a {"label": "table", "node_id":
                      "table", "location": "On the floor", "support": None} entry, so the table
                      is a selectable, draggable item in the preview like any placed object --
                      see kitchen_preview.pickObject()/floor_dragged. "support" is None rather
                      than a scene-graph node because a table is not placed ON a support, it
                      stands on the floor and drags against a mathematical plane.
      supports     -- [str, ...] scene-graph node names of every support surface labeled
                      for this kitchen type (countertops, island top, fridge shelves).
                      This is the geometry set the preview's raycast is restricted to, so
                      a drag can only land on a real support surface.
    """
    builder = KITCHEN_BUILDERS.get(kitchen_name)
    if builder is None:
        raise ValueError(f"Unknown kitchen type: {kitchen_name!r}")

    kitchen = builder(seed=seed, counter_height=counter_height)
    kitchen.unwrap_geometries("(sink_cabinet/sink_countertop|countertop_.*|.*countertop)")
    # Fixtures before supports: _label_supports walks the scene graph, so anything added
    # after it would carry no support labels and be undraggable in the preview.
    add_fixtures(kitchen, kitchen_name, np.random.default_rng(seed), counter_height)
    if table is not None:
        # Before _label_supports on purpose: that pass walks the scene graph, so a table added
        # after it carries no support label and nothing can ever be placed on it -- loc 42 would
        # be a menu entry that silently never works.
        add_table(kitchen, table, scale=table_scale)
    # Normalised UP FRONT, before a single chair is added, so a malformed entry anywhere in the
    # list is refused while the scene is still the kitchen alone. Half a set of chairs placed and
    # then a ValueError leaves the caller holding a scene it has no way to describe.
    chair_specs = [chair_spec(spec) for spec in (chairs or [])]
    chair_ids = []
    # Every chair's own footprint, up front, so each add_chair below lays out the SAME ring. A
    # per-call reading would only know the chairs placed so far, so chair 0 and chair 5 would size
    # their rings differently and land at different radii -- the defect Task 1 measured at
    # DEFAULT_CHAIR_ROW (three rings, 0.2125 m apart against a 0.85 m pitch). See _seat_slots.
    #
    # SCALED, because the ring is what reserves the space a chair takes up: chair_footprint_m
    # multiplies the plan diagonal by this chair's own scale, so a set of mixed scales reflows the
    # same way a set of mixed chairs does and the enlarged chair gets the room it now needs.
    chair_footprints = [chair_footprint_m(spec["uid"], spec["scale"]) for spec in chair_specs]
    for index, spec in enumerate(chair_specs):
        # Before _label_supports for the same reason the table is: that pass walks the scene
        # graph, so anything added after it carries no support label. Chairs are not support
        # surfaces themselves, but keeping every add_object call on the same side of that pass
        # is what keeps this ordering constraint from having to be rediscovered per fixture type.
        chair_ids.append(add_chair(kitchen, spec["uid"], transform=spec["transform"], index=index,
                                   footprints=chair_footprints, scale=spec["scale"],
                                   dynamic=spec["dynamic"]))
    supports = _label_supports(kitchen, kitchen_name)

    kitchen_data = {"kitchen_type": kitchen_name}
    objects = []

    if table is not None:
        # The table is user-placed, not a BODex object_spec, so the loop below never sees it --
        # without this entry the preview's JS builds its selectable/draggable items from
        # `objects` and could never offer the table, no matter how correct pickObject()'s
        # ground-drag branch is. node_id "table" is exactly what add_table() above returned
        # and what kitchen_preview.render_page's floor_dragged filters on
        # (`obj["node_id"] == "table"`), so this has to stay in sync with that string.
        #
        # support=None, not a scene-graph node: "support" exists so the preview's
        # pickSurface() can disambiguate a drag ray that crosses several stacked SUPPORT
        # surfaces (countertops, shelves...). A table doesn't rest on one of those -- it
        # stands on the floor and drags against a mathematical plane (pickGround() in
        # kitchen_preview.py) -- so there is no support node to name here, and inventing one
        # would misrepresent what the table is actually resting on.
        objects.append(
            {
                "label": "table",
                "node_id": "table",
                "location": "On the floor",
                "support": None,
            }
        )

    for chair_id in chair_ids:
        # One entry per chair, for exactly the reason the table gets one: without it the preview
        # cannot select or drag the chair, and drag-to-place is the feature the chairs exist for.
        # Everything here is the table's reasoning applied unchanged --
        #
        #   support=None, because a chair stands on the FLOOR. `support` exists so the preview's
        #   pickSurface() can disambiguate a drag ray crossing several stacked support surfaces; a
        #   chair rests on none of them and drags against a mathematical plane (pickGround()), so
        #   naming one would misrepresent what it is standing on.
        #
        #   node_id is whatever add_chair returned, because that is the scene-graph node the
        #   preview posts back and apply_placements looks up.
        #
        # The label comes from _furniture_label, NOT from a format string of its own, and that is
        # load-bearing twice over. It is the string the placement gate uses when it rejects a
        # position ("chair 0 overlaps chair 1 by 39 mm"), so the author can find the offender in
        # the preview's own list -- deriving both from one function is what stops them drifting.
        # And it is DISTINCT per chair: kitchen_preview keys roots/objectMeshes/homeSupportNames
        # by label, so two chairs sharing one would share a three.js root and dragging either
        # would move (and post) the same one.
        objects.append(
            {
                "label": _furniture_label(chair_id),
                "node_id": chair_id,
                "location": "On the floor",
                "support": None,
            }
        )

    for spec in object_specs:
        obj_n, obj_type, loc = spec["obj_n"], spec["type"], spec["loc"]

        support = _support_for(kitchen_name, loc, kitchen=kitchen)
        if not support:
            continue

        # A spec may name its exact mesh (the wizard's object gallery does); otherwise pick a random
        # one matching the type, which is what every caller did before the field existed.
        fname = spec.get("mesh") or resolve_mesh(obj_type, mesh_files)
        up, front, origin = _object_pose(obj_type, fname)

        # PER-OBJECT SCALE, defaulting to the 0.1 every caller got before this key existed.
        #
        # 0.1 suits most of the library but not a sphere: at 0.1 the apple mesh is 0.0759 m across
        # (BODex manifest, median_width_m) against a 0.080 m gripper opening -- 4 mm of clearance,
        # on a shape with no flat to close on. Measured consequence on kitchen 1204 (job 2096839):
        # 390 of ~420 episodes ended with the apple knocked to the FLOOR and the best lift was
        # 0.109 m, where the mug (0.0687 m, cylindrical body) lifts 0.197 m.
        _obj_scale = float(spec.get("scale", 0.1))
        kitchen.place_objects(
            obj_id_iterator=utils.object_id_generator(obj_n),
            obj_asset_iterator=synth.assets.asset_generator(
                itertools.repeat(fname, 1),
                scale=_obj_scale,
                up=up,
                front=front,
                origin=origin,
                align=True,
            ),
            obj_support_id_iterator=kitchen.support_generator(support_ids=support),
            obj_position_iterator=PositionIteratorGridStartXCenter(
                step_x=0.06, step_y=0.06, noise_std_x=0.04, noise_std_y=0.04
            ),
            obj_orientation_iterator=utils.orientation_generator_uniform_around_z(
                lower=0.0, upper=0.0
            ),
        )
        kitchen_data[obj_n] = fname

        # object_id_generator("bowl0") yields "bowl00" for the single placed instance.
        node_id = f"{obj_n}0"
        if node_id not in kitchen.scene.graph.nodes:
            print(f"[build] {obj_n} did not fit on support {support!r}; not placed.")
            continue

        objects.append(
            {
                "label": obj_n,
                "node_id": node_id,
                "location": LOCATION_LABELS[loc],
                # The support this object was actually placed on. The preview's pickSurface()
                # prefers it when a drag ray crosses several supports, so an object resting on
                # a fridge shelf is not yanked onto the fridge top that occludes it.
                "support": support,
            }
        )

    return kitchen, kitchen_data, objects, supports


#: How far a wall stands off the furniture on a side that HAS a counter run backing onto it,
#: in metres, as (min, max) of the openness draw. Skirting scale: a kitchen unit stands against
#: its wall, and a rendered kitchen with a metre of floor BEHIND the cabinets is the report this
#: exists to answer. 0.02 m rather than 0.0 keeps the wall strictly outside the furniture (see
#: test_the_shell_encloses_the_kitchen, which is a strict inequality on purpose).
ROOM_SHELL_FLUSH_M: tuple[float, float] = (0.02, 0.10)

#: How far a wall stands off the furniture on a side with nothing against it -- the walkway.
#: The floor is 0.20 m, which is exactly the low end of the single uniform offset this replaces:
#: no open side of any room is tighter than the tightest room this generator has ever shipped.
#: The ceiling is 0.75 m rather than that range's 1.20 m, which is the change the user asked for.
ROOM_SHELL_WALKWAY_M: tuple[float, float] = (0.20, 0.75)

#: The narrowest strip of wall a window may leave beside or above itself, in metres.
#:
#: Not cosmetic. BoxWithHoleAsset frames its opening with four boxes whose extents are
#: (span - hole)/2 -/+ offset (assets.py:2113); at zero those boxes are degenerate, and
#: scene_synthesizer 1.15 dies computing their centre of mass ("Primitive face normals are
#: immutable"), inside commit_kitchen, at the first byte written. 0.10 m is a tenth of a metre of
#: masonry, which is the smallest strip that still reads as a wall rather than as a rendering
#: artifact.
ROOM_SHELL_HOLE_MARGIN_M = 0.10

#: A wall-MOUNTED cabinet, exactly: "wall_cabinet" and "wall_cabinet_0" .. "wall_cabinet_5".
#: Anchored at both ends because the room shell's own walls are "wall_x" .. "wall_-y" and a
#: `"wall" in name` test catches those too -- the measurement trap this rule is built on.
_WALL_CABINET_RE = re.compile(r"wall_cabinet(_\d+)?$")


def room_shell_params(seed: int) -> dict:
    """Every random choice the shell makes, from one seeded Generator. Pure: builds nothing.

    Split out of add_room_shell so the draws can be tested (and the room reproduced) without a
    5-second kitchen build, and so "what does seed 42 decide" has exactly one answer.

    The draw ORDER is part of the contract. The room-size draw, then the hole spec, then
    thickness, then overhang is the order this has always used, and `holed_dim` is appended at
    the END on purpose: a Generator is a stream, so drawing the wall choice any earlier would
    shift every value after it and silently re-roll the thickness and room size of every kitchen
    in the corpus. Drawn last, and only when there is a hole to place, it leaves the geometry of
    every solid-walled kitchen exactly as it was.

    ONE draw still sizes the room, in the same stream position the single `offset` occupied, and
    that is deliberate. `openness` is the same uniform() call rescaled to [0, 1) -- verified over
    seeds 0..1999 that the old draw is exactly `0.2 + openness` and that thickness, overhang,
    the hole spec and holed_dim are bit-identical per seed. So the ONLY thing this change moves
    is how far the walls stand off, which is the thing it is for. Two draws here (one per band)
    would have re-rolled every wall thickness and every window in the corpus for nothing.

    The two bands it feeds are NOT interchangeable and which side gets which is not decided
    here: it is measured off the built scene by wall_backed_sides(), because this function is
    pure and a wall's proper standoff depends on whether there is a counter run against it.
    """
    rng = np.random.default_rng(seed)

    # How roomy this kitchen is, 0 (tightest) to 1 (roomiest), applied to BOTH bands so a roomy
    # seed is roomy all round. One number, so a room is still a rectangle rather than four
    # independently jittered walls -- and so this consumes exactly the one draw it always did.
    openness = float(rng.uniform(0.0, 1.0))

    # A window or a doorway. hole_extents is (width, height) in metres, hole_offset its centre
    # relative to the wall's. Zero extents means a solid wall, which is why a third of rooms get
    # one: an opening in every wall of every kitchen would be its own kind of monotony.
    if rng.random() < 0.33:
        hole_extents = (float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.8, 1.4)))
        hole_offset = (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0.2, 0.8)))
    else:
        hole_extents = (0.0, 0.0)
        hole_offset = (0.0, 0.0)

    thickness = float(rng.uniform(0.08, 0.20))
    overhang = float(rng.uniform(0.0, 0.4))

    # WHICH wall gets the opening. add_walls applies one hole spec to every dimension in the
    # call, so the single call this used to make punched the same window through all four walls
    # -- and there is nothing behind them. Through a 200 degree fisheye that is four holes onto
    # the void in a third of all kitchens.
    holed_dim = None
    if all(hole_extents):
        holed_dim = ROOM_SHELL_DIMENSIONS[int(rng.integers(len(ROOM_SHELL_DIMENSIONS)))]

    lo, hi = ROOM_SHELL_FLUSH_M
    flush_offset = float(lo + openness * (hi - lo))
    lo, hi = ROOM_SHELL_WALKWAY_M
    walkway_offset = float(lo + openness * (hi - lo))

    return {
        "openness": openness,
        "flush_offset": flush_offset,
        "walkway_offset": walkway_offset,
        "thickness": thickness,
        "overhang": overhang,
        "hole_extents": hole_extents,
        "hole_offset": hole_offset,
        "holed_dim": holed_dim,
    }


def wall_backed_sides(kitchen, tolerance: float = 0.01) -> frozenset[str]:
    """Which of ROOM_SHELL_DIMENSIONS have a counter run backing onto them. MEASURED.

    A wall cabinet hangs on a wall. Its footprint is a shallow, wide box, so the axis of its
    SMALLER horizontal extent is the axis it faces along, and its two faces on that axis are its
    front and its BACK -- the back being the wall plane. A side of the scene is wall-backed when
    some wall cabinet reaches that side's bounding face on the axis it faces along.

    Measured rather than tabulated per layout, for two reasons. The layouts do not agree (see
    the table below), and neither does one layout with itself once the author has dragged the
    furniture: the wizard lets a chair be put anywhere, and a rule that assumed "+y is always the
    back wall" would stand a wall in the middle of the room the first time somebody moved one.

    What the five layouts measure as, with no table and no chairs (seeds 0-3, identical on all
    four, and the wall cabinet backs sit EXACTLY on the bounding face -- 0.0 mm, not "within a
    tolerance", which is why 0.01 m is generous rather than tuned):

        island       +x, +y
        l_shaped     +x, +y
        peninsula    +x, +y, -y
        u_shaped     +x, +y, -y
        single_wall  +y

    Add default_table_transform's table and its chairs and every layout measures +x, +y alone --
    the table goes beyond the furniture's -y face, so on peninsula and u_shaped the -y counter
    run is no longer what the -y bounding face is made of, and the wall has to stand out past
    the chairs instead. That is a real property of where the table is put, not an artifact here:
    default_table_transform's docstring says -y "is the room side on every layout this generator
    builds", and on those two layouts it is not. Reported, not worked around.

    NOTE the direction test is deliberately absent. A wall cabinet is ~0.35 m deep and the
    counter beneath it is ~0.75 m, so the cabinet's FRONT can never be the outermost thing on
    its axis; only its back can touch a bounding face, and testing both faces needs no reading
    of which way the cabinet is turned (an earlier version compared against the scene CENTRE and
    got peninsula's -y run wrong the moment a table shifted that centre).

    Returns an empty set for a scene with no wall cabinets, which gives every side a walkway --
    the safe direction, and what the shell did for every side before this existed.
    """
    bounds = np.asarray(kitchen.scene.bounds, dtype=float)

    # Union the geometry per top-level object first. A wall cabinet is a group (body, door,
    # handle, shelves); the handle alone is shallow on a different axis than the cabinet is, so
    # measuring the argmin per GEOMETRY node picks the wrong facing axis about half the time.
    cabinets: dict[str, np.ndarray] = {}
    for node in kitchen.scene.graph.nodes_geometry:
        top = node.split("/")[0]
        if not _WALL_CABINET_RE.match(top):
            continue
        box = _world_bounds(kitchen, node)
        if box is None:
            continue
        if top in cabinets:
            cabinets[top] = np.array([np.minimum(cabinets[top][0], box[0]),
                                      np.maximum(cabinets[top][1], box[1])])
        else:
            cabinets[top] = box

    backed = set()
    for box in cabinets.values():
        axis = int(np.argmin(box[1][:2] - box[0][:2]))       # the axis the cabinet faces along
        name = "xy"[axis]
        if abs(box[1][axis] - bounds[1][axis]) <= tolerance:
            backed.add(name)
        if abs(box[0][axis] - bounds[0][axis]) <= tolerance:
            backed.add(f"-{name}")
    return frozenset(backed & set(ROOM_SHELL_DIMENSIONS))


def room_shell_offsets(kitchen, params) -> list[float]:
    """How far each of ROOM_SHELL_DIMENSIONS stands off the furniture, in add_walls' own order.

    A list, not a scalar, and that IS the change: add_walls takes `offset` as a per-dimension
    list of len(dimensions) and applies it face by face, so the wall behind the cabinets can sit
    against them while the wall across the walkway does not.

    Every entry is strictly positive, which is what makes containment unconditional: add_walls
    puts a wall's inner face at (bounding face + offset) on its own axis, so a positive offset
    cannot cut into anything that bounding box holds -- whichever table, whichever chairs, how
    many, and wherever they were dragged to.

    THE BOUNDING BOX add_walls MEASURES IS NOT THE FURNITURE'S. Scene.get_bounds()
    (scene_synthesizer/scene.py:449) puts each geometry's LOCAL AABB through the node transform
    as TWO corner points and min/maxes those. For an axis-aligned node that is exact. For a
    ROTATED one -- and every chair in a seat ring is rotated -- it is an approximation that errs
    in EITHER direction, because the two points it carries are not the box's other six. Measured
    on `island` with dining_long and four chairs (the numbers are the same on all five layouts,
    which is expected: the same chairs at the same yaws are what is rotated):

        face   get_bounds     scene.bounds    get_bounds reads
        +x       2.5660         2.5660         exact  (the counter run, axis-aligned)
        +y       0.3635         0.3635         exact
        -x      -1.8873        -1.9757         88.4 mm SHORT of the furniture
        -y      -6.1192        -5.9884        130.9 mm PAST the furniture

    So the number drawn is not the clearance you get: on -x it bought 88 mm less than it said.
    Inherited, not introduced -- the single 0.2-1.2 m offset was wrong by the same amounts -- but
    it is what stops "the wall stands off by X" from being checkable, and 88 mm is a large enough
    error that a small flush offset could not be trusted without correcting it.

    Hence the per-face correction: the gap is measured against trimesh's own scene.bounds (the
    tight AABB over the real vertices, and the box test_the_shell_encloses_the_kitchen has always
    used), and the offset handed to add_walls is bent by whatever get_bounds is going to be
    wrong by. The inner face then lands at exactly (scene.bounds face + the offset drawn), which
    is a contract a test can pin to 1e-9. A correction may be negative, and on -y it is; a
    negative entry in a per-dimension list is fine (a scalar one is not -- see
    _add_walls_with_one_window).
    """
    backed = wall_backed_sides(kitchen)
    approx = np.asarray(kitchen.get_bounds(), dtype=float)      # what add_walls will measure
    exact = np.asarray(kitchen.scene.bounds, dtype=float)       # what the gap is measured from

    offsets = []
    for dim in ROOM_SHELL_DIMENSIONS:
        axis = 1 if dim.endswith("y") else 0
        wanted = params["flush_offset"] if dim in backed else params["walkway_offset"]
        skew = (approx[0][axis] - exact[0][axis]) if dim.startswith("-") \
            else (exact[1][axis] - approx[1][axis])
        offsets.append(float(wanted + skew))
    return offsets


def add_room_shell(kitchen, *, seed: int) -> list[str]:
    """Put four walls around a built kitchen. Mutates `kitchen`; returns the wall node names.

    Called from commit_kitchen, never from build_kitchen — see ROOM_SHELL's comment for why
    the preview must stay open-topped.

    Everything random here is drawn from a Generator seeded by the caller's kitchen number
    (see room_shell_params), so regenerating kitchen 42 twice gives the same room, down to
    which wall its window is in. Nothing reads the global `random` state.

    The walls are visual only (use_collision_geometry=False): the room exists to fill the head
    camera's 200 degree fisheye view, not to constrain the arm, and a collidable box around a
    manipulator is a trap waiting to happen.

    The four walls do NOT stand off by the same amount. A side with a counter run backing onto
    it gets ROOM_SHELL_FLUSH_M, a side with nothing against it gets ROOM_SHELL_WALKWAY_M, and
    which is which is measured off this very scene by wall_backed_sides -- see
    room_shell_offsets.

    AT MOST ONE wall carries an opening, which is why a windowed room takes two add_walls
    calls -- see _add_walls_with_one_window for what that costs and how it is paid for.

    Returns [] when ROOM_SHELL is False, having changed nothing.
    """
    if not ROOM_SHELL:
        return []

    params = room_shell_params(seed)
    offsets = room_shell_offsets(kitchen, params)

    if params["holed_dim"] is None:
        # No window: one call, and the only call. Two thirds of kitchens take this path.
        kitchen.add_walls(
            dimensions=list(ROOM_SHELL_DIMENSIONS),
            thickness=params["thickness"],
            offset=offsets,
            overhang=params["overhang"],
            hole_extents=params["hole_extents"],
            hole_offset=params["hole_offset"],
            use_collision_geometry=False,
        )
    else:
        _add_walls_with_one_window(kitchen, params, offsets)

    # add_walls names its objects f"wall_{dim}" (scene_synthesizer/scene.py). That naming is not
    # incidental: GEOMETRY2MATERIAL's wall entry matches exactly it, which is how apply_materials
    # binds one of the 30 MATERIALS['wall'] finishes without any new code. Returning the names
    # rather than recomputing them elsewhere keeps that join in one place.
    return [f"wall_{dim}" for dim in ROOM_SHELL_DIMENSIONS]


def fit_hole_to_wall(hole_extents, hole_offset, span: float, height: float,
                     margin: float = ROOM_SHELL_HOLE_MARGIN_M):
    """A window shrunk and re-centred until it fits the wall it is punched in.

    Returns (hole_extents, hole_offset), unchanged when the drawn window already fits.

    `span` is the wall's in-plane extent and `height` its vertical one, both as add_walls will
    build them -- the scene extents on those axes plus the overhang at each end. The two
    conditions are the ones BoxWithHoleAsset's four frame boxes need to have a positive extent
    (assets.py:2113-2117), with ROOM_SHELL_HOLE_MARGIN_M of frame to spare:

        hole width  + 2*|offset across| <= span   - 2*margin
        hole height + 2*|offset up|     <= height - 2*margin

    This exists because the rooms got smaller. room_shell_params draws a window up to 1.6 m
    wide, offset up to 0.5 m across, from a range calibrated when every wall stood 0.2-1.2 m off
    the furniture on all four sides. A single_wall kitchen with no table is 0.77 m deep, so its
    end walls are now about 1.4 m across, and a 1.6 m window in a 1.4 m wall is not a window --
    it is the crash above, which seed 27 hits on `single_wall` with the -x wall holed.

    Shrink rather than re-draw: the seed decided how big a window this kitchen wants, and a
    kitchen whose wall cannot hold it should get the biggest one that wall can hold, in the same
    place. Re-drawing would make the window depend on the room and the room depend on the
    window. Both extents shrink independently; the offset is then clipped to whatever frame is
    left, so a window that had to be shrunk ends up centred rather than jammed against a corner.
    """
    width, tall = (float(v) for v in hole_extents)
    across, up = (float(v) for v in hole_offset)

    room_across = max(0.0, float(span) - 2.0 * margin)
    room_up = max(0.0, float(height) - 2.0 * margin)
    width = min(width, room_across)
    tall = min(tall, room_up)

    across = float(np.clip(across, -(room_across - width) / 2.0, (room_across - width) / 2.0))
    up = float(np.clip(up, -(room_up - tall) / 2.0, (room_up - tall) / 2.0))
    return (width, tall), (across, up)


def _add_walls_with_one_window(kitchen, params, offsets) -> None:
    """One holed wall plus three solid ones, standing where a single call would have put them.

    add_walls reads `scene_bounds = self.get_bounds()` at CALL time, so a second call measures a
    scene that ALREADY CONTAINS the first call's walls. Left uncorrected, the second call's wall
    is pushed out by its own side's offset plus the overhang -- measured on an island kitchen
    with offset 0.8370, thickness 0.1528, overhang 0.2918: a room roughly a metre too big.

    The correction: capture the bounds BEFORE the first call, measure how far they grew, and
    hand the second call a PER-DIMENSION offset list that gives that growth back. add_walls
    accepts `offset` as a list of len(dimensions), and its own arithmetic then lands the wall
    exactly where a single call would have -- proven to 1e-9 on all four choices of holed wall
    by test_kitchen_room_shell.py's test_the_two_call_shell_stands_where_a_single_call_would,
    parametrized one seed per dimension. The compensated value is routinely negative -- that is
    the point; it comes out at exactly -overhang.

    ALL FOUR WALLS GO UP SOLID FIRST, in ONE call with the per-side offsets, and the holed one
    is then removed and rebuilt with its opening. That ordering is the change per-side offsets
    forced, and the reason is the wall's LENGTH rather than its position. add_walls sizes a wall
    across its perpendicular axis from the scene extents PLUS the offsets in this call's own
    list, and a list only carries the dimensions being added: called first and alone, the holed
    wall would be sized to the bare furniture, stopping short of the perpendicular walls' inner
    faces by their offsets -- up to 0.65 m of open slot at each corner, looking out onto
    nothing. Second, it measures a scene that already spans those walls, so it reaches past them
    with no arithmetic at all. (The earlier order also passed a SCALAR offset, which sized the
    wall symmetrically; that was exactly right while all four sides shared one offset and cannot
    be right once they do not.)

    Two residuals, both measured and both deliberate -- the same two the previous ordering had,
    now borne by the holed wall instead of the three solid ones:

      * the holed wall extends past the corner along its perpendicular axis, by at most
        max(thickness, overhang), because the scene it measures includes the other walls' outer
        faces. It meets past the corner instead of stopping short of it, which is invisible from
        inside the room and strictly better than a gap.
      * it is `overhang` taller at top and bottom, for the same reason on the z axis, which
        add_walls gives no per-dimension control over (z is not one of the dimensions being
        added, so a z offset cannot be passed). A visual shell that fills more of a 200 degree
        fisheye is not a defect.

    The three solid walls are now EXACT -- they come out of the single call that a windowless
    kitchen makes, so there is nothing left for them to drift by.

    On the old ordering's stated reason for going holed-first ("add_walls with offset=0.0 and a
    non-zero hole raises 'primitive faces are immutable'"): measured against scene_synthesizer
    1.15, that is true only of the SCALAR offset form, which shrinks the wall's own z extent to
    a degenerate box. offset=0.0 scalar warns but survives; offset=-0.3 scalar raises;
    offset=[0.0] and offset=[-0.3] -- the per-dimension list form used here -- both build a
    perfectly good holed wall, because a length-1 list touches only that wall's own axis and
    leaves z alone. The compensated offset below is a list, and always is.
    """
    dim_to_axis = {"x": 0, "-x": 0, "y": 1, "-y": 1, "z": 2, "-z": 2}
    holed = params["holed_dim"]
    axis = dim_to_axis[holed]

    before = np.asarray(kitchen.get_bounds(), dtype=float)

    kitchen.add_walls(
        dimensions=list(ROOM_SHELL_DIMENSIONS),
        thickness=params["thickness"],
        offset=offsets,
        overhang=params["overhang"],
        hole_extents=(0.0, 0.0),
        hole_offset=(0.0, 0.0),
        use_collision_geometry=False,
    )

    # An exact object id, never a regex: scene_synthesizer treats a query with no special
    # characters as a literal key (utils.is_regex allows '-' and '_'), so "wall_-y" removes that
    # wall and only that wall.
    kitchen.remove_object(f"wall_{holed}")

    after = np.asarray(kitchen.get_bounds(), dtype=float)

    # How much did the other three walls grow the scene on the holed wall's own axis? Subtracting
    # it from that side's offset cancels it out of add_walls' arithmetic.
    grew = before[0][axis] - after[0][axis] if holed.startswith("-") \
        else after[1][axis] - before[1][axis]
    compensated = float(offsets[ROOM_SHELL_DIMENSIONS.index(holed)] - grew)

    # The wall this window is going into, sized exactly as add_walls is about to size it: the
    # scene extents on the two axes it spans, plus the overhang at each end. A length-1 offset
    # list contributes to the holed wall's OWN axis only, so neither of these is touched by
    # `compensated`. See fit_hole_to_wall for why a window can now be too big for its wall.
    extents = np.asarray(kitchen.get_extents(), dtype=float)
    hole_extents, hole_offset = fit_hole_to_wall(
        params["hole_extents"], params["hole_offset"],
        span=extents[1 - axis] + 2.0 * params["overhang"],
        height=extents[2] + 2.0 * params["overhang"],
    )

    kitchen.add_walls(
        dimensions=[holed],
        thickness=params["thickness"],
        offset=[compensated],
        overhang=params["overhang"],
        hole_extents=hole_extents,
        hole_offset=hole_offset,
        use_collision_geometry=False,
    )


def room_shell_prim_names() -> frozenset[str]:
    """Every name a shell wall can go by, for consumers that read the EXPORTED USD.

    add_room_shell returns the scene-graph names ("wall_-x"); USD forbids '-' in a prim name,
    so the same wall lands on the stage as "wall__x". Both spellings are in here because the
    pair IS the join between the exporter and everything downstream that has to tell the room
    apart from the furniture, and a consumer that guessed only one of them would silently miss
    half the shell.

    Never a "wall_" prefix test. wall_cabinet, wall_cabinet_0 .. wall_cabinet_3 are wall-MOUNTED
    cabinets -- real furniture, with real rigid bodies and a real footprint on the floor plan.

    Deliberately NOT gated on ROOM_SHELL. This answers "which names belong to a shell", a fact
    about naming; whether a given kitchen HAS one is recorded per-kitchen in kitchen_data
    ["room_shell"], because a USD exported while the flag was on keeps its walls after it goes
    off, and the emitter reads the USD, not the flag.
    """
    names = {f"wall_{dim}" for dim in ROOM_SHELL_DIMENSIONS}
    return frozenset(names | {name.replace("-", "_") for name in names})


def apply_placements(kitchen, placements):
    """Replay transforms the user dragged in the preview back into the scene graph.

    ``placements`` is {node_id: row-major 4x4 world-frame matrix}, as posted by the
    browser preview. The object's *world* pose is what must round-trip into the exported
    USD — but a placed object is not parented to "world", it is parented to its support
    geometry (e.g. "countertop_base_cabinet/geometry_0"), and that parent edge carries the
    metadata the USD exporter's joint gate uses to recognize a free-floating rigid body
    (a "floating" joint). Re-parenting the node under "world" would create a fresh edge
    with no metadata: the position would still be correct, but the exporter would never
    apply RigidBodyAPI/MassAPI and the object would silently ship as a static prop.

    So: keep the object on its existing parent edge, carry that edge's data (the joint
    metadata) forward unchanged, and re-express the dragged world-frame matrix in the
    parent's frame before writing it back.
    """
    graph = kitchen.scene.graph
    for node_id, matrix in placements.items():
        if node_id not in graph.nodes:
            print(f"[commit] ignoring placement for unknown node {node_id!r}")
            continue

        world_matrix = np.asarray(matrix, dtype=float)
        parent = graph.transforms.parents.get(node_id)
        if parent is None:
            # Already a root (no support edge to preserve): world frame is the only frame.
            graph.update(frame_to=node_id, frame_from="world", matrix=world_matrix)
            continue

        edge = graph.transforms.edge_data[(parent, node_id)]
        preserved = {k: v for k, v in edge.items() if k != "matrix"}
        parent_world, _ = graph.get(parent)
        local = np.linalg.inv(parent_world) @ world_matrix
        graph.update(frame_to=node_id, frame_from=parent, matrix=local, **preserved)


def collect_joints(kitchen):
    """The articulated joints of the kitchen furniture, for the preview's joint sliders.

    scene_synthesizer hangs joint data off the scene-graph edge that connects a moving part
    to its body: type, the axis and origin frame, and the limits. At q=0 the edge's local
    matrix IS the joint origin (verified), so a viewer can drive the part by setting

        child.matrix = origin @ T(q)

    with T a rotation about `axis` (revolute) or a translation along it (prismatic).

    Only revolute and prismatic joints are returned — 'fixed' cannot move, and 'floating' is
    what holds a placed object on its support (see apply_placements).

    Returns [{name, type, child, group, axis, origin, lower, upper}], where `origin` is a
    row-major 4x4 and `group` is the furniture the part belongs to (e.g. 'refrigerator'), so
    the preview can group the sliders instead of listing 22 of them flat.
    """
    joints = []
    for (_parent, child), edge in kitchen.scene.graph.transforms.edge_data.items():
        joint = (edge.get("metadata") or {}).get("joint")
        if not joint or joint.get("type") not in ("revolute", "prismatic"):
            continue

        lower = float(joint.get("limit_lower", 0.0))
        upper = float(joint.get("limit_upper", 0.0))
        if upper <= lower:                       # nothing to drive
            continue

        joints.append(
            {
                "name": joint.get("name", child),
                "type": joint["type"],
                "child": child,
                "group": child.split("/")[0],
                "axis": [float(a) for a in joint["axis"]],
                "origin": np.asarray(joint["origin"], dtype=float).tolist(),
                "lower": lower,
                "upper": upper,
            }
        )
    return joints


def pick_materials():
    """Roll one material per group. Returns {group: (mdl_url, mtl_name, texture_scale)}."""
    return {group: random.choice(choices) for group, choices in MATERIALS.items() if choices}


def material_display_names(picks):
    """{group: human-readable name} for the preview panel."""
    return {group: (name or Path(url).stem) for group, (url, name, _scale) in picks.items()}
