"""
Combined kitchen pipeline:

1. Step1 GUI: build a kitchen and place objects
2. Step2: for the same kitchen number, add BODex metadata, create 12 rotated USDs, and register gym environments
   -> uses a Tkinter dialog to choose which prim to rotate.
3. Step3: run fixed-joint cleanup on all rotated USDs for that kitchen
4. Step4: make wall_cabinet* static-only by stripping physics/collision APIs from wall_cabinet prims only
"""

import tkinter as tk
from tkinter import ttk, messagebox, Listbox
import random
import itertools
import json
import os
import glob
import torch

from scene_synthesizer import procedural_scenes as ps
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer.usd_import import get_scene_paths
from scene_synthesizer.exchange.usd_export import add_mdl_material, bind_material_to_prims

#: See goal_generator._safe_texture_scale for the full account. scene_synthesizer's
#: add_mdl_material always writes texture_scale as Float2; Paint_Eggshell.mdl declares it a float,
#: so the binding is refused and every prim using it -- which is EVERY wall variant -- renders
#: black. This is the path build_furnished_mug_kitchen actually takes (ksg.commit_kitchen), so the
#: fix has to exist here too, not only in goal_generator.
_FLOAT_TEXTURE_SCALE_MDLS = ("Paint_Eggshell.mdl",)


def _safe_texture_scale(mtl_url, texture_scale):
    return None if any(m in mtl_url for m in _FLOAT_TEXTURE_SCALE_MDLS) else texture_scale

from scene_synthesizer import utils
from scene_synthesizer import datasets
import scene_synthesizer as synth


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


# ==================== COLLISION APPROX (convexDecomposition) ====================
# If True, set UsdPhysics.MeshCollisionAPI.approximation="convexDecomposition" for object meshes
# so that the exported rotated kitchens inherit the same collision approximation.
CONVEX_DECOMP_ENABLE = True
# Optional: also apply to any prim path/name containing these substrings (e.g., kitchen fixture meshes)
CONVEX_DECOMP_EXTRA_SUBSTRINGS = tuple()  # e.g., ("sink_cabinet", "dishwasher")
CONVEX_DECOMP_VERBOSE = False


def ensure_convexDecomposition_approx_on_prim(prim) -> bool:
    """Ensure MeshCollisionAPI.approximation == 'convexDecomposition' on a prim (Mesh only).

    Returns True if changed, False otherwise.
    """
    # Only meaningful on Mesh prims
    try:
        if not prim.IsA(UsdGeom.Mesh):
            return False
    except Exception:
        return False

    # Apply MeshCollisionAPI if missing
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        UsdPhysics.MeshCollisionAPI.Apply(prim)

    mesh_api = UsdPhysics.MeshCollisionAPI(prim)
    if not mesh_api:
        return False

    approx_attr = mesh_api.GetApproximationAttr()
    if not approx_attr:
        approx_attr = mesh_api.CreateApproximationAttr()

    current = approx_attr.Get()
    if current != UsdPhysics.Tokens.convexDecomposition:
        if CONVEX_DECOMP_VERBOSE:
            print(f"[convexDecomposition][UPDATE] {prim.GetPath()}: {current!r} -> 'convexDecomposition'")
        approx_attr.Set(UsdPhysics.Tokens.convexDecomposition)
        return True
    else:
        if CONVEX_DECOMP_VERBOSE:
            print(f"[convexDecomposition][OK] {prim.GetPath()}: already 'convexDecomposition'")
        return False


def apply_convex_decomposition_approx(
    stage,
    root_paths=None,
    extra_match_substrings=(),
) -> tuple[int, int]:
    """Apply convexDecomposition approximation under given root paths, plus optional substring matches.

    Returns: (changed_count, checked_mesh_count)
    """
    changed = 0
    checked = 0
    import ipdb
    ipdb.set_trace()
    # (A) Under explicit roots (fast + precise)
    if root_paths:
        for root in root_paths:
            root_prim = stage.GetPrimAtPath(root)
            if not root_prim or not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                # PrimRange includes root and all descendants
                try:
                    if prim.IsA(UsdGeom.Mesh):
                        checked += 1
                        if ensure_convexDecomposition_approx_on_prim(prim):
                            changed += 1
                except Exception:
                    continue

    # (B) Optional: also apply by substring match anywhere on stage
    if extra_match_substrings:
        extra_match_substrings = tuple(s.lower() for s in extra_match_substrings)
        for prim in stage.Traverse():
            try:
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                name = (prim.GetName() or "").lower()
                path_str = str(prim.GetPath()).lower()
                if any(s in name or s in path_str for s in extra_match_substrings):
                    checked += 1
                    if ensure_convexDecomposition_approx_on_prim(prim):
                        changed += 1
            except Exception:
                continue

    return changed, checked

# ==================== MATERIAL SETUP ====================
DEFAULT_TEXTURE_SCALE = 0.25
url_mdl_material = 'http://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/'

materials = {
    'sink': [
        ('vMaterials_2/Metal/Aluminum_Brushed.mdl', None, DEFAULT_TEXTURE_SCALE)
#        ('Base/Stone/Ceramic_Smooth_Fired.mdl', None, DEFAULT_TEXTURE_SCALE)
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

geometry2material = {
    "(.*cabinet.*corpus.*|.*cabinet.*door|.*drawer.*board.*|.*cabinet.*closed.*|/world/kitchen_island/.*)|/world/corner.*": "cabinet",
    "(.*refrigerator.*|.*range_hood.*|.*range.*|.*dishwasher.*)": "appliances",
    "/world/range/corpus/heater.*": 'rusted metal',
    "/world/range/corpus/top": 'glossy black',
    ".*/corpus/sink": 'sink',
    ".*countertop.*": "countertop",
    "/world/range/.*window": 'tinted glass',
    "(.*glass|.*_window)": 'glass',
    "/world/plate.*": 'sink',
    "(/world/wall/geometry.*|/world/wall_(x|y|_y|_x)/geometry_0)": 'wall',
    "/world/floor/geometry.*": 'floor',
    ".*dishwasher.*basket": 'white plastic',
    ".*handle.*": "handle",
}


# ====================== STEP 1: GUI APP ======================
class KitchenBuilderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kitchen Scene Generator")
        self.geometry("600x480")

        self.objects_to_place = []
        self.obj_dict = {"bottle": 0, "bowl": 0, "apple": 0, "sodacan": 0, "nutella": 0, "mug": 0}

        self.kitchen_options = {
            0: (ps.kitchen_island, "island"),
            1: (ps.kitchen_l_shaped, "l_shaped"),
            2: (ps.kitchen_peninsula, "peninsula"),
            3: (ps.kitchen_u_shaped, "u_shaped"),
            4: (ps.kitchen_single_wall, "single_wall"),
        }
        self.kitchen_name = None

        self.generated = False
        self.generated_kitchen_num = None

        try:
            self.bodex_data = datasets.load_dataset("BODex")
            self.mesh_files = self.bodex_data.get_filenames()
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Could not load BODex dataset. Make sure it's installed correctly.\n\n{e}",
            )
            self.destroy()
            return

        self.create_widgets()
        self.select_new_kitchen()

    def select_new_kitchen(self):
        kitchen_type_key = random.choice(list(self.kitchen_options.keys()))
        _, self.kitchen_name = self.kitchen_options[kitchen_type_key]
        self.kitchen_type_var.set(
            f"Current Kitchen: {self.kitchen_name.replace('_', ' ').capitalize()}"
        )
        self.objects_to_place.clear()
        self.objects_listbox.delete(0, tk.END)
        self.obj_dict = {key: 0 for key in self.obj_dict}

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.LabelFrame(main_frame, text="1. Kitchen Setup", padding="10")
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Label(top_frame, text="Kitchen Number:").grid(row=0, column=0, padx=5, sticky="w")
        self.kitchen_num_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.kitchen_num_var, width=10).grid(
            row=0, column=1, sticky="w"
        )

        self.kitchen_type_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.kitchen_type_var, font=("Helvetica", 10, "bold")).grid(
            row=1, column=0, pady=(10, 0), sticky="w"
        )
        ttk.Button(
            top_frame, text="Change Kitchen Type", command=self.select_new_kitchen
        ).grid(row=1, column=1, pady=(10, 0), sticky="w")

        middle_frame = ttk.LabelFrame(main_frame, text="2. Object Placement", padding="10")
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        controls_frame = ttk.Frame(middle_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(controls_frame, text="Object Type:").pack(anchor=tk.W, pady=(0, 2))
        self.object_var = tk.StringVar()
        object_choices = ["bottle", "bowl", "apple", "sodacan", "nutella", "mug"]
        object_menu = ttk.Combobox(
            controls_frame,
            textvariable=self.object_var,
            values=object_choices,
            state="readonly",
            width=25,
        )
        object_menu.pack(fill=tk.X, pady=(0, 10))
        object_menu.set(object_choices[0])

        ttk.Label(controls_frame, text="Placement Location:").pack(anchor=tk.W, pady=(0, 2))
        self.location_var = tk.StringVar()
        location_choices = [
            "1. Above cabinet",
            "2. Above dishwasher",
            "3. In refrigerator",
            "4. Above Island (if available)",
        ]
        location_menu = ttk.Combobox(
            controls_frame,
            textvariable=self.location_var,
            values=location_choices,
            state="readonly",
            width=25,
        )
        location_menu.pack(fill=tk.X, pady=(0, 15))
        location_menu.set(location_choices[0])

        ttk.Button(
            controls_frame, text="Add Object to List", command=self.add_object_to_list
        ).pack(fill=tk.X)

        listbox_frame = ttk.Frame(middle_frame)
        listbox_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(listbox_frame, text="Objects to be Added:").pack(anchor=tk.W)
        self.objects_listbox = Listbox(listbox_frame, height=10)
        self.objects_listbox.pack(fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        generate_button = ttk.Button(
            bottom_frame,
            text="Generate Kitchen Scene",
            command=self.run_generation_process,
            style="Accent.TButton",
        )
        generate_button.pack(pady=5)

        style = ttk.Style()
        style.configure("Accent.TButton", font=("Helvetica", 12, "bold"))

    def add_object_to_list(self):
        obj = self.object_var.get()
        loc_str = self.location_var.get()
        loc_num = int(loc_str.split(".")[0])

        if loc_num == 4 and self.kitchen_name != "island":
            messagebox.showwarning(
                "Invalid Location",
                "The 'Above Island' location is only available for the 'Island' kitchen type.",
            )
            return

        if not obj or not loc_str:
            messagebox.showwarning("Warning", "Please select both an object and a location.")
            return

        obj_list = [elmt for elmt in self.mesh_files if obj in elmt.lower()]
        if not obj_list:
            messagebox.showerror("Error", f"No mesh files found for object type: {obj}")
            return
        fname = random.choice(obj_list)

        obj_n = f"{obj}{self.obj_dict[obj]}"
        self.obj_dict[obj] += 1

        self.objects_to_place.append(
            {"obj_n": obj_n, "type": obj, "fname": fname, "loc": loc_num}
        )
        self.objects_listbox.insert(tk.END, f"{obj_n} -> placing on '{loc_str}'")

    def run_generation_process(self):
        kitchen_num_str = self.kitchen_num_var.get()
        if not kitchen_num_str.isdigit():
            messagebox.showerror("Error", "Please enter a valid integer for the kitchen number.")
            return

        kitchen_num = int(kitchen_num_str)

        if not self.objects_to_place:
            messagebox.showwarning(
                "Warning", "Please add at least one object before generating the scene."
            )
            return

        try:
            usd_filename = (
                f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num:02d}.usd"
            )
            seed = None
            random.seed(seed)
            kitchen_data = {}

            kitchen_name = self.kitchen_name
            kitchen_data["kitchen_type"] = kitchen_name

            kitchen_func = None
            for func, name in self.kitchen_options.values():
                if name == kitchen_name:
                    kitchen_func = func
                    break
            if kitchen_func is None:
                raise ValueError("Could not find the kitchen generation function.")

            kitchen = kitchen_func(seed=seed, counter_height=0.95)
            kitchen.unwrap_geometries(
                "(sink_cabinet/sink_countertop|countertop_.*|.*countertop)"
            )

            if kitchen_name == "island":
                kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
                kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
                kitchen.label_support(label="island", geom_ids="kitchen_island/countertop")
            elif kitchen_name == "l_shaped":
                kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
                kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
            elif kitchen_name == "peninsula":
                kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
#                kitchen.label_support(label="base_cabinet_0", geom_ids="countertop_base_cabinet_0")
#                kitchen.label_support(label="base_cabinet_1", geom_ids="countertop_base_cabinet_1")
                kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
            elif kitchen_name == "u_shaped":
                kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
#                kitchen.label_support(label="base_cabinet_0", geom_ids="countertop_base_cabinet_0")
                kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")
            elif kitchen_name == "single_wall":
                kitchen.label_support(label="base_cabinet", geom_ids="countertop_base_cabinet")
                kitchen.label_support(label="dishwasher", geom_ids="countertop_dishwasher")

            kitchen.label_support(label="refrigerator_1st", geom_ids="refrigerator/shelf_3")
            kitchen.label_support(label="refrigerator_2nd", geom_ids="refrigerator/shelf_2")
            kitchen.label_support(label="refrigerator_3nd", geom_ids="refrigerator/shelf_1")
            kitchen.label_support(label="refrigerator_4nd", geom_ids="refrigerator/shelf_0")

            for obj_info in self.objects_to_place:
                obj_n = obj_info["obj_n"]
                obj_type = obj_info["type"]
                fname = obj_info["fname"]
                loc = obj_info["loc"]

                kitchen_data[obj_n] = fname
                if obj_type in ["apple", "sodacan"]:
                    up, front, origin = (0, 0, 1), (0, 1, 0), ("com", "com", "bottom")
                elif obj_type in ["bottle", "bowl", "mug", "nutella"]:
                    up, front, origin = (0, 1, 0), (0, 0, -1), ("com", "bottom", "com")
                else:
                    up, front, origin = (0, 1, 0), (0, 0, -1), ("com", "bottom", "com")

                place = ""
                if loc == 1:
                    if kitchen_name in ["island", "l_shaped", "single_wall"]:
                        place = "base_cabinet"
                    elif kitchen_name == "peninsula":
                        place = "base_cabinet"
                    elif kitchen_name == "u_shaped":
                        place = "base_cabinet"
                elif loc == 2:
                    place = "dishwasher"
                elif loc == 3:
                    place = random.choice(["refrigerator_1st", "refrigerator_2nd"])
                elif loc == 4:
                    if kitchen_name == "island":
                        place = "island"
                    else:
                        continue
                if not place:
                    continue

                kitchen.place_objects(
                    obj_id_iterator=utils.object_id_generator(obj_n),
                    obj_asset_iterator=synth.assets.asset_generator(
                        itertools.repeat(fname, 1),
                        scale=0.1,
                        up=up,
                        front=front,
                        origin=origin,
                        align=True,
                    ),
                    obj_support_id_iterator=kitchen.support_generator(support_ids=place),
                    obj_position_iterator=utils.PositionIteratorGrid(
                        step_x=0.06,
                        step_y=0.06,
                        noise_std_x=0.04,
                        noise_std_y=0.04,
                    ),
                    obj_orientation_iterator=utils.orientation_generator_uniform_around_z(
                        lower=0.0, upper=0.0
                    ),
                )

            stage = kitchen.export(file_type="usd")

            for geom_regex, material_group in geometry2material.items():
                paths = get_scene_paths(
                    stage=stage,
                    prim_types=["Mesh", "Capsule", "Cube", "Cylinder", "Sphere"],
                    scene_path_regex=geom_regex,
                )
                if len(materials[material_group]) == 0:
                    print(f"Warning: No materials for {material_group}")
                    continue
                mtl_url, mtl_name, texture_scale = random.choice(materials[material_group])
                texture_scale = _safe_texture_scale(mtl_url, texture_scale)
                mtl = add_mdl_material(
                    stage=stage,
                    mtl_url=url_mdl_material + mtl_url,
                    mtl_name=mtl_name,
                    texture_scale=texture_scale,
                )
                bind_material_to_prims(stage=stage, material=mtl, prim_paths=paths)

            stage.Export(usd_filename)
            bodex_dir = os.path.expanduser(
                f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen/bodex"
            )
            os.makedirs(bodex_dir, exist_ok=True)
            json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num:02d}.json")
            with open(json_path, "w") as f:
                json.dump(kitchen_data, f, indent=4)

            self.generated = True
            self.generated_kitchen_num = kitchen_num

            messagebox.showinfo(
                "Success!",
                f"Scene generation complete!\n\n"
                f"Kitchen Type: {kitchen_name.replace('_', ' ').capitalize()}\n\n"
                f"USD saved to: {usd_filename}\n"
                f"JSON saved to: {json_path}",
            )

        except Exception as e:
            messagebox.showerror(
                "An Error Occurred", f"Failed during scene generation:\n\n{e}"
            )


# ================= SMALL DIALOG TO CHOOSE PRIM =================
def choose_prim_dialog(prim_paths):
    if not prim_paths:
        return None

    root = tk.Tk()
    root.title("Choose prim to rotate")

    ttk.Label(root, text="Select the prim to rotate:").pack(padx=10, pady=5)

    listbox = tk.Listbox(root, height=min(10, len(prim_paths)), width=60)
    for p in prim_paths:
        listbox.insert(tk.END, p)
    listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    result = {"value": None}

    def on_ok(event=None):
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Please select a prim to rotate.")
            return
        result["value"] = prim_paths[selection[0]]
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=10)

    ok_btn = ttk.Button(btn_frame, text="OK", command=on_ok)
    ok_btn.pack(side=tk.LEFT, padx=5)
    cancel_btn = ttk.Button(btn_frame, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.LEFT, padx=5)

    listbox.bind("<Double-Button-1>", on_ok)

    root.mainloop()
    return result["value"]


# =============== STEP 2: rotate & register (with dialog) ===============
def rotate_and_register_envs(kitchen_num: int):
    bodex_dir = os.path.expanduser(f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen/bodex")
    json_path = os.path.join(bodex_dir, f"kitchen_data_{kitchen_num:02d}.json")

    if not os.path.exists(json_path):
        print(f"[Step2] JSON not found: {json_path}")
        return

    with open(json_path, "r") as f:
        kitchen_data = json.load(f)

    base_usd = (
        f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num:02d}.usd"
    )

    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(base_usd)
    init_file = f"{SIMVLA_REPO_ROOT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/kitchen/__init__.py"

    if not success:
        print(f"[Step2] Failed to open base USD: {base_usd}")
        return

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

    if not candidate_prims:
        print("[Step2] No candidate prims found to rotate.")
        return


# ---- (Optional) Force convexDecomposition collision approximation on placed object meshes ----
# This runs once on the base stage; all 12 exported rotated USDs inherit it.
    if CONVEX_DECOMP_ENABLE:
        changed, checked = apply_convex_decomposition_approx(
            stage,
            root_paths=candidate_prims,
            extra_match_substrings=CONVEX_DECOMP_EXTRA_SUBSTRINGS,
        )
        print(f"[Step2] convexDecomposition applied: checked {checked} mesh prim(s), updated {changed}.")

        rotate_obj = choose_prim_dialog(candidate_prims)
        if rotate_obj is None:
            print("[Step2] Rotation cancelled by user.")
            return

        print(f"[Step2] Selected prim to rotate: {rotate_obj}")

        obj_prim = stage.GetPrimAtPath(rotate_obj)
        if not obj_prim:
            print(f"[Step2] Prim not found: {rotate_obj}")
            return

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

        for i in range(12):
            angle = i * 30.0
            rotate_op.Set(Gf.Vec3f(0.0, 0.0, angle))

            usd_out = f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen/kitchen_{kitchen_num:02d}_{i:02d}.usd"
            stage.GetRootLayer().Export(usd_out)
            print(f"[Step2] Saved rotated kitchen USD: {usd_out}")

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


# =============== HELPERS FOR WALL_CABINET STATICIZATION ===============
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


# =============== STEP 3+4: fixed joints + wall_cabinet cleanup ===============
def fix_missing_fixed_joint_targets_and_wall_cabinets(kitchen_num: int):
    folder_path = f"{SIMVLA_REPO_ROOT}/source/isaaclab_assets/data/Kitchen"
    pattern = f"{folder_path}/kitchen_{kitchen_num:02d}_*.usd"
    files = glob.glob(pattern)

    if not files:
        print(f"[Step3] No rotated USDs found for kitchen {kitchen_num:02d} at {pattern}")
        return

    for file in files:
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
        ensured_dynamic = ensured_collision = 0
        for prim in stage.Traverse():
            p = str(prim.GetPath())

            if is_dynamic_exception(p):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(False)
                    ensured_dynamic += 1
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    ensured_collision += 1
                if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    ensured_collision += 1

            elif is_collision_exception(p):
                need = False
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim); need = True
                if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    PhysxSchema.PhysxCollisionAPI.Apply(prim); need = True
                if need:
                    ensured_collision += 1

        print(f"[Step4] Ensured kinematicEnabled=False on {ensured_dynamic} dynamic-exception bodies in {file}.")
        print(f"[Step4] Ensured collision present on {ensured_collision} exception prim(s) in {file}.")

        omni.usd.get_context().save_as_stage(file)
        


# ======================= MAIN PIPELINE =======================
if __name__ == "__main__":
    app = KitchenBuilderApp()
    app.mainloop()

    if not getattr(app, "generated", False):
        print("[Main] No kitchen was generated (you may have closed the GUI without clicking 'Generate'). Exiting.")
    else:
        k_num = app.generated_kitchen_num
        print(f"[Main] Proceeding with kitchen number: {k_num:02d}")

        rotate_and_register_envs(k_num)
        fix_missing_fixed_joint_targets_and_wall_cabinets(k_num)

        print("[Main] Pipeline complete.")


