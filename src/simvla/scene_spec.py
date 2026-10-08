"""SimVLA: a task's scene — every placed object and its geometry, as pure data.

Stdlib only. The composer edits this; the generator samples it; neither needs the other's
heavy imports. Every geometry number is an Extent (a center that still jitters), so a task keeps
the per-kitchen variation that makes the dataset useful while still letting you pin things down.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


class SceneError(Exception):
    """A scene that cannot be placed. Raised loudly; never swallowed."""


#: Graspable types — they have BODex grasp data, so they grasp with the real graspdata logic.
#: (nutella is the exception; see NO_GRASPDATA below. It has always been in this set.)
GRASPABLE_TYPES = ("bottle", "bowl", "apple", "sodacan", "nutella", "mug")

#: Types assumed to lack BODex grasp data when no manifest is available. A FALLBACK only: the real
#: answer is per-mesh and lives in <BODEX_OBJ_DIR>/grasp_manifest.json, because a type's meshes need
#: not agree (toaster ships 4, of which 2 have grasps). Kept so a checkout with no dataset still
#: renders an honest menu.
NO_GRASPDATA = frozenset({"nutella"})

#: Placement-only clutter — meshes from the BODex bundle's processed_data. Historically these had no
#: grasp data and fell back to a bbox-center grasp; that is no longer true of this install's dataset
#: (see scripts/tools/build_grasp_manifest.py), so what grasps and what does not is now MEASURED per
#: mesh, and this table is only about placement. THE SINGLE SOURCE OF TRUTH for clutter: the type
#: processed_data category prefix its meshes come from, the base dims (mesh extents at grasp scale,
#: preserving real proportions), a default placement surface, and how many variants to link. Both the
#: registration below AND scripts/tools/expand_use_data.py read this, so adding a clutter object is a
#: single entry here — no hand-syncing three tables.
#:
#: Which meshes a type draws from is NOT this table's business any more -- see CATEGORIES. Type
#: names used to have to be unambiguous substrings of the use_data pool, and 'can' was not one: it
#: matched sem_Candle_* and sem_SodaCan_* as well, so a 'can' row could spawn a candle.
#: NOTE: clutter orientation is per mesh, not per type -- see mesh_orientation.MESH_UP, which
#: corrects 73 ShapeNetSem meshes that were being placed on their side.
CLUTTER = {
    "jar":             {"category": "core_jar",           "dims": (0.083, 0.165, 0.083), "placement": "base_cabinet", "link": 12},
    "plate":           {"category": "sem_Plate",          "dims": (0.142, 0.142, 0.020), "placement": "dishwasher",   "link": 2},
    "teapot":          {"category": "sem_Teapot",         "dims": (0.155, 0.101, 0.078), "placement": "base_cabinet", "link": 5},
    "cerealbox":       {"category": "sem_CerealBox",      "dims": (0.122, 0.035, 0.159), "placement": "base_cabinet", "link": 8},
    "milkcarton":      {"category": "sem_MilkCarton",     "dims": (0.066, 0.071, 0.179), "placement": "base_cabinet", "link": 8},
    "can":             {"category": "core_can",           "dims": (0.067, 0.179, 0.067), "placement": "base_cabinet", "link": 12},
    "fooditem":        {"category": "sem_FoodItem",       "dims": (0.133, 0.135, 0.067), "placement": "base_cabinet", "link": 12},
    "vase":            {"category": "sem_Vase",           "dims": (0.113, 0.113, 0.125), "placement": "base_cabinet", "link": 12},
    "tissuebox":       {"category": "sem_TissueBox",      "dims": (0.092, 0.162, 0.081), "placement": "base_cabinet", "link": 8},
    "toiletpaper":     {"category": "sem_ToiletPaper",    "dims": (0.095, 0.127, 0.127), "placement": "base_cabinet", "link": 8},
    "candle":          {"category": "sem_Candle",         "dims": (0.074, 0.074, 0.173), "placement": "base_cabinet", "link": 8},
    "soapbar":         {"category": "sem_SoapBar",        "dims": (0.159, 0.116, 0.050), "placement": "dishwasher",   "link": 5},
    "detergent":       {"category": "sem_Detergent",      "dims": (0.144, 0.064, 0.128), "placement": "base_cabinet", "link": 5},
    "ricecooker":      {"category": "sem_RiceCooker",     "dims": (0.112, 0.148, 0.082), "placement": "base_cabinet", "link": 4},
    "cap":             {"category": "sem_Cap",            "dims": (0.153, 0.106, 0.080), "placement": "base_cabinet", "link": 3},
    "kettle":          {"category": "sem_Kettle",         "dims": (0.112, 0.084, 0.146), "placement": "base_cabinet", "link": 1},
    "toaster":         {"category": "sem_Toaster",        "dims": (0.155, 0.094, 0.092), "placement": "base_cabinet", "link": 4},
    "blender":         {"category": "sem_Blender",        "dims": (0.098, 0.110, 0.139), "placement": "base_cabinet", "link": 2},
}

#: Every mesh-directory prefix a type can spawn. THE definition of what a type means, replacing a
#: substring match on the type name -- that match quietly gave `can` every `sem_Candle_*` and every
#: `sem_SodaCan_*` too, because "can" is inside both. Prefixes also let one type draw from several
#: source categories, which is how the merges below are expressed rather than by copying meshes.
#:
#: Merged on inspection of the placed scenes: sem_Fruit is an apple; sem_Cup and sem_DrinkingUtensil
#: are mugs; sem_Cookie is a food item. Their own types are gone -- one name per real thing.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "bottle":     ("core_bottle",),
    "bowl":       ("core_bowl",),
    "apple":      ("ddg_ycb_013_apple", "sem_Fruit"),
    "sodacan":    ("sem_SodaCan",),
    "nutella":    ("ddg_kit_NutellaGo",),
    "mug":        ("core_mug", "sem_DrinkingUtensil", "sem_Cup"),
    "jar":        ("core_jar",),
    "plate":      ("sem_Plate",),
    "teapot":     ("sem_Teapot",),
    "cerealbox":  ("sem_CerealBox",),
    "milkcarton": ("sem_MilkCarton",),
    "can":        ("core_can",),
    "fooditem":   ("sem_FoodItem", "sem_Cookie"),
    "vase":       ("sem_Vase",),
    "tissuebox":  ("sem_TissueBox",),
    "toiletpaper": ("sem_ToiletPaper",),
    "candle":     ("sem_Candle",),
    "soapbar":    ("sem_SoapBar",),
    "detergent":  ("sem_Detergent",),
    "ricecooker": ("sem_RiceCooker",),
    "cap":        ("sem_Cap",),
    "kettle":     ("sem_Kettle",),
    "toaster":    ("sem_Toaster", "sem_ToasterOven"),
    "blender":    ("sem_Blender",),
}

#: Meshes withdrawn on inspection: broken, duplicated, or not the thing their category claims.
#: Excluded from the type registry rather than deleted from use_data, so the decision is reversible
#: and their BODex grasp data survives in graspdata_final.
MESH_EXCLUDE = frozenset({
    "sem_Vase_3ca4bd70f299100acd6a36c601ac1ec2",
    "sem_Vase_418003a4a24e2c18265d1076b4b6c5c",
    "sem_ToiletPaper_44e96b3ccbfc0620bcf87cf50cf0a3e4",
    "sem_ToiletPaper_815e3a43c8a7ad634c5341ee07f41676",
    "sem_Detergent_266dd42d5c571ee65af983a27c524ad0",
    "sem_Detergent_2d9be843d39a545f5af983a27c524ad0",
    "sem_RiceCooker_5aa24fc0f25b61abb564bb0657e0d6",
    "sem_Cookie_6a4bcdda0de2810190a3f8bc630957a8",
    "sem_ToasterOven_3c357d1352d2d1811fddae104d1cd00e",
    # every pizza: flat discs that never sat right in a scene
    "sem_Pizza_99dadb83b9bcf2498af30108ea9ccb6c",
    "sem_Pizza_b3a8ebcf6e9ca5852721058c5c629b05",
    "sem_Pizza_df62f7b84fd210ba82cabcc4d06ff984",
})

#: Every placeable type: graspable + clutter. Derived, so the two sets can't drift from their sources.
OBJECT_TYPES = frozenset(GRASPABLE_TYPES) | set(CLUTTER)

# The five procedural kitchen layouts goal_generator.KITCHEN_FUNCS can build.
KITCHEN_TYPES = frozenset({"island", "l_shaped", "peninsula", "u_shaped", "single_wall"})


def kitchen_surfaces(kitchen_type: str, *, has_table: bool = False) -> "set[str]":
    """The support-surface labels a given kitchen layout provides, as pure data (no Omniverse).

    Derived from kitchen_build.PLACEMENTS so this and the generator's menu cannot drift; they
    were two independent tables before. Imported lazily because kitchen_build imports this
    module at import time.

    has_table forwards to available_locations()'s own has_table flag (default False, matching
    it): the table is not a property of the layout, it is something the user adds or does not,
    so a *particular* kitchen only offers 'table_top' when has_table=True is passed for it.
    Callers that describe a specific built (or about-to-build) kitchen -- goal_generator's
    _label_supports and its surface-availability check, neither of which ever adds a table --
    keep the default. A caller asking what 'table_top' could ever mean across every kitchen the
    generator can build (not one specific kitchen) should pass has_table=True instead.
    """
    from .placements import available_locations
    return {p.key for p in available_locations(kitchen_type, has_table=has_table)}


def support_surfaces() -> "frozenset[str]":
    """Every "placement" label goal_generator._label_supports could ever apply, across every
    kitchen type — the vocabulary validate_scene() and task_composer.py's surface dropdown
    accept.

    Derived from kitchen_build.PLACEMENT_BY_KEY (lazy import: kitchen_build imports this module
    at import time, same reason as kitchen_surfaces()). Was a hand-maintained 7-entry frozenset
    that carried the refrigerator_3nd/_4nd typos even after Task 9 corrected the registry's own
    keys to _3rd/_4th — a fresh instance of the exact two-tables-can-disagree drift this task
    exists to close, just at the validate_scene()/task_composer.py seam instead of the GUI/batch
    one. Deriving it here closes that seam too: it cannot name a label the registry doesn't.
    """
    from .placements import PLACEMENT_BY_KEY
    return frozenset(PLACEMENT_BY_KEY)


def __getattr__(name: str):
    """PEP 562 module __getattr__. SUPPORT_SURFACES used to be a plain module-level constant;
    every existing call site (task_composer.py, tests, validate_scene() below) still does
    `from scene_spec import SUPPORT_SURFACES` / `scene_spec.SUPPORT_SURFACES` / bare
    `SUPPORT_SURFACES`. A plain `SUPPORT_SURFACES = support_surfaces()` assignment here would
    import kitchen_build at scene_spec's own import time — exactly the module-scope cycle
    kitchen_surfaces() and support_surfaces() avoid by importing lazily inside a function body.
    This hook gets the same laziness (kitchen_build is only imported the first time
    SUPPORT_SURFACES is actually read, not when scene_spec itself is imported) without changing
    any of those call sites.
    """
    if name == "SUPPORT_SURFACES":
        return support_surfaces()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass(frozen=True)
class Extent:
    """center +/- spread, sampled uniformly per kitchen. spread=0 is exact."""
    center: float
    spread: float = 0.0

    def to_dict(self) -> dict:
        return {"center": self.center, "spread": self.spread}

    @staticmethod
    def from_dict(d: dict) -> "Extent":
        return Extent(center=float(d["center"]), spread=float(d.get("spread", 0.0)))


@dataclass
class SceneObject:
    name: str
    object_type: str
    size: Extent                       # uniform scale multiplier (center ~1.0)
    placement: str                     # a SUPPORT_SURFACES label
    lift: Extent | None = None         # set only when this object is manipulated
    #: An exact mesh directory name (e.g. "sem_Cup_dc1c220c…"), pinning WHICH object of this type
    #: gets placed. None means "any mesh of the type", which is what every template did before and
    #: still does. It matters when a type draws from many meshes: `mug` has 161, so a task authored
    #: against one particular cup gets a different object on every build unless it says which.
    mesh: str | None = None

    def to_dict(self) -> dict:
        # `mesh` is omitted when unset rather than written as null, so a template that never pins
        # one round-trips byte-identical -- 20 of them are on disk and read by other tests.
        d = {
            "name": self.name,
            "object_type": self.object_type,
            "size": self.size.to_dict(),
            "placement": self.placement,
            "lift": self.lift.to_dict() if self.lift is not None else None,
        }
        if self.mesh:
            d["mesh"] = self.mesh
        return d

    @staticmethod
    def from_dict(d: dict) -> "SceneObject":
        lift = d.get("lift")
        return SceneObject(
            name=d["name"],
            object_type=d["object_type"],
            size=Extent.from_dict(d["size"]),
            placement=d["placement"],
            lift=Extent.from_dict(lift) if lift is not None else None,
            mesh=d.get("mesh") or None,
        )


# Per type: nominal scale (center ~1.0), a default per-instance jitter, a default surface,
# and a default lift for types that are typically the manipulation target. lift=None means
# "not lifted by default" — a user may still set one in the GUI.
_DEFAULT_LIFT = Extent(center=0.775, spread=0.025)   # = the old target_z (0.75, 0.80)
SCENE_DEFAULTS: dict[str, dict] = {
    "bottle":  {"size": Extent(1.0, 0.05), "placement": "base_cabinet", "lift": _DEFAULT_LIFT},
    "bowl":    {"size": Extent(1.0, 0.05), "placement": "dishwasher",   "lift": _DEFAULT_LIFT},
    "mug":     {"size": Extent(1.0, 0.05), "placement": "island",       "lift": None},
    "apple":   {"size": Extent(1.0, 0.05), "placement": "base_cabinet", "lift": None},
    "sodacan": {"size": Extent(1.0, 0.05), "placement": "base_cabinet", "lift": None},
    "nutella": {"size": Extent(1.0, 0.05), "placement": "base_cabinet", "lift": None},
}
# Clutter defaults are derived from CLUTTER (placement-only, so never lifted) and merged in, so a new
# clutter object is one entry in CLUTTER — not another line here.
SCENE_DEFAULTS.update({
    t: {"size": Extent(1.0, 0.05), "placement": c["placement"], "lift": None}
    for t, c in CLUTTER.items()
})


def default_object(name: str, object_type: str) -> SceneObject:
    if object_type not in SCENE_DEFAULTS:
        raise SceneError(f"no default for object type {object_type!r}; known: {sorted(SCENE_DEFAULTS)}")
    d = SCENE_DEFAULTS[object_type]
    return SceneObject(name=name, object_type=object_type,
                       size=d["size"], placement=d["placement"], lift=d["lift"])


def scene_to_dicts(scene: list[SceneObject]) -> list[dict]:
    return [o.to_dict() for o in scene]


def scene_from_dicts(raw: list[dict]) -> list[SceneObject]:
    return [SceneObject.from_dict(d) for d in raw]


def _check_extent(e: Extent, where: str) -> None:
    if e.center < 0:
        raise SceneError(f"{where}: center must be >= 0, got {e.center}")
    if e.spread < 0:
        raise SceneError(f"{where}: spread must be >= 0, got {e.spread}")


def validate_scene(scene: list[SceneObject]) -> None:
    # Calls support_surfaces() directly rather than reading the bare global SUPPORT_SURFACES:
    # that name only exists via this module's __getattr__ (PEP 562), which fires for external
    # attribute access (scene_spec.SUPPORT_SURFACES / from scene_spec import SUPPORT_SURFACES)
    # but NOT for a plain global-name lookup inside a function defined in this same module.
    surfaces = support_surfaces()
    seen: set[str] = set()
    for o in scene:
        if o.name in seen:
            raise SceneError(f"duplicate scene object name {o.name!r}")
        seen.add(o.name)
        if o.object_type not in OBJECT_TYPES:
            raise SceneError(f"{o.name!r}: unknown object_type {o.object_type!r}; "
                             f"known: {sorted(OBJECT_TYPES)}")
        if o.placement not in surfaces:
            raise SceneError(f"{o.name!r}: unknown placement {o.placement!r}; "
                             f"known: {sorted(surfaces)}")
        if o.mesh is not None:
            # Checked against the type's categories, not against the filesystem: this module is
            # stdlib-only and runs where the dataset is not. It catches the mistake that actually
            # happens -- a pinned mesh that belongs to a different type than the row claims, which
            # would otherwise place something the task never meant.
            if o.mesh in MESH_EXCLUDE:
                raise SceneError(f"{o.name!r}: mesh {o.mesh!r} has been withdrawn from the registry")
            prefixes = CATEGORIES.get(o.object_type, ())
            if not o.mesh.startswith(prefixes):
                raise SceneError(
                    f"{o.name!r}: mesh {o.mesh!r} is not a {o.object_type!r}; "
                    f"that type is {' or '.join(prefixes) or 'undefined'}")
        _check_extent(o.size, f"{o.name!r} size")
        if o.lift is not None:
            _check_extent(o.lift, f"{o.name!r} lift")


def _sample(extent: "Extent", seed: int) -> float:
    if extent.spread == 0.0:
        return extent.center
    rng = random.Random(seed)
    return rng.uniform(extent.center - extent.spread, extent.center + extent.spread)


def sampled_dims(obj: "SceneObject", seed: int) -> list[float]:
    from .kitchen_build import object_scale        # lazy: keeps `import scene_spec` numpy-free
    size = _sample(obj.size, seed)
    return object_scale(obj.object_type, size)


def sampled_lift(lift: "Extent | None", seed: int) -> "float | None":
    if lift is None:
        return None
    return _sample(lift, seed ^ 0x9E3779B9)        # decorrelate lift from size at the same seed


def placement_decision(scene: "list[SceneObject]", seed: int) -> "list[dict]":
    """Per object: concrete dims and lift for one kitchen. Deterministic in seed; the pure
    half of generation, so goal_generator (Omniverse) is a thin consumer."""
    out = []
    for i, o in enumerate(scene):
        s = seed * 1000 + i                       # decorrelate objects within a kitchen
        out.append({
            "name": o.name,
            "object_type": o.object_type,
            "surface": o.placement,
            "dims": sampled_dims(o, s),
            "lift": sampled_lift(o.lift, s),
            # Carried through so a task that pinned an exact mesh gets that mesh on every kitchen
            # it is generated across, not a fresh draw from its type each time.
            "mesh": o.mesh,
        })
    return out
