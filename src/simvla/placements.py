"""Placement vocabulary shared by scene validation and construction."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Placement:
    """One entry in the generator's Placement Location menu.

    `patterns` are fnmatch patterns over scene-graph NODE names (not geometry names):
    label_support()'s geom_ids takes a node, which may be a group owning several
    geometries. Every matched node is labeled with its own unique support label —
    scene_synthesizer OVERWRITES when label_support() is called twice with the same
    label, so a shared label would silently discard all but the last surface.

    `pick` decides which matched node an object spawns on when several match:
      "random"  -- uniform choice, which is what loc 1 did for peninsula/u_shaped
      "largest" -- the node with the greatest horizontal area, for multi-mesh
                   interiors like the oven rack (79-88 near-identical sub-meshes)

    `needs_door` marks placements behind a door the robot must open first. It is
    metadata for task authors; the generator does not act on it.
    """

    loc: int
    key: str
    label: str
    group: str
    patterns: tuple[str, ...]
    pick: str = "random"
    needs_door: bool = False

WALL_CABINET_DYNAMIC = True

PLACEMENTS: tuple[Placement, ...] = (
    Placement(
        loc=1,
        key="base_cabinet",
        label="Above cabinet",
        group="Countertops",
        patterns=("countertop_base_cabinet", "countertop_base_cabinet_[0-9]"),
    ),
    Placement(
        loc=2,
        key="dishwasher",
        label="Above dishwasher",
        group="Countertops",
        patterns=("countertop_dishwasher",),
    ),
    Placement(
        loc=3,
        key="refrigerator_1st",
        label="Fridge shelf 1",
        group="Refrigerator",
        patterns=("refrigerator/shelf_3",),
        needs_door=True,
    ),
    Placement(
        loc=4,
        key="island",
        label="Above island",
        group="Countertops",
        patterns=("kitchen_island/countertop",),
    ),
    Placement(
        loc=5,
        key="sink_counter",
        label="Sink counter",
        group="Countertops",
        patterns=("sink_cabinet/sink_countertop",),
    ),
    Placement(
        loc=6,
        key="corner_counter",
        label="Corner counter",
        group="Countertops",
        patterns=("countertop_corner", "countertop_corner_[0-9]"),
    ),
    Placement(
        loc=7,
        key="stovetop",
        label="On stovetop",
        group="Appliance tops",
        patterns=("range/top",),
    ),
    Placement(
        loc=8,
        key="fridge_top",
        label="On fridge top",
        group="Appliance tops",
        patterns=("refrigerator/top",),
    ),
    Placement(
        loc=9,
        key="base_cabinet_interior",
        label="Inside base cabinet",
        group="Inside",
        # base_cabinet/shelf_* placed successfully as-is (verified by
        # test_new_location_actually_places_an_object); no *_surface_* fallback needed.
        patterns=("base_cabinet/shelf_*", "base_cabinet_[0-9]/shelf_*"),
        pick="largest",
        needs_door=True,
    ),
    Placement(
        loc=10,
        key="sink_cabinet_interior",
        label="Inside sink cabinet",
        group="Inside",
        # sink_cabinet/surface_* placed successfully as-is (verified by
        # test_new_location_actually_places_an_object); no widening needed.
        patterns=("sink_cabinet/surface_*",),
        pick="largest",
        needs_door=True,
    ),
    Placement(
        loc=11,
        key="dishwasher_interior",
        label="Inside dishwasher",
        group="Inside",
        # dishwasher/surface_* placed successfully as-is (verified by
        # test_new_location_actually_places_an_object); no widening needed.
        patterns=("dishwasher/surface_*",),
        pick="largest",
        needs_door=True,
    ),
    Placement(
        loc=12,
        key="oven",
        label="In oven",
        group="Inside",
        # range/shelf_* is the wire rack: 88 near-identical sub-meshes, individually too
        # thin for place_object to find a pose (verified: every one raised "No supports
        # found"). range/surface_* -- the three burner surfaces -- doubles as the oven
        # cavity floor on these assets and is what actually places an object.
        patterns=("range/surface_*",),
        pick="largest",
        needs_door=True,
    ),
    Placement(loc=20, key="refrigerator_2nd", label="Fridge shelf 2", group="Refrigerator",
              patterns=("refrigerator/shelf_2",), needs_door=True),
    Placement(loc=21, key="refrigerator_3rd", label="Fridge shelf 3", group="Refrigerator",
              patterns=("refrigerator/shelf_1",), needs_door=True),
    Placement(loc=22, key="refrigerator_4th", label="Fridge shelf 4", group="Refrigerator",
              patterns=("refrigerator/shelf_0",), needs_door=True),
    Placement(loc=23, key="refrigerator_5th", label="Fridge shelf 5", group="Refrigerator",
              patterns=("refrigerator/shelf_4",), needs_door=True),
    Placement(loc=30, key="fridge_door_shelf_1", label="Fridge door shelf 1", group="Refrigerator",
              patterns=("refrigerator/door_shelf_2",), needs_door=True),
    Placement(loc=31, key="fridge_door_shelf_2", label="Fridge door shelf 2", group="Refrigerator",
              patterns=("refrigerator/door_shelf_1",), needs_door=True),
    Placement(loc=32, key="fridge_door_shelf_3", label="Fridge door shelf 3", group="Refrigerator",
              patterns=("refrigerator/door_shelf_0",), needs_door=True),
    Placement(loc=40, key="microwave_interior", label="Inside microwave", group="Inside",
              patterns=("microwave/bottom",), needs_door=True),
    Placement(loc=42, key="table_top", label="On the table", group="Countertops",
              patterns=("table/top",)),
)

_WALL_CABINET_PLACEMENTS: tuple[Placement, ...] = (
    Placement(
        loc=13,
        key="wall_cabinet_top",
        label="On wall cabinet top",
        group="Appliance tops",
        patterns=("wall_cabinet/top", "wall_cabinet_[0-9]/top"),
    ),
    Placement(
        loc=14,
        key="wall_cabinet_interior",
        label="Inside wall cabinet",
        group="Inside",
        patterns=("wall_cabinet/shelf_*", "wall_cabinet_[0-9]/shelf_*"),
        pick="largest",
        needs_door=True,
    ),
)

if WALL_CABINET_DYNAMIC:
    PLACEMENTS = PLACEMENTS + _WALL_CABINET_PLACEMENTS

PLACEMENT_BY_LOC: dict[int, Placement] = {p.loc: p for p in PLACEMENTS}

LOCATION_LABELS: dict[int, str] = {p.loc: p.label for p in PLACEMENTS}

PLACEMENT_BY_KEY: dict[str, Placement] = {p.key: p for p in PLACEMENTS}

KITCHEN_TYPES: tuple[str, ...] = ("island", "l_shaped", "peninsula", "u_shaped", "single_wall")

_ALL = frozenset(KITCHEN_TYPES)

LOCATION_KITCHENS: dict[int, frozenset[str]] = {
    1: _ALL,
    2: _ALL,
    3: _ALL,
    4: frozenset({"island"}),
    5: _ALL,
    6: _ALL - frozenset({"single_wall"}),   # single_wall has no corner countertop
    7: _ALL,
    8: _ALL,
    9: _ALL,
    10: _ALL,
    11: _ALL,
    12: _ALL,
    20: _ALL,
    21: _ALL,
    22: _ALL,
    23: _ALL,
    30: _ALL,
    31: _ALL,
    32: _ALL,
    40: _ALL,
    42: _ALL,      # a user-placed table can go in any layout; presence is has_table, not type
}

if WALL_CABINET_DYNAMIC:
    LOCATION_KITCHENS[13] = _ALL
    LOCATION_KITCHENS[14] = _ALL

def available_locations(kitchen_name: str, *, has_table: bool = False) -> list[Placement]:
    """The placements this kitchen type can actually offer, in menu order.

    has_table is separate from LOCATION_KITCHENS because a table is not a property of the
    layout any more -- the user adds one or does not. Everything else here is still decided by
    the kitchen type alone, which is what lets the wizard build its menu before any kitchen
    exists (a build costs 5-7s).
    """
    if kitchen_name not in KITCHEN_TYPES:
        raise ValueError(f"Unknown kitchen type: {kitchen_name!r}")
    table_loc = PLACEMENT_BY_KEY["table_top"].loc
    return [
        p for p in PLACEMENTS
        if kitchen_name in LOCATION_KITCHENS[p.loc] and (p.loc != table_loc or has_table)
    ]

