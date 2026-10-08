"""scene_spec is the pure model for a task's objects and their geometry. Stdlib only —
importing it must boot neither Omniverse nor torch, so the whole thing stays unit-testable.

Run: pytest scripts/simvla/test_scene_spec.py -v
"""

import pytest

from simvla.scene_spec import (
    Extent, SceneObject, SceneError, SCENE_DEFAULTS, OBJECT_TYPES, SUPPORT_SURFACES,
    KITCHEN_TYPES, default_object, kitchen_surfaces, scene_to_dicts, scene_from_dicts,
    validate_scene,
)


def test_an_extent_is_a_center_and_a_spread():
    e = Extent(center=1.0, spread=0.1)
    assert e.center == 1.0 and e.spread == 0.1


def test_extent_spread_defaults_to_zero_meaning_exact():
    assert Extent(0.08).spread == 0.0


def test_extent_round_trips_through_a_dict():
    e = Extent(0.775, 0.025)
    assert Extent.from_dict(e.to_dict()) == e


def test_default_object_fills_from_the_defaults_table():
    obj = default_object("bowl0", "bowl")
    assert obj.name == "bowl0"
    assert obj.object_type == "bowl"
    assert obj.size == SCENE_DEFAULTS["bowl"]["size"]
    assert obj.placement == SCENE_DEFAULTS["bowl"]["placement"]


def test_every_declared_object_type_has_a_default():
    """A type the GUI can add must have a default, or 'add object' hands the user a blank."""
    assert set(SCENE_DEFAULTS) == OBJECT_TYPES


def test_a_scene_round_trips_through_dicts():
    scene = [
        SceneObject("bowl0", "bowl", Extent(1.0, 0.05), "dishwasher", lift=Extent(0.775, 0.025)),
        SceneObject("mug0", "mug", Extent(1.0, 0.05), "island"),
    ]
    assert scene_from_dicts(scene_to_dicts(scene)) == scene


def test_validate_rejects_a_duplicate_name():
    scene = [default_object("x", "bowl"), default_object("x", "mug")]
    with pytest.raises(SceneError, match="duplicate"):
        validate_scene(scene)


def test_validate_rejects_an_unknown_object_type():
    scene = [SceneObject("t", "spaceship", Extent(1.0), "island")]   # not a registered object type
    with pytest.raises(SceneError, match="spaceship"):
        validate_scene(scene)


def test_validate_rejects_an_unknown_placement():
    scene = [SceneObject("b", "bowl", Extent(1.0), "ceiling")]
    with pytest.raises(SceneError, match="ceiling"):
        validate_scene(scene)


def test_validate_rejects_a_negative_spread():
    scene = [SceneObject("b", "bowl", Extent(1.0, -0.1), "island")]
    with pytest.raises(SceneError, match="spread"):
        validate_scene(scene)


def test_the_support_surfaces_are_spelled_exactly_as_the_generator_labels_them():
    """SUPPORT_SURFACES must be exactly the set of labels goal_generator._label_supports could
    ever apply, across every kitchen type it can build. It used to be a hand-maintained 7-entry
    frozenset (with a refrigerator_2nd/_3nd/_4nd typo-preserving contract that "_label_supports
    uses" — that contract stopped being true the moment _label_supports was rewired to derive
    corrected spellings from the registry, which made this test pass for the wrong reason: it
    was only checking an untouched constant, not anything _label_supports actually does).
    SUPPORT_SURFACES is now itself derived from kitchen_build.PLACEMENT_BY_KEY (the exact table
    _label_supports reads through kitchen_surfaces()), so the two cannot drift again.

    kitchen_surfaces() takes has_table=True here, not the default False. The table is optional
    now -- a *particular* kitchen only has 'table_top' if the user added one, which is what the
    default (no table) is for and why every real call site in goal_generator.py keeps it. But
    this test asks a different question: every label _label_supports could EVER apply, across
    every kitchen the generator can build -- and a kitchen it builds CAN have a table, so
    'table_top' belongs in that "ever" set same as any other label. has_table=False here would
    make this test's own union stop matching what it claims to compute (label_support only ever
    registers 'table_top' when a table exists), not just paper over the table becoming optional.
    """
    from simvla.kitchen_build import KITCHEN_TYPES as KB_KITCHEN_TYPES, PLACEMENT_BY_KEY

    # Every label _label_supports could apply to ANY kitchen it builds, across all 5 layouts --
    # including one with a user-placed table, since that is still a kitchen the generator builds.
    labelable = set()
    for kt in KB_KITCHEN_TYPES:
        labelable |= kitchen_surfaces(kt, has_table=True)

    assert SUPPORT_SURFACES == labelable, (
        f"SUPPORT_SURFACES disagrees with what _label_supports actually labels: "
        f"only-in-SUPPORT_SURFACES={sorted(SUPPORT_SURFACES - labelable)}, "
        f"only-labelable={sorted(labelable - SUPPORT_SURFACES)}"
    )
    assert SUPPORT_SURFACES == set(PLACEMENT_BY_KEY)

    assert "refrigerator_3rd" in SUPPORT_SURFACES
    assert "refrigerator_3nd" not in SUPPORT_SURFACES   # the old typo must not survive
    assert "island" in SUPPORT_SURFACES


def test_every_kitchen_type_provides_the_four_refrigerator_shelves():
    """The refrigerator shelves are created for every layout — a placement there is always valid.

    kitchen_surfaces() is now derived from kitchen_build.PLACEMENTS (Task 9), whose keys
    correct the old refrigerator_3nd/_4nd typos to _3rd/_4th; no template ever persisted the
    typo'd spelling, so the rename is safe.
    """
    fridge = {"refrigerator_1st", "refrigerator_2nd", "refrigerator_3rd", "refrigerator_4th"}
    for kt in KITCHEN_TYPES:
        assert fridge <= kitchen_surfaces(kt), f"{kt} is missing a refrigerator shelf"


def test_only_the_island_kitchen_provides_the_island_surface():
    assert "island" in kitchen_surfaces("island")
    for kt in KITCHEN_TYPES - {"island"}:
        assert "island" not in kitchen_surfaces(kt), f"{kt} must not provide 'island'"


def test_a_non_island_kitchen_still_provides_base_cabinet_and_dishwasher():
    surfaces = kitchen_surfaces("l_shaped")
    assert {"base_cabinet", "dishwasher"} <= surfaces
    assert "island" not in surfaces


def test_kitchen_surfaces_matches_the_registry():
    """kitchen_surfaces() and the placement registry must agree on which labels a layout offers;
    they were two independent tables before, which is the drift this closes."""
    from simvla.kitchen_build import KITCHEN_TYPES as KB_KITCHEN_TYPES, available_locations
    for kitchen_type in KB_KITCHEN_TYPES:
        assert kitchen_surfaces(kitchen_type) == {
            p.key for p in available_locations(kitchen_type)
        }, f"{kitchen_type}: kitchen_surfaces disagrees with available_locations"


def test_kitchen_surfaces_never_invents_a_label_outside_the_vocabulary():
    """Every surface a layout provides is a key from kitchen_build.PLACEMENT_BY_KEY, so the guard
    that reads kitchen_surfaces and the labels _label_supports registers share one vocabulary.

    Was checked against scene_spec.SUPPORT_SURFACES (a separate, hand-maintained 7-entry set);
    since Task 9, kitchen_surfaces derives from the registry instead, so the real vocabulary is
    kitchen_build.PLACEMENT_BY_KEY."""
    from simvla.kitchen_build import PLACEMENT_BY_KEY
    for kt in KITCHEN_TYPES:
        assert kitchen_surfaces(kt) <= set(PLACEMENT_BY_KEY)


def test_an_unknown_kitchen_type_raises():
    """kitchen_surfaces() now derives from kitchen_build.available_locations(), which raises
    ValueError for a kitchen type outside kitchen_build.KITCHEN_TYPES rather than silently
    degrading to a default surface set — the same contract available_locations() itself is
    held to (test_available_locations_rejects_unknown_kitchen in test_kitchen_build.py)."""
    with pytest.raises(ValueError, match="igloo"):
        kitchen_surfaces("igloo")


def test_sampled_dims_is_exact_when_spread_is_zero():
    from simvla.scene_spec import SceneObject, Extent, sampled_dims
    obj = SceneObject("b", "bowl", Extent(1.0, 0.0), "island")
    # bowl base dims (0.08, 0.08, 0.10) x size 1.0
    assert sampled_dims(obj, seed=7) == [0.08, 0.08, 0.10]


def test_sampled_dims_is_deterministic_in_the_seed():
    from simvla.scene_spec import SceneObject, Extent, sampled_dims
    obj = SceneObject("b", "bowl", Extent(1.0, 0.2), "island")
    assert sampled_dims(obj, seed=7) == sampled_dims(obj, seed=7)
    assert sampled_dims(obj, seed=7) != sampled_dims(obj, seed=8)


def test_sampled_dims_stays_within_the_spread():
    from simvla.scene_spec import SceneObject, Extent, sampled_dims
    obj = SceneObject("b", "bowl", Extent(1.0, 0.25), "island")
    for seed in range(200):
        w, d, h = sampled_dims(obj, seed)
        assert 0.08 * 0.75 <= w <= 0.08 * 1.25   # size in [0.75, 1.25]


def test_the_mug_base_height_is_now_concrete():
    from simvla.kitchen_build import OBJECT_BASE_DIMS, object_scale
    assert OBJECT_BASE_DIMS["mug"][2] == 0.08
    # object_scale needs no mug_height any more
    assert object_scale("mug", 1.0) == [0.06, 0.06, 0.08]


def test_sampled_lift_is_none_when_absent():
    from simvla.scene_spec import sampled_lift
    assert sampled_lift(None, seed=1) is None


def test_sampled_lift_stays_within_the_spread():
    from simvla.scene_spec import Extent, sampled_lift
    for seed in range(200):
        v = sampled_lift(Extent(0.775, 0.025), seed)
        assert 0.75 <= v <= 0.80


def test_sampled_lift_decorrelates_from_the_size_draw_at_the_same_seed():
    """sampled_lift XORs the seed before sampling, so an object's lift draw does not lock
    to its size draw even though both start from the same per-kitchen seed. Pin that here:
    with the same Extent used for both, the raw _sample() draw (what size sampling uses)
    and the sampled_lift() draw must differ. If the XOR were dropped, sampled_lift would
    call _sample(extent, seed) directly and this would be an equality, not an inequality.
    """
    from simvla.scene_spec import Extent, _sample, sampled_lift
    ext = Extent(1.0, 0.3)
    seed = 7
    assert sampled_lift(ext, seed) != _sample(ext, seed)


def test_importing_scene_spec_boots_nothing():
    import subprocess, sys, textwrap, os
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            from simvla import scene_spec
            banned = ("omni", "isaaclab", "pxr", "torch", "tkinter", "numpy")
            pulled = sorted(m for m in sys.modules if m.split(".")[0] in banned)
            assert not pulled, f"scene_spec pulled in {pulled}"
            print("clean")
        """)],
        cwd=here, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.split() == ["clean"]


# --- pinning an exact mesh -------------------------------------------------------------------

def _obj(**kw):
    from simvla.scene_spec import Extent, SceneObject
    base = dict(name="mug0", object_type="mug", size=Extent(1.0, 0.05), placement="island")
    base.update(kw)
    return SceneObject(**base)


def test_a_scene_object_pins_no_mesh_by_default():
    """Every template written before this field existed means 'any mesh of the type'."""
    assert _obj().mesh is None


def test_an_unpinned_object_round_trips_without_growing_a_mesh_key():
    """20 templates are on disk and read by other tests; writing "mesh": null into all of them
    would be a diff with no meaning."""
    from simvla.scene_spec import SceneObject
    d = _obj().to_dict()
    assert "mesh" not in d
    assert SceneObject.from_dict(d).mesh is None


def test_a_pinned_mesh_survives_the_round_trip():
    from simvla.scene_spec import SceneObject
    d = _obj(mesh="sem_Cup_dc1c220c8ef89a5d1a57566a9a7e5976").to_dict()
    assert d["mesh"] == "sem_Cup_dc1c220c8ef89a5d1a57566a9a7e5976"
    assert SceneObject.from_dict(d).mesh == d["mesh"]


def test_a_pinned_mesh_must_belong_to_the_type_that_names_it():
    """The mistake worth catching: a task says 'mug' but pins a vase, and every kitchen it
    generates places the wrong object without ever failing."""
    from simvla.scene_spec import SceneError, validate_scene
    validate_scene([_obj(mesh="sem_Cup_dc1c220c8ef89a5d1a57566a9a7e5976")])   # a mug, per CATEGORIES
    with pytest.raises(SceneError, match="is not a 'mug'"):
        validate_scene([_obj(mesh="sem_Vase_1a8f9295b44b48895e8c5748ca5ef3ea")])


def test_a_withdrawn_mesh_cannot_be_pinned():
    from simvla.scene_spec import MESH_EXCLUDE, SceneError, validate_scene
    withdrawn = next(m for m in MESH_EXCLUDE if m.startswith("sem_Vase"))
    with pytest.raises(SceneError, match="withdrawn"):
        validate_scene([_obj(name="vase0", object_type="vase",
                             placement="base_cabinet", mesh=withdrawn)])


def test_the_pinned_mesh_reaches_the_placement_decision():
    """placement_decision is what goal_generator turns into build rows; a mesh that stops here
    would be honoured by the composer and silently dropped by the batch path."""
    from simvla.scene_spec import placement_decision
    d = placement_decision([_obj(mesh="core_mug_1038e4eac0e18dcce02ae6d2a21d494a")], seed=0)
    assert d[0]["mesh"] == "core_mug_1038e4eac0e18dcce02ae6d2a21d494a"
