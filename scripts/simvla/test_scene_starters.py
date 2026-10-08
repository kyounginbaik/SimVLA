"""The reproduction invariant: each starter scene must reproduce the object population of the
TASK_PROFILE it came from. If it does not, the conversion is wrong — do not adjust the expected
values. (The analogue of the composer's v813 proof.)

Run: pytest scripts/simvla/test_scene_starters.py -v
"""

from collections import Counter

from scene_spec import Extent
from scene_starters import PROFILE_SCENES, LOC_TO_SURFACE

# The profiles, verbatim from goal_generator.TASK_PROFILES (objects only; loc -> surface below).
PROFILE_OBJECTS = {
    "1":   [("mug", 1, 4), ("bowl", 1, 2)],
    "2":   [("bowl", 1, 1), ("mug", 1, 1)],
    "3":   [("bottle", 1, 1), ("mug", 1, 2)],
    "3_1": [("bottle", 1, 2), ("mug", 1, 1)],
    "4":   [("bottle", 1, 1), ("mug", 1, 2)],
}


def test_every_profile_has_a_starter():
    assert set(PROFILE_SCENES) == set(PROFILE_OBJECTS)


def test_each_starter_reproduces_its_profiles_types_counts_and_surfaces():
    for pid, objs in PROFILE_OBJECTS.items():
        want = Counter()
        for type_, count, loc in objs:
            want[(type_, LOC_TO_SURFACE[loc])] += count
        got = Counter((o.object_type, o.placement) for o in PROFILE_SCENES[pid])
        assert got == want, f"profile {pid}: got {got}, want {want}"


def test_every_starter_object_has_a_unique_name():
    for pid, scene in PROFILE_SCENES.items():
        names = [o.name for o in scene]
        assert len(names) == len(set(names)), f"profile {pid}: duplicate names {names}"


def test_the_mug_is_canonicalized_and_the_profile_1_change_is_explicit():
    # After canonicalization every mug scales at center 1.0 (base height 0.08). Profile "1" used
    # mug_height 0.09; its starter mug is size center 1.0 too -- the deliberate 1mm change.
    for pid, scene in PROFILE_SCENES.items():
        for o in scene:
            if o.object_type == "mug":
                assert o.size.center == 1.0, f"profile {pid} mug size center != 1.0"


def test_every_starter_scene_validates():
    from scene_spec import validate_scene
    for scene in PROFILE_SCENES.values():
        validate_scene(scene)


def test_placement_decision_is_one_entry_per_object_deterministic_in_seed():
    from scene_spec import placement_decision
    scene = PROFILE_SCENES["3"]
    d1 = placement_decision(scene, seed=11)
    d2 = placement_decision(scene, seed=11)
    assert d1 == d2
    assert len(d1) == len(scene)
    assert {e["name"] for e in d1} == {o.name for o in scene}
    for e in d1:
        assert len(e["dims"]) == 3


def test_placement_decision_lift_is_present_only_where_the_object_has_one():
    from scene_spec import placement_decision, SceneObject, Extent
    scene = [
        SceneObject("bowl0", "bowl", Extent(1.0), "dishwasher", lift=Extent(0.775, 0.025)),
        SceneObject("mug0", "mug", Extent(1.0), "island"),
    ]
    by_name = {e["name"]: e for e in placement_decision(scene, seed=3)}
    assert by_name["bowl0"]["lift"] is not None
    assert by_name["mug0"]["lift"] is None
