"""The five legacy TASK_PROFILES, converted to loadable starter scenes.

Replaces `goal_generator --task N`: you load one of these and tweak it, instead of picking a
hardcoded profile. The reproduction invariant in test_scene_starters.py proves each reproduces
its profile's object population.
"""

from scene_spec import SceneObject, Extent

# _choose_place_support's canonical mapping (loc 4 -> island on island kitchens).
LOC_TO_SURFACE = {1: "base_cabinet", 2: "dishwasher", 3: "refrigerator_1st", 4: "island"}

_DEFAULT_SIZE = Extent(1.0, 0.05)          # canonical scale; mug height now lives in OBJECT_BASE_DIMS

# (profile_id, [(type, count, loc), ...]) verbatim from TASK_PROFILES.
_PROFILES = {
    "1":   [("mug", 1, 4), ("bowl", 1, 2)],
    "2":   [("bowl", 1, 1), ("mug", 1, 1)],
    "3":   [("bottle", 1, 1), ("mug", 1, 2)],
    "3_1": [("bottle", 1, 2), ("mug", 1, 1)],
    "4":   [("bottle", 1, 1), ("mug", 1, 2)],
}


def _scene_for(objs):
    scene = []
    counts: dict[str, int] = {}
    for type_, count, loc in objs:
        for _ in range(count):
            i = counts.get(type_, 0)
            counts[type_] = i + 1
            scene.append(SceneObject(
                name=f"{type_}{i}",
                object_type=type_,
                size=_DEFAULT_SIZE,
                placement=LOC_TO_SURFACE[loc],
            ))
    return scene


PROFILE_SCENES = {pid: _scene_for(objs) for pid, objs in _PROFILES.items()}
