"""Binding roles to prims. Ambiguity and absence are REPORTED, never guessed — a fan-out that
silently skips kitchens reads as 'it covered everything'.

Run: pytest scripts/simvla/test_task_bind.py -v
"""

import pytest

import skills  # noqa: F401
from task_bind import BindingError, BoundPrim, KitchenScene, bind_many, bind_roles
from scripts.simvla.test_task_template import bowl_to_drawer


def kitchen(objects, articulations=None) -> KitchenScene:
    return KitchenScene(
        objects=objects,
        articulations=articulations
        if articulations is not None
        else {"/world/base_cabinet/drawer_0_0": ["drawer", "handle"]},
    )


def test_a_kitchen_with_one_bowl_and_a_drawer_binds_with_no_clicks():
    scene = kitchen([("mug", "/world/mug0"), ("bowl", "/world/bowl0")])
    bound = bind_roles(bowl_to_drawer(), scene)

    assert bound["target"].prim_path == "/world/bowl0"
    assert bound["target"].is_rigid_body is True
    assert bound["target"].object_index == 1, "the placement order — NOT --target_idx, which is a step"
    assert bound["container"].prim_path == "/world/base_cabinet/drawer_0_0"
    assert bound["container"].is_rigid_body is False
    assert bound["container_handle"].prim_path.endswith("/door_handle")
    assert bound["container_handle"].is_rigid_body is False


def test_bound_prim_name_strips_the_path():
    assert BoundPrim("/world/bowl0", True, 0).name == "bowl0"


def test_a_kitchen_with_no_bowl_is_reported_not_guessed():
    scene = kitchen([("mug", "/world/mug0")])
    with pytest.raises(BindingError, match="target"):
        bind_roles(bowl_to_drawer(), scene)


def test_a_kitchen_with_two_bowls_is_ambiguous_and_says_so():
    scene = kitchen([("bowl", "/world/bowl0"), ("bowl", "/world/bowl1")])
    with pytest.raises(BindingError, match="ambiguous"):
        bind_roles(bowl_to_drawer(), scene)


def test_a_kitchen_with_no_drawer_is_reported():
    scene = kitchen([("bowl", "/world/bowl0")], articulations={"/world/fridge": ["door", "handle"]})
    with pytest.raises(BindingError, match="container"):
        bind_roles(bowl_to_drawer(), scene)


def test_two_drawers_but_only_one_handled_binds_the_handled_one():
    """bowl_to_drawer has a container_handle role (handle_of=container), so the task opens the drawer
    by its handle. When a kitchen has two drawers but only one has a handle (the real case: a
    base_cabinet drawer is handled, a range drawer is not), bind to the handled one instead of
    refusing as ambiguous — the handle-less drawer could never satisfy the handle step."""
    scene = kitchen(
        [("bowl", "/world/bowl0")],
        articulations={
            "/world/base_cabinet/drawer_0_0": ["drawer", "handle"],
            "/world/range/drawer_0_2": ["drawer"],                   # no handle
        },
    )
    bound = bind_roles(bowl_to_drawer(), scene)
    assert bound["container"].prim_path == "/world/base_cabinet/drawer_0_0"
    assert bound["container_handle"].prim_path == "/world/base_cabinet/drawer_0_0/door_handle"


def test_only_drawer_is_handle_less_is_skipped_not_bound():
    """A kitchen whose only drawer has no handle can't do a task that opens the drawer by its handle.
    Report it (naming the handle requirement), don't bind a drawer the handle step would then fail on."""
    scene = kitchen(
        [("bowl", "/world/bowl0")],
        articulations={"/world/range/drawer_0_2": ["drawer"]},      # no handle
    )
    with pytest.raises(BindingError, match="handle"):
        bind_roles(bowl_to_drawer(), scene)


def test_two_handled_drawers_are_still_ambiguous():
    """The handle filter narrows, it doesn't invent a choice: two drawers that BOTH have a handle are
    genuinely ambiguous for a role that names neither, so still refuse loudly."""
    scene = kitchen(
        [("bowl", "/world/bowl0")],
        articulations={
            "/world/base_cabinet/drawer_0_0": ["drawer", "handle"],
            "/world/base_cabinet/drawer_1_0": ["drawer", "handle"],
        },
    )
    with pytest.raises(BindingError, match="ambiguous"):
        bind_roles(bowl_to_drawer(), scene)


def test_bind_many_skips_and_reports_rather_than_dropping():
    scenes = {
        "Isaac-Kitchen-v813-00": kitchen([("mug", "/world/mug0"), ("bowl", "/world/bowl0")]),
        "Isaac-Kitchen-v999-00": kitchen([("mug", "/world/mug0")]),          # no bowl
        "Isaac-Kitchen-v998-00": kitchen([("bowl", "/world/b0"), ("bowl", "/world/b1")]),  # two
    }
    bound, skipped = bind_many(bowl_to_drawer(), scenes)

    assert set(bound) == {"Isaac-Kitchen-v813-00"}
    assert len(skipped) == 2
    assert any("v999" in s and "target" in s for s in skipped)
    assert any("v998" in s and "ambiguous" in s for s in skipped)


# ==================================================================================================
# Naming WHICH articulation: a conjunction of features
# ==================================================================================================
#
# A single feature string cannot name a cabinet door in a real kitchen. Measured on a generated
# l_shaped kitchen: eleven articulations carry "door" (both sink-cabinet doors, five wall-cabinet
# doors, the fridge, the freezer, the microwave, the oven), and the base cabinet carries "drawer"
# on some seeds and "door" on others. `articulation_with="door"` is therefore always ambiguous, and
# `articulation_with="base_cabinet"` binds a PRISMATIC DRAWER on half the seeds -- silently, to a
# template whose whole purpose is to arc a base about a hinge.
#
# A list means "every one of these features", which names the intersection exactly.

def test_a_list_of_features_binds_their_intersection():
    from task_template import Role, TaskTemplate, TemplateStep
    scene = KitchenScene(objects=[], articulations={
        "/world/base_cabinet/drawer_0_0": ["drawer", "handle", "base_cabinet"],
        "/world/base_cabinet/door_0_0": ["door", "handle", "base_cabinet"],
        "/world/sink_cabinet/door_0_1": ["door", "handle", "sink_cabinet"],
    })
    t = TaskTemplate(
        name="t", language="Open the cabinet door.",
        roles=[Role(name="cabinet", articulation_with=["door", "base_cabinet"]),
               Role(name="cabinet_handle", handle_of="cabinet")],
        steps=[TemplateStep(skill="arm.reset", action="A_l")],
        success={"all": []},
    )
    bound = bind_roles(t, scene)
    assert bound["cabinet"].prim_path == "/world/base_cabinet/door_0_0"
    assert bound["cabinet_handle"].prim_path == "/world/base_cabinet/door_0_0/door_handle"


def test_a_single_feature_string_still_works():
    """Every template in the repo passes a bare string; none of them may change behaviour."""
    scene = kitchen([("bowl", "/world/bowl0")])
    bound = bind_roles(bowl_to_drawer(), scene)
    assert bound["container"].prim_path == "/world/base_cabinet/drawer_0_0"


def test_a_conjunction_that_matches_nothing_is_reported():
    from task_template import Role, TaskTemplate, TemplateStep
    scene = KitchenScene(objects=[], articulations={
        "/world/base_cabinet/drawer_0_0": ["drawer", "handle", "base_cabinet"],
    })
    t = TaskTemplate(
        name="t", language="Open the cabinet door.",
        roles=[Role(name="cabinet", articulation_with=["door", "base_cabinet"])],
        steps=[TemplateStep(skill="arm.reset", action="A_l")], success={"all": []},
    )
    with pytest.raises(BindingError, match="door.*base_cabinet|base_cabinet.*door"):
        bind_roles(t, scene)


def test_a_conjunction_matching_two_articulations_is_still_ambiguous():
    """The point is to narrow, not to pick the first. Two base-cabinet doors that BOTH have a
    handle must still be reported rather than silently resolved."""
    from task_template import Role, TaskTemplate, TemplateStep
    scene = KitchenScene(objects=[], articulations={
        "/world/base_cabinet/door_0_0": ["door", "handle", "base_cabinet"],
        "/world/base_cabinet/door_1_0": ["door", "handle", "base_cabinet"],
    })
    t = TaskTemplate(
        name="t", language="Open the cabinet door.",
        roles=[Role(name="cabinet", articulation_with=["door", "base_cabinet"]),
               Role(name="cabinet_handle", handle_of="cabinet")],
        steps=[TemplateStep(skill="arm.reset", action="A_l")], success={"all": []},
    )
    with pytest.raises(BindingError, match="ambiguous"):
        bind_roles(t, scene)
