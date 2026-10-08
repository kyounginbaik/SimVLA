import importlib.util
import ast
import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).parents[1] / "scripts/simvla/carried_collision.py"
spec = importlib.util.spec_from_file_location("carried_collision", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_proxy_covers_entire_box_not_just_its_surface():
    lower, upper = [-.03, -.04, 0], [.03, .05, .086]
    spheres = module.box_sphere_cover(lower, upper)
    assert len(spheres) == 8
    for point in itertools.product(*[[lo + (hi-lo)*i/8 for i in range(9)]
                                     for lo, hi in zip(lower, upper)]):
        assert any(math.dist(point, sphere[:3]) <= sphere[3] for sphere in spheres)


@pytest.mark.parametrize("lower,upper,padding", [
    ([0, 0, 0], [0, 1, 1], .003), ([0, 0, 0], [-1, 1, 1], .003),
    ([0, 0], [1, 1, 1], .003), ([0, 0, 0], [1, 1, float("nan")], .003),
    ([0, 0, 0], [1, 1, 1], -.001),
])
def test_invalid_payload_geometry_fails_closed(lower, upper, padding):
    with pytest.raises(ValueError):
        module.box_sphere_cover(lower, upper, padding)


def test_only_hand_contacts_are_ignored_and_capacity_is_preallocated():
    kin = {"ee_link": "ee_link2", "collision_link_names": ["arm_l_link7", "gripper_l_base"],
           "self_collision_ignore": {"arm_l_link7": ["gripper_l_base"]}}
    module.configure_payload_proxy({"kinematics": kin})
    assert kin["extra_collision_spheres"][module.PAYLOAD_LINK] == 8
    assert kin["self_collision_ignore"][module.PAYLOAD_LINK] == ["gripper_l_base"]
    assert kin["extra_links"][module.PAYLOAD_LINK]["parent_link_name"] == "ee_link2"
    assert kin["collision_link_names"][-1] == module.PAYLOAD_LINK
    with pytest.raises(ValueError, match="already"):
        module.configure_payload_proxy({"kinematics": kin})


def test_wrong_robot_is_not_silently_configured():
    with pytest.raises(ValueError, match="AI Worker"):
        module.configure_payload_proxy({"kinematics": {"ee_link": "other", "collision_link_names": []}})


@pytest.mark.parametrize("closed", [True, False])
def test_locked_joint_refresh_cannot_erase_live_payload(closed):
    torch = pytest.importorskip("torch")
    source = path.with_name("simvla_gen.py")
    tree = ast.parse(source.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == "_repose_world_to_base")
    calls = []
    class Planner:
        spheres = None
        def update_locked_joints(self, locks, cfg):
            calls.append("refresh")
            self.spheres = None  # cuRobo copies configured link_spheres here.
        def attach_spheres_to_robot(self, sphere_tensor, link_name):
            calls.append("attach")
            self.spheres = sphere_tensor.clone()
        def detach_spheres_from_robot(self, link_name):
            calls.append("detach")
            self.spheres = None
    planner = Planner()
    planner.kinematics = SimpleNamespace(kinematics_config=SimpleNamespace(
        get_link_spheres=lambda link_name: planner.spheres))
    cfg = {"kinematics": {"lock_joints": {"lift_joint": -.3}}}
    robot = SimpleNamespace(joint_names=["lift_joint"], data=SimpleNamespace(
        joint_pos=torch.tensor([[-.285]]), body_quat_w=torch.tensor([[[1., 0, 0, 0]]]),
        body_pos_w=torch.zeros(1, 1, 3)))
    namespace = dict(torch=torch, motion_gen_l=planner, motion_gen_r=planner,
                     l_robot_cfg=cfg, r_robot_cfg=cfg, robot=robot,
                     measured_locked_joints=lambda *args: {"lift_joint": -.285},
                     _payload_collision_l=True, new_gripper_commands_L=[closed],
                     _payload_asset=SimpleNamespace(data=SimpleNamespace(
                         root_quat_w=torch.tensor([[1., 0, 0, 0]]), root_pos_w=torch.zeros(1, 3))),
                     _payload_local_centers=torch.ones(8, 3), _payload_radii=torch.ones(8, 1) * .03,
                     math_utils=SimpleNamespace(quat_apply=lambda q, v: v,
                                                quat_rotate_inverse=lambda q, v: v),
                     l_eef_idx=0, PAYLOAD_LINK=module.PAYLOAD_LINK,
                     _target_collision_mode="none", _world_reposers={})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace["_repose_world_to_base"]("l", 0)
    assert calls == ["refresh", "attach" if closed else "detach"]
    if closed:
        assert planner.spheres.shape == (8, 4)
        assert bool((planner.spheres[:, 3] > 0).all())
