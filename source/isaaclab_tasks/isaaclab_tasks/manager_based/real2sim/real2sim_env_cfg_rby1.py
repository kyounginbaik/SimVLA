"""Real2Sim env for the RB-Y1 in absolute joint-position control.

Replays a recorded teleop LEADER stream (the command) into the sim arms and compares the sim's
joint trajectory against the recorded FOLLOWER stream, across envs that each carry a different
random draw of arm actuator gains. Survivors' gains are written out by `mdp.sim2ruin_named`.
Driven by scripts/simvla/systemid.py --robot rby1 --control joint --npz <log>.
"""
import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.rby1 import RBY1_CFG

from . import mdp

# Explicit lists, in the order the dataset columns arrive (systemid_core.load_rby1_npz):
# right arm, then left arm; torso after. preserve_order=True on the action terms keeps the
# articulation's own joint order from silently re-sorting them.
ARM_JOINTS = [f"right_arm_{i}" for i in range(7)] + [f"left_arm_{i}" for i in range(7)]
TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
FIT_JOINTS = ARM_JOINTS + TORSO_JOINTS  # observation.state column order


@configclass
class Real2SimSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = RBY1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    # Head camera, only read under --record. Same mount as the kitchens' `front` camera (the
    # 50 deg pitch lives in the offset; the real log has head_1 = 0.872 rad = 50 deg with the
    # camera flat on the head). Pinhole at ~100 deg HFOV and 640x480 to match the ZED frames in
    # the recorder's cameras.h5, so real and sim can sit side by side.
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_head_2/head_cam",
        update_period=1 / 20,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            focus_distance=1.0,
            horizontal_aperture=4.6,
            clipping_range=(0.01, 1000.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.18, 0.0, 0.05), rot=(0.9063078, 0, 0.4226183, 0), convention="world"
        ),
    )
    # THIRD-PERSON view, for judging whether a replayed motion looks right -- the head camera
    # only sees the hands when they are in front of the face, and a reaching trajectory takes
    # them out of frame. convention="world" means the camera looks along its own +X with +Z up,
    # so the rotation is the columns [forward, left, up]; this quaternion is a look-at from
    # (2.0, -1.6, 1.5) to (0, 0, 1.05) built by that rule and checked against the kitchens'
    # shipped scene_cam, which it reproduces exactly.
    scene_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam",
        update_period=1 / 20,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 1.0e6)),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(2.0, -1.6, 1.5),
            rot=(0.3297562, -0.0819547, 0.0287482, 0.9400627),
            convention="world",
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )


@configclass
class ActionsCfg:
    """22 dims, in this order: arms(14) torso(6) gripR(1) gripL(1) -- the loader's action layout."""

    arm_action = JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, preserve_order=True, scale=1.0,
        use_default_offset=False,
    )
    # The torso is commanded to its recorded (constant) posture. It must be in the action space:
    # an initialised-but-uncommanded joint has a zero position target and its implicit actuator
    # drags it to zero over the first steps (the head_1 lesson in rby1.py).
    torso_action = JointPositionActionCfg(
        asset_name="robot", joint_names=TORSO_JOINTS, preserve_order=True, scale=1.0,
        use_default_offset=False,
    )
    # Binary: action < 0 closes. The log has a gripper COMMAND but no measured gripper joint,
    # so the grippers are driven for realism and excluded from the fit.
    gripperR_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_finger_r.*"],
        open_command_expr={"gripper_finger_r1": -0.04, "gripper_finger_r2": 0.04},
        close_command_expr={"gripper_finger_r1": 0.0, "gripper_finger_r2": 0.0},
    )
    gripperL_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_finger_l.*"],
        open_command_expr={"gripper_finger_l1": -0.04, "gripper_finger_l2": 0.04},
        close_command_expr={"gripper_finger_l1": 0.0, "gripper_finger_l2": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # One draw per env per arm joint on every reset, parameterised by the DAMPING RATIO.
    # Independent log-uniform K and D make most draws physically impossible (measured: 31% of
    # surviving joints below zeta 0.2, single arms spanning zeta 0.02..20), and the replay
    # objective cannot tell them apart -- so it happily returns an undamped shoulder. Sampling
    # zeta instead keeps every draw a realisable actuator. Reference inertias from
    # scripts/simvla/joint_inertia.py (locked chain about each axis at zero configuration).
    arm_gains = EventTerm(
        func=mdp.randomize_gains_by_damping_ratio,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "joint_inertia": {"right_arm_0": 1.344701, "right_arm_1": 1.505387, "right_arm_2": 0.009777, "right_arm_3": 0.377204, "right_arm_4": 0.003484, "right_arm_5": 0.037383, "right_arm_6": 0.001155, "left_arm_0": 1.344115, "left_arm_1": 1.505387, "left_arm_2": 0.009777, "left_arm_3": 0.377228, "left_arm_4": 0.003484, "left_arm_5": 0.037383, "left_arm_6": 0.001155},
            "stiffness_range": (30.0, 20000.0),
            # 0.2 .. 3.0 covers usefully underdamped through solidly overdamped without
            # admitting the 2%-of-critical corners the old sampling lived in.
            "zeta_range": (0.2, 3.0),
        },
    )


@configclass
class RewardsCfg:
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)


@configclass
class TerminationsCfg:
    # `qpos`, `termination_t`, `json_file` are filled in by systemid.py.
    sim2ruin = DoneTerm(
        func=mdp.sim2ruin_named,
        params={
            "qpos": torch.zeros(1),
            "joint_names": FIT_JOINTS,
            # abs-sum over the 20 fitted joints. The real follower tracks its leader to
            # ~0.02 rad/joint (~0.3 summed); this only prunes gross failures early, the
            # per-env MSE in the json does the ranking.
            "qpos_thres": 2.5,
            # Grace window after each reset: the reset frame's error is one frame of the REAL
            # robot's motion and no gain can change it, so enforcing a tight threshold there
            # rejects every draw. 20 steps = 1 s at the 20 Hz control rate.
            "settle_steps": 20,
        },
    )


@configclass
class Real2SimEnvCfg(ManagerBasedRLEnvCfg):
    scene: Real2SimSceneCfg = Real2SimSceneCfg(num_envs=1024, env_spacing=2.8)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 120 Hz physics, control at 20 Hz -- exactly the kitchens' rates. Implicit PD gains are
        # only meaningful at the physics rate they were identified at.
        self.decimation = 6
        self.episode_length_s = 120.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
