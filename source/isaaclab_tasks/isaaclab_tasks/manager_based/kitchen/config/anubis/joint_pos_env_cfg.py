from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.kitchen import mdp

from isaaclab_tasks.manager_based.kitchen.kitchen_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    KitchenEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip

@configclass
class AnubisKitchenEnvCfg(KitchenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set anubis as robot
        self.scene.robot = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (anubis)
        self.actions.armR_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "arm1_base_link_joint": (-0.523599, 0.523599),
                "link11_joint": (-0.523599, 1.91986),
                "link12_joint": (0.174533, 2.79253),
                "link13_joint": (-1.5708, 1.74533),
                "link14_joint": (-1.5708, 1.57085),
                "link15_joint": (-1.74533, 1.74533),
            }
        )
        self.actions.armL_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link2.*","arm2.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "arm2_base_link_joint": (-0.523599, 0.523599),
                "link21_joint": (-0.523599, 1.91986),
                "link22_joint": (0.174533, 2.79253),
                "link23_joint": (-1.5708, 1.74533),
                "link24_joint": (-1.5708, 1.57085),
                "link25_joint": (-1.74533, 1.74533),
            }
        )
        self.actions.gripperR_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper1.*"],
            open_command_expr={"gripper1.*": 0.04},
            close_command_expr={"gripper1.*": 0.0},
        )

        self.actions.gripperL_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper2.*"],
            open_command_expr={"gripper2.*": 0.04},
            close_command_expr={"gripper2.*": 0.0},
        )

        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["Omni.*"],
        )
        


        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_R_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link1",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )
        self.scene.ee_L_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link2",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )

@configclass
class AnubisKitchenEnvCfg_PLAY(AnubiskitchenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 7
        # disable randomization for play
        self.observations.policy.enable_corruption = False
