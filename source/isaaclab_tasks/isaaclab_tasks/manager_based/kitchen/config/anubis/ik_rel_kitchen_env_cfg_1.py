# Environment configuration for the Anubis robot in the Cabinet task for teleoperation.
  
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis_wheels import ANUBIS_PD_CFG  # isort:skip

right_arm_joint_names = [
    "arm1_base_link_joint",
    "link11_joint",
    "link12_joint",
    "link13_joint",
    "link14_joint",
    "link15_joint",
]

left_arm_joint_names = [
    "arm2_base_link_joint",
    "link21_joint",
    "link22_joint",
    "link23_joint",
    "link24_joint",
    "link25_joint",
]
@configclass
class AnubisKitchenEnvCfg(joint_pos_env_cfg.AnubisKitchenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # We switch here to a stiffer PD controller for IK tracking to be better.
#        self.scene.robot = ANUBIS_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.armR_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            body_name="ee_link1",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_PD_CFG.init_state.joint_pos[name] for name in right_arm_joint_names],
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
        )

        self.actions.armL_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["link2.*", "arm2.*"],
            body_name="ee_link2",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                init_joint_pos=[ANUBIS_PD_CFG.init_state.joint_pos[name] for name in left_arm_joint_names],
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0]),
        )

@configclass
class AnubisKitchenEnvCfg_PLAY(AnubisKitchenEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 6
        # disable randomization for play
        self.observations.policy.enable_corruption = False


