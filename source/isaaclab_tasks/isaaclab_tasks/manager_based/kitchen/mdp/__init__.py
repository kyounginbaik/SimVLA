# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
# The SceneSmith deploy scenes (scene_001 … scene_207) declare their success/retry terms as
# mdp.pushchair / mdp.OOB_chair / mdp.mug2sink, which live in terminations_other.py — without
# this, every one of those envs dies at import with
# "module ... .mdp has no attribute 'pushchair'".
# Imported BY NAME, never `import *`: terminations_other also defines OOB, make_logger,
# navigation, task1, task2, task3 and task4, and a star-import would shadow terminations.py's
# versions of those for the ~1700 kitchen envs that depend on them.
from .terminations_other import (  # noqa: F401
    OOB_chair,
    OOB_chair_2,
    mug2sink,
    pushchair,
    pushchair_2,
)
# RB-Y1 forms of ee_6d_pos / task1_molmospace. Imported BY NAME, never `import *`, for the
# same reason terminations_other is: a star-import here would risk shadowing the shared
# Anubis versions that ~9,310 kitchen envs depend on.
from .rby1 import ee_6d_pos_rby1, task1_molmospace_rby1  # noqa: F401
from .composed import composed  # noqa: F401
from .contact_gripper import ContactHoldingBinaryJointPositionActionCfg  # noqa: F401
