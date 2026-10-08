# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure unit tests for robot_schemas.py — no Isaac Sim, no lerobot imports."""

import unittest

from isaaclab.utils.datasets.robot_schemas import (
    ROBOT_SCHEMAS,
    get_robot_schema,
)


class TestRobotSchemas(unittest.TestCase):
    def test_anubis_shapes(self):
        s = get_robot_schema("anubis")
        self.assertEqual(len(s["joint_angles"]), 14)
        self.assertEqual(len(s["action_joint"]), 19)

    def test_rby1_shapes(self):
        s = get_robot_schema("rby1")
        self.assertEqual(len(s["joint_angles"]), 24)
        self.assertEqual(len(s["action_joint"]), 29)

    def test_all_names_are_strings(self):
        for robot, s in ROBOT_SCHEMAS.items():
            for key in ("joint_angles", "action_joint"):
                for name in s[key]:
                    self.assertIsInstance(
                        name, str, f"{robot}.{key} contains non-str: {name!r}"
                    )

    def test_no_duplicate_names_within_a_list(self):
        for robot, s in ROBOT_SCHEMAS.items():
            for key in ("joint_angles", "action_joint"):
                names = s[key]
                self.assertEqual(
                    len(names),
                    len(set(names)),
                    f"{robot}.{key} has duplicates: {names}",
                )

    def test_unknown_robot_raises_with_helpful_message(self):
        # The example used to be "aiworker", which is now a CONFIGURED robot (FFW_SG2). The
        # test is about the unknown-robot path, so it needs a name that will not become real.
        with self.assertRaises(ValueError) as cm:
            get_robot_schema("no_such_robot")
        self.assertIn("no_such_robot", str(cm.exception))
        self.assertIn("robot_schemas.py", str(cm.exception))

    def test_aiworker_is_configured(self):
        schema = get_robot_schema("aiworker")
        # 28 joints in the USD; joint_angles drops the 3 base joints and the two mirror
        # fingers, matching kitchen/mdp/observations.py:joint_angles for this robot.
        self.assertEqual(len(schema["action_joint"]), 28)
        self.assertEqual(len(schema["joint_angles"]), 23)
        self.assertTrue(set(schema["joint_angles"]).issubset(set(schema["action_joint"])))


if __name__ == "__main__":
    unittest.main()
