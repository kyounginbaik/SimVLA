"""Minimal planned skill extension for the existing gripper action channel.

Importing this file registers only the example skill in the current process.
It is not loaded by the public CLI or simulator until a contributor moves the
declaration into the canonical src/simvla/skills.py (see docs/task-authoring.md).
"""

from simvla.skill_contract import skill


@skill(id="gripper.open", actions=("G_r", "G_l"), label="Open gripper", params=[])
class GripperOpen:
    def plan(self, app, action: str, params: dict) -> bool:
        # The existing G_r/G_l executor writes a negative command for False.
        return False
