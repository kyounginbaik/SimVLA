"""Simulator-independent validation of force sensor to gripper-joint mappings."""


def resolve_pad_joint_indices(joint_names, pad_joint_names):
    """Resolve two explicit jaw groups without assuming articulation joint order."""
    names = list(joint_names)
    groups = [list(group) for group in pad_joint_names]
    if len(groups) != 2 or not all(groups):
        raise ValueError("force balance requires two nonempty pad_joint_names groups")
    flattened = [name for group in groups for name in group]
    if (len(names) != len(set(names)) or len(flattened) != len(set(flattened))
            or set(flattened) != set(names)):
        raise ValueError("pad_joint_names must cover each controlled joint exactly once")
    return tuple(tuple(names.index(name) for name in group) for group in groups)
