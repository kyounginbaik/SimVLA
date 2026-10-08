"""Ten fixed, mirrored above-gripper wrist-camera mounts for visual selection."""
import copy
import math

# Height on the upper side of the hand, setback behind the wrist, downward pitch.
CANDIDATES = (
    (.06, .08, 15), (.08, .10, 20), (.10, .10, 25), (.12, .10, 30),
    (.14, .10, 35), (.08, .16, 15), (.10, .16, 20), (.12, .16, 25),
    (.14, .16, 30), (.16, .16, 35),
)


def add_rby1_wrist_candidates(scene):
    report = {}
    for side, sign in (("right", 1), ("left", -1)):
        source = getattr(scene, f"wrist_{side}")
        for index, (height, setback, pitch) in enumerate(CANDIDATES, 1):
            camera = copy.deepcopy(source)
            name = f"candidate_{index:02d}_{side}"
            camera.prim_path = source.prim_path + f"_above_{index:02d}"
            camera.offset.pos = (0., sign * height, setback)
            angle = math.radians(-sign * pitch) / 2
            # Restore the original right-camera roll; no extra optical-axis flip.
            camera.offset.rot = (math.cos(angle), math.sin(angle), 0., 0.)
            if side == "left":
                # The mirrored left tool has local -Y upward. Compose local Z(pi)
                # so both above-hand views are upright; right keeps its original roll.
                camera.offset.rot = (0., 0., -math.sin(angle), math.cos(angle))
            camera.offset.convention = "opengl"
            setattr(scene, name, camera)
            report[name] = dict(side=side, height_m=height, setback_m=setback,
                                downward_pitch_deg=pitch, pos=camera.offset.pos,
                                rot_wxyz=camera.offset.rot, focal_length=camera.spawn.focal_length)
    return report
