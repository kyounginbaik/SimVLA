"""The planning hand must match USD frame rotations, not just body centres."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_aiworker", ROOT / "scripts/tools/build_aiworker_urdf.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def generated(tmp_path):
    source = ET.Element("robot", name="fixture")
    for name in sorted(builder.KEEP_LINKS):
        ET.SubElement(source, "link", name=name)
    for name in sorted(builder.KEEP_JOINTS):
        joint = ET.SubElement(source, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link="arm_base_link")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    src, dst = tmp_path / "source.urdf", tmp_path / "generated.urdf"
    ET.ElementTree(source).write(src)
    builder.main(["--source", str(src), "--output", str(dst)])
    return ET.parse(dst).getroot()


@pytest.mark.parametrize("side", ["l", "r"])
def test_all_jaw_frames_match_measured_usd_in_tool_frame(tmp_path, side):
    root = generated(tmp_path)
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}

    def pose(link):
        if link == f"arm_{side}_link7":
            return np.eye(4)
        joint = joints[link]
        origin = joint.find("origin")
        roll, pitch, yaw = map(float, origin.get("rpy").split())
        assert pitch == yaw == 0
        matrix = np.eye(4)
        c, s = np.cos(roll), np.sin(roll)
        matrix[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        matrix[:3, 3] = list(map(float, origin.get("xyz").split()))
        return pose(joint.find("parent").get("link")) @ matrix

    ee = np.linalg.inv(pose("ee_link2" if side == "l" else "ee_link1"))
    expected = {"base": [0, 0, -.1], "r1": [0, .008, -.052],
                "l1": [0, -.008, -.052], "r2": [0, .0573634, -.0235],
                "l2": [0, -.0573634, -.0235]}
    for suffix, position in expected.items():
        result = ee @ pose(f"gripper_{side}_rh_p12_rn_{suffix}")
        np.testing.assert_allclose(result[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result[:3, 3], position, atol=1e-8)


def test_research_lock_identifies_corrected_generator_output():
    lock = json.loads((ROOT / "docs/research-asset-lock.json").read_text())
    assert lock["generated_aiworker_urdf_sha256"] == "796dc31d6d3d7e607e7ba5662cbea742906268393a7bb9758e6051d79ac21fe5"
