"""Emit an FFW_SG2 URDF for cuRobo, from the BG2 URDF plus the SG2 USD's own jaw geometry.

WHY THIS EXISTS. cuRobo and grasp_clearance both parse a URDF, and there is no SG2 one in the
tree -- only `ffw_bg2_rev5_follower/ffw_bg2_follower.urdf`. That file carries THIS ARM exactly:
every arm and head link of the two variants agrees to 3.7e-9 m when both stages are resolved
against arm_base_link. What it does not carry is this robot's HAND: BG2 wears a 3-finger
dexterous hand, SG2 an RH-P12-RN parallel jaw.

So this script keeps the arm chain verbatim and grafts on the jaw, at offsets measured off
Robots/MM/aiworker/ffw_sg2.usd:

    gripper_?_rh_p12_rn_base   (0,       0,       -0.078 ) from arm_?_link7, Rx(pi)
    gripper_?_rh_p12_rn_r1     (0,      +0.008,   +0.048 ) from the rotated base
    gripper_?_rh_p12_rn_l1     (0,      -0.008,   +0.048 )
    gripper_?_rh_p12_rn_r2     (0,      +0.0493634,+0.0285) from r1
    gripper_?_rh_p12_rn_l2     (0,      -0.0493634,+0.0285) from l1

This reproduces both positions AND orientations. The original position-only
graft omitted Rx(pi) and used negated offsets, hiding a 180-degree collision-shape
error behind apparently correct finger centres and wrist FK.

THE JAWS ARE MODELLED AS PRISMATIC, AND THAT IS A DELIBERATE ABSTRACTION. The real RH-P12-RN
is a four-bar linkage driven by revolute joints -- that is what the USD simulates and what
AIWORKER_CFG actuates. But this URDF is a PLANNING model, evaluated only at a locked joint
value, and two consumers need the jaws to be prismatic:

  * grasp_clearance.tool_spheres finds the pinch point as the mean of the spheres on links
    that are "children of a LOCKED PRISMATIC joint" (grasp_clearance.py:155). With revolute
    fingers that set is empty, the pinch point silently falls back to the centroid of every
    tool sphere, and every clearance raise is biased by roughly the 0.17 m from link7 to the
    fingertips -- in the direction that drives the hand INTO the counter.
  * a locked joint of any type makes rel_to_ee treat the chain as rigid, which is what gets
    the finger spheres included at all.

Nothing dynamic reads this file. The simulation uses the USD's revolute linkage unchanged.

    python scripts/tools/build_aiworker_urdf.py \
      --source /path/to/ai_worker_min/.../ffw_bg2_follower.urdf \
      --output /path/to/ai_worker_min/.../ffw_sg2_follower.urdf

Defaults preserve the original in-repository source/output locations. Pure stdlib XML -- no
pxr, no Isaac, no GPU.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "ai_worker_min/ffw_description/urdf/ffw_bg2_rev5_follower/ffw_bg2_follower.urdf"
DST = REPO / "ai_worker_min/ffw_description/urdf/ffw_sg2_follower/ffw_sg2_follower.urdf"

#: Everything the arm chain needs, and nothing else. The zedm_* camera frames and the whole
#: dexterous hand go: cuRobo plans the arm, and the tool geometry is supplied as spheres.
#: base_link is the ROOT, not `world`: the source URDF's `world` link and its `world_fixed`
#: joint are both commented out (lines 7-12), so base_link is already free. That is also the
#: root cuRobo's configs name, and the frame nav's base pose is expressed in.
KEEP_LINKS = (
    {"base_link", "arm_base_link", "head_link1", "head_link2"}
    | {f"arm_{s}_link{i}" for s in "lr" for i in range(1, 8)}
)
KEEP_JOINTS = (
    {"lift_joint", "head_joint1", "head_joint2"}
    | {f"arm_{s}_joint{i}" for s in "lr" for i in range(1, 8)}
)

#: (child, parent, xyz, joint type, axis) for the grafted jaw, per side. Measured on the USD.
JAW = [
    ("gripper_{s}_rh_p12_rn_base", "arm_{s}_link7", (0.0, 0.0, -0.078), "fixed", None),
    ("gripper_{s}_rh_p12_rn_r1", "gripper_{s}_rh_p12_rn_base", (0.0, 0.008, 0.048), "fixed", None),
    ("gripper_{s}_rh_p12_rn_l1", "gripper_{s}_rh_p12_rn_base", (0.0, -0.008, 0.048), "fixed", None),
    ("gripper_{s}_rh_p12_rn_r2", "gripper_{s}_rh_p12_rn_r1", (0.0, 0.0493634, 0.0285),
     "prismatic", (0.0, 1.0, 0.0)),
    ("gripper_{s}_rh_p12_rn_l2", "gripper_{s}_rh_p12_rn_l1", (0.0, -0.0493634, 0.0285),
     "prismatic", (0.0, -1.0, 0.0)),
]

#: Joint names the cuRobo configs lock the jaws at. 0.0 is the geometry as measured, i.e. the
#: pose the USD ships in, so a locked value of 0 reproduces the asset exactly.
JAW_JOINT = "gripper_{s}_jaw_{f}"

#: THE TOOL FRAME, COPIED FROM THE USD RATHER THAN INVENTED.
#:
#: The MM USD ALREADY CARRIES ee_link1 and ee_link2 as rigid bodies, and simvla_video.py looks
#: them up by name (find_bodies("ee_link1")). So this URDF does not get to choose where the
#: tool frame is -- if it disagrees with the USD, cuRobo plans to one frame while the
#: simulation reads another, and the hand misses by the difference. Measured off the USD:
#:
#:     ee_link1 rel arm_r_link7 : t = (0, 0, -0.178), R = 180 degrees about X
#:
#: The 180-degree roll is the important half. It makes the ee frame's +Z point along link7's
#: -Z, i.e. ALONG THE APPROACH -- the Anubis convention, not RB-Y1's. An earlier version of
#: this file put the frame at -0.1620 with no rotation, on the reasoning that the contact patch
#: is the natural pinch; that measurement is still right (the pads' inner faces span z in
#: [-0.1945, -0.1519] and centre at -0.1620) but it is not this robot's tool frame, and
#: skills.GRASP_TOOL_FRAME had to be re-derived once the real one was read.
#:
#: In these coordinates the contact patch sits at (0, 0, -0.016) -- alongside Anubis's +0.0072
#: and RB-Y1's +0.0050, which is the check that says the frame is in the right place.
EE_LINK = {"r": "ee_link1", "l": "ee_link2"}
EE_Z = -0.178
EE_RPY = "3.141592653589793 0 0"


#: Joint origins that SG2 authors differently from BG2, read off
#: source/isaaclab_assets/data/Robots/MM/aiworker/ffw_sg2.usd (UsdPhysics.Joint localPos0).
#: Verified by scripts/tools/check_aiworker_urdf_vs_usd.py.
SG2_JOINT_ORIGINS = {
    "lift_joint": (-0.0199, 0.0, 1.4316),
}


def _inertial(mass=0.05, i=1e-4):
    el = ET.Element("inertial")
    ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(el, "mass", value=str(mass))
    ET.SubElement(el, "inertia", ixx=str(i), ixy="0", ixz="0", iyy=str(i), iyz="0", izz=str(i))
    return el


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SRC,
                        help=f"BG2 source URDF (default: {SRC})")
    parser.add_argument("--output", type=Path, default=DST,
                        help=f"Generated SG2 planning URDF (default: {DST})")
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"missing source URDF: {source}")
    root = ET.parse(source).getroot()

    out = ET.Element("robot", name="ffw_sg2_follower")
    # No "--" anywhere in this text: a double hyphen is illegal inside an XML comment and
    # makes the whole URDF unparseable, which is exactly how the first generated file failed.
    out.append(ET.Comment(
        " GENERATED by scripts/tools/build_aiworker_urdf.py. Do not hand edit.\n"
        "     Arm and head chain copied verbatim from ffw_bg2_follower.urdf (bit identical\n"
        "     between the two variants); RH-P12-RN jaw grafted on from the SG2 USD's own\n"
        "     measurements. The jaws are PRISMATIC here on purpose; see the script's\n"
        "     docstring. This is a planning model, and nothing dynamic reads it. "
    ))

    kept_links = 0
    for link in root.findall("link"):
        if link.get("name") in KEEP_LINKS:
            out.append(link)
            kept_links += 1
    kept_joints = 0
    for joint in root.findall("joint"):
        if joint.get("name") in KEEP_JOINTS:
            out.append(joint)
            kept_joints += 1

    # SG2'S OWN JOINT ORIGINS, WHERE THEY DIFFER FROM BG2'S.
    #
    # The arm and head chains are bit identical between the two variants -- but that was measured
    # AGAINST arm_base_link, and arm_base_link's own placement on the lift column is NOT the same
    # robot to robot. BG2 puts it at x = +0.0055; SG2's USD authors lift_joint localPos0 =
    # (-0.0199, 0, 1.4316). Copying BG2's joint wholesale therefore shifted the ENTIRE planning
    # model 25.4 mm in +x against the robot cuRobo was planning for.
    #
    # That is half of simvla_gen's 0.05 m "A_r not reached to goal" gate spent before the arm
    # moves, on every plan, in every episode. Measured with
    # scripts/tools/check_aiworker_urdf_vs_usd.py, which compares base_link -> ee_link1 forward
    # kinematics computed from each description and is what found this.
    for joint in out.findall("joint"):
        override = SG2_JOINT_ORIGINS.get(joint.get("name"))
        if override is None:
            continue
        o = joint.find("origin")
        if o is None:
            o = ET.SubElement(joint, "origin", rpy="0 0 0")
        was = o.get("xyz")
        o.set("xyz", " ".join(repr(v) for v in override))
        print(f"[urdf] {joint.get('name')}: origin xyz ({was}) -> ({o.get('xyz')})  "
              f"[SG2 USD, not BG2]", flush=True)

    missing_l = KEEP_LINKS - {l.get("name") for l in out.findall("link")}
    missing_j = KEEP_JOINTS - {j.get("name") for j in out.findall("joint")}
    if missing_l or missing_j:
        raise SystemExit(f"source URDF is missing links {sorted(missing_l)} "
                         f"joints {sorted(missing_j)}")

    grafted = 0
    for side in ("r", "l"):
        for child_t, parent_t, xyz, jtype, axis in JAW:
            child, parent = child_t.format(s=side), parent_t.format(s=side)
            link = ET.SubElement(out, "link", name=child)
            link.append(_inertial())
            finger = child.rsplit("_", 1)[-1]
            jname = (JAW_JOINT.format(s=side, f=finger) if jtype == "prismatic"
                     else f"{child}_fixed")
            j = ET.SubElement(out, "joint", name=jname, type=jtype)
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=child)
            rpy = "3.141592653589793 0 0" if child.endswith("_base") else "0 0 0"
            ET.SubElement(j, "origin", xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}", rpy=rpy)
            if jtype == "prismatic":
                ET.SubElement(j, "axis", xyz=f"{axis[0]} {axis[1]} {axis[2]}")
                # The travel a real RH-P12-RN pad has. Only the LOCKED value is ever evaluated,
                # but a limit of zero would make some URDF parsers drop the joint entirely.
                ET.SubElement(j, "limit", lower="0.0", upper="0.04", effort="100",
                              velocity="1.0")
            grafted += 1

        # The tool frame itself, fixed to link7 at the measured contact patch.
        ee = EE_LINK[side]
        link = ET.SubElement(out, "link", name=ee)
        link.append(_inertial(mass=1e-3, i=1e-6))
        j = ET.SubElement(out, "joint", name=f"{ee}_fixed", type="fixed")
        ET.SubElement(j, "parent", link=f"arm_{side}_link7")
        ET.SubElement(j, "child", link=ee)
        ET.SubElement(j, "origin", xyz=f"0.0 0.0 {EE_Z}", rpy=EE_RPY)
        grafted += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(out, space="  ")
    destination.write_text('<?xml version="1.0"?>\n' + ET.tostring(out, encoding="unicode") + "\n")
    print(f"[urdf] kept {kept_links} links / {kept_joints} joints from BG2, "
          f"grafted {grafted} jaw links", flush=True)
    print(f"[urdf] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
