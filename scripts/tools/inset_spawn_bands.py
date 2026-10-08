"""Replace a goal file's spawn bands with ones the robot actually FITS IN.

THE DEFECT. isaaclab/simvla/utils.py:find_free_spaces_grid decomposes the floor into cells
between obstacle boundaries and keeps a cell when its CENTRE POINT is outside every obstacle. It
never inflates the obstacles by the robot's size, so a "free" cell runs right up to the obstacle
that bounds it. Kitchen 1215 emitted two bands 0.111 m and 0.219 m wide in x for a robot whose
base is 0.628 x 0.606 m: it cannot stand in either without overlapping the thing next door.

MEASURED CONSEQUENCE. The robot arrives already in contact and the base cannot move at all --
commanded 0.463 rad/s of yaw, achieved 0.000, all 16 envs frozen on goal step 0 forever. Lifted
5 m clear of the kitchen the same base drives at 95.6%, which is what proves it is contact and
not the drive.

WHAT THIS DOES. Rewrites initial_pos_ranges to a box centred on a spot chosen to clear every
obstacle by a stated margin. It is a WORKAROUND for one campaign, not the fix: the fix is for the
free-space search to inflate obstacles by the robot's footprint, which changes where Anubis and
RB-Y1 spawn too and therefore is not a change to make in passing.

    python scripts/tools/inset_spawn_bands.py --goals <dir> --x 1.0 --y -1.6 --half 0.10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goals", required=True, help="directory of Isaac-Kitchen-*.json")
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--half", type=float, default=0.10, help="half-width of the new band, m")
    a = ap.parse_args()

    files = sorted(Path(a.goals).glob("Isaac-Kitchen-*.json"))
    if not files:
        raise SystemExit(f"no goal files in {a.goals}")
    band = [[["x", a.x - a.half, a.x + a.half], ["y", a.y - a.half, a.y + a.half]]]
    for f in files:
        d = json.loads(f.read_text())
        old = d.get("initial_pos_ranges")
        d["initial_pos_ranges"] = band
        f.write_text(json.dumps(d))
        spans = [f"x{round(r[0][2] - r[0][1], 3)}/y{round(r[1][2] - r[1][1], 3)}" for r in (old or [])]
        print(f"[bands] {f.name}: {len(old or [])} band(s) [{', '.join(spans)}] -> "
              f"1 band centred ({a.x}, {a.y}) +-{a.half}", flush=True)
    print(f"[bands] rewrote {len(files)} goal files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
