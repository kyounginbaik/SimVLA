#!/usr/bin/env python3
"""Import BODex grasp-synthesis output into SimVLA's graspdata layout, ready to grasp.

After BODex synthesizes grasps for an object (the `sim_parallel` gripper, left+right), this drops the
`.npy` files into the exact path plan_arm_grasp reads:

    <BODEX_OBJ_DIR>/graspdata_final/sim_parallel/<obj>/floating/
        scale010_grasp_left.npy
        scale010_grasp_right.npy
        scale010_grasp_segments_thumbnails/            (folder must exist; plan_arm_grasp checks)
            selected_indices.json                      (which grasps to use)
    <BODEX_OBJ_DIR>/grasp_manifest.json               (refreshed unless --no-manifest)

The authors' pipeline renders per-grasp thumbnails so a human can pick in a GUI. That renderer is not
in the repo — but the grasp `.npy` carries `grasp_error`, so we can pick the best grasps
PROGRAMMATICALLY (keep the lowest-error fraction, the same ~65% a real mug selection keeps) and write
`selected_indices.json` directly. No rendering needed; the object grasps with the real graspdata logic
immediately, and the choice is still editable later in the composer's Grasps panel.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def select_grasps_by_quality(grasp_error, dist_error=None, keep_fraction: float = 0.65) -> list[int]:
    """Grasp indices to keep, best-first-by-quality then sorted ascending.

    Ranks the N grasps by the L2 norm of `grasp_error` (lower = better force closure); `dist_error`,
    if given, is added in as a secondary term (the fingers actually reaching the surface). Keeps the
    best `keep_fraction` — matching the ~65% a released mug selection retains — and returns those
    indices in ascending order, the order selected_indices.json stores them in.
    """
    ge = np.asarray(grasp_error)
    if ge.ndim == 3:               # (1, N, K) -> (N, K), the on-disk shape
        ge = ge[0]
    score = np.linalg.norm(ge, axis=1)
    if dist_error is not None:
        de = np.asarray(dist_error)
        if de.ndim == 3:
            de = de[0]
        score = score + np.linalg.norm(de, axis=1)
    n = len(score)
    keep = max(1, int(round(n * keep_fraction)))
    best = np.argsort(score)[:keep]
    return sorted(int(i) for i in best)


def _quality_rank(npy_path: Path):
    """All grasp indices, best-first by quality (same metric as select_grasps_by_quality)."""
    d = np.load(npy_path, allow_pickle=True).item()
    ge = np.asarray(d["grasp_error"])
    if ge.ndim == 3:
        ge = ge[0]
    score = np.linalg.norm(ge, axis=1)
    de = d.get("dist_error")
    if de is not None:
        de = np.asarray(de)
        if de.ndim == 3:
            de = de[0]
        score = score + np.linalg.norm(de, axis=1)
    return [int(i) for i in np.argsort(score)]


def _selection_for(npy_path: Path, hand: str, keep_fraction: float) -> list[list]:
    d = np.load(npy_path, allow_pickle=True).item()
    idx = select_grasps_by_quality(d["grasp_error"], d.get("dist_error"), keep_fraction)
    return [[i, hand] for i in idx]


def _split_selection(npy_path: Path, keep_fraction: float) -> list[list]:
    """When ONE synthesis feeds both hands, left and right must use DISJOINT grasp indices — else the
    planner sees every pose twice (eef_data_left[i] == eef_data_right[i]) and its uniqueness check
    (sum != 1) rejects all of them. Rank by quality, keep the best fraction, then interleave: even
    ranks -> left, odd ranks -> right. Both hands get good grasps, and no index is shared."""
    ranked = _quality_rank(npy_path)
    keep = max(2, int(round(len(ranked) * keep_fraction)))
    best = ranked[:keep]
    left = sorted(best[0::2])
    right = sorted(best[1::2])
    return [[i, "left"] for i in left] + [[i, "right"] for i in right]


def import_object(obj_name: str, left_npy, right_npy, bodex_obj_dir, keep_fraction: float = 0.65,
                   *, repick: bool = False) -> Path:
    """Place an object's BODex output into graspdata_final/... and write a quality-ranked selection.
    Returns the floating dir written. Both hands' .npy are required (plan_arm_grasp reads both).

    An existing selected_indices.json is KEPT, not overwritten, unless repick=True. Five to seven
    objects in this dataset carry hand-curated selections (as few as 1 grasp of 200) whose
    provenance is still an open question -- silently replacing one with a quality-ranked pick would
    destroy information nobody can recover. The grasp .npy files themselves are always refreshed;
    only the human's choice of which grasps to use is protected.
    """
    left_npy, right_npy = Path(left_npy), Path(right_npy)
    floating = Path(bodex_obj_dir) / "graspdata_final" / "sim_parallel" / obj_name / "floating"
    floating.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(left_npy, floating / "scale010_grasp_left.npy")
    shutil.copyfile(right_npy, floating / "scale010_grasp_right.npy")

    thumbs = floating / "scale010_grasp_segments_thumbnails"
    thumbs.mkdir(exist_ok=True)     # plan_arm_grasp requires the folder to exist, even without PNGs
    selection_path = thumbs / "selected_indices.json"
    if selection_path.exists() and not repick:
        print(f"kept existing selection: {selection_path} (pass --repick to overwrite)")
        return floating
    if Path(left_npy).resolve() == Path(right_npy).resolve():
        # one synthesis for both hands -> split disjointly so no grasp is used for BOTH left and right
        selected = _split_selection(floating / "scale010_grasp_left.npy", keep_fraction)
    else:
        selected = (_selection_for(floating / "scale010_grasp_left.npy", "left", keep_fraction)
                    + _selection_for(floating / "scale010_grasp_right.npy", "right", keep_fraction))
    selection_path.write_text(
        json.dumps({"selected_indices": selected}, indent=2), encoding="utf-8"
    )
    return floating


def _default_bodex_obj_dir() -> str:
    """The same default build_grasp_manifest.py and grasp_manifest._root() resolve to -- see there.

    Was hardcoded to a fixed path here, ignoring $BODEX_OBJ_DIR: on a machine where that variable
    points elsewhere, this importer would write grasps into the wrong tree while refreshing the
    wrong manifest, even though every reader honours the variable.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_grasp_manifest import default_bodex_obj_dir
    return default_bodex_obj_dir()


def refresh_manifest(bodex_obj_dir) -> Path:
    """Rebuild the grasp manifest for this dataset. Returns the path written.

    Called at the end of an import because an import is exactly what invalidates the manifest:
    every consumer reads a cached file, and a dataset that gained grasps while the manifest said
    otherwise is the one staleness bug this design can have.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_grasp_manifest import write
    return write(str(bodex_obj_dir))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--obj", required=True, help="object dir name (e.g. core_jar_<hash>)")
    ap.add_argument("--left", required=True, help="BODex left-hand grasp .npy")
    ap.add_argument("--right", required=True, help="BODex right-hand grasp .npy")
    ap.add_argument("--bodex-obj-dir", default=_default_bodex_obj_dir())
    ap.add_argument("--keep-fraction", type=float, default=0.65,
                    help="fraction of best-quality grasps to keep (default 0.65)")
    ap.add_argument("--no-manifest", action="store_true",
                    help="skip rebuilding grasp_manifest.json (for batch imports; run "
                         "build_grasp_manifest.py once at the end instead)")
    ap.add_argument("--repick", action="store_true",
                    help="overwrite an existing selected_indices.json with a fresh quality-ranked "
                         "pick (default: keep it -- some selections are hand-curated)")
    args = ap.parse_args()
    out = import_object(args.obj, args.left, args.right, args.bodex_obj_dir, args.keep_fraction,
                         repick=args.repick)
    n = len(json.loads((out / "scale010_grasp_segments_thumbnails" / "selected_indices.json")
                       .read_text())["selected_indices"])
    print(f"imported {args.obj}: {n} grasps selected -> {out}")
    if not args.no_manifest:
        print(f"refreshed {refresh_manifest(args.bodex_obj_dir)}")


if __name__ == "__main__":
    main()
