#!/usr/bin/env python3
"""Download the SimVLA BODex preprocessed mesh bundle from HuggingFace.

The dataset ships as 3 tarballs (~2.1 GB compressed, ~6.3 GB extracted):

    graspdata_final.tar.gz  (~1.6 GB)  grasp pose data
    processed_data.tar.gz   (~0.5 GB)  full object set, 2,339 dirs
    use_data.tar.gz         (~28 MB)   curated 129-object kitchen subset

`use_data/` is what `~/.config/scene-synth.datasets/config.yaml` points at —
the set `kitchen_scene_generator.py` picks from. If you only need to place
objects in kitchens, pull just that one tarball.

Usage:
    # Reads destination from $BODEX_OBJ_DIR (see installation.md §11):
    export BODEX_OBJ_DIR=/path/to/your/bodex

    # Full bundle (all 3 tarballs):
    python scripts/tools/download_bodex.py

    # Only the curated kitchen subset:
    python scripts/tools/download_bodex.py --allow-patterns 'use_data.tar.gz'

    # Explicit destination instead of $BODEX_OBJ_DIR:
    python scripts/tools/download_bodex.py --local-dir /path/to/your/bodex

    # Pull from a fork:
    python scripts/tools/download_bodex.py --repo-id <user>/<repo>

After the snapshot finishes, the script extracts each `.tar.gz` in place and
removes the archive. Final layout:

    $BODEX_OBJ_DIR/graspdata_final/...
    $BODEX_OBJ_DIR/processed_data/core_*/{mesh,urdf,info}/...
    $BODEX_OBJ_DIR/use_data/core_*/{mesh,urdf,info}/...

Which matches the `root_dir: ${BODEX_OBJ_DIR}/use_data` line in the
scene-synthesizer config (installation.md §7).
"""

import argparse
import glob
import os
import sys
import tarfile
import time

DEFAULT_REPO_ID = "kyounginbaik/SimVLA-BODex"

# Rate limits can still occur on shared cluster IPs. Keep the outer retry
# loop so an interrupted download can resume without re-fetching files.
OUTER_RETRIES = 20            # ~ a few hours total even at worst case
OUTER_BACKOFF_BASE_SEC = 300  # 5 min between outer retries
OUTER_BACKOFF_MAX_SEC = 900   # cap at 15 min


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    if "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower():
        return True
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 429


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required; install it with "
            "'python -m pip install huggingface_hub'"
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("BODEX_HF_REPO", DEFAULT_REPO_ID),
        help=f"HuggingFace dataset repo (default: {DEFAULT_REPO_ID}; or $BODEX_HF_REPO)",
    )
    parser.add_argument(
        "--local-dir",
        default=os.environ.get("BODEX_OBJ_DIR"),
        help="Download destination. Defaults to $BODEX_OBJ_DIR.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch / tag / commit to fetch (default: main).",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional glob(s) restricting what to fetch (e.g. 'use_data/core_bottle_*/*').",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent file downloads (default: 4 — kept low to ease HF API rate limits).",
    )
    parser.add_argument(
        "--keep-tarballs",
        action="store_true",
        help="Keep the *.tar.gz files after extraction (default: delete to save disk).",
    )
    args = parser.parse_args()

    if not args.local_dir:
        sys.exit(
            "Error: no destination set. Export BODEX_OBJ_DIR or pass --local-dir <path>."
        )

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"[download_bodex] repo: {args.repo_id} @ {args.revision}")
    print(f"[download_bodex] dest: {args.local_dir}")
    if args.allow_patterns:
        print(f"[download_bodex] patterns: {args.allow_patterns}")
    print(f"[download_bodex] max-workers: {args.max_workers}")

    for attempt in range(1, OUTER_RETRIES + 1):
        try:
            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.local_dir,
                allow_patterns=args.allow_patterns,
                max_workers=args.max_workers,
            )
            print(f"[download_bodex] snapshot -> {path}")
            break
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == OUTER_RETRIES:
                raise
            wait = min(OUTER_BACKOFF_BASE_SEC * attempt, OUTER_BACKOFF_MAX_SEC)
            print(
                f"[download_bodex] HF rate-limited (outer attempt {attempt}/{OUTER_RETRIES}); "
                f"sleeping {wait}s before retrying. Already-downloaded files will be skipped.",
                flush=True,
            )
            time.sleep(wait)

    _extract_tarballs(args.local_dir, keep=args.keep_tarballs)
    print("[download_bodex] done")


def _extract_tarballs(local_dir: str, *, keep: bool) -> None:
    """Extract any *.tar.gz files found in `local_dir` in place.

    The dataset is published as 3 tarballs; this transparently unpacks them
    so callers end up with $BODEX_OBJ_DIR/{graspdata_final,processed_data,use_data}/.
    Already-extracted dirs are skipped (idempotent re-runs).
    """
    tarballs = sorted(glob.glob(os.path.join(local_dir, "*.tar.gz")))
    if not tarballs:
        return
    print(f"[download_bodex] found {len(tarballs)} tarball(s) to extract")
    for tb in tarballs:
        name = os.path.basename(tb)
        target = tb[: -len(".tar.gz")]
        if os.path.isdir(target) and os.listdir(target):
            print(f"[download_bodex]   {name}: {os.path.basename(target)}/ already populated, skipping")
            if not keep:
                os.remove(tb)
                print(f"[download_bodex]   {name}: removed (already extracted)")
            continue
        print(f"[download_bodex]   {name}: extracting...")
        with tarfile.open(tb) as tf:
            _safe_extract(tf, local_dir)
        if not keep:
            os.remove(tb)
            print(f"[download_bodex]   {name}: removed (extracted to {os.path.basename(target)}/)")


def _safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    """Extract regular files/directories without allowing archive path traversal."""
    root = os.path.realpath(destination)
    members = archive.getmembers()
    for member in members:
        output = os.path.realpath(os.path.join(root, member.name))
        if os.path.commonpath((root, output)) != root:
            raise ValueError(f"Unsafe archive path: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not supported: {member.name!r}")
        if member.isdev() or member.isfifo():
            raise ValueError(f"Unsafe archive entry type: {member.name!r}")
    if sys.version_info >= (3, 12):
        # Python 3.12+ asks callers to choose a filter explicitly. Paths and entry types were
        # already validated above, so retain the archive metadata after those checks.
        archive.extractall(root, members=members, filter="fully_trusted")
    else:
        archive.extractall(root, members=members)


if __name__ == "__main__":
    main()
