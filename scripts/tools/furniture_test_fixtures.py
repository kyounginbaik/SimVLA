"""Verify external furniture regression fixtures; download only with --download.

Meshes remain external and retain their individual Objaverse source licences.
The checked-in lock pins the exact bytes used for the geometry measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LOCK = Path(__file__).resolve().parents[2] / "docs/evidence/furniture-test-fixtures.json"


def verify(cache: Path, fixtures: list[dict]) -> list[str]:
    missing = []
    for row in fixtures:
        paths = list((cache / "hf-objaverse-v1/glbs").glob(f"*/{row['uid']}.glb"))
        if not paths:
            missing.append(row["uid"])
            continue
        if len(paths) != 1:
            raise ValueError(f"ambiguous fixture in cache: {row['uid']}")
        data = paths[0].read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"fixture checksum mismatch (not overwritten): {paths[0]}")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    cache = args.cache_root.expanduser().resolve()
    fixtures = json.loads(LOCK.read_text())["fixtures"]
    missing = verify(cache, fixtures)
    if missing and args.download:
        import objaverse
        # Objaverse 0.1.7 exposes no cache argument; match the furniture importer.
        objaverse.BASE_PATH = str(cache)
        objaverse._VERSIONED_PATH = str(cache / "hf-objaverse-v1")
        objaverse.load_objects(missing, download_processes=1)
        missing = verify(cache, fixtures)
    if missing:
        raise SystemExit(f"Missing {len(missing)} fixtures: {', '.join(missing)}; use --download")
    print(f"Verified {len(fixtures)} hash-pinned furniture fixtures in {cache}")


if __name__ == "__main__":
    main()
