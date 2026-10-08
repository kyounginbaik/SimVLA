#!/usr/bin/env python3
"""Verify every extracted simulator asset against its bundle manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


ASSET_ROOTS = {"assets", "models", "rby1m"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(root: Path) -> dict[str, int]:
    """Validate safe relative paths, exact file set, sizes, and content hashes."""
    root = root.resolve(strict=True)
    try:
        manifest = json.loads((root / "BUNDLE-MANIFEST.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid BUNDLE-MANIFEST.json in {root}") from exc
    if manifest.get("schema") != "simvla-assets/v1":
        raise ValueError(f"unsupported asset manifest schema: {manifest.get('schema')!r}")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("asset manifest 'files' must be a list")

    expected: set[str] = set()
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("asset manifest contains a non-object entry")
        raw = entry.get("path")
        if not isinstance(raw, str):
            raise ValueError("asset manifest entry lacks a relative path")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError(f"unsafe bundle path: {raw!r}")
        if relative.parts[0] not in ASSET_ROOTS or relative.as_posix() != raw:
            raise ValueError(f"unexpected bundle path: {raw!r}")
        if raw in expected:
            raise ValueError(f"duplicate bundle path: {raw}")
        expected.add(raw)
        path = root.joinpath(*relative.parts)
        parent = path.parent
        while parent != root:
            if parent.is_symlink():
                raise ValueError(f"unexpected symlink in bundle path: {parent.relative_to(root)}")
            parent = parent.parent
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or non-regular bundle file: {raw}")
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"bundle path escapes extraction root: {raw}")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid size for bundle file: {raw}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 for bundle file: {raw}")
        if path.stat().st_size != size:
            raise ValueError(f"size mismatch for {raw}: expected {size}, found {path.stat().st_size}")
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {raw}: expected {digest}, found {actual}")
        total_bytes += size

    actual_paths: set[str] = set()
    for root_name in ASSET_ROOTS:
        asset_root = root / root_name
        if not asset_root.is_dir() or asset_root.is_symlink():
            raise ValueError(f"asset directory missing or not a directory: {root_name}")
        for current, dirs, files in os.walk(asset_root, followlinks=False):
            current_path = Path(current)
            for name in dirs:
                child = current_path / name
                if child.is_symlink():
                    raise ValueError(f"unexpected symlink in bundle: {child.relative_to(root)}")
            for name in files:
                child = current_path / name
                if child.is_symlink() or not child.is_file():
                    raise ValueError(f"unexpected non-regular file in bundle: {child.relative_to(root)}")
                actual_paths.add(child.relative_to(root).as_posix())
    if actual_paths != expected:
        missing = sorted(expected - actual_paths)[:5]
        extra = sorted(actual_paths - expected)[:5]
        raise ValueError(f"manifest file set differs (missing={missing}, extra={extra})")
    if manifest.get("file_count") != len(expected):
        raise ValueError("manifest file_count does not match its entries")
    if manifest.get("total_bytes") != total_bytes:
        raise ValueError("manifest total_bytes does not match its entries")
    return {"file_count": len(expected), "total_bytes": total_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted asset bundle directory")
    args = parser.parse_args()
    try:
        result = verify_bundle(args.root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"verified": True, **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
