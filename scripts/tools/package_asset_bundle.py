#!/usr/bin/env python3
"""Build a deterministic, checksummed archive from a verified staged asset tree."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import zstandard


REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "docs/research-asset-lock.json"
ROOTS = ("assets", "models", "rby1m")
ASSET_README = """# SimVLA external simulator assets

This archive contains the external robot and kitchen inputs used by SimVLA. Verify the archive
SHA-256 before extraction, then verify each extracted file against `BUNDLE-MANIFEST.json`.
The manifest records source revisions and per-file hashes without workstation-specific paths.

The source checkout's BSD-3-Clause license applies to project-authored material where applicable.
Individual upstream assets retain their own terms: for example, Rainbow Robotics SDK material is
Apache-2.0, and NVIDIA Omniverse/Isaac Sim materials remain subject to NVIDIA's license. The
SimVLA maintainer has authorized publication of project-controlled assets; that authorization
does not override upstream third-party terms. Accept NVIDIA's license before using Isaac Sim.

The public BODex object/grasp subset is downloaded separately using the revision-pinned command
in `docs/installation.md`. This bundle is not a trained checkpoint or a collection of episodes.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_asset_files(stage: Path):
    """Yield portable asset paths and resolved sources, excluding VCS/cache metadata."""
    for root_name in ROOTS:
        root = stage / root_name
        if not root.is_dir():
            raise ValueError(f"staged asset root is missing: {root}")
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(name for name in dirs if name not in {".git", "__pycache__"})
            for name in sorted(files):
                if name.startswith(".nfs"):
                    continue
                path = Path(current) / name
                if not path.is_file():
                    raise ValueError(f"unsupported asset entry: {path}")
                relative = PurePosixPath(path.relative_to(stage).as_posix())
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe archive path: {relative}")
                yield relative, path.resolve()


def file_manifest(stage: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(relative),
            "size": source.stat().st_size,
            "mode": stat.S_IMODE(source.stat().st_mode),
            "sha256": sha256(source),
        }
        for relative, source in iter_asset_files(stage)
    ]


def verify_stage_manifest(stage: Path, lock: dict) -> dict:
    try:
        manifest = json.loads((stage / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"staged tree lacks a valid manifest: {stage}") from exc
    for key in ("source_commit", "rby1_source_commit", "sha256",
                "generated_aiworker_urdf_sha256", "relativized_rby1_usd_sha256"):
        if manifest.get(key) != lock.get(key):
            raise ValueError(f"staged tree does not match the asset lock: {key}")
    return manifest


def tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


class HashingReader:
    """Stream a file into tar while detecting changes since manifest creation."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def build_bundle(stage: Path, output: Path) -> dict:
    if zstandard.__version__ != "0.23.0":
        raise ValueError(
            f"asset archives require zstandard==0.23.0 for byte reproducibility; "
            f"found {zstandard.__version__}"
        )
    stage = stage.resolve()
    output = output.resolve()
    if output == stage or stage in output.parents:
        raise ValueError("bundle archive must be outside the staged asset tree")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing asset archive: {output}")
    lock = json.loads(LOCK.read_text())
    staged = verify_stage_manifest(stage, lock)
    files = file_manifest(stage)
    bundle_manifest = {
        "schema": "simvla-assets/v1",
        "source_commit": staged["source_commit"],
        "rby1_source_commit": staged["rby1_source_commit"],
        "locked_inputs_sha256": lock["sha256"],
        "generated_aiworker_urdf_sha256": staged["generated_aiworker_urdf_sha256"],
        "relativized_rby1_usd_sha256": staged["relativized_rby1_usd_sha256"],
        "file_count": len(files),
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    manifest_bytes = (json.dumps(bundle_manifest, sort_keys=True, indent=2) + "\n").encode()
    readme_bytes = ASSET_README.encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="simvla-assets-", dir=output.parent) as temporary:
        temporary_output = Path(temporary) / output.name
        compressor = zstandard.ZstdCompressor(
            level=10, threads=0, write_checksum=True, write_content_size=True
        )
        with temporary_output.open("wb") as compressed:
            with compressor.stream_writer(compressed, closefd=False) as writer:
                with tarfile.open(fileobj=writer, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                    archive.addfile(tar_info("ASSET-README.md", len(readme_bytes), 0o644),
                                    io.BytesIO(readme_bytes))
                    archive.addfile(tar_info("BUNDLE-MANIFEST.json", len(manifest_bytes), 0o644),
                                    io.BytesIO(manifest_bytes))
                    for item in files:
                        relative = PurePosixPath(str(item["path"]))
                        source = (stage / Path(*relative.parts)).resolve()
                        with source.open("rb") as stream:
                            checked = HashingReader(stream)
                            archive.addfile(
                                tar_info(str(relative), int(item["size"]), int(item["mode"])), checked
                            )
                            if checked.digest.hexdigest() != item["sha256"]:
                                raise ValueError(f"asset changed while packaging: {relative}")
        shutil.move(str(temporary_output), output)
    return {
        **bundle_manifest,
        "archive": str(output),
        "archive_sha256": sha256(output),
        "archive_bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True, help="Verified staged asset tree")
    parser.add_argument("--output", type=Path, required=True, help="Output .tar.zst archive")
    args = parser.parse_args()
    try:
        result = build_bundle(args.stage, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: result[key] for key in (
        "file_count", "total_bytes", "archive", "archive_bytes", "archive_sha256"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
