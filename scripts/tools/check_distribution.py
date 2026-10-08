#!/usr/bin/env python3
"""Check release archive scope and version agreement without installing it."""

from __future__ import annotations

import argparse
from email.parser import Parser
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import zipfile


REQUIRED_SDIST = {
    "README.md", "LICENSE", "CITATION.cff", "docs/releasing.md",
    "examples/README.md", "examples/vqa/vqa.jsonl",
    "tests/test_scene_spec.py", "tests/test_vqa_package.py",
}
FORBIDDEN_SDIST_DIRS = {"source", "scripts", "third_party", ".github", "outputs"}


def _citation_version(path: Path) -> str:
    matches = re.findall(r"(?m)^version:\s*['\"]?([^\s'\"]+)", path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one version field in {path}")
    return matches[0]


def _project_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    project = re.search(r"(?ms)^\[project\]\s*$([\s\S]*?)(?=^\[|\Z)", text)
    if not project:
        raise ValueError(f"missing [project] table in {path}")
    matches = re.findall(r"(?m)^version\s*=\s*['\"]([^'\"]+)['\"]", project.group(1))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one project version in {path}")
    return matches[0]


def check(root: Path, wheel: Path, sdist: Path) -> dict:
    root, wheel, sdist = root.resolve(), wheel.resolve(), sdist.resolve()
    project_version = _project_version(root / "pyproject.toml")
    citation_version = _citation_version(root / "CITATION.cff")

    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one .dist-info/METADATA file")
        wheel_version = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))["Version"]

    with tarfile.open(sdist, "r:*") as archive:
        names = [PurePosixPath(member.name) for member in archive.getmembers() if member.name]
    roots = {name.parts[0] for name in names}
    if len(roots) != 1:
        raise ValueError(f"sdist must have one top-level directory; found {sorted(roots)}")
    prefix = next(iter(roots))
    relative = {str(PurePosixPath(*name.parts[1:])) for name in names if len(name.parts) > 1}
    missing = sorted(REQUIRED_SDIST - relative)
    forbidden = sorted(
        name for name in relative if PurePosixPath(name).parts
        and PurePosixPath(name).parts[0] in FORBIDDEN_SDIST_DIRS
    )
    if missing:
        raise ValueError(f"sdist is missing required files: {', '.join(missing)}")
    if forbidden:
        raise ValueError(f"sdist contains excluded paths: {', '.join(forbidden[:5])}")

    versions = {"pyproject": project_version, "citation": citation_version, "wheel": wheel_version}
    if len(set(versions.values())) != 1:
        raise ValueError("version mismatch: " + ", ".join(f"{key}={value}" for key, value in versions.items()))
    expected_prefix = f"simvla-{project_version}"
    if prefix != expected_prefix:
        raise ValueError(f"sdist prefix {prefix!r} does not match {expected_prefix!r}")
    return {"valid": True, "version": project_version, "versions": versions,
            "sdist_prefix": prefix, "sdist_files": len(relative)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check(args.repo_root, args.wheel, args.sdist), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"check-distribution: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
