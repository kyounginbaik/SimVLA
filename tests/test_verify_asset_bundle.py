"""Checks extracted asset bundle manifests before robot simulation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/verify_asset_bundle.py"
_SPEC = importlib.util.spec_from_file_location("verify_asset_bundle", _SCRIPT)
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


def make_bundle(root: Path) -> Path:
    (root / "assets/Kitchen").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "rby1m/model").mkdir(parents=True)
    contents = {
        "assets/Kitchen/kitchen.usd": b"kitchen",
        "models/robot.urdf": b"robot",
        "rby1m/model/model.usd": b"rby1",
    }
    entries = []
    for relative, content in contents.items():
        path = root / relative
        path.write_bytes(content)
        entries.append({
            "path": relative,
            "size": len(content),
            "mode": 0o644,
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    (root / "BUNDLE-MANIFEST.json").write_text(json.dumps({
        "schema": "simvla-assets/v1",
        "file_count": len(entries),
        "total_bytes": sum(item["size"] for item in entries),
        "files": entries,
    }))
    return root


def test_verifies_complete_extracted_bundle(tmp_path):
    root = make_bundle(tmp_path / "bundle")

    assert verify.verify_bundle(root) == {"file_count": 3, "total_bytes": 16}


def test_rejects_tampered_file(tmp_path):
    root = make_bundle(tmp_path / "bundle")
    (root / "models/robot.urdf").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify.verify_bundle(root)


def test_rejects_unmanifested_file(tmp_path):
    root = make_bundle(tmp_path / "bundle")
    (root / "assets/Kitchen/untracked.usd").write_bytes(b"extra")

    with pytest.raises(ValueError, match="file set differs"):
        verify.verify_bundle(root)


def test_rejects_path_traversal(tmp_path):
    root = make_bundle(tmp_path / "bundle")
    manifest_path = root / "BUNDLE-MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "assets/../../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unsafe bundle path"):
        verify.verify_bundle(root)
