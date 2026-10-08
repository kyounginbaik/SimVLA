"""Checks for the local research-asset staging boundary."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/stage_research_assets.py"
_SPEC = importlib.util.spec_from_file_location("stage_research_assets", _SCRIPT)
stage_assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_assets)


def test_lock_rejects_a_changed_asset_before_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "robot.usd"
    asset.write_bytes(b"pinned")
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    lock = {"sha256": {"assets/robot.usd": expected}}
    stage_assets.verify_lock({"assets": source}, lock)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_assets.verify_lock({"assets": source}, lock)


def test_source_revisions_are_checked_and_returned(tmp_path):
    roots = [tmp_path / "source", tmp_path / "rby1"]
    revisions = []
    for root in roots:
        root.mkdir()
        import subprocess

        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
        subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
        (root / "input.txt").write_text("pinned")
        subprocess.run(["git", "-C", str(root), "add", "input.txt"], check=True)
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
        revisions.append(stage_assets.repository_head(root))

    lock = {"source_commit": revisions[0], "rby1_source_commit": revisions[1]}
    assert stage_assets.verify_revisions(*roots, lock) == lock
    lock["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="revision mismatch"):
        stage_assets.verify_revisions(*roots, lock)


def test_link_tree_keeps_generated_paths_local(tmp_path):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / ".git").mkdir()
    (source / "nested/source.txt").write_text("source")
    (source / "nested/generated.txt").write_text("old")
    (source / ".git/config").write_text("local repository metadata")
    output = tmp_path / "output"
    count = stage_assets.link_tree(source, output, skip={Path("nested/generated.txt")})
    assert count == 1
    assert (output / "nested/source.txt").is_symlink()
    assert not (output / ".git").exists()
    assert not (output / "nested/generated.txt").exists()
    (output / "nested/generated.txt").write_text("new")
    assert (source / "nested/generated.txt").read_text() == "old"


def test_env_file_quotes_paths_with_spaces(tmp_path):
    text = stage_assets.env_text(tmp_path / "asset stage", None)
    assert "SIMVLA_ASSETS_DIR='" in text
    assert "asset stage/assets'" in text
