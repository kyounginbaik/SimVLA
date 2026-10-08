import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/release_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("release_evidence", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Release Test"], check=True)
    (path / "tracked.txt").write_text("release source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return path


def test_collect_binds_artifact_to_git_revision(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    artifact = tmp_path / "sample.whl"
    artifact.write_bytes(b"release artifact\n")
    report = _module().collect(repo, [artifact])

    expected_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert report["schema_version"] == 1
    assert report["source"]["commit"] == expected_commit
    assert len(report["source"]["tree"]) == 40
    assert isinstance(report["source"]["dirty"], bool)
    assert report["artifacts"] == [{
        "path": str(artifact),
        "bytes": len(b"release artifact\n"),
        "sha256": hashlib.sha256(b"release artifact\n").hexdigest(),
    }]


def test_cli_writes_report_and_rejects_missing_artifact(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo), "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["source"]["commit"]

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo),
         "--artifact", str(tmp_path / "missing.whl")],
        capture_output=True, text=True,
    )
    assert missing.returncode == 1
    assert "artifact is not a file" in missing.stderr


def test_runtime_records_import_origin_without_importing_gpu_packages(monkeypatch):
    monkeypatch.setenv("SIMVLA_PREGRASP_STANDOFF", ".08")
    module = _module()
    monkeypatch.setenv("SIMVLA_CUROBO_TARGET_COLLISION", "approach")
    monkeypatch.setenv("SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S", ".3")
    monkeypatch.setenv("SIMVLA_PRIVATE_TOKEN", "do-not-record-this")
    monkeypatch.setenv("HF_TOKEN", "nor-this")
    before = set(sys.modules)
    report = module.runtime_environment()
    assert report["settings"]["SIMVLA_PREGRASP_STANDOFF"] == ".08"
    assert report["settings"]["SIMVLA_CUROBO_TARGET_COLLISION"] == "approach"
    assert report["settings"]["SIMVLA_AIWORKER_GRIPPER_SPEED_RAD_S"] == ".3"
    assert "do-not-record-this" not in json.dumps(report)
    assert "nor-this" not in json.dumps(report)
    assert report["packages"]["simvla"]["origin"]
    assert not ({"torch", "isaacsim", "curobo"} - before) & set(sys.modules)


def test_dirty_source_content_is_identified_not_just_filenames(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    (repo / "src").mkdir()
    new = repo / "src/new.py"
    new.write_text("VALUE = 1\n")
    (repo / "tracked.txt").write_text("first edit\n")
    first = _module().collect(repo)["source"]
    new.write_text("VALUE = 2\n")
    (repo / "tracked.txt").write_text("second edit\n")
    second = _module().collect(repo)["source"]
    assert first["changes"] == second["changes"]
    assert first["tracked_diff_sha256"] != second["tracked_diff_sha256"]
    assert first["untracked_source_files"] != second["untracked_source_files"]
    assert "VALUE" not in json.dumps(second)


def test_archive_evidence_does_not_borrow_parent_git_revision(tmp_path):
    parent = _git_repo(tmp_path / "parent")
    archive = parent / "extracted"
    (archive / "src/simvla").mkdir(parents=True)
    (archive / "pyproject.toml").write_text('[project]\nname = "simvla"\n')
    code = archive / "src/simvla/example.py"
    code.write_text("VALUE = 1\n")
    first = _module().collect(archive)["source"]
    assert first["kind"] == "archive"
    assert first["commit"] is None and first["dirty"] is None
    assert len(first["archive_source_files"]) == 2
    code.write_text("VALUE = 2\n")
    assert first["archive_source_files"] != _module().collect(archive)["source"]["archive_source_files"]
    result = subprocess.run([sys.executable, str(SCRIPT), "--repo-root", str(archive), "--require-clean"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "verified clean Git" in result.stderr


def _dataset(root):
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes": 1}')
    (root / "actions.parquet").write_bytes(b"action fixture")
    (root / "front.mp4").write_bytes(b"video fixture")
    return root


def test_dataset_fingerprint_is_complete_and_relocatable(tmp_path):
    module = _module()
    original = _dataset(tmp_path / "original")
    clone = tmp_path / "clone"
    shutil.copytree(original, clone)
    first = module.dataset_fingerprint(original)
    assert len(first["files"]) == 3
    assert first["content_sha256"] == module.dataset_fingerprint(clone)["content_sha256"]
    for name in ("actions.parquet", "front.mp4", "meta/info.json"):
        path = clone / name
        saved = path.read_bytes()
        path.write_bytes(saved + b"changed")
        assert first["content_sha256"] != module.dataset_fingerprint(clone)["content_sha256"]
        path.write_bytes(saved)
    (clone / "extra.bin").write_bytes(b"extra")
    assert first["content_sha256"] != module.dataset_fingerprint(clone)["content_sha256"]


@pytest.mark.parametrize("link_kind", ["file", "directory", "root"])
def test_dataset_fingerprint_rejects_symlinks(tmp_path, link_kind):
    root = _dataset(tmp_path / "dataset")
    if link_kind == "root":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    else:
        (root / "link").symlink_to(root / "meta" if link_kind == "directory"
                                  else root / "front.mp4")
    with pytest.raises(ValueError, match="symlink"):
        _module().dataset_fingerprint(root)


def test_dataset_fingerprint_rejects_changes_during_hash(tmp_path, monkeypatch):
    module = _module()
    root = _dataset(tmp_path / "dataset")
    original = module._sha256
    def mutate(path):
        value = original(path)
        path.write_bytes(path.read_bytes() + b"change")
        return value
    monkeypatch.setattr(module, "_sha256", mutate)
    with pytest.raises(ValueError, match="changed while hashing"):
        module.dataset_fingerprint(root)


def test_dataset_cli_rejects_self_modifying_report(tmp_path):
    root = _dataset(tmp_path / "dataset")
    assert _module().main(["--dataset", str(root), "--output", str(root / "report.json")]) == 1
    assert not (root / "report.json").exists()


def test_dataset_provenance_cli_and_missing_metadata(tmp_path):
    repo = _git_repo(tmp_path / "repo")
    root = _dataset(repo / "dataset")
    module = _module()
    output = tmp_path / "report.json"
    assert module.main(["--repo-root", str(repo), "--dataset", "dataset", "--output", str(output)]) == 0
    assert len(json.loads(output.read_text())["datasets"][0]["files"]) == 3
    assert module.main(["--repo-root", str(repo), "--dataset", "dataset",
                        "--output", str(root / "report.json")]) == 1
    assert not (root / "report.json").exists()
    with pytest.raises(ValueError, match="missing meta/info.json"):
        module.dataset_fingerprint(tmp_path / "missing")
