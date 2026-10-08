"""Checks that external asset packages omit VCS metadata and carry exact file identities."""

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/package_asset_bundle.py"
_SPEC = importlib.util.spec_from_file_location("package_asset_bundle", _SCRIPT)
bundle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle)


def test_bundle_manifest_is_relative_and_skips_git(tmp_path):
    stage = tmp_path / "stage"
    (stage / "assets/.git").mkdir(parents=True)
    (stage / "assets/.git/config").write_text("do not publish")
    (stage / "assets/Kitchen").mkdir()
    (stage / "assets/Kitchen/kitchen.usd").write_bytes(b"kitchen")
    (stage / "models").mkdir()
    (stage / "models/robot.urdf").write_text("robot")
    (stage / "rby1m").mkdir()
    (stage / "rby1m/robot.usd").write_bytes(b"rby1")

    files = bundle.file_manifest(stage)

    assert [item["path"] for item in files] == [
        "assets/Kitchen/kitchen.usd", "models/robot.urdf", "rby1m/robot.usd"
    ]
    assert all(not Path(item["path"]).is_absolute() for item in files)
    assert all(len(item["sha256"]) == 64 for item in files)
