import hashlib
import importlib.util
from pathlib import Path

import pytest


def test_fixture_verifier_reports_missing_and_refuses_changed_bytes(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/tools/furniture_test_fixtures.py"
    spec = importlib.util.spec_from_file_location("furniture_test_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = b"fixture mesh"
    row = {"uid": "1" * 32, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert module.verify(tmp_path, [row]) == [row["uid"]]
    mesh = tmp_path / "hf-objaverse-v1/glbs/000-000" / (row["uid"] + ".glb")
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(data)
    assert module.verify(tmp_path, [row]) == []
    mesh.write_bytes(b"changed mesh")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify(tmp_path, [row])
    assert mesh.read_bytes() == b"changed mesh"
