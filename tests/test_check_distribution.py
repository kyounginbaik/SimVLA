import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/check_distribution.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_distribution", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archives(tmp_path, *, version="1.0.0", omit=(), extra=()):
    wheel = tmp_path / "simvla.whl"
    metadata = f"Metadata-Version: 2.4\nName: simvla\nVersion: {version}\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"simvla-{version}.dist-info/METADATA", metadata)

    required = set(_module().REQUIRED_SDIST) - set(omit)
    sdist = tmp_path / "simvla.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for relative in sorted(required | set(extra)):
            data = b"fixture\n"
            info = tarfile.TarInfo(f"simvla-{version}/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return wheel, sdist


def test_accepts_scoped_distribution_with_matching_versions(tmp_path):
    wheel, sdist = _archives(tmp_path)
    report = _module().check(ROOT, wheel, sdist)
    assert report["valid"] is True
    assert report["version"] == "1.0.0"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"omit": {"CITATION.cff"}}, "missing required files"),
        ({"extra": {"scripts/private.py"}}, "contains excluded paths"),
        ({"version": "9.9.9"}, "version mismatch"),
    ],
)
def test_rejects_incomplete_overscoped_or_mismatched_distribution(tmp_path, options, message):
    wheel, sdist = _archives(tmp_path, **options)
    with pytest.raises(ValueError, match=message):
        _module().check(ROOT, wheel, sdist)
