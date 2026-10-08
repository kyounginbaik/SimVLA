"""Tests for the standalone BODex acquisition helper."""

import importlib.util
import io
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_downloader():
    path = ROOT / "scripts/tools/download_bodex.py"
    spec = importlib.util.spec_from_file_location("simvla_download_bodex_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(path: Path, name: str, *, linkname: str | None = None):
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        if linkname is None:
            body = b"mesh"
            item.size = len(body)
            archive.addfile(item, io.BytesIO(body))
        else:
            item.type = tarfile.SYMTYPE
            item.linkname = linkname
            archive.addfile(item)


def test_extracts_normal_archive_and_removes_tarball(tmp_path):
    downloader = _load_downloader()
    bundle = tmp_path / "use_data.tar.gz"
    _archive(bundle, "use_data/core_mug/mesh/simplified.obj")

    downloader._extract_tarballs(str(tmp_path), keep=False)

    assert (tmp_path / "use_data/core_mug/mesh/simplified.obj").read_bytes() == b"mesh"
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("name", "linkname"),
    [("../escape.txt", None), ("use_data/link", "../../escape.txt")],
)
def test_rejects_unsafe_archive_entries(tmp_path, name, linkname):
    downloader = _load_downloader()
    bundle = tmp_path / "use_data.tar.gz"
    _archive(bundle, name, linkname=linkname)

    with pytest.raises(ValueError, match="Unsafe archive path|Archive links"):
        downloader._extract_tarballs(str(tmp_path), keep=True)
    assert not (tmp_path.parent / "escape.txt").exists()
