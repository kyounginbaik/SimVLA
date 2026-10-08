"""The grasp manifest's read side. stdlib only; no dataset needed.
Run: cd scripts/simvla && pytest test_grasp_manifest.py -v"""
import json
import os

import pytest

from grasp_manifest import (
    is_stale, load, manifest_path, object_name, stale_warning, strategy_for_mesh, strategy_for_type,
)


def _manifest(**meshes):
    return {"generated_from": "/data", "graspdata_mtime": 0, "aperture_m": 0.09,
            "margin_m": 0.005, "min_candidates": 30, "shrink_floor": 0.7, "meshes": meshes}


def test_object_name_matches_load_grasp_files_own_derivation():
    """load_grasp_file uses input_path.split('/')[-3]; drift here silently mismatches the data."""
    assert object_name("/data/use_data/core_mug_1038e4/mesh/simplified.obj") == "core_mug_1038e4"


def test_manifest_path_sits_beside_the_data():
    assert manifest_path("/data/BODex_obj") == "/data/BODex_obj/grasp_manifest.json"


def test_load_returns_none_when_there_is_no_manifest(tmp_path):
    """A fresh clone without the dataset must still run."""
    assert load(str(tmp_path)) is None


def test_load_returns_none_for_a_corrupt_manifest(tmp_path):
    (tmp_path / "grasp_manifest.json").write_text("{not json", encoding="utf-8")
    assert load(str(tmp_path)) is None


def test_load_reads_a_written_manifest(tmp_path):
    (tmp_path / "grasp_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    assert load(str(tmp_path))["min_candidates"] == 30


def test_strategy_for_mesh_reads_the_entry():
    m = _manifest(core_mug_aaa={"type": "mug", "strategy": "single"})
    assert strategy_for_mesh(m, "/d/use_data/core_mug_aaa/mesh/simplified.obj") == "single"


def test_a_mesh_absent_from_the_manifest_is_unknown():
    assert strategy_for_mesh(_manifest(), "/d/use_data/core_mug_zzz/mesh/simplified.obj") == "unknown"


def test_no_manifest_at_all_is_unknown_not_a_crash():
    assert strategy_for_mesh(None, "/d/use_data/core_mug_aaa/mesh/simplified.obj") == "unknown"
    assert strategy_for_type(None, "mug") == "unknown"


def test_strategy_for_type_agrees_when_every_mesh_agrees():
    m = _manifest(a={"type": "mug", "strategy": "single"}, b={"type": "mug", "strategy": "single"})
    assert strategy_for_type(m, "mug") == "single"


def test_strategy_for_type_is_partial_when_meshes_disagree():
    """toaster is the real case: 2 of 4 meshes have grasps."""
    m = _manifest(a={"type": "toaster", "strategy": "single"},
                  b={"type": "toaster", "strategy": "none"})
    assert strategy_for_type(m, "toaster") == "partial"


def test_a_type_whose_every_mesh_lacks_grasps_is_none():
    m = _manifest(a={"type": "nutella", "strategy": "none"})
    assert strategy_for_type(m, "nutella") == "none"


def test_a_type_with_no_meshes_is_unknown():
    assert strategy_for_type(_manifest(), "mug") == "unknown"


def test_a_manifest_older_than_the_graspdata_is_stale(tmp_path):
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    os.utime(gd, (2_000_000_000, 2_000_000_000))
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime=1_000_000_000)
    assert is_stale(m) is True


def test_a_current_manifest_is_not_stale(tmp_path):
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    os.utime(gd, (1_000_000_000, 1_000_000_000))
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime=1_000_000_000)
    assert is_stale(m) is False


def test_staleness_of_a_vanished_dataset_is_not_an_error(tmp_path):
    m = dict(_manifest(), generated_from=str(tmp_path / "gone"))
    assert is_stale(m) is False


def test_a_hand_edited_non_numeric_graspdata_mtime_is_not_an_error(tmp_path):
    """A manifest is JSON on disk -- someone can hand-edit graspdata_mtime into garbage. is_stale
    must say 'not stale' rather than raise, same as any other unreadable-staleness state."""
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime="not-a-number")
    assert is_stale(m) is False


def test_a_null_graspdata_mtime_is_not_an_error(tmp_path):
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime=None)
    assert is_stale(m) is False


def test_stale_warning_is_none_when_the_manifest_is_current(tmp_path):
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    os.utime(gd, (1_000_000_000, 1_000_000_000))
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime=1_000_000_000)
    assert stale_warning(m) is None


def test_stale_warning_names_both_timestamps_and_the_fix(tmp_path):
    gd = tmp_path / "graspdata_final" / "sim_parallel"
    gd.mkdir(parents=True)
    os.utime(gd, (2_000_000_000, 2_000_000_000))
    m = dict(_manifest(), generated_from=str(tmp_path), graspdata_mtime=1_000_000_000)
    warning = stale_warning(m)
    assert warning is not None
    assert "1000000000" in warning
    assert "2000000000" in warning
    assert "build_grasp_manifest.py" in warning


def test_stale_warning_is_none_without_a_manifest():
    assert stale_warning(None) is None
