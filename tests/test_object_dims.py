"""object_dims holds the pure size tables + real-world scaling, extracted from kitchen_build so
they import without booting Omniverse. Run: cd scripts/simvla && pytest test_object_dims.py -v
"""
from simvla.object_dims import OBJECT_BASE_DIMS, REAL_HEIGHT, _up_axis_index, real_placement_dims


def test_up_axis_is_the_odd_one_out_for_a_surface_of_revolution():
    assert _up_axis_index((0.098, 0.098, 0.148)) == 2   # cup: x≈y, z tall
    assert _up_axis_index((0.067, 0.179, 0.067)) == 1   # can: x≈z, y tall


def test_real_placement_scales_uniformly_to_real_height_preserving_aspect():
    # can real height 0.12; mesh up-axis 0.179; preserve aspect ratio
    placed = real_placement_dims("can", (0.067, 0.179, 0.067))
    assert placed[1] == pytest.approx(0.12, abs=1e-6)
    assert placed[0] == pytest.approx(0.067 * (0.12 / 0.179), abs=1e-6)


def test_types_without_a_real_height_keep_their_fallback_dims():
    base = OBJECT_BASE_DIMS["bowl"]        # graspable, no REAL_HEIGHT entry
    assert real_placement_dims("bowl", base) == list(base)


import pytest  # noqa: E402  (kept at bottom to mirror the existing suite's import style)
