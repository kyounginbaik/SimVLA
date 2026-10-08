"""Tests for spawn_select: pinning which initial_pos_ranges band a run spawns in.

Run with: pytest scripts/simvla/test_spawn_select.py -v
Stdlib only -- no Omniverse, no scene_synthesizer.
"""

import pytest

from spawn_select import (
    ENV_VAR,
    SpawnSelectError,
    choose_init_pos,
    describe,
    nearest_band,
)

#: The two bands kitchen 1202 actually emits, measured on 2026-08-19: one adjacent to the counter
#: the mug stands on, one 3.07 m away on the far side of the table.
FAR = [["x", 0.87, 1.41], ["y", -4.06, -4.04]]
NEAR = [["x", 1.30, 1.41], ["y", -2.78, -0.69]]
BANDS = [FAR, NEAR]

MUG_XY = (1.90, -0.97)


class _FixedRng:
    """A stand-in for `random` that always returns the first option, so the unpinned path is
    deterministic in a test without reaching for a seed."""

    @staticmethod
    def choice(seq):
        return seq[0]


def test_unset_falls_back_to_the_random_draw():
    idx, band = choose_init_pos(BANDS, env_value="", rng=_FixedRng)
    assert (idx, band) == (0, FAR)


def test_none_env_value_reads_the_environment(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "1")
    idx, band = choose_init_pos(BANDS)
    assert (idx, band) == (1, NEAR)


def test_a_pinned_index_selects_that_band():
    assert choose_init_pos(BANDS, env_value="1") == (1, NEAR)
    assert choose_init_pos(BANDS, env_value="0") == (0, FAR)


def test_a_negative_index_counts_from_the_end():
    idx, band = choose_init_pos(BANDS, env_value="-1")
    assert band is NEAR
    assert idx == 1, "the reported index must be the positive one, for the log line"


def test_an_out_of_range_pin_raises_rather_than_falling_back():
    """Silently choosing at random would look identical in the log while running something else."""
    with pytest.raises(SpawnSelectError) as e:
        choose_init_pos(BANDS, env_value="5")
    assert "2 spawn band(s)" in str(e.value)


def test_a_non_integer_pin_raises():
    with pytest.raises(SpawnSelectError):
        choose_init_pos(BANDS, env_value="near")


def test_no_bands_at_all_raises():
    with pytest.raises(SpawnSelectError):
        choose_init_pos([], env_value="")


def test_nearest_band_finds_the_counter_side_one():
    """How an operator derives the index to pin, without hard-coding one a re-emit could renumber."""
    assert nearest_band(BANDS, MUG_XY) == 1


def test_describe_names_the_band():
    assert describe(1, NEAR) == "[spawn] band 1: x[1.30,1.41] y[-2.78,-0.69]"
