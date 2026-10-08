"""Which of a kitchen's initial_pos_ranges the robot spawns in.

Stdlib only, so it can be unit-tested; simvla_video.py cannot be imported at all (it calls
AppLauncher at module scope), which is why this is its own module rather than a function in there.

WHY THIS EXISTS. simvla_video picks the spawn band with a bare `random.choice` over
`initial_pos_ranges`, ONCE per run -- so every env in a 32-env run shares whichever band that one
draw returned. On a kitchen with a single band that is invisible. On a kitchen with several it
makes the run bimodal: kitchen 1202 (an l_shaped kitchen with a table and two chairs) has two
bands, one adjacent to the counter and one 3.07 m away on the far side of the table, and a run
that draws the far band starts all 32 robots behind the furniture.

The bands come from the free-space search in simvla_data_generator._generate_env_config, which
treats every /world/<name> prim as an obstacle -- so ADDING FURNITURE IS WHAT CREATES THE SECOND
BAND. The coin flip is not a property of the kitchen being badly built; it is what happens the
moment a room has furniture in it.

SIMVLA_INIT_POS_IDX pins the choice. Unset, this is byte-identical to the bare random.choice it
replaces -- same call, same rng, same distribution.
"""

from __future__ import annotations

import os
import random as _random

#: The environment variable. An integer index into initial_pos_ranges; negative indices count from
#: the end, as they do everywhere else in Python. Unset or empty = choose uniformly at random.
ENV_VAR = "SIMVLA_INIT_POS_IDX"


class SpawnSelectError(ValueError):
    """A pin that names no band. Raised rather than silently falling back to random.

    Falling back would be worse than useless here: the caller asked for a REPRODUCIBLE spawn, and
    a silent random one looks identical in the log while producing a different run.
    """


def choose_init_pos(ranges, env_value=None, rng=None):
    """Return (index, range) for the band to spawn in.

    ranges     -- initial_pos_ranges, as loaded from the goal file: a list of bands, each of the
                  form [["x", lo, hi], ["y", lo, hi]].
    env_value  -- the raw SIMVLA_INIT_POS_IDX string. None means read the environment; pass a
                  value explicitly in tests so they do not depend on the ambient environment.
    rng        -- something with .choice, defaulting to the `random` module, so the unpinned path
                  stays exactly the draw simvla_video made before this existed.
    """
    if not ranges:
        raise SpawnSelectError("initial_pos_ranges is empty; this kitchen has nowhere to spawn")

    raw = os.environ.get(ENV_VAR, "") if env_value is None else env_value
    raw = (raw or "").strip()
    if not raw:
        chosen = (rng or _random).choice(ranges)
        return ranges.index(chosen), chosen

    try:
        idx = int(raw)
    except ValueError:
        raise SpawnSelectError(
            f"{ENV_VAR}={raw!r} is not an integer; it indexes initial_pos_ranges "
            f"(0..{len(ranges) - 1})"
        ) from None
    if not -len(ranges) <= idx < len(ranges):
        raise SpawnSelectError(
            f"{ENV_VAR}={idx} is out of range; this kitchen has {len(ranges)} spawn band(s), "
            f"so the valid indices are 0..{len(ranges) - 1} (or -1..-{len(ranges)})"
        )
    return idx % len(ranges), ranges[idx]


def describe(index, band) -> str:
    """One log line naming the band that was chosen, so a run's spawn is recoverable from its log.

    Without this the choice is invisible: two runs of the same command that land in different
    bands look identical until their yields differ.
    """
    try:
        (_, x0, x1), (_, y0, y1) = band[0], band[1]
        return (f"[spawn] band {index}: x[{float(x0):.2f},{float(x1):.2f}] "
                f"y[{float(y0):.2f},{float(y1):.2f}]")
    except Exception:
        return f"[spawn] band {index}: {band!r}"


def nearest_band(ranges, point_xy):
    """The index of the band whose rectangle is closest to `point_xy`.

    Not used to choose anything automatically -- it is how an operator works out what to pass to
    SIMVLA_INIT_POS_IDX for a given target, and how a test states "the counter-adjacent one"
    without hard-coding an index that a re-emit could renumber.
    """
    px, py = float(point_xy[0]), float(point_xy[1])

    def gap(band):
        (_, x0, x1), (_, y0, y1) = band[0], band[1]
        dx = max(float(x0) - px, px - float(x1), 0.0)
        dy = max(float(y0) - py, py - float(y1), 0.0)
        return (dx * dx + dy * dy) ** 0.5

    return min(range(len(ranges)), key=lambda i: gap(ranges[i]))
