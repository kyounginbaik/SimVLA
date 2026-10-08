"""How many simulation steps one episode gets before it is called a timeout.

Stdlib only, so it can be unit-tested. simvla_gen.py launches Omniverse at import and cannot be
imported in a test at all, so the parsing lives here and only the comparison -- `timestep > steps`
-- stays there. Same split, and the same reason, as spawn_select.

WHY THIS EXISTS. The budget was the literal 1500 in `timeout_mask = timestep > 1500`, and a script
that outgrew it did not report having outgrown it: every env simply reset with `reason=timeout`,
which reads exactly like a task that failed. Job 2108770 cost a round to that. Its bimanual
push-chair script -- ten legs, two cuRobo arm plans per chair -- reached step 5 of 10 at t=1500 on
an env whose every completed leg had gone perfectly.

THE DEFAULT IS TIGHT AND THAT IS THE EVIDENCE FOR MAKING IT SETTABLE, not a reason to raise it.
The CHASSIS push-chair script -- six legs, no arm planning at all -- finishes at **1434 of 1500**.
Sixty-six steps, about 3 seconds, a 4% margin. Anything longer than that script does not fit, and
nothing in the old code said so. So the default stays 1500, byte-identical for every task that
already runs, and the horizon becomes something a run states in its own log.

NO UPPER BOUND. An episode budget is a property of the script being run, and this module has no way
to know what that script is; a cap here would be a guess dressed as a check. What guards against a
mistyped extra zero is `describe`, which puts the number in the log where the operator can see it.
"""

from __future__ import annotations

import os

#: The environment variable. A positive integer number of simulation steps. Unset or empty means
#: DEFAULT_STEPS, which is what the code did before this module existed.
ENV_VAR = "SIMVLA_EPISODE_STEPS"

#: The literal that used to be inline at simvla_gen.py's `timeout_mask = timestep > 1500`.
#: Unchanged on purpose: every existing task keeps the horizon it was tuned against.
DEFAULT_STEPS = 1500

#: What the six-leg CHASSIS push-chair script actually takes, measured. It is recorded here rather
#: than in a comment on a job because it is the number that says how much room DEFAULT_STEPS has:
#: 1434 of 1500 is 66 steps, a 4% margin, on the shortest script anyone is still running.
CHASSIS_PUSHCHAIR_STEPS = 1434


class EpisodeBudgetError(ValueError):
    """The variable is set to something that is not a step count. Raised, never fallen back from.

    A silent fall back to 1500 would be worse than useless, for the reason spawn_select gives about
    a pinned spawn band: the operator asked for a specific horizon, and a run that quietly used a
    different one looks identical in the log until its yield does not make sense.
    """


def episode_steps(env_value=None) -> int:
    """How many steps an episode may take before it times out.

    env_value -- the raw SIMVLA_EPISODE_STEPS string. None means read the environment; pass a value
                 explicitly in tests so they do not depend on the ambient environment. An empty or
                 whitespace-only value means "unset", exactly as spawn_select treats its own.

    READ ONCE, AT STARTUP. Not per step: the loop runs at 20-60 Hz and an os.environ lookup per
    frame would put a syscall-shaped cost in the hot path for a value that cannot change, and --
    worse -- a horizon that CAN change mid-run is one that no single log line can honestly report.
    """
    raw = os.environ.get(ENV_VAR, "") if env_value is None else env_value
    raw = (raw or "").strip()
    if not raw:
        return DEFAULT_STEPS

    try:
        steps = int(raw)
    except (TypeError, ValueError):
        raise EpisodeBudgetError(
            f"{ENV_VAR}={raw!r} is not an integer number of simulation steps. It is compared "
            f"against a per-env integer step counter, so 2000 is a value and 2e3, 2000.0 and "
            f"'2000 steps' are not."
        ) from None
    if steps <= 0:
        raise EpisodeBudgetError(
            f"{ENV_VAR}={steps} is not a positive number of steps; every episode would time out on "
            f"its first frame and the run would record nothing."
        )
    return steps


def describe(steps: int, env_value=None) -> str:
    """One log line naming the horizon in use, so a run's timeouts are readable from its own log.

    Without it a timeout is indistinguishable from a failure: the reset says `reason=timeout` and
    nothing anywhere says what the budget was, so a script that simply needed more room reads as a
    script that did not work. That is the round job 2108770 cost.
    """
    raw = os.environ.get(ENV_VAR, "") if env_value is None else env_value
    raw = (raw or "").strip()
    source = f"{ENV_VAR}={raw}" if raw else f"default (set {ENV_VAR} to change)"
    tail = ""
    if steps < CHASSIS_PUSHCHAIR_STEPS:
        tail = (f" -- BELOW the {CHASSIS_PUSHCHAIR_STEPS} the chassis push-chair script takes; "
                f"scripts that long will all time out")
    return f"[episode] budget {steps} steps, {source}{tail}"
