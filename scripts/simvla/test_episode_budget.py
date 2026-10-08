"""episode_budget: the per-episode step horizon, its parsing, and its refusals.

Run with: pytest scripts/simvla/test_episode_budget.py -q

The comparison itself -- `timestep > episode_steps` -- lives in simvla_gen.py, which launches
Omniverse at import and cannot be imported here at all. So the arithmetic is tested directly and
the WIRING is checked statically, off the source, the way test_skill_dispatch checks that
simvla_gen imports `skills`.
"""

import ast
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import episode_budget as eb  # noqa: E402


# ---------------------------------------------------------------------------------------------
# The default, and why it is the number it is
# ---------------------------------------------------------------------------------------------

def test_unset_is_the_literal_that_used_to_be_inline():
    """Byte-identical for every task that already runs. This is the whole compatibility promise:
    the change is meant to make the horizon SETTABLE, not to move it."""
    assert eb.DEFAULT_STEPS == 1500
    assert eb.episode_steps("") == 1500
    assert eb.episode_steps("   ") == 1500
    assert eb.episode_steps(None if False else "") == 1500


def test_unset_means_the_environment_when_no_value_is_passed(monkeypatch):
    """env_value=None is 'read the environment'; a test passes a value so it does not depend on the
    ambient one. Both paths are exercised, because the production call site uses neither default."""
    monkeypatch.delenv(eb.ENV_VAR, raising=False)
    assert eb.episode_steps() == eb.DEFAULT_STEPS
    monkeypatch.setenv(eb.ENV_VAR, "2750")
    assert eb.episode_steps() == 2750


def test_the_default_has_only_a_four_percent_margin_on_the_shortest_script_still_run():
    """The evidence for making this settable at all, kept next to the number it justifies.

    The six-leg CHASSIS push-chair script finishes at 1434 of 1500 -- 66 steps, about 3 seconds. A
    ten-leg bimanual script with two cuRobo arm plans per chair does not fit, and job 2108770
    reached step 5 of 10 at t=1500 with every completed leg perfect.
    """
    assert eb.CHASSIS_PUSHCHAIR_STEPS == 1434
    margin = eb.DEFAULT_STEPS - eb.CHASSIS_PUSHCHAIR_STEPS
    assert margin == 66
    assert margin / eb.DEFAULT_STEPS < 0.05, "the default is not tight; re-read why this is settable"


# ---------------------------------------------------------------------------------------------
# A pin is used verbatim
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [("3000", 3000), (" 3000 ", 3000), ("1", 1),
                                          ("1500", 1500), ("100000", 100000)])
def test_a_pinned_value_is_used_exactly(raw, expected):
    assert eb.episode_steps(raw) == expected


def test_there_is_no_upper_bound_and_that_is_deliberate():
    """A cap here would be a guess dressed as a check: this module has no way to know what script
    is being run. What guards a mistyped extra zero is describe(), which puts the number in the log.
    """
    assert eb.episode_steps("30000") == 30000
    assert str(30000) in eb.describe(30000, "30000")


# ---------------------------------------------------------------------------------------------
# Refusals. NOT fallbacks.
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["abc", "2e3", "1500.0", "1_500 steps", "2000 steps", "0x10", "∞"])
def test_a_value_that_is_not_an_integer_is_refused(raw):
    with pytest.raises(eb.EpisodeBudgetError) as exc:
        eb.episode_steps(raw)
    assert eb.ENV_VAR in str(exc.value), "the message must name the variable the operator set"


@pytest.mark.parametrize("raw", ["0", "-1", "-1500"])
def test_a_non_positive_value_is_refused(raw):
    """Zero or negative is not a short episode, it is an episode that times out on its first frame
    and a run that records nothing."""
    with pytest.raises(eb.EpisodeBudgetError) as exc:
        eb.episode_steps(raw)
    assert "positive" in str(exc.value)


def test_a_bad_value_never_silently_becomes_the_default():
    """THE POINT OF THE REFUSAL, asserted as the absence of a fallback rather than as the presence
    of an exception. spawn_select's reasoning applies unchanged: the operator asked for a specific
    horizon, and a run that quietly used a different one looks identical in the log until its yield
    stops making sense.
    """
    for raw in ("abc", "0", "-5", "1500.0"):
        try:
            got = eb.episode_steps(raw)
        except eb.EpisodeBudgetError:
            continue
        pytest.fail(f"{raw!r} returned {got} instead of refusing; a silent fallback is the bug")


def test_the_error_is_a_value_error_so_an_unprepared_caller_still_dies_loudly():
    assert issubclass(eb.EpisodeBudgetError, ValueError)


# ---------------------------------------------------------------------------------------------
# The log line
# ---------------------------------------------------------------------------------------------

def test_describe_states_the_horizon_and_where_it_came_from():
    """A timeout is indistinguishable from a failure unless the log says what the budget was. That
    ambiguity is what cost job 2108770 a round."""
    line = eb.describe(eb.episode_steps(""), "")
    assert "1500" in line and "default" in line and eb.ENV_VAR in line

    line = eb.describe(eb.episode_steps("3000"), "3000")
    assert "3000" in line and f"{eb.ENV_VAR}=3000" in line
    assert "default" not in line, "a pinned run must not read as a defaulted one"


def test_describe_says_when_the_budget_is_below_the_shortest_known_script():
    """A value under the chassis script's own 1434 makes every such run time out. Saying so in the
    line is cheaper than a second round of diagnosis."""
    assert "BELOW" in eb.describe(900, "900")
    assert "BELOW" not in eb.describe(1500, "")
    assert "BELOW" not in eb.describe(3000, "3000")


# ---------------------------------------------------------------------------------------------
# The wiring, checked off the source
# ---------------------------------------------------------------------------------------------

def _gen_source() -> str:
    return (HERE / "simvla_gen.py").read_text()


def test_no_ORDERING_comparison_measures_the_timestep_against_a_LITERAL():
    """Not just the one at `timeout_mask` -- ANY of them, wherever it sits.

    Deliberately the AST and not a substring search. The first version of this test looked for
    "timestep > 1500" in the text and failed on the comment that EXPLAINS the change, which quotes
    the old line; a text check here cannot tell a live comparison from a description of one.

    ORDERING operators only. `timestep == 1` is live and correct -- _chairtrace_tick uses it for
    "this env has taken no step since its reset" -- and it is not a horizon: an equality against a
    small constant cannot be an episode budget, while `>` against one is exactly what this change
    removed.
    """
    ordered = (ast.Gt, ast.GtE, ast.Lt, ast.LtE)
    tree = ast.parse(_gen_source())
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
            continue
        if node.left.id != "timestep":
            continue
        for op, cmp_node in zip(node.ops, node.comparators):
            if not isinstance(op, ordered):
                continue
            if isinstance(cmp_node, ast.Constant) and isinstance(cmp_node.value, (int, float)):
                literals.append(cmp_node.value)
    assert literals == [], (
        f"simvla_gen orders the step counter against the literal(s) {literals}; a horizon written "
        f"inline is one no run can state in its own log and no operator can change"
    )


def test_simvla_gen_compares_the_timestep_against_a_NAME_and_not_a_constant():
    """AST, not text: a comment mentioning `episode_steps` would satisfy a substring check.

    What must be true is that the timeout mask is built from a variable, so the value can come from
    episode_budget and be printed once at startup.
    """
    tree = ast.parse(_gen_source())
    compares = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "timeout_mask" not in names:
            continue
        assert isinstance(node.value, ast.Compare), ast.dump(node.value)
        assert isinstance(node.value.left, ast.Name) and node.value.left.id == "timestep"
        (rhs,) = node.value.comparators
        assert isinstance(rhs, ast.Name), (
            f"timeout_mask compares timestep against {ast.dump(rhs)}; it must be a variable so the "
            f"budget can be read once from {eb.ENV_VAR} and logged"
        )
        compares.append(rhs.id)

    assert compares == ["episode_steps"], compares


def test_simvla_gen_imports_episode_budget():
    """It boots Omniverse, so it cannot be imported here -- but the line either is in the file or
    it is not."""
    tree = ast.parse(_gen_source())
    imported = {alias.name
                for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "episode_budget" in imported, (
        "simvla_gen.py does not import episode_budget, so the horizon is whatever is written inline "
        "and SIMVLA_EPISODE_STEPS reaches nothing"
    )


def test_the_budget_is_read_once_and_not_per_frame():
    """The loop runs at 20-60 Hz. An os.environ lookup per frame would put a syscall in the hot
    path for a value that cannot change -- and a horizon that CAN change mid-run is one no single
    log line can honestly report.

    Asserted as: the variable name appears exactly once in an environment lookup in the whole file.
    """
    src = _gen_source()
    assert src.count(f"os.environ.get(episode_budget.ENV_VAR") == 1, (
        "SIMVLA_EPISODE_STEPS is looked up more than once; read it at startup and pass the value"
    )
    assert f'"{eb.ENV_VAR}"' not in src, (
        "simvla_gen names the variable by its literal string; it should go through "
        "episode_budget.ENV_VAR so the two cannot drift"
    )
