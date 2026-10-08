from pathlib import Path

import nav_tuning


def test_aiworker_closed_loop_approach_is_wired_into_collector():
    """The profile flag must change phase-1 motion, where long kitchen crossings happen."""
    assert nav_tuning.profile_for("aiworker").closed_loop_approach is True

    source = (Path(__file__).with_name("simvla_gen.py")).read_text()
    phase1 = source.split("# ---- Phase 1:", 1)[1].split("# ---- Phase 2:", 1)[0]
    assert "if _nav.closed_loop_approach:" in phase1
    assert "nav_actions[:, :2] =" in phase1
    assert "nav_actions[:, 1] *= -1" in phase1


def test_aiworker_object_parking_preserves_counter_clearance_but_extends_arm_reach():
    """The 813 collector plateaued 7-12 cm short from a 0.76 m base-frame reach target.

    Object-on-counter goals use safety + the nav standoff as their edge gap. A -0.10 m robot
    adjustment moves the base toward the counter while retaining 0.185 m nominal clearance.
    """
    profile = nav_tuning.profile_for("aiworker")
    nominal_counter_clearance = 0.28 + 0.23 + profile.object_extra_standoff_m - 0.225
    assert profile.object_extra_standoff_m == -0.10
    assert abs(nominal_counter_clearance - 0.185) < 1e-9

    # Anubis is similarly reach-limited on public kitchen 813; its shorter 0.2325 m
    # forward footprint preserves about 0.178 m from the counter edge.
    anubis = nav_tuning.profile_for("anubis")
    assert anubis.object_extra_standoff_m == -0.10
    assert abs((0.28 + 0.23 + anubis.object_extra_standoff_m - 0.2325) - 0.1775) < 1e-9

    # RB-Y1's measured reach misses the first public target by 0.216 m. A second 5 cm approach
    # adjustment targets ~0.49 m reach distance while retaining positive nominal clearance.
    rby1 = nav_tuning.profile_for("rby1")
    assert rby1.object_extra_standoff_m == -0.10
    assert abs(0.28 + 0.23 + rby1.object_extra_standoff_m - 0.345 - 0.065) < 1e-9
