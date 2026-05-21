import math
from pathlib import Path

from directional_upper_bound import DirectionalUpperBoundSolver, run_directional_upper_bound


def test_directional_upper_bound_lands_instance():
    result = run_directional_upper_bound("lands_instance", base_dir=Path(__file__).resolve().parent)
    assert result["best_direction"]["direction_label"] == "xi_5_minus"
    assert math.isclose(result["directional_upper_bound_gamma_bar"], 1.7619047619047619, rel_tol=1e-8, abs_tol=1e-8)
    assert result["standard_form_dimensions"]["num_basis_candidates_combination"] == 50388
    assert result["standard_form_dimensions"]["num_dual_feasible_bases_enumerated"] == 63


def test_directional_upper_bound_stores_basis_intervals():
    solver = DirectionalUpperBoundSolver("lands_instance", base_dir=Path(__file__).resolve().parent)
    result = solver.solve_single_direction(5, -1)
    assert result["direction_label"] == "xi_5_minus"
    assert result["num_intervals"] == 2
    assert result["num_basis_changes"]["star"] == 1
    assert result["num_basis_changes"]["EV"] == 0
    assert result["intervals"][0]["event_at_end"] == ["star"]
    assert result["intervals"][1]["event_at_end"] == ["violation_crossing"]
    assert result["basis_change_events"][0]["updates"][0]["selection_rule"] == "lexicographically_first_right_feasible_optimal_basis"
