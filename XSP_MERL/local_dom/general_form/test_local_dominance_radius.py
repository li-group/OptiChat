from pathlib import Path
import json
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_dominance_radius import (
    LocalDominanceData,
    LocalDominanceRadiusProblem,
    find_cost_gap_dominating_scenarios,
)

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_NAME = "lands_instance"
TOL = 1e-6


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_find_cost_gap_dominating_scenarios_writes_positive_gap_list():
    result = find_cost_gap_dominating_scenarios(INSTANCE_NAME, base_dir=BASE_DIR)
    assert result["num_positive_gap_scenarios"] == 1
    assert result["default_selected_scenario"] == "scenario_3"
    assert result["cost_gap_dominating_scenarios"][0]["scenario_index_1_based"] == 3
    assert result["cost_gap_dominating_scenarios"][0]["cost_gap"] > 0

    saved = _load_json(BASE_DIR / INSTANCE_NAME / "cost_gap_dominating_scenarios.json")
    assert saved["default_selected_scenario"] == "scenario_3"


def test_local_dominance_data_dimensions_and_default_selection():
    data = LocalDominanceData.load(INSTANCE_NAME, base_dir=BASE_DIR)
    assert data.num_first_stage_vars == 4
    assert data.num_second_stage_vars == 12
    assert data.num_recourse_constraints == 7
    assert data.xi_dim == 7

    problem = LocalDominanceRadiusProblem(INSTANCE_NAME, base_dir=BASE_DIR)
    assert problem.selected_scenario_name == "scenario_3"
    assert problem.selected_scenario_index_1_based == 3
    assert problem.initial_cost_gap > 0
    assert problem.xi_hat.tolist() == [0, 0, 0, 0, 7, 3, 2]


def test_pyomo_model_dimensions_if_pyomo_is_available():
    pytest.importorskip("pyomo.environ")
    problem = LocalDominanceRadiusProblem(INSTANCE_NAME, base_dir=BASE_DIR, scenario="scenario_3")
    model = problem.build_pyomo_model()
    assert len(list(model.K)) == 7
    assert len(list(model.J)) == 12
    assert len(list(model.R)) == 7
    assert len(list(model.I)) == 4


def test_dual_extreme_enumeration_cross_check_for_dummy_instance():
    # This is an exact small-instance LP cross-check that does not require Pyomo/Gurobi.
    problem = LocalDominanceRadiusProblem(INSTANCE_NAME, base_dir=BASE_DIR, scenario=3)
    result = problem.solve_by_dual_extreme_enumeration()
    assert math.isclose(result["gamma_l1_radius"], 1.7619047619047619, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(result["nearest_violation_xi"][4], 5.238095238095238, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(result["gap_at_nearest_violation"], 0.0, rel_tol=TOL, abs_tol=TOL)
    assert result["num_dual_extreme_points"] == 63
