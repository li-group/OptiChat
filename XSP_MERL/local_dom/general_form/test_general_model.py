# test_general_model.py
from pathlib import Path
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from general_model import (
    TSSPInstance,
    run_full_workflow,
    create_cost_gap_diagnostics,
)

BASE_DIR = Path(__file__).resolve().parent
# INSTANCE_NAME = "lands_instance"
INSTANCE_NAME = "lands_instance_naive"
TOL = 1e-5
X_TOL = 1e-3


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_close(actual, expected, tol=TOL):
    assert math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol), (actual, expected)


def _assert_vector_close(actual, expected, tol=X_TOL):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert math.isclose(float(a), float(e), rel_tol=tol, abs_tol=tol), (actual, expected)


def test_data_consistency_checks_pass():
    inst = TSSPInstance.load(INSTANCE_NAME, base_dir=BASE_DIR)
    report = inst.consistency_report()
    assert report["status"] == "passed"
    assert report["num_first_stage_vars"] == 4
    assert report["num_second_stage_vars"] == 12
    assert report["num_scenarios"] == 3
    _assert_close(report["probability_sum"], 1.0)


def test_full_workflow_against_provided_solution_data():
    # backend="auto" uses Pyomo+Gurobi when installed and scipy linprog otherwise.
    run_full_workflow(INSTANCE_NAME, base_dir=BASE_DIR, solver="gurobi", backend="auto")

    test_data = _load_json(BASE_DIR / INSTANCE_NAME / "test_data"/ "test_data.json")
    stochastic = _load_json(BASE_DIR / INSTANCE_NAME / "output_data" / "stochastic_results.json")
    expected_value = _load_json(BASE_DIR / INSTANCE_NAME / "output_data" / "expectedvalue_results.json")
    diagnostics = _load_json(BASE_DIR / INSTANCE_NAME / "output_data" / "cost_gap_diagnostics.json")

    st_expected = test_data["stochastic_solution"]
    _assert_close(stochastic["objective"], st_expected["objective"])
    _assert_close(stochastic["first_stage_cost"], st_expected["first_stage_cost"])
    _assert_close(stochastic["expected_second_stage_cost"], st_expected["expected_second_stage_cost"])
    _assert_vector_close(stochastic["x_star"], st_expected["x_vector"])
    for name, expected_cost in st_expected["scenario_second_stage_costs"].items():
        _assert_close(stochastic["scenario_second_stage_costs"][name], expected_cost)
        _assert_close(stochastic["recourse_evaluations"][name]["Q_value"], expected_cost)

    ev_expected = test_data["expected_value_solution"]
    _assert_close(expected_value["EV_objective"], ev_expected["objective"])
    _assert_close(expected_value["first_stage_cost"], ev_expected["first_stage_cost"])
    _assert_close(
        expected_value["expected_scenario_second_stage_cost"],
        ev_expected["expected_second_stage_cost"],
    )

    eev_expected = test_data["eev_evaluation"]
    _assert_close(expected_value["EEV_objective"], eev_expected["objective"])
    _assert_close(expected_value["EEV_expected_second_stage_cost"], eev_expected["expected_second_stage_cost"])
    for name, expected_cost in eev_expected["scenario_second_stage_costs"].items():
        _assert_close(expected_value["EEV_scenario_second_stage_costs"][name], expected_cost)
        _assert_close(expected_value["recourse_evaluations"][name]["Q_value"], expected_cost)

    metrics_expected = test_data["metrics"]
    _assert_close(diagnostics["metrics"]["RP"], metrics_expected["RP"])
    _assert_close(diagnostics["metrics"]["EV_objective"], metrics_expected["EV_objective"])
    _assert_close(diagnostics["metrics"]["EEV"], metrics_expected["EEV"])
    _assert_close(diagnostics["metrics"]["VSS"], metrics_expected["VSS"])

    for name, expected_gap_payload in test_data["scenario_cost_gaps"].items():
        _assert_close(diagnostics["scenario_cost_gaps"][name]["cost_gap"], expected_gap_payload["cost_gap"])
        _assert_close(
            diagnostics["scenario_cost_gaps"][name]["weighted_cost_gap"],
            expected_gap_payload["weighted_cost_gap"],
        )


def test_cost_gap_diagnostics_uses_saved_jsons_only():
    # This test intentionally does not call any solve class. It recomputes diagnostics
    # from stochastic_results.json and expectedvalue_results.json only.
    run_full_workflow(INSTANCE_NAME, base_dir=BASE_DIR, solver="gurobi", backend="auto")
    diagnostics = create_cost_gap_diagnostics(INSTANCE_NAME, base_dir=BASE_DIR)
    assert diagnostics["note"].startswith("This file was computed only from saved result JSONs")
    _assert_close(diagnostics["metrics"]["expected_cost_gap"], diagnostics["metrics"]["VSS"])


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))