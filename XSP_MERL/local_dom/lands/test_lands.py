"""Regression tests for the LandS deterministic equivalent.

Run:
    python test_lands.py
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import pyomo.environ as pyo

from lands_model import LandSDeterministicEquivalent


class TestLandSDeterministicEquivalent(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(__file__).with_name("lands_data.json")
        self.solver_name = self.available_solver()

    @staticmethod
    def available_solver():
        for solver_name in ("gurobi", "highs", "appsi_highs"):
            if pyo.SolverFactory(solver_name).available(False):
                return solver_name
        raise RuntimeError("No supported LP solver is available. Install Gurobi or HiGHS/highspy.")

    def assert_close(self, actual, expected, msg=""):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8),
            msg=msg or f"expected {expected}, got {actual}",
        )

    def test_reported_stochastic_solution(self):
        problem = LandSDeterministicEquivalent.from_json(self.data_path)
        problem.solve(solver_name=self.solver_name)

        expected = problem.data["expected_solution"]
        actual_x = problem.x_values()
        actual_obj = problem.objective_value()

        for i, expected_value in expected["x"].items():
            self.assert_close(
                actual_x[i],
                expected_value,
                msg=f"x[{i}] expected {expected_value}, got {actual_x[i]}",
            )

        self.assert_close(
            actual_obj,
            expected["objective"],
            msg=f"objective expected {expected['objective']}, got {actual_obj}",
        )

    def test_expected_value_solution_eev_and_vss(self):
        problem = LandSDeterministicEquivalent.from_json(self.data_path)
        diagnostics = problem.compute_diagnostics(solver_name=self.solver_name)
        expected = problem.data["expected_value_reference"]

        self.assertEqual(diagnostics["expected_demand"], expected["expected_demand"])
        self.assert_close(diagnostics["metrics"]["EV_objective"], expected["EV_objective"])
        self.assert_close(diagnostics["metrics"]["EEV"], expected["EEV"])
        self.assert_close(diagnostics["metrics"]["VSS"], expected["VSS"])

        actual_x_ev = diagnostics["expected_value_solution"]["first_stage"]["x_by_i"]
        for i, expected_value in expected["x_EV"].items():
            self.assert_close(
                actual_x_ev[i],
                expected_value,
                msg=f"x_EV[{i}] expected {expected_value}, got {actual_x_ev[i]}",
            )

        # The EEV evaluation must use the EV first-stage vector, fixed across all
        # original scenarios, while re-optimizing the scenario-dependent y values.
        actual_eev_x = diagnostics["eev_evaluation"]["first_stage"]["x_by_i"]
        self.assertEqual(actual_eev_x, actual_x_ev)
        self.assertIn("y_vector", diagnostics["eev_evaluation"]["second_stage"])
        self.assertIn("feasibility", diagnostics["eev_evaluation"])

    def test_save_diagnostics_json(self):
        problem = LandSDeterministicEquivalent.from_json(self.data_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lands_diagnostics.json"
            diagnostics = problem.save_diagnostics(path, solver_name=self.solver_name)
            self.assertTrue(path.exists())

            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["metrics"], diagnostics["metrics"])
            self.assertIn("stochastic_solution", loaded)
            self.assertIn("expected_value_solution", loaded)
            self.assertIn("eev_evaluation", loaded)

    def test_cost_gap_diagnostics(self):
        problem = LandSDeterministicEquivalent.from_json(self.data_path)
        diagnostics = problem.compute_cost_gap_diagnostics(solver_name=self.solver_name)
        expected = problem.data["cost_gap_reference"]

        for s, expected_gap in expected["scenario_cost_gaps"].items():
            actual_gap = diagnostics["scenario_cost_gaps"][s]["cost_gap"]
            self.assert_close(
                actual_gap,
                expected_gap,
                msg=f"Delta[{s}] expected {expected_gap}, got {actual_gap}",
            )

        self.assert_close(
            diagnostics["summary"]["expected_cost_gap"],
            expected["expected_cost_gap"],
        )
        self.assert_close(
            diagnostics["summary"]["expected_cost_gap"],
            diagnostics["summary"]["VSS"],
        )
        self.assert_close(
            diagnostics["summary"]["expected_cost_gap_minus_VSS"],
            0.0,
        )

    def test_save_cost_gap_diagnostics_json(self):
        problem = LandSDeterministicEquivalent.from_json(self.data_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cost_gap_diagnostics.json"
            diagnostics = problem.save_cost_gap_diagnostics(
                path, solver_name=self.solver_name
            )
            self.assertTrue(path.exists())

            loaded = json.loads(path.read_text())
            self.assertEqual(
                loaded["summary"]["expected_cost_gap"],
                diagnostics["summary"]["expected_cost_gap"],
            )
            self.assertIn("scenario_cost_gaps", loaded)
            self.assertIn("x_star", loaded["first_stage_solutions"])
            self.assertIn("x_EV", loaded["first_stage_solutions"])


if __name__ == "__main__":
    unittest.main()