"""
farmer_tssp/tests/test_model.py
────────────────────────────────
Comprehensive unit and integration tests.

Run with:
    python -m pytest farmer_tssp/tests/ -v
or:
    python -m unittest discover farmer_tssp/tests
"""

import json
import os
import sys
import math
import tempfile
import unittest
from pathlib import Path

# Make parent importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from farmer_tssp.data_loader import load_data, FarmerData, _validate
from farmer_tssp.model import (
    build_extensive_form,
    build_wait_and_see,
    build_mean_value,
    solve_model,
)
from farmer_tssp.solver import solve_rp, solve_ws, solve_ev_and_eev, compute_metrics


# from data_loader import load_data, FarmerData, _validate
# from model import (
#     build_extensive_form,
#     build_wait_and_see,
#     build_mean_value,
#     solve_model,
# )
# from solver import solve_rp, solve_ws, solve_ev_and_eev, compute_metrics


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farmer_data.json"
EXPECTED_RP = -118600.0   # known optimal for LP relaxation (equal yields avg)
EXPECTED_RP_INT = -118600.0  # same value for this data
TOLERANCE = 500.0          # allow $500 numerical tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# Data-loader tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.data = load_data(DATA_PATH)

    def test_sets_loaded(self):
        self.assertEqual(len(self.data.crop_ids),     3)
        self.assertEqual(len(self.data.purch_ids),    2)
        self.assertEqual(len(self.data.sell_ids),     4)
        self.assertEqual(len(self.data.scenario_ids), 3)

    def test_budget_positive(self):
        self.assertGreater(self.data.budget, 0)
        self.assertEqual(self.data.budget, 500)

    def test_beet_quota_positive(self):
        self.assertGreater(self.data.beet_quota, 0)

    def test_probabilities_sum_to_one(self):
        total = sum(self.data.probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_yields_positive(self):
        for s in self.data.scenario_ids:
            for c in self.data.crop_ids:
                self.assertGreater(self.data.yield_of(s, c), 0,
                                   msg=f"yield[s={s},c={c}] must be >0")

    def test_planting_cost_positive(self):
        for c in self.data.crop_ids:
            self.assertGreater(self.data.planting_cost[c], 0)

    def test_purchase_price_positive(self):
        for j in self.data.purch_ids:
            self.assertGreater(self.data.purchase_price[j], 0)

    def test_sell_price_nonneg(self):
        for k in self.data.sell_ids:
            self.assertGreaterEqual(self.data.sell_price[k], 0)

    def test_scenario_names_exist(self):
        for s in self.data.scenario_ids:
            self.assertIn(s, self.data.scen_names)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_data("/nonexistent/path.json")

    def test_bad_probabilities_raise(self):
        """Validation should reject probabilities that don't sum to 1."""
        import copy
        d = copy.deepcopy(self.data)
        d.scenarios[1] = d.scenarios[1].__class__(
            sid=1, name="Bad", probability=0.9,
            yields=d.scenarios[1].yields
        )
        with self.assertRaises(ValueError):
            _validate(d)

    def test_negative_budget_raises(self):
        import copy
        d = copy.deepcopy(self.data)
        object.__setattr__(d, 'budget', -100)
        with self.assertRaises((ValueError, AttributeError)):
            d2 = FarmerData(**{k: (v if k != 'budget' else -100)
                               for k, v in d.__dict__.items()})
            _validate(d2)


# ═══════════════════════════════════════════════════════════════════════════════
# Model-structure tests  (no solve required)
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelStructure(unittest.TestCase):

    def setUp(self):
        self.data = load_data(DATA_PATH)
        self.m = build_extensive_form(self.data, integer_x=False)

    def test_variables_exist(self):
        self.assertTrue(hasattr(self.m, 'x'))
        self.assertTrue(hasattr(self.m, 'y'))
        self.assertTrue(hasattr(self.m, 'w'))

    def test_x_size(self):
        self.assertEqual(len(list(self.m.x)), len(self.data.crop_ids))

    def test_y_size(self):
        expected = len(self.data.scenario_ids) * len(self.data.purch_ids)
        self.assertEqual(len(list(self.m.y)), expected)

    def test_w_size(self):
        expected = len(self.data.scenario_ids) * len(self.data.sell_ids)
        self.assertEqual(len(list(self.m.w)), expected)

    def test_constraints_exist(self):
        self.assertTrue(hasattr(self.m, 'TotalAcreage'))
        self.assertTrue(hasattr(self.m, 'MeetRequirement'))
        self.assertTrue(hasattr(self.m, 'BeetRequirement'))
        self.assertTrue(hasattr(self.m, 'BeetQuotaLimit'))

    def test_objective_exists(self):
        self.assertTrue(hasattr(self.m, 'ExpectedCost'))

    def test_ws_model_structure(self):
        ws = build_wait_and_see(self.data, 1)
        self.assertFalse(hasattr(ws, 'SCENARIOS'),
                         "WS model should NOT have a SCENARIOS set")
        self.assertTrue(hasattr(ws, 'x'))

    def test_mean_value_model_structure(self):
        mv = build_mean_value(self.data)
        self.assertFalse(hasattr(mv, 'SCENARIOS'))
        self.assertTrue(hasattr(mv, 'x'))


# ═══════════════════════════════════════════════════════════════════════════════
# Solve tests  (requires GLPK)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolveLP(unittest.TestCase):
    """LP relaxation tests – faster than MIP."""

    def setUp(self):
        self.data = load_data(DATA_PATH)

    def test_rp_lp_solves(self):
        res = solve_rp(self.data, integer_x=False)
        self.assertEqual(res["status"], "optimal")
        self.assertIsNotNone(res["objective"])

    def test_rp_lp_objective_range(self):
        res = solve_rp(self.data, integer_x=False)
        obj = res["objective"]
        # known range for this data set
        self.assertLess(obj,    50_000,  "objective too large")
        self.assertGreater(obj, -200_000, "objective too small (unrealistic)")

    def test_acreage_feasible(self):
        res  = solve_rp(self.data, integer_x=False)
        total = sum(res["x"].values())
        self.assertLessEqual(total, self.data.budget + 1e-6,
                             "Total acreage exceeds budget")

    def test_x_nonneg(self):
        res = solve_rp(self.data, integer_x=False)
        for i, v in res["x"].items():
            self.assertGreaterEqual(v, -1e-6, f"x[{i}] is negative")

    def test_y_nonneg(self):
        res = solve_rp(self.data, integer_x=False)
        for k, v in res["y"].items():
            self.assertGreaterEqual(v, -1e-6, f"y{k} is negative")

    def test_w_nonneg(self):
        res = solve_rp(self.data, integer_x=False)
        for k, v in res["w"].items():
            self.assertGreaterEqual(v, -1e-6, f"w{k} is negative")

    def test_beet_quota_respected(self):
        res = solve_rp(self.data, integer_x=False)
        for s in self.data.scenario_ids:
            beet_under = res["w"][(s, 3)]
            self.assertLessEqual(beet_under, self.data.beet_quota + 1e-3,
                                 f"Beet quota violated in scenario {s}")

    def test_ws_less_than_rp(self):
        """WS (perfect info) must be ≤ RP (stochastic)."""
        rp  = solve_rp(self.data, integer_x=False)["objective"]
        ws  = solve_ws(self.data, integer_x=False)["ws"]
        self.assertLessEqual(ws, rp + 1.0,
                             "WS should be ≤ RP (perfect info is cheaper)")

    def test_rp_less_than_eev(self):
        """RP (stochastic) must be ≤ EEV (using EV solution in full model)."""
        rp  = solve_rp(self.data, integer_x=False)["objective"]
        eev = solve_ev_and_eev(self.data, integer_x=False)["eev"]
        self.assertLessEqual(rp, eev + 1.0,
                             "RP should be ≤ EEV (stochastic solution is better)")

    def test_evpi_nonneg(self):
        m = compute_metrics(self.data, integer_x=False)
        self.assertGreaterEqual(m["EVPI"], -1.0)

    def test_vss_nonneg(self):
        m = compute_metrics(self.data, integer_x=False)
        self.assertGreaterEqual(m["VSS"], -1.0)

    def test_scenario_costs_computed(self):
        res = solve_rp(self.data, integer_x=False)
        for s in self.data.scenario_ids:
            self.assertIn(s, res["scenario_costs"])

    def test_first_stage_cost_positive(self):
        res = solve_rp(self.data, integer_x=False)
        self.assertGreater(res["first_stage_cost"], 0)


class TestSolveMIP(unittest.TestCase):
    """Integer x tests (slower – run separately if needed)."""

    def setUp(self):
        self.data = load_data(DATA_PATH)

    def test_x_integer(self):
        res = solve_rp(self.data, integer_x=True)
        for i, v in res["x"].items():
            self.assertAlmostEqual(v, round(v), places=3,
                                   msg=f"x[{i}]={v} is not integer")

    def test_mip_objective_close_to_lp(self):
        rp_lp  = solve_rp(self.data, integer_x=False)["objective"]
        rp_mip = solve_rp(self.data, integer_x=True)["objective"]
        # MIP obj >= LP relaxation, but should be within ~2%
        self.assertGreaterEqual(rp_mip, rp_lp - 1.0)
        gap_pct = abs(rp_mip - rp_lp) / (abs(rp_lp) + 1e-9) * 100
        self.assertLess(gap_pct, 5.0, f"MIP-LP gap too large: {gap_pct:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint-violation tests  (sanity-check solutions manually)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstraintSanity(unittest.TestCase):

    def setUp(self):
        self.data = load_data(DATA_PATH)
        self.res  = solve_rp(self.data, integer_x=False)

    def test_feed_requirement_met(self):
        """yield*x + y - w[1 or 2] >= minreq for wheat & corn, each scenario."""
        x = self.res["x"]
        y = self.res["y"]
        w = self.res["w"]
        for s in self.data.scenario_ids:
            for j in self.data.purch_ids:
                lhs = (self.data.yield_of(s, j) * x[j]
                       + y[(s, j)] - w[(s, j)])
                req = self.data.min_requirement[j]
                self.assertGreaterEqual(lhs + 1e-4, req,
                    f"Feed req violated: s={s} j={j}: {lhs:.2f} < {req}")

    def test_sell_not_more_than_harvested(self):
        """w[s,1] <= yield*x[1] for wheat, etc."""
        x = self.res["x"]
        w = self.res["w"]
        for s in self.data.scenario_ids:
            for j in self.data.purch_ids:
                sold      = w[(s, j)]
                harvested = self.data.yield_of(s, j) * x[j]
                self.assertLessEqual(sold, harvested + 1e-4,
                    f"Sold more than harvested: s={s} j={j}")


if __name__ == "__main__":
    unittest.main(verbosity=2)