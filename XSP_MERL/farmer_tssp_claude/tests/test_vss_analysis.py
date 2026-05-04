"""
farmer_tssp/tests/test_vss_analysis.py
───────────────────────────────────────
Unit and integration tests for the four-stage VSS explanation pipeline.

Run with:
    python -m pytest farmer_tssp/tests/test_vss_analysis.py -v
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ..data_loader import load_data
from ..vss_analysis import (
    BadScenarioSet,
    CompensationTestResult,
    CompensationTester,
    GlobalInfeasibilityResult,
    GlobalInfeasibilityTester,
    MinBadScenarioFinder,
    Prerequisites,
    PrerequisitesSolver,
    ScenarioInfeasibilityResult,
    ScenarioInfeasibilityTester,
    VSSExplainer,
    _recourse_cost_given_x,
)

# DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farmer_data.json"
# DATA_PATH = Path(__file__).parent / "data" / "farmer_data.json"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farmer_data.json"
TOL = 1.0   # $1 numerical tolerance throughout


def _prereqs(data=None):
    """Cached helper – solve prerequisites once per module load."""
    if data is None:
        data = load_data(DATA_PATH)
    return PrerequisitesSolver(data, solver="glpk", integer_x=False).solve()


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 0 – Prerequisites
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrerequisites(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)

    # ── Fundamental SP inequality: z_SP ≤ z_EEV ───────────────────────────────

    def test_z_sp_le_z_eev(self):
        """z_SP ≤ z_EEV must always hold."""
        self.assertLessEqual(
            self.prereqs.z_sp, self.prereqs.z_eev + TOL,
            "z_SP must be ≤ z_EEV (stochastic solution no worse than EV)"
        )

    def test_vss_nonneg(self):
        """VSS = z_EEV - z_SP ≥ 0."""
        self.assertGreaterEqual(self.prereqs.vss, -TOL,
                                "VSS must be non-negative")

    def test_vss_identity(self):
        """VSS = z_EEV - z_SP exactly."""
        computed = self.prereqs.z_eev - self.prereqs.z_sp
        self.assertAlmostEqual(self.prereqs.vss, computed, delta=TOL)

    # ── Recourse cost consistency ──────────────────────────────────────────────

    def test_qs_keys_match_scenarios(self):
        """Q_s dicts must have one entry per scenario."""
        ids = set(self.data.scenario_ids)
        self.assertEqual(set(self.prereqs.Qs_sp.keys()), ids)
        self.assertEqual(set(self.prereqs.Qs_ev.keys()), ids)

    def test_delta_s_identity(self):
        """Δ_s = Q_s(x_EV) - Q_s(x*) for every s."""
        for s in self.data.scenario_ids:
            expected = self.prereqs.Qs_ev[s] - self.prereqs.Qs_sp[s]
            self.assertAlmostEqual(
                self.prereqs.delta_s[s], expected, delta=TOL,
                msg=f"delta_s[{s}] identity failed"
            )

    def test_vss_decomposition(self):
        """
        VSS = Σ_s p_s Δ_s + Δ₀  (full decomposition from the paper).
        """
        probs = self.prereqs.probabilities
        total = self.prereqs.delta_0 + sum(
            probs[s] * self.prereqs.delta_s[s]
            for s in self.data.scenario_ids
        )
        self.assertAlmostEqual(self.prereqs.vss, total, delta=TOL,
                               msg="VSS decomposition Σ p_s Δ_s + Δ₀ failed")

    def test_delta_0_identity(self):
        """Δ₀ = c^T x_EV - c^T x* ."""
        expected = self.prereqs.c_x_ev - self.prereqs.c_x_sp
        self.assertAlmostEqual(self.prereqs.delta_0, expected, delta=TOL)

    def test_z_sp_reconstructed_from_x_star(self):
        """
        z_SP = c^T x* + Σ_s p_s Q_s(x*)  (extensive-form objective identity).
        """
        probs = self.prereqs.probabilities
        reconstructed = self.prereqs.c_x_sp + sum(
            probs[s] * self.prereqs.Qs_sp[s]
            for s in self.data.scenario_ids
        )
        self.assertAlmostEqual(self.prereqs.z_sp, reconstructed, delta=TOL,
                               msg="z_SP reconstruction from Qs_sp failed")

    def test_z_eev_reconstructed_from_x_ev(self):
        """
        z_EEV = c^T x_EV + Σ_s p_s Q_s(x_EV).
        """
        probs = self.prereqs.probabilities
        reconstructed = self.prereqs.c_x_ev + sum(
            probs[s] * self.prereqs.Qs_ev[s]
            for s in self.data.scenario_ids
        )
        self.assertAlmostEqual(self.prereqs.z_eev, reconstructed, delta=TOL,
                               msg="z_EEV reconstruction from Qs_ev failed")

    def test_x_sp_feasible_acreage(self):
        """x* must respect the acreage budget."""
        total = sum(self.prereqs.x_sp.values())
        self.assertLessEqual(total, self.data.budget + TOL)

    def test_x_ev_feasible_acreage(self):
        """x_EV must respect the acreage budget."""
        total = sum(self.prereqs.x_ev.values())
        self.assertLessEqual(total, self.data.budget + TOL)

    def test_x_sp_nonneg(self):
        for i, v in self.prereqs.x_sp.items():
            self.assertGreaterEqual(v, -1e-6, f"x_sp[{i}] negative")

    def test_x_ev_nonneg(self):
        for i, v in self.prereqs.x_ev.items():
            self.assertGreaterEqual(v, -1e-6, f"x_ev[{i}] negative")


# ═══════════════════════════════════════════════════════════════════════════════
#  Recourse subproblem helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecourseHelper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)

    def test_qs_sp_matches_prereqs(self):
        """_recourse_cost_given_x with x* must match Qs_sp from Prerequisites."""
        for s in self.data.scenario_ids:
            recomputed = _recourse_cost_given_x(
                self.data, self.prereqs.x_sp, s
            )
            self.assertAlmostEqual(
                recomputed, self.prereqs.Qs_sp[s], delta=TOL,
                msg=f"Q_s(x*) mismatch for s={s}"
            )

    def test_qs_ev_matches_prereqs(self):
        """_recourse_cost_given_x with x_EV must match Qs_ev from Prerequisites."""
        for s in self.data.scenario_ids:
            recomputed = _recourse_cost_given_x(
                self.data, self.prereqs.x_ev, s
            )
            self.assertAlmostEqual(
                recomputed, self.prereqs.Qs_ev[s], delta=TOL,
                msg=f"Q_s(x_EV) mismatch for s={s}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 – Global infeasibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalInfeasibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)
        cls.result  = GlobalInfeasibilityTester(
            cls.data, cls.prereqs
        ).test()

    def test_returns_correct_type(self):
        self.assertIsInstance(self.result, GlobalInfeasibilityResult)

    def test_budget_rhs_identity(self):
        """Budget RHS = z_SP - c^T x_EV."""
        expected = self.prereqs.z_sp - self.prereqs.c_x_ev
        self.assertAlmostEqual(self.result.budget_rhs, expected, delta=TOL)

    def test_infeasible_when_vss_positive(self):
        """
        When VSS > 0, the global system must be infeasible:
        x_EV cannot match z_SP with any recourse.
        """
        if self.prereqs.vss > TOL:
            self.assertTrue(
                self.result.is_infeasible,
                "Expected infeasibility when VSS > 0"
            )

    def test_model_has_budget_constraint(self):
        """The Pyomo model must include the GlobalBudget constraint."""
        self.assertTrue(hasattr(self.result.model, "GlobalBudget"))

    def test_model_has_recourse_vars(self):
        self.assertTrue(hasattr(self.result.model, "y"))
        self.assertTrue(hasattr(self.result.model, "w"))

    def test_model_has_all_scenarios(self):
        sc = list(self.result.model.SCENARIOS)
        self.assertEqual(set(sc), set(self.data.scenario_ids))

    def test_str_representation(self):
        """__str__ must run without error."""
        s = str(self.result)
        self.assertIn("Global infeasibility", s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 – Scenario-level infeasibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestScenarioInfeasibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)
        cls.result  = ScenarioInfeasibilityTester(
            cls.data, cls.prereqs
        ).test_all()

    def test_returns_correct_type(self):
        self.assertIsInstance(self.result, ScenarioInfeasibilityResult)

    def test_partition_covers_all_scenarios(self):
        """Infeasible ∪ feasible = all scenarios (disjoint partition)."""
        all_s = set(self.data.scenario_ids)
        union = self.result.infeasible_scenarios | self.result.feasible_scenarios
        inter = self.result.infeasible_scenarios & self.result.feasible_scenarios
        self.assertEqual(union, all_s)
        self.assertEqual(inter, set())

    def test_infeasible_iff_delta_positive(self):
        """
        Scenario s is infeasible iff Δ_s > 0, i.e., x_EV is strictly
        more expensive than x* in that scenario.
        """
        for s in self.data.scenario_ids:
            delta = self.prereqs.delta_s[s]
            infeas = s in self.result.infeasible_scenarios
            if delta > TOL:
                self.assertTrue(infeas,
                    f"s={s}: Δ_s={delta:.2f}>0 but not flagged infeasible")
            elif delta < -TOL:
                self.assertFalse(infeas,
                    f"s={s}: Δ_s={delta:.2f}<0 but flagged infeasible")

    def test_per_scenario_keys(self):
        """per_scenario must have an entry for every scenario."""
        self.assertEqual(
            set(self.result.per_scenario.keys()),
            set(self.data.scenario_ids)
        )

    def test_per_scenario_has_model(self):
        for s, d in self.result.per_scenario.items():
            self.assertIn("model", d, f"s={s} missing 'model' key")

    def test_individual_scenario_test_matches_all(self):
        """test_scenario(s) must match the batch result from test_all()."""
        tester = ScenarioInfeasibilityTester(self.data, self.prereqs)
        for s in self.data.scenario_ids:
            single = tester.test_scenario(s)
            batch  = self.result.per_scenario[s]
            self.assertEqual(single["is_infeasible"], batch["is_infeasible"],
                             f"s={s}: single vs batch infeasibility mismatch")

    def test_str_representation(self):
        s = str(self.result)
        self.assertIn("Scenario-level", s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 – Minimum bad-scenario set
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinBadScenarioFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)
        cls.bss     = MinBadScenarioFinder(cls.prereqs).find()

    def test_returns_correct_type(self):
        self.assertIsInstance(self.bss, BadScenarioSet)

    def test_G_B_partition_of_scenarios(self):
        """G ∪ B = all scenarios, G ∩ B = ∅."""
        all_s = frozenset(self.data.scenario_ids)
        self.assertEqual(self.bss.G | self.bss.B, all_s)
        self.assertEqual(self.bss.G & self.bss.B, frozenset())

    def test_G_contains_non_positive_delta(self):
        for s in self.bss.G:
            self.assertLessEqual(
                self.prereqs.delta_s[s], 1e-6,
                f"G contains s={s} but Δ_s={self.prereqs.delta_s[s]:.4f} > 0"
            )

    def test_B_contains_positive_delta(self):
        for s in self.bss.B:
            self.assertGreater(
                self.prereqs.delta_s[s], -1e-6,
                f"B contains s={s} but Δ_s={self.prereqs.delta_s[s]:.4f} ≤ 0"
            )

    def test_K_subset_of_B(self):
        """K must be a subset of B."""
        self.assertTrue(
            set(self.bss.K) <= self.bss.B,
            f"K={self.bss.K} is not a subset of B={self.bss.B}"
        )

    def test_K_nonempty(self):
        self.assertGreater(len(self.bss.K), 0)

    def test_K_satisfies_compensation_condition(self):
        """Σ_{s∈K} p_s Δ_s + C_comp > 0 (the defining condition of K)."""
        probs = self.prereqs.probabilities
        total = sum(
            probs[s] * self.prereqs.delta_s[s]
            for s in self.bss.K
        ) + self.bss.C_comp
        self.assertGreater(total, -1e-6,
            "K does not satisfy the compensation condition")

    def test_K_minimality(self):
        """
        No strict prefix of K satisfies the condition.
        (K is the smallest prefix that works.)
        """
        if len(self.bss.K) <= 1:
            return   # trivially minimal
        probs = self.prereqs.probabilities
        # All strict prefixes should fail
        for size in range(1, len(self.bss.K)):
            prefix = self.bss.K[:size]
            partial = sum(
                probs[s] * self.prereqs.delta_s[s]
                for s in prefix
            ) + self.bss.C_comp
            self.assertLessEqual(
                partial, 1e-6,
                f"Prefix of size {size} already satisfies condition "
                f"– K is not minimal"
            )

    def test_C_comp_identity(self):
        """C_comp = Δ₀ + Σ_{s∈G} p_s Δ_s."""
        probs = self.prereqs.probabilities
        expected = self.prereqs.delta_0 + sum(
            probs[s] * self.prereqs.delta_s[s]
            for s in self.bss.G
        )
        self.assertAlmostEqual(self.bss.C_comp, expected, delta=TOL)

    def test_vss_decomposition_via_K_and_G(self):
        """
        VSS = Σ_{s∈B} p_s Δ_s + C_comp  (paper's decomposition).
        """
        probs = self.prereqs.probabilities
        bad_part = sum(probs[s] * self.prereqs.delta_s[s] for s in self.bss.B)
        self.assertAlmostEqual(
            self.prereqs.vss,
            bad_part + self.bss.C_comp,
            delta=TOL,
            msg="VSS decomposition via B and C_comp failed"
        )

    def test_weighted_regrets_keys_match_B(self):
        self.assertEqual(set(self.bss.weighted_regrets.keys()), self.bss.B)

    def test_weighted_regrets_nonneg(self):
        for s, wr in self.bss.weighted_regrets.items():
            self.assertGreaterEqual(wr, -1e-6,
                f"weighted_regret[{s}]={wr:.4f} negative")

    def test_zero_vss_raises(self):
        """MinBadScenarioFinder must raise if VSS ≤ 0."""
        import copy, dataclasses
        p0 = copy.copy(self.prereqs)
        # Manually set VSS to 0
        p0 = dataclasses.replace(p0, vss=0.0, z_eev=p0.z_sp)
        with self.assertRaises(ValueError):
            MinBadScenarioFinder(p0).find()

    def test_str_representation(self):
        s = str(self.bss)
        self.assertIn("Minimum bad-scenario", s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4 – Compensation test
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompensationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data    = load_data(DATA_PATH)
        cls.prereqs = _prereqs(cls.data)
        cls.bss     = MinBadScenarioFinder(cls.prereqs).find()
        cls.result  = CompensationTester(
            cls.data, cls.prereqs, cls.bss
        ).test()

    def test_returns_correct_type(self):
        self.assertIsInstance(self.result, CompensationTestResult)

    def test_K_matches_bad_set(self):
        self.assertEqual(self.result.K, self.bss.K)

    def test_budget_rhs_identity(self):
        """
        budget_rhs = Σ_{s∈K} p_s Q_s(x*) - C_comp.
        """
        probs  = self.prereqs.probabilities
        Qs_sp  = self.prereqs.Qs_sp
        K      = self.bss.K
        C_comp = self.bss.C_comp
        expected = sum(probs[s] * Qs_sp[s] for s in K) - C_comp
        self.assertAlmostEqual(self.result.budget_rhs, expected, delta=TOL)

    def test_infeasible_when_vss_positive(self):
        """
        When VSS > 0 and K is correctly computed, the compensation test
        must be infeasible (the whole point of K).
        """
        if self.prereqs.vss > TOL:
            self.assertTrue(
                self.result.is_infeasible,
                "Compensation test should be infeasible when VSS > 0"
            )

    def test_model_has_compensation_budget_constraint(self):
        self.assertTrue(hasattr(self.result.model, "CompensationBudget"))

    def test_model_restricts_to_K(self):
        """Model's K_SET must match the bad-scenario set K."""
        model_K = set(self.result.model.K_SET)
        self.assertEqual(model_K, set(self.bss.K))

    def test_model_has_recourse_vars(self):
        self.assertTrue(hasattr(self.result.model, "y"))
        self.assertTrue(hasattr(self.result.model, "w"))

    def test_str_representation(self):
        s = str(self.result)
        self.assertIn("Compensation test", s)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class TestVSSExplainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data   = load_data(DATA_PATH)
        cls.result = VSSExplainer(cls.data, solver="glpk", integer_x=False).run()

    def test_all_fields_present(self):
        r = self.result
        self.assertIsNotNone(r.prereqs)
        self.assertIsNotNone(r.global_test)
        self.assertIsNotNone(r.scenario_test)
        self.assertIsNotNone(r.bad_set)
        self.assertIsNotNone(r.compensation_test)

    def test_summary_runs(self):
        s = self.result.summary()
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 100)

    def test_end_to_end_infeasibility_chain(self):
        """
        All three infeasibility tests must agree:
        global, at least one scenario, and compensation.
        """
        r = self.result
        p = r.prereqs
        if p.vss > TOL:
            self.assertTrue(r.global_test.is_infeasible,
                            "Global test must be infeasible when VSS > 0")
            self.assertTrue(len(r.scenario_test.infeasible_scenarios) > 0,
                            "At least one scenario must be infeasible when VSS > 0")
            self.assertTrue(r.compensation_test.is_infeasible,
                            "Compensation test must be infeasible when VSS > 0")

    def test_pipeline_vss_matches_independent_prereqs(self):
        """VSS from pipeline matches a freshly computed Prerequisites."""
        fresh = _prereqs(self.data)
        self.assertAlmostEqual(
            self.result.prereqs.vss, fresh.vss, delta=TOL
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)