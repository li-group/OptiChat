"""
farmer_tssp/tests/test_iis.py
──────────────────────────────
Unit and integration tests for the IIS extraction module.

Run with:
    python -m pytest farmer_tssp/tests/test_iis.py -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from farmer_tssp.data_loader import load_data
from farmer_tssp.vss_analysis import VSSExplainer
from farmer_tssp.iis import (
    ConstraintID,
    DeletionFilter,
    GrowingMethod,
    IISExtractor,
    IISResult,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "farmer_data.json"

# Known correct IIS for the farmer problem (both algorithms must agree)
KNOWN_IIS = {
    "MeetRequirement[(1, 1)]",
    "MeetRequirement[(1, 2)]",
    "BeetRequirement[1]",
    "BeetQuotaLimit[1]",
    "CompensationBudget",
}


def _setup():
    data   = load_data(DATA_PATH)
    result = VSSExplainer(data, solver="glpk", integer_x=False).run()
    return data, result


class TestConstraintID(unittest.TestCase):

    def test_str_with_tuple_index(self):
        cid = ConstraintID(block="MeetRequirement", index=(1, 2))
        self.assertEqual(str(cid), "MeetRequirement[(1, 2)]")

    def test_str_with_scalar_index(self):
        cid = ConstraintID(block="BeetQuotaLimit", index=1)
        self.assertEqual(str(cid), "BeetQuotaLimit[1]")

    def test_str_with_none_index(self):
        cid = ConstraintID(block="CompensationBudget", index=None)
        self.assertEqual(str(cid), "CompensationBudget")

    def test_frozen_immutable(self):
        cid = ConstraintID(block="A", index=1)
        with self.assertRaises((AttributeError, TypeError)):
            cid.block = "B"

    def test_hashable(self):
        cid = ConstraintID(block="A", index=(1, 2))
        s = {cid}
        self.assertIn(cid, s)


class TestDeletionFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, cls.result = _setup()
        model      = cls.result.compensation_test.model
        algo       = DeletionFilter(model, solver="glpk")
        cls.iis_list, cls.n_solves = algo.run()

    def test_returns_list(self):
        self.assertIsInstance(self.iis_list, list)

    def test_iis_nonempty(self):
        self.assertGreater(len(self.iis_list), 0)

    def test_iis_matches_known(self):
        found = {str(c) for c in self.iis_list}
        self.assertEqual(found, KNOWN_IIS)

    def test_n_solves_positive(self):
        self.assertGreater(self.n_solves, 0)

    def test_model_restored_after_run(self):
        """All constraints must be active after the algorithm finishes."""
        from pyomo.environ import Constraint
        model = self.result.compensation_test.model
        for c in model.component_objects(Constraint, active=True):
            for idx in c:
                self.assertTrue(c[idx].active,
                                f"{c.name}[{idx}] was left deactivated")

    def test_iis_is_irreducible(self):
        """
        Removing any single IIS constraint must make the system feasible.
        This is the definition of irreducibility.
        """
        from pyomo.environ import SolverFactory
        model = self.result.compensation_test.model
        slv   = SolverFactory("glpk")

        for cid in self.iis_list:
            comp = model.component(cid.block)
            idx  = cid.index
            comp[idx].deactivate()
            r  = slv.solve(model, tee=False)
            st = str(r.solver.termination_condition)
            comp[idx].activate()
            self.assertIn(st, ("optimal", "feasible"),
                f"Removing {cid} did not make system feasible (status={st}) "
                "— IIS is not irreducible")

    def test_iis_is_infeasible(self):
        """
        The full IIS constraint set must be collectively infeasible.
        """
        from pyomo.environ import Constraint, SolverFactory
        model = self.result.compensation_test.model
        slv   = SolverFactory("glpk")

        # Deactivate everything not in IIS
        iis_set = {(c.block, c.index) for c in self.iis_list}
        deactivated = []
        for comp in model.component_objects(Constraint, active=True):
            for idx in comp:
                if (comp.name, idx) not in iis_set:
                    comp[idx].deactivate()
                    deactivated.append((comp, idx))

        r  = slv.solve(model, tee=False)
        st = str(r.solver.termination_condition)

        for (comp, idx) in deactivated:
            comp[idx].activate()

        self.assertNotIn(st, ("optimal", "feasible"),
            "IIS constraint set alone is not infeasible")


class TestGrowingMethod(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, cls.result = _setup()
        model       = cls.result.compensation_test.model
        algo        = GrowingMethod(model, solver="glpk")
        cls.iis_list, cls.n_solves = algo.run()

    def test_returns_list(self):
        self.assertIsInstance(self.iis_list, list)

    def test_iis_nonempty(self):
        self.assertGreater(len(self.iis_list), 0)

    def test_iis_matches_known(self):
        found = {str(c) for c in self.iis_list}
        self.assertEqual(found, KNOWN_IIS)

    def test_n_solves_positive(self):
        self.assertGreater(self.n_solves, 0)

    def test_model_restored_after_run(self):
        from pyomo.environ import Constraint
        model = self.result.compensation_test.model
        for c in model.component_objects(Constraint, active=True):
            for idx in c:
                self.assertTrue(c[idx].active,
                                f"{c.name}[{idx}] was left deactivated")

    def test_iis_is_irreducible(self):
        from pyomo.environ import SolverFactory
        model = self.result.compensation_test.model
        slv   = SolverFactory("glpk")
        for cid in self.iis_list:
            comp = model.component(cid.block)
            idx  = cid.index
            comp[idx].deactivate()
            r  = slv.solve(model, tee=False)
            st = str(r.solver.termination_condition)
            comp[idx].activate()
            self.assertIn(st, ("optimal", "feasible"),
                f"Removing {cid} did not restore feasibility — not irreducible")

    def test_agrees_with_deletion_filter(self):
        """Both algorithms must produce the same IIS set."""
        model = self.result.compensation_test.model
        df    = DeletionFilter(model, solver="glpk")
        df_iis, _ = df.run()
        self.assertEqual(
            {str(c) for c in self.iis_list},
            {str(c) for c in df_iis},
            "GrowingMethod and DeletionFilter disagree on IIS"
        )


class TestIISExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data, cls.result = _setup()

    def test_returns_iis_result(self):
        iis = IISExtractor(self.data, self.result).extract()
        self.assertIsInstance(iis, IISResult)

    def test_default_algorithm_is_deletion(self):
        iis = IISExtractor(self.data, self.result).extract()
        self.assertEqual(iis.algorithm, "deletion")

    def test_growing_algorithm_works(self):
        iis = IISExtractor(self.data, self.result, algorithm="growing").extract()
        self.assertEqual(iis.algorithm, "growing")

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            IISExtractor(self.data, self.result, algorithm="fancy_algo")

    def test_feasible_model_raises(self):
        """Extracting IIS from a feasible model must raise."""
        # Build a trivially feasible model
        from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, minimize
        import copy
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=m.x, sense=minimize)

        import dataclasses
        # Patch vss_result to have is_infeasible=False
        fake_ct = dataclasses.replace(
            self.result.compensation_test,
            is_infeasible=False
        )
        fake_result = dataclasses.replace(
            self.result,
            compensation_test=fake_ct
        )
        with self.assertRaises(ValueError):
            IISExtractor(self.data, fake_result).extract()

    def test_iis_size_matches_known(self):
        iis = IISExtractor(self.data, self.result).extract()
        self.assertEqual(len(iis.constraints), len(KNOWN_IIS))

    def test_iis_set_matches_known(self):
        iis = IISExtractor(self.data, self.result).extract()
        self.assertEqual({str(c) for c in iis.constraints}, KNOWN_IIS)

    def test_str_repr(self):
        iis = IISExtractor(self.data, self.result).extract()
        s = str(iis)
        self.assertIn("IIS", s)
        for cname in ["MeetRequirement", "BeetQuotaLimit", "CompensationBudget"]:
            self.assertIn(cname, s)

    def test_explain_runs(self):
        iis = IISExtractor(self.data, self.result).extract()
        text = iis.explain()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 200)

    def test_explain_contains_key_sections(self):
        iis = IISExtractor(self.data, self.result).extract()
        text = iis.explain()
        for section in [
            "VSS ROOT-CAUSE ANALYSIS",
            "FIRST-STAGE DECISIONS",
            "COMPENSATION BUDGET",
            "IIS",
            "ROOT CAUSE SYNTHESIS",
        ]:
            self.assertIn(section, text, f"Missing section: {section}")

    def test_explain_contains_constraint_names(self):
        iis = IISExtractor(self.data, self.result).extract()
        text = iis.explain()
        for cname in ["MeetRequirement", "BeetRequirement",
                      "BeetQuotaLimit", "CompensationBudget"]:
            self.assertIn(cname, text, f"Missing constraint name: {cname}")

    def test_explain_mentions_vss(self):
        iis = IISExtractor(self.data, self.result).extract()
        text = iis.explain()
        self.assertIn("1,150", text)   # VSS value

    def test_data_and_result_stored(self):
        iis = IISExtractor(self.data, self.result).extract()
        self.assertIs(iis.data, self.data)
        self.assertIs(iis.vss_result, self.result)


class TestIISEndToEnd(unittest.TestCase):
    """Full pipeline: load → VSS analysis → IIS → explain."""

    def test_full_pipeline(self):
        data   = load_data(DATA_PATH)
        result = VSSExplainer(data, solver="glpk", integer_x=False).run()
        iis    = IISExtractor(data, result, algorithm="deletion").extract()
        text   = iis.explain()
        # The narrative must confirm VSS is positive
        self.assertIn("VSS", text)
        # Must identify scenario 1 as the bad scenario
        self.assertIn("Good", text)
        # Must mention wheat specifically
        self.assertIn("Wheat", text)
        # Must mention the beet quota
        self.assertIn("6,000", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)