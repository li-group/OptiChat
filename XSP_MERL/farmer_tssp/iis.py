"""
farmer_tssp/iis.py
──────────────────
Irreducible Infeasible Subsystem (IIS) extraction and human-readable
explanation for the VSS compensation feasibility model  [eq. (6)].

Since GLPK does not natively expose an IIS, we implement two algorithms:

  1. DeletionFilter   – the classic Chinneck (2007) deletion-filter algorithm.
                        Iterates over all constraints; deactivates each one,
                        re-solves, and keeps it in the IIS if its removal
                        makes the problem feasible.  O(|C|) solves.

  2. GrowingMethod    – a faster additive approach (Amaldi et al. 2003).
                        Starts from the empty set and greedily adds constraints
                        until the system becomes infeasible; the last added
                        constraint is always in the IIS.  Repeated to find a
                        full IIS.  Generally fewer solves on sparse instances.

Both return an :class:`IISResult` dataclass whose `.explain()` method prints
the full human-readable root-cause narrative.

Usage
─────
    from farmer_tssp.data_loader import load_data
    from farmer_tssp.vss_analysis import VSSExplainer
    from farmer_tssp.iis import IISExtractor, DeletionFilter

    data   = load_data("farmer_tssp/data/farmer_data.json")
    result = VSSExplainer(data).run()

    iis = IISExtractor(data, result).extract()
    print(iis.explain())
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pyomo.environ import (
    Constraint, ConcreteModel, Objective, SolverFactory, value
)

from .data_loader import FarmerData
from .vss_analysis import VSSAnalysisResult


# ── Constraint descriptor ────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConstraintID:
    """Uniquely identifies one constraint row in a Pyomo model."""
    block:  str          # component name, e.g. "MeetRequirement"
    index:  object       # index tuple / scalar, e.g. (1, 2) or None

    def __str__(self) -> str:
        idx = f"[{self.index}]" if self.index is not None else ""
        return f"{self.block}{idx}"


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class IISResult:
    """
    The IIS of the compensation feasibility model together with
    a human-readable explanation.

    Attributes
    ----------
    constraints   : ordered list of :class:`ConstraintID` in the IIS
    algorithm     : name of the algorithm that produced this IIS
    n_solves      : number of LP solves performed
    data          : FarmerData (for names / parameters)
    vss_result    : full VSSAnalysisResult (for numbers)
    """
    constraints:  List[ConstraintID]
    algorithm:    str
    n_solves:     int
    data:         FarmerData
    vss_result:   VSSAnalysisResult

    # ── pretty printing ───────────────────────────────────────────────────────

    def __str__(self) -> str:
        lines = [
            f"IIS  ({self.algorithm}, {self.n_solves} solves)",
            f"{'─'*48}",
        ]
        for cid in self.constraints:
            lines.append(f"  {cid}")
        return "\n".join(lines)

    def explain(self) -> str:
        """
        Return the full root-cause narrative: what each IIS constraint means
        operationally, why it binds under x_EV, and why that explains VSS > 0.
        """
        p      = self.vss_result.prereqs
        bss    = self.vss_result.bad_set
        ct     = self.vss_result.compensation_test
        data   = self.data

        x_ev   = p.x_ev
        x_sp   = p.x_sp
        probs  = p.probabilities

        W = 64

        def hr(c="─"): return c * W
        def wrap(txt, indent=4):
            return textwrap.fill(txt, width=W, initial_indent=" "*indent,
                                 subsequent_indent=" "*indent)

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            hr("═"),
            "  VSS ROOT-CAUSE ANALYSIS via IIS",
            hr("═"),
            "",
            wrap(
                f"VSS = {p.vss:,.2f}  (stochastic model saves ${abs(p.vss):,.2f} "
                f"over the EV solution). "
                f"The IIS below proves why x_EV cannot match z_SP = {p.z_sp:,.2f} "
                f"even when granted the full compensation budget of "
                f"${abs(bss.C_comp):,.2f}.", indent=2),
            "",
        ]

        # ── Stage decisions ───────────────────────────────────────────────────
        lines += [
            hr(),
            "  FIRST-STAGE DECISIONS",
            hr(),
            f"  {'Crop':<18} {'x*  (stochastic)':>18} {'x_EV (mean-value)':>18} {'Diff':>10}",
            hr(),
        ]
        for i in data.crop_ids:
            diff = x_ev[i] - x_sp[i]
            sign = "+" if diff >= 0 else ""
            lines.append(
                f"  {data.crop_names[i]:<18} {x_sp[i]:>17.1f}  "
                f"{x_ev[i]:>17.1f}  {sign}{diff:>8.1f}"
            )
        lines += [hr(), ""]

        # ── Compensation budget ───────────────────────────────────────────────
        lines += [
            hr(),
            "  COMPENSATION BUDGET  (eq. 5 / 6)",
            hr(),
            f"  Δ₀  (first-stage cost diff, c^T x_EV - c^T x*)  : {p.delta_0:>12,.2f}",
        ]
        for s in data.scenario_ids:
            ps  = probs[s]
            ds  = p.delta_s[s]
            tag = "← G" if s in bss.G else "← B"
            lines.append(
                f"  p_{s}·Δ_{s}  (s={s} {data.scen_names[s]:<8})              "
                f": {ps*ds:>12,.2f}  {tag}"
            )
        lines += [
            f"  {'─'*54}",
            f"  C_comp  (total good-side compensation)           : {bss.C_comp:>12,.2f}",
            f"  Σ_K p_s·Δ_s  (bad-scenario weighted regret)      : "
            f"{sum(probs[s]*p.delta_s[s] for s in bss.K):>12,.2f}",
            f"  VSS                                              : {p.vss:>12,.2f}",
            "",
            f"  Budget RHS for compensation test (eq. 6c)        : {ct.budget_rhs:>12,.2f}",
            "",
        ]

        # ── IIS ────────────────────────────────────────────────────────────────
        lines += [
            hr("═"),
            f"  IIS  ({len(self.constraints)} constraints, {self.algorithm}, {self.n_solves} solves)",
            hr("═"),
            "",
            wrap(
                "Each constraint below is NECESSARY for infeasibility: removing "
                "any single one makes the compensation problem feasible. Together "
                "they form the tightest possible certificate that x_EV cannot "
                "match z_SP.", indent=2),
            "",
        ]

        for cid in self.constraints:
            lines += self._explain_constraint(cid)
            lines.append("")

        # ── Synthesis ─────────────────────────────────────────────────────────
        lines += [
            hr("═"),
            "  ROOT CAUSE SYNTHESIS",
            hr("═"),
            "",
        ]
        lines.append(wrap(
            f"x_EV allocates {x_ev[1]:.0f} acres to wheat vs {x_sp[1]:.0f} for x*. "
            f"In the good-harvest scenario (s={bss.K[0]}, yield={data.yield_of(bss.K[0],1)} T/ac) "
            f"this gap costs {abs(p.delta_s[bss.K[0]]):,.0f} in forgone wheat revenue "
            f"(160T less to sell at ${data.sell_price[1]}/T sub-quota). "
            f"The beet quota ({data.beet_quota:,.0f}T) caps upside from the extra beet "
            f"acreage x_EV planted instead, so the shortfall cannot be offset. "
            f"The compensation budget of ${abs(bss.C_comp):,.2f} (from good performance "
            f"in scenarios {sorted(bss.G)}) is insufficient to close the gap, "
            f"confirming VSS = ${p.vss:,.2f}.", indent=2))
        lines.append("")

        return "\n".join(lines)

    # ── per-constraint narrative ───────────────────────────────────────────────

    def _explain_constraint(self, cid: ConstraintID) -> List[str]:
        """Return a list of explanation lines for one IIS constraint."""
        p    = self.vss_result.prereqs
        data = self.data
        x_ev = p.x_ev
        bss  = self.vss_result.bad_set
        ct   = self.vss_result.compensation_test

        block = cid.block
        idx   = cid.index

        lines = [f"  ▸ {cid}"]

        if block == "MeetRequirement":
            s, j = idx
            sc   = data.scenarios[s]
            harv = sc.yields[j] * x_ev[j]
            req  = data.min_requirement[j]
            name = data.crop_names[j]
            slack = harv - req
            lines += [
                f"    Type : feed-requirement (recourse)",
                f"    Crop : {name}  |  Scenario : {s} ({data.scen_names[s]})",
                f"    Form : yield[{s},{j}] * x_EV[{j}] + y[{s},{j}] - w[{s},{j}] >= MinReq[{j}]",
                f"           {sc.yields[j]} × {x_ev[j]:.1f} + y - w  >=  {req}",
                f"           {harv:.1f} + y - w  >=  {req}",
                f"    Note : harvest alone gives {harv:.1f}T — {'+' if slack>=0 else ''}{slack:.1f}T "
                f"{'above' if slack>=0 else 'below'} the minimum.",
                f"    Role : forces a floor on how much recourse can net-sell,"
                f" limiting revenue.",
            ]

        elif block == "BeetRequirement":
            s    = idx
            sc   = data.scenarios[s]
            harv = sc.yields[3] * x_ev[3]
            lines += [
                f"    Type : beet feed-requirement (recourse)",
                f"    Scenario : {s} ({data.scen_names[s]})",
                f"    Form : yield[{s},3] * x_EV[3] - w[{s},3] - w[{s},4] >= 0",
                f"           {sc.yields[3]} × {x_ev[3]:.1f} - w3 - w4  >=  0",
                f"           Total beet harvest = {harv:.1f}T",
                f"    Role : caps total beet sales at {harv:.1f}T; "
                f"together with BeetQuotaLimit this limits high-price quota sales.",
            ]

        elif block == "BeetQuotaLimit":
            s   = idx
            qta = data.beet_quota
            rev = data.sell_price[3] * qta
            lines += [
                f"    Type : beet price-quota cap (recourse)",
                f"    Scenario : {s} ({data.scen_names[s]})",
                f"    Form : w[{s},3] <= {qta:,.0f}",
                f"    Note : maximum revenue at high beet price = "
                f"{qta:,.0f}T × ${data.sell_price[3]}/T = ${rev:,.0f}.",
                f"    Role : hard ceiling on beet revenue — prevents compensation"
                f" for losses elsewhere.",
            ]

        elif block == "SellOnlyHarvested":
            s, j = idx
            sc   = data.scenarios[s]
            name = data.crop_names[j]
            harv = sc.yields[j] * x_ev[j]
            lines += [
                f"    Type : no-short-selling (recourse)",
                f"    Crop : {name}  |  Scenario : {s} ({data.scen_names[s]})",
                f"    Form : w[{s},{j}] <= yield[{s},{j}] * x_EV[{j}]",
                f"           w[{s},{j}] <= {sc.yields[j]} × {x_ev[j]:.1f} = {harv:.1f}T",
                f"    Role : sales cannot exceed harvest; "
                f"x_EV's smaller {name.lower()} area "
                f"({x_ev[j]:.1f}ac vs {p.x_sp[j]:.1f}ac for x*) "
                f"directly limits {name.lower()} revenue.",
            ]

        elif block == "CompensationBudget":
            K      = bss.K
            probs  = p.probabilities
            budget = ct.budget_rhs
            Qs_sum = sum(probs[s] * p.Qs_sp[s] for s in K)
            lines += [
                f"    Type : global budget constraint  (eq. 6c)",
                f"    Form : Σ_{{s∈K}} p_s · recourse(y,w)  <=  {budget:,.2f}",
                f"         = Σ_K p_s·Q_s(x*)  -  C_comp",
                f"         = {Qs_sum:,.2f}  -  ({bss.C_comp:,.2f})",
                f"         = {budget:,.2f}",
                f"    Note : recourse under x_EV cannot be made cheap enough "
                f"to satisfy this limit given all the operational constraints above.",
                f"    Role : the binding constraint that closes the IIS — "
                f"jointly with the others it makes the system infeasible.",
            ]

        else:
            lines.append(f"    (No detailed explanation registered for '{block}'.)")

        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm 1: Deletion Filter
# ═══════════════════════════════════════════════════════════════════════════════

class DeletionFilter:
    """
    Classic deletion-filter IIS algorithm (Chinneck 2007).

    For each constraint c in the model (in insertion order):
      • Deactivate c temporarily.
      • Re-solve.
      • If FEASIBLE  → c is essential; reactivate and add to IIS.
      • If INFEASIBLE → c is redundant; leave deactivated.

    Complexity: O(|C|) LP solves.  Always finds a valid IIS.
    """

    def __init__(self, model: ConcreteModel, solver: str = "glpk"):
        self.model  = model
        self.solver = solver

    def run(self) -> Tuple[List[ConstraintID], int]:
        """
        Returns
        -------
        (iis_list, n_solves) where iis_list is the ordered IIS.
        """
        model   = self.model
        slv     = SolverFactory(self.solver)
        n_solves = 0

        # Collect all active constraints
        all_constraints: List[Tuple] = []
        for comp in model.component_objects(Constraint, active=True):
            for idx in comp:
                all_constraints.append((comp, idx))

        # Verify infeasibility
        r      = slv.solve(model, tee=False)
        status = str(r.solver.termination_condition)
        n_solves += 1
        if status in ("optimal", "feasible"):
            return [], n_solves   # not infeasible

        iis: List[ConstraintID] = []

        for (comp, idx) in all_constraints:
            comp[idx].deactivate()

            r      = slv.solve(model, tee=False)
            status = str(r.solver.termination_condition)
            n_solves += 1

            if status in ("optimal", "feasible"):
                # Essential: reactivate and record
                comp[idx].activate()
                iis.append(ConstraintID(block=comp.name, index=idx))
            # else: redundant, leave deactivated

        # Restore everything so the model is unchanged for the caller
        for (comp, idx) in all_constraints:
            comp[idx].activate()

        return iis, n_solves


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm 2: Growing (Additive) Method
# ═══════════════════════════════════════════════════════════════════════════════
class GrowingMethod:
    """
    Additive / growing IIS algorithm (Amaldi et al. 2003).

    Uses a budget-last ordering: operational constraints first, the
    objective-budget constraint last.  Adds constraints one by one until the
    system becomes infeasible, then runs a deletion filter on the active set
    to trim to irreducibility.

    The budget-last heuristic ensures operational constraints that together
    with the budget make the system infeasible are discovered before the budget
    is added — giving a richer and more informative IIS.
    """

    BUDGET_NAMES = {"CompensationBudget", "GlobalBudget", "RecourseCostCap"}

    def __init__(self, model: ConcreteModel, solver: str = "glpk"):
        self.model  = model
        self.solver = solver

    def run(self) -> Tuple[List[ConstraintID], int]:
        model    = self.model
        slv      = SolverFactory(self.solver)
        n_solves = 0

        # Collect; budget-type constraints go last
        operational: List[Tuple] = []
        budget:      List[Tuple] = []
        for comp in model.component_objects(Constraint, active=True):
            for idx in comp:
                if comp.name in self.BUDGET_NAMES:
                    budget.append((comp, idx))
                else:
                    operational.append((comp, idx))
        ordered = operational + budget

        # Deactivate all
        for (comp, idx) in ordered:
            comp[idx].deactivate()

        iis:       List[ConstraintID] = []
        candidates: List[Tuple]       = list(ordered)
        activated:  List[Tuple]       = []   # track what has been activated so far

        while candidates:
            comp, idx = candidates.pop(0)
            comp[idx].activate()
            activated.append((comp, idx))

            r      = slv.solve(model, tee=False)
            status = str(r.solver.termination_condition)
            n_solves += 1

            if status not in ("optimal", "feasible"):
                # Deactivate remaining candidates (they were never added)
                for (c2, i2) in candidates:
                    c2[i2].deactivate()

                # Deletion filter over the currently-active set to trim to IIS
                # activated contains exactly the constraints that are now active
                for (c2, i2) in list(activated):
                    c2[i2].deactivate()
                    r2       = slv.solve(model, tee=False)
                    n_solves += 1
                    if str(r2.solver.termination_condition) in ("optimal", "feasible"):
                        # Essential — put back
                        c2[i2].activate()
                        iis.append(ConstraintID(block=c2.name, index=i2))
                    # else redundant — leave deactivated
                break

        # Restore all constraints to their original active state
        for (comp, idx) in ordered:
            comp[idx].activate()

        return iis, n_solves



# ═══════════════════════════════════════════════════════════════════════════════
#  High-level extractor
# ═══════════════════════════════════════════════════════════════════════════════

class IISExtractor:
    """
    Extracts an IIS from the compensation feasibility model of a
    :class:`VSSAnalysisResult` and returns a fully annotated
    :class:`IISResult`.

    Parameters
    ----------
    data       : FarmerData
    vss_result : VSSAnalysisResult  (from VSSExplainer.run())
    algorithm  : "deletion" (default) or "growing"
    solver     : Pyomo solver name (default "glpk")

    Example
    -------
    >>> from farmer_tssp.data_loader import load_data
    >>> from farmer_tssp.vss_analysis import VSSExplainer
    >>> from farmer_tssp.iis import IISExtractor
    >>>
    >>> data   = load_data("farmer_tssp/data/farmer_data.json")
    >>> result = VSSExplainer(data).run()
    >>>
    >>> iis = IISExtractor(data, result).extract()
    >>> print(iis)               # compact list
    >>> print(iis.explain())     # full narrative
    """

    ALGORITHMS = {
        "deletion": DeletionFilter,
        "growing":  GrowingMethod,
    }

    def __init__(self, data: FarmerData, vss_result: VSSAnalysisResult,
                 algorithm: str = "deletion", solver: str = "glpk"):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Choose from: {list(self.ALGORITHMS)}"
            )
        self.data       = data
        self.vss_result = vss_result
        self.algorithm  = algorithm
        self.solver     = solver

    def extract(self) -> IISResult:
        """Run the IIS algorithm and return an annotated :class:`IISResult`."""
        ct = self.vss_result.compensation_test

        if not ct.is_infeasible:
            raise ValueError(
                "Compensation test is NOT infeasible — "
                "IIS extraction requires an infeasible model. "
                f"(VSS = {self.vss_result.prereqs.vss:.4f})"
            )

        algo_cls            = self.ALGORITHMS[self.algorithm]
        algo                = algo_cls(ct.model, solver=self.solver)
        iis_constraints, n  = algo.run()

        return IISResult(
            constraints = iis_constraints,
            algorithm   = self.algorithm,
            n_solves    = n,
            data        = self.data,
            vss_result  = self.vss_result,
        )# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm 2: Growing (Additive) Method
# ═══════════════════════════════════════════════════════════════════════════════