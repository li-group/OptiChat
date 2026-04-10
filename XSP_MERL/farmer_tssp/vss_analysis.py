"""
farmer_tssp/vss_analysis.py
───────────────────────────
Implements the VSS explanation framework from the paper:
  "Two-stage stochastic programs with finite support"

The pipeline proceeds in four stages:

  Stage 0  –  Prerequisites
               Solve RP → x*, z_SP, Q_s(x*)
               Solve EV → x_EV
               Evaluate EEV → Q_s(x_EV), z_EEV
               Compute VSS = z_EEV - z_SP

  Stage 1  –  Global infeasibility  [eq. (3)]
               Can ANY recourse policy under x_EV match z_SP?
               Build feasibility problem: fix x=x_EV, add budget constraint
               ≤ z_SP.  If infeasible → confirmed x_EV cannot match stochastic
               optimum regardless of recourse.

  Stage 2  –  Scenario-level infeasibility  [eq. (4)]
               For each scenario s separately: can recourse under x_EV match
               Q_s(x*)?  If infeasible in scenario s → EV strictly worse in s.

  Stage 3  –  Minimum bad-scenario set  [eq. (5)]
               Partition scenarios into G (good: Δ_s ≤ 0) and B (bad: Δ_s > 0).
               Compute compensation budget C_comp.
               Rank bad scenarios by p_s·Δ_s desc; greedily pick minimum prefix
               K ⊆ B such that Σ_{s∈K} p_s·Δ_s + C_comp > 0.

  Stage 4  –  Compensation test  [eq. (6)]
               Given K, build the restricted feasibility problem: can x_EV be
               rescued within K by granting it the full compensation budget?
               If infeasible → K scenarios alone force EV to exceed the stochastic
               benchmark.  IIS of this system (left to caller) identifies
               which operational constraints are responsible.

All classes are stateless given their constructor inputs; call .solve() or
.run() to populate results.  All arithmetic follows the paper's sign convention:
costs are positive when the problem is framed as a minimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from pyomo.environ import (
    ConcreteModel, Constraint, NonNegativeReals, Objective,
    Param, Set as PySet, SolverFactory, Var, minimize, value,
)

from .data_loader import FarmerData
from .model import build_extensive_form, build_mean_value, build_wait_and_see


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_lp(model: ConcreteModel, solver: str = "glpk") -> Tuple[str, float]:
    """Solve *model* and return (termination_condition, objective_value)."""
    slv = SolverFactory(solver)
    result = slv.solve(model, tee=False)
    status = str(result.solver.termination_condition)
    for obj in model.component_objects(Objective, active=True):
        return status, value(obj)
    raise RuntimeError("No active Objective found in model.")


def _is_optimal(status: str) -> bool:
    return status in ("optimal", "feasible")


def _recourse_cost_given_x(
    data: FarmerData,
    x_fixed: Dict[int, float],
    scenario_id: int,
    solver: str = "glpk",
) -> float:
    """
    Compute Q_s(x_fixed) – the optimal second-stage cost in scenario *s*
    when first-stage decisions are fixed to *x_fixed*.

    Corresponds to:
        Q_s(x) = min_{y_s} { q_s^T y_s : T_s x + W_s y_s >= h_s, y_s ∈ Y_s }
    """
    m = _build_recourse_subproblem(data, x_fixed, scenario_id)
    status, obj = _solve_lp(m, solver)
    if not _is_optimal(status):
        raise RuntimeError(
            f"Recourse subproblem infeasible/unbounded in scenario {scenario_id}: {status}"
        )
    return obj


def _build_recourse_subproblem(
    data: FarmerData,
    x_fixed: Dict[int, float],
    scenario_id: int,
) -> ConcreteModel:
    """
    Build a pure recourse LP for scenario *scenario_id* with x fixed.
    Variables: y (purchases) and w (sales).
    Objective: second-stage cost only (no planting cost).
    """
    s  = scenario_id
    sc = data.scenarios[s]

    m = ConcreteModel(name=f"Recourse_s{s}")
    m.PURCH = PySet(initialize=data.purch_ids, ordered=True)
    m.SELL  = PySet(initialize=data.sell_ids,  ordered=True)
    m.CROPS = PySet(initialize=data.crop_ids,  ordered=True)

    m.PurchPrice = Param(m.PURCH, initialize=data.purchase_price)
    m.SellPrice  = Param(m.SELL,  initialize=data.sell_price)
    m.MinReq     = Param(m.CROPS, initialize=data.min_requirement)
    m.BeetQuota  = Param(initialize=data.beet_quota)
    m.Yield      = Param(m.CROPS, initialize=sc.yields)
    m.X          = Param(m.CROPS, initialize=x_fixed)

    m.y = Var(m.PURCH, domain=NonNegativeReals)
    m.w = Var(m.SELL,  domain=NonNegativeReals)

    @m.Constraint(m.PURCH)
    def MeetRequirement(m, j):
        return m.Yield[j] * m.X[j] + m.y[j] - m.w[j] >= m.MinReq[j]

    @m.Constraint()
    def BeetRequirement(m):
        return m.Yield[3] * m.X[3] - m.w[3] - m.w[4] >= m.MinReq[3]

    @m.Constraint()
    def BeetQuotaLimit(m):
        return m.w[3] <= m.BeetQuota

    @m.Constraint(m.PURCH)
    def SellOnlyHarvested(m, j):
        return m.w[j] <= m.Yield[j] * m.X[j]

    @m.Objective(sense=minimize)
    def RecourseCost(m):
        return (sum(m.PurchPrice[j] * m.y[j] for j in m.PURCH)
                - sum(m.SellPrice[k]  * m.w[k] for k in m.SELL))

    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Data containers  (returned by each stage)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Prerequisites:
    """
    All quantities that every later stage needs.

    Attributes
    ----------
    x_sp        : first-stage solution of the recourse problem
    x_ev        : first-stage solution of the EV (mean-value) problem
    z_sp        : optimal stochastic objective  z^SP
    z_eev       : expected cost of using x_EV   z^EEV
    vss         : Value of the Stochastic Solution = z_EEV - z_SP
    c_x_sp      : first-stage cost  c^T x*
    c_x_ev      : first-stage cost  c^T x_EV
    delta_0     : first-stage difference  c^T x_EV - c^T x*
    Qs_sp       : {s: Q_s(x*)}   recourse costs under stochastic solution
    Qs_ev       : {s: Q_s(x_EV)} recourse costs under EV solution
    delta_s     : {s: Q_s(x_EV) - Q_s(x*)}  scenario-wise regret
    probabilities: {s: p_s}
    """
    x_sp:          Dict[int, float]
    x_ev:          Dict[int, float]
    z_sp:          float
    z_eev:         float
    vss:           float
    c_x_sp:        float
    c_x_ev:        float
    delta_0:       float
    Qs_sp:         Dict[int, float]
    Qs_ev:         Dict[int, float]
    delta_s:       Dict[int, float]
    probabilities: Dict[int, float]

    def scenario_ids(self) -> List[int]:
        return sorted(self.delta_s.keys())

    def __str__(self) -> str:
        lines = [
            "── Prerequisites ────────────────────────────────────",
            f"  z_SP  (stochastic optimum)   : {self.z_sp:>12.4f}",
            f"  z_EEV (EV solution evaluated) : {self.z_eev:>12.4f}",
            f"  VSS                           : {self.vss:>12.4f}",
            f"  Δ₀  (first-stage diff)        : {self.delta_0:>12.4f}",
            "",
            "  Scenario-wise regret  Δ_s = Q_s(x_EV) - Q_s(x*):",
        ]
        for s in self.scenario_ids():
            p  = self.probabilities[s]
            ds = self.delta_s[s]
            lines.append(
                f"    s={s}  p={p:.4f}  Q_s(x*)={self.Qs_sp[s]:>10.4f}"
                f"  Q_s(x_EV)={self.Qs_ev[s]:>10.4f}  Δ_s={ds:>10.4f}"
            )
        return "\n".join(lines)


@dataclass
class GlobalInfeasibilityResult:
    """
    Result of Stage 1: global infeasibility test  [eq. (3)].

    Attributes
    ----------
    is_infeasible  : True ⟺ no recourse policy can rescue x_EV to match z_SP
    status         : solver termination condition
    model          : the Pyomo feasibility model (useful for IIS extraction)
    budget_rhs     : right-hand-side of the budget constraint  (z_SP - c^T x_EV)
    """
    is_infeasible: bool
    status:        str
    model:         ConcreteModel
    budget_rhs:    float

    def __str__(self) -> str:
        verdict = "INFEASIBLE – x_EV cannot match z_SP" if self.is_infeasible \
                  else "FEASIBLE – x_EV *could* match z_SP with suitable recourse"
        return (
            "── Global infeasibility test (eq. 3) ───────────────\n"
            f"  Budget RHS (z_SP - c^T x_EV) : {self.budget_rhs:>12.4f}\n"
            f"  Solver status                 : {self.status}\n"
            f"  Verdict                       : {verdict}"
        )


@dataclass
class ScenarioInfeasibilityResult:
    """
    Result of Stage 2: scenario-level infeasibility  [eq. (4)].

    Attributes
    ----------
    infeasible_scenarios : set of scenario ids where x_EV strictly worse than x*
    feasible_scenarios   : set of scenario ids where x_EV can match Q_s(x*)
    per_scenario         : {s: {"is_infeasible": bool, "status": str, "model": ...}}
    """
    infeasible_scenarios: Set[int]
    feasible_scenarios:   Set[int]
    per_scenario:         Dict[int, dict]

    def __str__(self) -> str:
        lines = ["── Scenario-level infeasibility (eq. 4) ────────────"]
        for s in sorted(self.per_scenario):
            flag = "INFEASIBLE" if s in self.infeasible_scenarios else "feasible  "
            lines.append(f"  s={s}  {flag}")
        if self.infeasible_scenarios:
            lines.append(
                f"  → EV is strictly worse in scenarios: {sorted(self.infeasible_scenarios)}"
            )
        return "\n".join(lines)


@dataclass
class BadScenarioSet:
    """
    Result of Stage 3: minimum set of bad scenarios K  [eq. (5)].

    Attributes
    ----------
    G               : good scenario ids  (Δ_s ≤ 0)
    B               : bad  scenario ids  (Δ_s > 0)
    K               : minimum prefix of B that explains VSS > 0
    C_comp          : compensation budget from G and Δ₀
    weighted_regrets: {s: p_s * Δ_s} for s ∈ B, sorted desc
    cumulative_vss  : running sum Σ_{s∈K} p_s Δ_s + C_comp at each prefix size
    """
    G:               FrozenSet[int]
    B:               FrozenSet[int]
    K:               List[int]          # ordered: highest p_s*Δ_s first
    C_comp:          float
    weighted_regrets: Dict[int, float]  # s → p_s * Δ_s  for s ∈ B
    cumulative_vss:  List[float]        # one entry per prefix length

    def __str__(self) -> str:
        lines = [
            "── Minimum bad-scenario set (eq. 5) ────────────────",
            f"  Good scenarios G         : {sorted(self.G)}",
            f"  Bad  scenarios B         : {sorted(self.B)}",
            f"  Compensation budget C_comp: {self.C_comp:>12.4f}",
            "",
            "  Bad scenarios ranked by p_s·Δ_s (desc):",
        ]
        for i, s in enumerate(sorted(self.weighted_regrets,
                                     key=lambda x: -self.weighted_regrets[x])):
            in_K = "← in K" if s in self.K else ""
            lines.append(
                f"    rank {i+1}  s={s}  p_s·Δ_s={self.weighted_regrets[s]:>10.4f}"
                f"  cumulative+C_comp={self.cumulative_vss[i] if i < len(self.cumulative_vss) else '':>10.4f}"
                f"  {in_K}"
            )
        lines.append(f"\n  Minimum K = {self.K}  (|K|={len(self.K)})")
        return "\n".join(lines)


@dataclass
class CompensationTestResult:
    """
    Result of Stage 4: compensation feasibility test  [eq. (6)].

    Attributes
    ----------
    is_infeasible  : True ⟺ K scenarios alone force EV to exceed stochastic
                     benchmark even after granting full compensation budget
    status         : solver termination condition
    model          : the Pyomo feasibility model (IIS-ready for caller)
    K              : the bad-scenario set used
    budget_rhs     : Σ_{s∈K} p_s Q_s(x*) - C_comp  (RHS of budget constraint)
    """
    is_infeasible: bool
    status:        str
    model:         ConcreteModel
    K:             List[int]
    budget_rhs:    float

    def __str__(self) -> str:
        verdict = (
            "INFEASIBLE – K alone forces EV to exceed stochastic benchmark"
            if self.is_infeasible else
            "FEASIBLE – compensation budget is sufficient within K"
        )
        return (
            "── Compensation test (eq. 6) ────────────────────────\n"
            f"  Bad-scenario set K        : {self.K}\n"
            f"  Budget RHS                 : {self.budget_rhs:>12.4f}\n"
            f"  Solver status              : {self.status}\n"
            f"  Verdict                    : {verdict}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 0  –  Prerequisites
# ═══════════════════════════════════════════════════════════════════════════════

class PrerequisitesSolver:
    """
    Computes all quantities required by Stages 1–4.

    Steps
    -----
    1. Solve extensive form (RP) → x*, z_SP, c^T x*
    2. Solve EV (mean-value)     → x_EV, c^T x_EV
    3. Evaluate recourse costs   → Q_s(x*) and Q_s(x_EV) for all s
    4. Derive Δ₀, Δ_s, z_EEV, VSS
    """

    def __init__(self, data: FarmerData, solver: str = "glpk",
                 integer_x: bool = False):
        """
        Parameters
        ----------
        data       : FarmerData
        solver     : solver name recognised by Pyomo (default "glpk")
        integer_x  : whether x is integer; LP relaxation is usually sufficient
                     for the analysis and is faster
        """
        self.data      = data
        self.solver    = solver
        self.integer_x = integer_x

    # ── public ────────────────────────────────────────────────────────────────

    def solve(self) -> Prerequisites:
        """Run all prerequisite solves and return a :class:`Prerequisites`."""
        data = self.data

        # ── Step 1: RP ────────────────────────────────────────────────────────
        ef = build_extensive_form(data, integer_x=self.integer_x)
        status_rp, z_sp = _solve_lp(ef, self.solver)
        if not _is_optimal(status_rp):
            raise RuntimeError(f"RP solve failed: {status_rp}")

        x_sp       = {i: value(ef.x[i]) for i in data.crop_ids}
        c_x_sp     = value(ef.FirstStageCost)

        # ── Step 2: EV ────────────────────────────────────────────────────────
        mv         = build_mean_value(data, integer_x=False)
        status_ev, _ = _solve_lp(mv, self.solver)
        if not _is_optimal(status_ev):
            raise RuntimeError(f"EV solve failed: {status_ev}")

        x_ev       = {i: value(mv.x[i]) for i in data.crop_ids}
        c_x_ev     = sum(data.planting_cost[i] * x_ev[i] for i in data.crop_ids)

        # ── Step 3: recourse costs Q_s(x*) and Q_s(x_EV) ─────────────────────
        Qs_sp: Dict[int, float] = {}
        Qs_ev: Dict[int, float] = {}

        for s in data.scenario_ids:
            Qs_sp[s] = _recourse_cost_given_x(data, x_sp,  s, self.solver)
            Qs_ev[s] = _recourse_cost_given_x(data, x_ev,  s, self.solver)

        # ── Step 4: derived quantities ────────────────────────────────────────
        probs   = data.probabilities
        z_eev   = c_x_ev + sum(probs[s] * Qs_ev[s] for s in data.scenario_ids)
        vss     = z_eev - z_sp
        delta_0 = c_x_ev - c_x_sp
        delta_s = {s: Qs_ev[s] - Qs_sp[s] for s in data.scenario_ids}

        return Prerequisites(
            x_sp          = x_sp,
            x_ev          = x_ev,
            z_sp          = z_sp,
            z_eev         = z_eev,
            vss           = vss,
            c_x_sp        = c_x_sp,
            c_x_ev        = c_x_ev,
            delta_0       = delta_0,
            Qs_sp         = Qs_sp,
            Qs_ev         = Qs_ev,
            delta_s       = delta_s,
            probabilities = probs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1  –  Global infeasibility  [eq. (3)]
# ═══════════════════════════════════════════════════════════════════════════════

class GlobalInfeasibilityTester:
    """
    Tests whether x_EV can be paired with recourse decisions that achieve z_SP.

    Builds the feasibility system (eq. 3):

        find   {y_s}_{s ∈ S}
        s.t.   T_s x_EV + W_s y_s >= h_s,  y_s ∈ Y_s,  ∀ s
               c^T x_EV + Σ_s p_s q_s^T y_s ≤ z_SP

    Infeasibility ⟺ no recourse policy can rescue x_EV sufficiently.
    """

    def __init__(self, data: FarmerData, prereqs: Prerequisites,
                 solver: str = "glpk"):
        self.data    = data
        self.prereqs = prereqs
        self.solver  = solver

    # ── public ────────────────────────────────────────────────────────────────

    def test(self) -> GlobalInfeasibilityResult:
        """Build and solve the global feasibility problem."""
        m          = self._build_model()
        status, _  = self._solve_feasibility(m)
        infeasible = not _is_optimal(status)
        budget_rhs = self.prereqs.z_sp - self.prereqs.c_x_ev

        return GlobalInfeasibilityResult(
            is_infeasible = infeasible,
            status        = status,
            model         = m,
            budget_rhs    = budget_rhs,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_model(self) -> ConcreteModel:
        """
        Construct the feasibility LP:
            Fix x = x_EV in the EF recourse constraints.
            Add budget constraint Σ p_s * (recourse cost) ≤ z_SP - c^T x_EV.
            Objective: minimise 0 (pure feasibility).
        """
        data    = self.data
        prereqs = self.prereqs
        x_ev    = prereqs.x_ev

        m = ConcreteModel(name="GlobalFeasibility")
        m.CROPS     = PySet(initialize=data.crop_ids,     ordered=True)
        m.PURCH     = PySet(initialize=data.purch_ids,    ordered=True)
        m.SELL      = PySet(initialize=data.sell_ids,     ordered=True)
        m.SCENARIOS = PySet(initialize=data.scenario_ids, ordered=True)

        m.PurchPrice  = Param(m.PURCH,              initialize=data.purchase_price)
        m.SellPrice   = Param(m.SELL,               initialize=data.sell_price)
        m.MinReq      = Param(m.CROPS,              initialize=data.min_requirement)
        m.BeetQuota   = Param(initialize=data.beet_quota)
        m.Probability = Param(m.SCENARIOS,          initialize=data.probabilities)
        m.Yield       = Param(m.SCENARIOS, m.CROPS,
                              initialize={(s, c): data.yield_of(s, c)
                                          for s in data.scenario_ids
                                          for c in data.crop_ids})
        m.X_EV        = Param(m.CROPS,              initialize=x_ev)

        # Recourse variables
        m.y = Var(m.SCENARIOS, m.PURCH, domain=NonNegativeReals)
        m.w = Var(m.SCENARIOS, m.SELL,  domain=NonNegativeReals)

        # ── Recourse constraints (eq. 3b) ─────────────────────────────────────
        @m.Constraint(m.SCENARIOS, m.PURCH)
        def MeetRequirement(m, s, j):
            return (m.Yield[s, j] * m.X_EV[j] + m.y[s, j]
                    - m.w[s, j] >= m.MinReq[j])

        @m.Constraint(m.SCENARIOS)
        def BeetRequirement(m, s):
            return m.Yield[s, 3] * m.X_EV[3] - m.w[s, 3] - m.w[s, 4] >= m.MinReq[3]

        @m.Constraint(m.SCENARIOS)
        def BeetQuotaLimit(m, s):
            return m.w[s, 3] <= m.BeetQuota

        @m.Constraint(m.SCENARIOS, m.PURCH)
        def SellOnlyHarvested(m, s, j):
            return m.w[s, j] <= m.Yield[s, j] * m.X_EV[j]

        # ── Budget constraint (eq. 3c): Σ p_s * recourse_cost ≤ z_SP - c^T x_EV
        budget_rhs = prereqs.z_sp - prereqs.c_x_ev

        @m.Constraint()
        def GlobalBudget(m):
            recourse_expr = sum(
                m.Probability[s] * (
                    sum(m.PurchPrice[j] * m.y[s, j] for j in m.PURCH)
                    - sum(m.SellPrice[k]  * m.w[s, k] for k in m.SELL)
                )
                for s in m.SCENARIOS
            )
            return recourse_expr <= budget_rhs

        # Feasibility objective
        @m.Objective(sense=minimize)
        def FeasObj(m):
            return 0.0

        return m

    def _solve_feasibility(self, model: ConcreteModel) -> Tuple[str, float]:
        return _solve_lp(model, self.solver)


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2  –  Scenario-level infeasibility  [eq. (4)]
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioInfeasibilityTester:
    """
    For each scenario s, tests whether recourse under x_EV can match Q_s(x*).

    Feasibility problem per scenario (eq. 4):

        find   y_s
        s.t.   T_s x_EV + W_s y_s >= h_s,  y_s ∈ Y_s
               q_s^T y_s ≤ Q_s(x*)

    Infeasibility in scenario s ⟺ EV strictly more expensive in s.
    """

    def __init__(self, data: FarmerData, prereqs: Prerequisites,
                 solver: str = "glpk"):
        self.data    = data
        self.prereqs = prereqs
        self.solver  = solver

    # ── public ────────────────────────────────────────────────────────────────

    def test_all(self) -> ScenarioInfeasibilityResult:
        """Run scenario-level test for every scenario and aggregate."""
        per_scenario: Dict[int, dict] = {}
        infeasible_set: Set[int] = set()
        feasible_set:   Set[int] = set()

        for s in self.data.scenario_ids:
            model, status = self._test_scenario(s)
            inf = not _is_optimal(status)
            per_scenario[s] = {
                "is_infeasible": inf,
                "status":        status,
                "model":         model,
                "Q_s_star":      self.prereqs.Qs_sp[s],
                "Q_s_ev":        self.prereqs.Qs_ev[s],
                "delta_s":       self.prereqs.delta_s[s],
            }
            (infeasible_set if inf else feasible_set).add(s)

        return ScenarioInfeasibilityResult(
            infeasible_scenarios = infeasible_set,
            feasible_scenarios   = feasible_set,
            per_scenario         = per_scenario,
        )

    def test_scenario(self, s: int) -> dict:
        """Run scenario-level test for a single scenario *s*."""
        model, status = self._test_scenario(s)
        inf = not _is_optimal(status)
        return {
            "scenario":      s,
            "is_infeasible": inf,
            "status":        status,
            "model":         model,
            "Q_s_star":      self.prereqs.Qs_sp[s],
            "Q_s_ev":        self.prereqs.Qs_ev[s],
            "delta_s":       self.prereqs.delta_s[s],
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _test_scenario(self, s: int) -> Tuple[ConcreteModel, str]:
        model  = self._build_scenario_model(s)
        status, _ = _solve_lp(model, self.solver)
        return model, status

    def _build_scenario_model(self, s: int) -> ConcreteModel:
        """
        Feasibility LP for scenario s (eq. 4):
            Fix x = x_EV.
            Add: recourse cost ≤ Q_s(x*).
        """
        data    = self.data
        prereqs = self.prereqs
        sc      = data.scenarios[s]
        x_ev    = prereqs.x_ev
        q_cap   = prereqs.Qs_sp[s]     # Q_s(x*) – the cap

        m = ConcreteModel(name=f"ScenFeas_s{s}")
        m.PURCH = PySet(initialize=data.purch_ids, ordered=True)
        m.SELL  = PySet(initialize=data.sell_ids,  ordered=True)
        m.CROPS = PySet(initialize=data.crop_ids,  ordered=True)

        m.PurchPrice = Param(m.PURCH, initialize=data.purchase_price)
        m.SellPrice  = Param(m.SELL,  initialize=data.sell_price)
        m.MinReq     = Param(m.CROPS, initialize=data.min_requirement)
        m.BeetQuota  = Param(initialize=data.beet_quota)
        m.Yield      = Param(m.CROPS, initialize=sc.yields)
        m.X_EV       = Param(m.CROPS, initialize=x_ev)

        m.y = Var(m.PURCH, domain=NonNegativeReals)
        m.w = Var(m.SELL,  domain=NonNegativeReals)

        # Recourse constraints (eq. 4b)
        @m.Constraint(m.PURCH)
        def MeetRequirement(m, j):
            return m.Yield[j] * m.X_EV[j] + m.y[j] - m.w[j] >= m.MinReq[j]

        @m.Constraint()
        def BeetRequirement(m):
            return m.Yield[3] * m.X_EV[3] - m.w[3] - m.w[4] >= m.MinReq[3]

        @m.Constraint()
        def BeetQuotaLimit(m):
            return m.w[3] <= m.BeetQuota

        @m.Constraint(m.PURCH)
        def SellOnlyHarvested(m, j):
            return m.w[j] <= m.Yield[j] * m.X_EV[j]

        # Cost cap (eq. 4c): q_s^T y_s ≤ Q_s(x*)
        @m.Constraint()
        def RecourseCostCap(m):
            recourse = (sum(m.PurchPrice[j] * m.y[j] for j in m.PURCH)
                        - sum(m.SellPrice[k]  * m.w[k] for k in m.SELL))
            return recourse <= q_cap

        @m.Objective(sense=minimize)
        def FeasObj(m):
            return 0.0

        return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3  –  Minimum bad-scenario set  [eq. (5)]
# ═══════════════════════════════════════════════════════════════════════════════

class MinBadScenarioFinder:
    """
    Identifies the minimum cardinality subset K ⊆ B of bad scenarios whose
    weighted regrets exceed the compensation budget (eq. 5).

    The partition is:
        G = { s : Δ_s ≤ 0 }   (good scenarios)
        B = { s : Δ_s > 0 }   (bad  scenarios)

    Compensation budget:
        C_comp = Δ₀ + Σ_{s∈G} p_s Δ_s

    We rank B by p_s Δ_s descending and take the smallest prefix K such that:
        Σ_{s∈K} p_s Δ_s + C_comp > 0
    """

    def __init__(self, prereqs: Prerequisites, tol: float = 1e-8):
        """
        Parameters
        ----------
        prereqs : Prerequisites
        tol     : numerical tolerance for Δ_s > 0 classification
        """
        self.prereqs = prereqs
        self.tol     = tol

    # ── public ────────────────────────────────────────────────────────────────

    def find(self) -> BadScenarioSet:
        """Compute and return the minimum bad-scenario set."""
        prereqs = self.prereqs

        if prereqs.vss <= self.tol:
            raise ValueError(
                f"VSS = {prereqs.vss:.6f} ≤ 0: stochastic solution offers no "
                "advantage; bad-scenario analysis is not meaningful."
            )

        probs   = prereqs.probabilities
        delta_s = prereqs.delta_s

        # ── Partition ─────────────────────────────────────────────────────────
        G = frozenset(s for s in delta_s if delta_s[s] <= self.tol)
        B = frozenset(s for s in delta_s if delta_s[s]  > self.tol)

        # ── Compensation budget ───────────────────────────────────────────────
        C_comp = prereqs.delta_0 + sum(probs[s] * delta_s[s] for s in G)

        # ── Rank B by p_s * Δ_s descending ───────────────────────────────────
        weighted_regrets: Dict[int, float] = {s: probs[s] * delta_s[s] for s in B}
        ranked_B = sorted(B, key=lambda s: -weighted_regrets[s])

        # ── Greedy prefix to satisfy Σ p_s Δ_s + C_comp > 0 ─────────────────
        cumulative: List[float] = []
        running = C_comp
        K: List[int] = []

        for s in ranked_B:
            running += weighted_regrets[s]
            cumulative.append(running)
            K.append(s)
            if running > self.tol:
                break
        else:
            # Should not reach here if VSS > 0 by paper's decomposition
            raise RuntimeError(
                "Could not find K satisfying the compensation condition "
                "despite VSS > 0. Check numerical tolerances."
            )

        return BadScenarioSet(
            G                = G,
            B                = B,
            K                = K,
            C_comp           = C_comp,
            weighted_regrets = weighted_regrets,
            cumulative_vss   = cumulative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4  –  Compensation test  [eq. (6)]
# ═══════════════════════════════════════════════════════════════════════════════

class CompensationTester:
    """
    Tests whether x_EV can match the stochastic benchmark when restricted to
    bad scenarios K and granted the full compensation budget  [eq. (6)].

    Feasibility problem:

        find   {y_s}_{s∈K}
        s.t.   T_s x_EV + W_s y_s >= h_s,  y_s ∈ Y_s,   ∀ s ∈ K        (6b)
               Σ_{s∈K} p_s q_s^T y_s ≤ Σ_{s∈K} p_s Q_s(x*) - C_comp    (6c)

    Infeasibility ⟺ the bad scenarios in K alone force x_EV to exceed
    the stochastic benchmark even with the full compensation budget.
    The Pyomo model is returned so the caller can extract an IIS.
    """

    def __init__(self, data: FarmerData, prereqs: Prerequisites,
                 bad_set: BadScenarioSet, solver: str = "glpk"):
        self.data    = data
        self.prereqs = prereqs
        self.bad_set = bad_set
        self.solver  = solver

    # ── public ────────────────────────────────────────────────────────────────

    def test(self) -> CompensationTestResult:
        """Build and solve the compensation feasibility problem."""
        m          = self._build_model()
        status, _  = _solve_lp(m, self.solver)
        infeasible = not _is_optimal(status)

        budget_rhs = self._compute_budget_rhs()

        return CompensationTestResult(
            is_infeasible = infeasible,
            status        = status,
            model         = m,
            K             = self.bad_set.K,
            budget_rhs    = budget_rhs,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_budget_rhs(self) -> float:
        """
        RHS of the budget constraint (eq. 6c):
            Σ_{s∈K} p_s Q_s(x*) - C_comp
        """
        probs   = self.prereqs.probabilities
        Qs_sp   = self.prereqs.Qs_sp
        K       = self.bad_set.K
        C_comp  = self.bad_set.C_comp
        return sum(probs[s] * Qs_sp[s] for s in K) - C_comp

    def _build_model(self) -> ConcreteModel:
        """
        Construct the compensation feasibility LP (eq. 6):
            Restrict to scenarios K.
            Fix x = x_EV.
            Add budget constraint over K only.
        """
        data    = self.data
        prereqs = self.prereqs
        K       = self.bad_set.K
        x_ev    = prereqs.x_ev
        probs   = prereqs.probabilities

        m = ConcreteModel(name="CompensationFeasibility")
        m.K_SET  = PySet(initialize=K,                 ordered=True)
        m.PURCH  = PySet(initialize=data.purch_ids,    ordered=True)
        m.SELL   = PySet(initialize=data.sell_ids,     ordered=True)
        m.CROPS  = PySet(initialize=data.crop_ids,     ordered=True)

        m.PurchPrice  = Param(m.PURCH,           initialize=data.purchase_price)
        m.SellPrice   = Param(m.SELL,            initialize=data.sell_price)
        m.MinReq      = Param(m.CROPS,           initialize=data.min_requirement)
        m.BeetQuota   = Param(initialize=data.beet_quota)
        m.Probability = Param(m.K_SET,           initialize={s: probs[s] for s in K})
        m.Yield       = Param(m.K_SET, m.CROPS,
                              initialize={(s, c): data.yield_of(s, c)
                                          for s in K
                                          for c in data.crop_ids})
        m.X_EV        = Param(m.CROPS,           initialize=x_ev)

        # Recourse variables for scenarios in K only
        m.y = Var(m.K_SET, m.PURCH, domain=NonNegativeReals)
        m.w = Var(m.K_SET, m.SELL,  domain=NonNegativeReals)

        # ── Recourse constraints (eq. 6b) ─────────────────────────────────────
        @m.Constraint(m.K_SET, m.PURCH)
        def MeetRequirement(m, s, j):
            return (m.Yield[s, j] * m.X_EV[j] + m.y[s, j]
                    - m.w[s, j] >= m.MinReq[j])

        @m.Constraint(m.K_SET)
        def BeetRequirement(m, s):
            return m.Yield[s, 3] * m.X_EV[3] - m.w[s, 3] - m.w[s, 4] >= m.MinReq[3]

        @m.Constraint(m.K_SET)
        def BeetQuotaLimit(m, s):
            return m.w[s, 3] <= m.BeetQuota

        @m.Constraint(m.K_SET, m.PURCH)
        def SellOnlyHarvested(m, s, j):
            return m.w[s, j] <= m.Yield[s, j] * m.X_EV[j]

        # ── Budget constraint (eq. 6c) ─────────────────────────────────────────
        budget_rhs = self._compute_budget_rhs()

        @m.Constraint()
        def CompensationBudget(m):
            weighted_recourse = sum(
                m.Probability[s] * (
                    sum(m.PurchPrice[j] * m.y[s, j] for j in m.PURCH)
                    - sum(m.SellPrice[k]  * m.w[s, k] for k in m.SELL)
                )
                for s in m.K_SET
            )
            return weighted_recourse <= budget_rhs

        @m.Objective(sense=minimize)
        def FeasObj(m):
            return 0.0

        return m


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator  –  full pipeline in one call
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VSSAnalysisResult:
    """
    Aggregated result of the full VSS explanation pipeline.

    Attributes
    ----------
    prereqs          : Prerequisites
    global_test      : GlobalInfeasibilityResult
    scenario_test    : ScenarioInfeasibilityResult
    bad_set          : BadScenarioSet
    compensation_test: CompensationTestResult
    """
    prereqs:           Prerequisites
    global_test:       GlobalInfeasibilityResult
    scenario_test:     ScenarioInfeasibilityResult
    bad_set:           BadScenarioSet
    compensation_test: CompensationTestResult

    def summary(self) -> str:
        sep = "\n" + "─" * 53 + "\n"
        return sep.join([
            str(self.prereqs),
            str(self.global_test),
            str(self.scenario_test),
            str(self.bad_set),
            str(self.compensation_test),
            (
                "IIS extraction: call .compensation_test.model\n"
                "to inspect the Pyomo model for constraint analysis."
            ),
        ])


class VSSExplainer:
    """
    Orchestrates the full four-stage VSS explanation pipeline.

    Usage
    -----
    >>> from farmer_tssp.data_loader import load_data
    >>> from farmer_tssp.vss_analysis import VSSExplainer
    >>> data   = load_data("farmer_tssp/data/farmer_data.json")
    >>> result = VSSExplainer(data).run()
    >>> print(result.summary())
    """

    def __init__(self, data: FarmerData, solver: str = "glpk",
                 integer_x: bool = False, tol: float = 1e-6):
        """
        Parameters
        ----------
        data       : FarmerData
        solver     : Pyomo solver name
        integer_x  : use integer x in RP (False = LP relaxation, faster)
        tol        : numerical tolerance for scenario classification
        """
        self.data      = data
        self.solver    = solver
        self.integer_x = integer_x
        self.tol       = tol

    def run(self) -> VSSAnalysisResult:
        """Execute all four stages and return a :class:`VSSAnalysisResult`."""
        # Stage 0
        prereqs = PrerequisitesSolver(
            self.data, self.solver, self.integer_x
        ).solve()

        # Stage 1
        global_test = GlobalInfeasibilityTester(
            self.data, prereqs, self.solver
        ).test()

        # Stage 2
        scenario_test = ScenarioInfeasibilityTester(
            self.data, prereqs, self.solver
        ).test_all()

        # Stage 3
        bad_set = MinBadScenarioFinder(prereqs, tol=self.tol).find()

        # Stage 4
        comp_test = CompensationTester(
            self.data, prereqs, bad_set, self.solver
        ).test()

        return VSSAnalysisResult(
            prereqs           = prereqs,
            global_test       = global_test,
            scenario_test     = scenario_test,
            bad_set           = bad_set,
            compensation_test = comp_test,
        )
    
