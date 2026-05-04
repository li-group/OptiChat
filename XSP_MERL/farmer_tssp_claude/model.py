"""
farmer_tssp/model.py
────────────────────
Two-Stage Stochastic Programming model for the Farmer problem.

Stage 1  (here-and-now):   x[i]   – acres devoted to crop i            (INTEGER)
Stage 2  (recourse / wait-and-see, indexed by scenario s):
           y[s,j]  – tonnes of crop j purchased
           w[s,k]  – tonnes sold in sell-slot k
           (slots: 1=wheat, 2=corn, 3=beets≤6000T, 4=beets>6000T)

The EEV and EVPI benchmarks are also provided as separate builders.
"""

from __future__ import annotations
from typing import Optional
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint, Expression,
    NonNegativeReals, NonNegativeIntegers, Integers,
    minimize, summation, value, SolverFactory
)

# from data_loader import FarmerData
from .data_loader import FarmerData

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  EXTENSIVE FORM  (EF)  – the main TSSP model
# ═══════════════════════════════════════════════════════════════════════════════

def build_extensive_form(data: FarmerData, integer_x: bool = True) -> ConcreteModel:
    """
    Build and return the extensive form of the two-stage stochastic
    farmer problem.

    Parameters
    ----------
    data       : FarmerData   – loaded from JSON
    integer_x  : bool         – if True x is integer (matching Julia model);
                                set False for LP relaxation

    Returns
    -------
    ConcreteModel (unsolved)
    """
    m = ConcreteModel(name="Farmer_TSSP_EF")

    # ── Sets ──────────────────────────────────────────────────────────────────
    m.CROPS     = Set(initialize=data.crop_ids,     ordered=True)
    m.PURCH     = Set(initialize=data.purch_ids,    ordered=True)
    m.SELL      = Set(initialize=data.sell_ids,     ordered=True)
    m.SCENARIOS = Set(initialize=data.scenario_ids, ordered=True)

    # ── Parameters ────────────────────────────────────────────────────────────
    m.Budget        = Param(initialize=data.budget)
    m.BeetQuota     = Param(initialize=data.beet_quota)
    m.PlantCost     = Param(m.CROPS,  initialize=data.planting_cost)
    m.PurchPrice    = Param(m.PURCH,  initialize=data.purchase_price)
    m.SellPrice     = Param(m.SELL,   initialize=data.sell_price)
    m.MinReq        = Param(m.CROPS,  initialize=data.min_requirement)
    m.Probability   = Param(m.SCENARIOS, initialize=data.probabilities)
    m.Yield         = Param(m.SCENARIOS, m.CROPS,
                            initialize={(s, c): data.yield_of(s, c)
                                        for s in data.scenario_ids
                                        for c in data.crop_ids})

    # ── Stage-1 variables  (shared across all scenarios) ──────────────────────
    dom = NonNegativeIntegers if integer_x else NonNegativeReals
    m.x = Var(m.CROPS, domain=dom, bounds=(0, data.budget))

    # ── Stage-2 variables  (one copy per scenario) ────────────────────────────
    m.y = Var(m.SCENARIOS, m.PURCH, domain=NonNegativeReals)   # purchase tonnes
    m.w = Var(m.SCENARIOS, m.SELL,  domain=NonNegativeReals)   # sell   tonnes

    # ── Constraints ───────────────────────────────────────────────────────────

    @m.Constraint()
    def TotalAcreage(m):
        """Stage-1: cannot plant more than the budget (total acreage)."""
        return sum(m.x[i] for i in m.CROPS) <= m.Budget

    @m.Constraint(m.SCENARIOS, m.PURCH)
    def MeetRequirement(m, s, j):
        """Stage-2: harvested + purchased – sold ≥ minimum feed requirement."""
        return (m.Yield[s, j] * m.x[j] + m.y[s, j]
                - m.w[s, j] >= m.MinReq[j])

    @m.Constraint(m.SCENARIOS)
    def BeetRequirement(m, s):
        """Stage-2: sugar-beet feed requirement (slot 3 = crop 3)."""
        return m.Yield[s, 3] * m.x[3] - m.w[s, 3] - m.w[s, 4] >= m.MinReq[3]

    @m.Constraint(m.SCENARIOS)
    def BeetQuotaLimit(m, s):
        """Stage-2: beets sold at high price capped by quota."""
        return m.w[s, 3] <= m.BeetQuota

    @m.Constraint(m.SCENARIOS, m.PURCH)
    def SellOnlyHarvested(m, s, j):
        """Stage-2: cannot sell more wheat/corn than harvested."""
        return m.w[s, j] <= m.Yield[s, j] * m.x[j]

    # ── Stage cost expressions  (useful for reporting) ─────────────────────────

    @m.Expression()
    def FirstStageCost(m):
        return sum(m.PlantCost[i] * m.x[i] for i in m.CROPS)

    @m.Expression(m.SCENARIOS)
    def SecondStageCost(m, s):
        purchase_cost = sum(m.PurchPrice[j] * m.y[s, j] for j in m.PURCH)
        sell_revenue  = sum(m.SellPrice[k]  * m.w[s, k] for k in m.SELL)
        return purchase_cost - sell_revenue

    @m.Expression(m.SCENARIOS)
    def ScenarioCost(m, s):
        return m.FirstStageCost + m.SecondStageCost[s]

    # ── Objective ─────────────────────────────────────────────────────────────

    @m.Objective(sense=minimize)
    def ExpectedCost(m):
        """Minimise E[total cost] = planting + expected recourse cost."""
        return m.FirstStageCost + sum(
            m.Probability[s] * m.SecondStageCost[s]
            for s in m.SCENARIOS
        )

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  WAIT-AND-SEE  (WS) – perfect information lower bound  → used for EVPI
# ═══════════════════════════════════════════════════════════════════════════════

def build_wait_and_see(data: FarmerData, scenario_id: int,
                       integer_x: bool = True) -> ConcreteModel:
    """
    Single-scenario deterministic model solved with *perfect* knowledge of
    scenario *scenario_id*.  Solving all scenarios and taking the
    probability-weighted average gives WS (= lower bound for EVPI).
    """
    s   = scenario_id
    sc  = data.scenarios[s]

    m = ConcreteModel(name=f"Farmer_WS_s{s}")
    m.CROPS = Set(initialize=data.crop_ids, ordered=True)
    m.PURCH = Set(initialize=data.purch_ids, ordered=True)
    m.SELL  = Set(initialize=data.sell_ids,  ordered=True)

    m.Budget      = Param(initialize=data.budget)
    m.BeetQuota   = Param(initialize=data.beet_quota)
    m.PlantCost   = Param(m.CROPS, initialize=data.planting_cost)
    m.PurchPrice  = Param(m.PURCH, initialize=data.purchase_price)
    m.SellPrice   = Param(m.SELL,  initialize=data.sell_price)
    m.MinReq      = Param(m.CROPS, initialize=data.min_requirement)
    m.Yield       = Param(m.CROPS, initialize=sc.yields)

    dom = NonNegativeIntegers if integer_x else NonNegativeReals
    m.x = Var(m.CROPS, domain=dom, bounds=(0, data.budget))
    m.y = Var(m.PURCH, domain=NonNegativeReals)
    m.w = Var(m.SELL,  domain=NonNegativeReals)

    @m.Constraint()
    def TotalAcreage(m):
        return sum(m.x[i] for i in m.CROPS) <= m.Budget

    @m.Constraint(m.PURCH)
    def MeetRequirement(m, j):
        return m.Yield[j] * m.x[j] + m.y[j] - m.w[j] >= m.MinReq[j]

    @m.Constraint()
    def BeetRequirement(m):
        return m.Yield[3] * m.x[3] - m.w[3] - m.w[4] >= m.MinReq[3]

    @m.Constraint()
    def BeetQuotaLimit(m):
        return m.w[3] <= m.BeetQuota

    @m.Constraint(m.PURCH)
    def SellOnlyHarvested(m, j):
        return m.w[j] <= m.Yield[j] * m.x[j]

    @m.Objective(sense=minimize)
    def TotalCost(m):
        return (sum(m.PlantCost[i] * m.x[i] for i in m.CROPS)
                + sum(m.PurchPrice[j] * m.y[j] for j in m.PURCH)
                - sum(m.SellPrice[k] * m.w[k] for k in m.SELL))

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  EEV  – expected value of using the EV solution
#     Step 1: solve the mean-value deterministic problem  (EV)
#     Step 2: fix x to EV solution, evaluate over all scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def build_mean_value(data: FarmerData, integer_x: bool = False) -> ConcreteModel:
    """
    Deterministic model that replaces stochastic yields with their
    probability-weighted expected value.  Used to compute the EV solution
    (and then EEV).
    """
    expected_yield = {
        c: sum(data.scenarios[s].probability * data.yield_of(s, c)
               for s in data.scenario_ids)
        for c in data.crop_ids
    }

    m = ConcreteModel(name="Farmer_EV")
    m.CROPS = Set(initialize=data.crop_ids, ordered=True)
    m.PURCH = Set(initialize=data.purch_ids, ordered=True)
    m.SELL  = Set(initialize=data.sell_ids,  ordered=True)

    m.Budget      = Param(initialize=data.budget)
    m.BeetQuota   = Param(initialize=data.beet_quota)
    m.PlantCost   = Param(m.CROPS, initialize=data.planting_cost)
    m.PurchPrice  = Param(m.PURCH, initialize=data.purchase_price)
    m.SellPrice   = Param(m.SELL,  initialize=data.sell_price)
    m.MinReq      = Param(m.CROPS, initialize=data.min_requirement)
    m.Yield       = Param(m.CROPS, initialize=expected_yield)

    dom = NonNegativeIntegers if integer_x else NonNegativeReals
    m.x = Var(m.CROPS, domain=dom, bounds=(0, data.budget))
    m.y = Var(m.PURCH, domain=NonNegativeReals)
    m.w = Var(m.SELL,  domain=NonNegativeReals)

    @m.Constraint()
    def TotalAcreage(m):
        return sum(m.x[i] for i in m.CROPS) <= m.Budget

    @m.Constraint(m.PURCH)
    def MeetRequirement(m, j):
        return m.Yield[j] * m.x[j] + m.y[j] - m.w[j] >= m.MinReq[j]

    @m.Constraint()
    def BeetRequirement(m):
        return m.Yield[3] * m.x[3] - m.w[3] - m.w[4] >= m.MinReq[3]

    @m.Constraint()
    def BeetQuotaLimit(m):
        return m.w[3] <= m.BeetQuota

    @m.Constraint(m.PURCH)
    def SellOnlyHarvested(m, j):
        return m.w[j] <= m.Yield[j] * m.x[j]

    @m.Objective(sense=minimize)
    def TotalCost(m):
        return (sum(m.PlantCost[i] * m.x[i] for i in m.CROPS)
                + sum(m.PurchPrice[j] * m.y[j] for j in m.PURCH)
                - sum(m.SellPrice[k] * m.w[k] for k in m.SELL))

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Solver wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def solve_model(model: ConcreteModel,
                solver: str = "glpk",
                tee:    bool = False) -> dict:
    """
    Solve *model* and return a result dict with status and objective value.

    Returns
    -------
    dict with keys: status, termination, objective
    """
    slv    = SolverFactory(solver)
    result = slv.solve(model, tee=tee)
    status = str(result.solver.termination_condition)
    obj    = value(model.component(model.component_map(Objective).keys().__iter__().__next__()))
    return {"status": status, "objective": obj, "result": result}