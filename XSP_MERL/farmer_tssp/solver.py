"""
farmer_tssp/solver.py
─────────────────────
High-level solve routines and stochastic-programming metrics:
  • RP   – Recourse Problem  (extensive form)
  • WS   – Wait-and-See      (perfect information)
  • EV   – Expected-Value problem
  • EEV  – Expected result of using EV solution
  • EVPI – Expected Value of Perfect Information  = RP − WS
  • VSS  – Value of Stochastic Solution           = EEV − RP
"""

from __future__ import annotations
from typing import Dict

from pyomo.environ import value, SolverFactory, Objective, ConcreteModel

# from farmer_tssp.data_loader import FarmerData
# from farmer_tssp.model import build_extensive_form, build_wait_and_see, build_mean_value
# from data_loader import FarmerData
# from model import build_extensive_form, build_wait_and_see, build_mean_value
from .data_loader import FarmerData
from .model import build_extensive_form, build_wait_and_see, build_mean_value

# ── internal helper ─────────────────────────────────────────────────────────

def _solve(model: ConcreteModel, solver: str = "glpk") -> tuple[str, float]:
    """Solve model; return (termination_condition, objective_value)."""
    slv    = SolverFactory(solver)
    result = slv.solve(model, tee=False)
    status = str(result.solver.termination_condition)
    for obj in model.component_objects(Objective, active=True):
        return status, value(obj)
    raise RuntimeError("No active objective found in model")


# ── RP ──────────────────────────────────────────────────────────────────────

def solve_rp(data: FarmerData,
             solver: str = "glpk",
             integer_x: bool = True) -> Dict:
    """Solve the Recourse Problem (full extensive form)."""
    m = build_extensive_form(data, integer_x=integer_x)
    status, obj = _solve(m, solver)

    x = {i: value(m.x[i]) for i in data.crop_ids}
    y = {(s, j): value(m.y[s, j])
         for s in data.scenario_ids for j in data.purch_ids}
    w = {(s, k): value(m.w[s, k])
         for s in data.scenario_ids for k in data.sell_ids}
    scenario_costs = {s: value(m.ScenarioCost[s]) for s in data.scenario_ids}

    return {
        "status":           status,
        "objective":        obj,
        "first_stage_cost": value(m.FirstStageCost),
        "x": x, "y": y, "w": w,
        "scenario_costs":   scenario_costs,
        "model":            m,
    }


# ── WS ──────────────────────────────────────────────────────────────────────

def solve_ws(data: FarmerData,
             solver: str = "glpk",
             integer_x: bool = True) -> Dict:
    """Solve each scenario under perfect information; return weighted WS cost."""
    ws_costs: Dict[int, float] = {}
    ws_solutions: Dict[int, dict] = {}

    for s in data.scenario_ids:
        ms = build_wait_and_see(data, s, integer_x=integer_x)
        status, c = _solve(ms, solver)
        ws_costs[s] = c
        ws_solutions[s] = {
            "status": status,
            "cost": c,
            "x": {i: value(ms.x[i]) for i in data.crop_ids},
            "y": {j: value(ms.y[j]) for j in data.purch_ids},
            "w": {k: value(ms.w[k]) for k in data.sell_ids},
        }

    ws = sum(data.scenarios[s].probability * ws_costs[s]
             for s in data.scenario_ids)
    return {"ws": ws, "per_scenario": ws_solutions}


# ── EV & EEV ────────────────────────────────────────────────────────────────

def solve_ev_and_eev(data: FarmerData,
                     solver: str = "glpk",
                     integer_x: bool = True) -> Dict:
    """1. Solve EV (mean-value). 2. Fix x to EV solution and evaluate EEV."""
    mv = build_mean_value(data, integer_x=False)
    _, ev_obj = _solve(mv, solver)
    x_ev = {i: value(mv.x[i]) for i in data.crop_ids}

    m_eev = build_extensive_form(data, integer_x=False)
    for i in data.crop_ids:
        m_eev.x[i].fix(x_ev[i])
    _, eev_obj = _solve(m_eev, solver)

    return {"ev_objective": ev_obj, "x_ev": x_ev, "eev": eev_obj}


# ── All metrics ─────────────────────────────────────────────────────────────

def compute_metrics(data: FarmerData,
                    solver: str = "glpk",
                    integer_x: bool = True) -> Dict:
    """Compute RP, WS, EV, EEV, EVPI, VSS in one call."""
    rp_res  = solve_rp(data, solver, integer_x)
    ws_res  = solve_ws(data, solver, integer_x)
    eev_res = solve_ev_and_eev(data, solver, integer_x)

    rp  = rp_res["objective"]
    ws  = ws_res["ws"]
    eev = eev_res["eev"]

    return {
        "RP":   rp,
        "WS":   ws,
        "EV":   eev_res["ev_objective"],
        "EEV":  eev,
        "EVPI": rp - ws,
        "VSS":  eev - rp,
        "rp_detail":  rp_res,
        "ws_detail":  ws_res,
        "eev_detail": eev_res,
    }