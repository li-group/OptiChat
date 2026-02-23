"""
One-call robustness analysis on a Pyomo model with Option-B param handling.

What this does
--------------
1) You provide a zero-arg model factory that returns:
       (model, uncertain_params, bounds, xi_set)
   where `uncertain_params` may contain scalar or indexed Param/ParamData, and
   `bounds` align 1:1 with those entries (see bounds formats below).

2) We generate `n_scenarios` samples for EACH scalar/indexed entry using your
   scenario generator. Internally we "flatten for sampling" but keep the public
   API in Option B style. The scenario table uses canonical column names:
      - scalar:     name
      - 1D index:   name[i]
      - multi-idx:  name[i,j,...]

3) Solve the model ONCE at baseline. Snapshot variable values.

4) For each scenario:
   - Set Param/ParamData values from the scenario row (scalar & indexed supported)
   - Reapply stored variable values (no re-solve)
   - Evaluate objective and constraint feasibility

5) Save:
   - Results  -> `out_path`
   - Scenarios-> `out_path + ".scenarios.csv"`

Bounds formats (per `uncertain_params` entry)
---------------------------------------------
If the entry is:
- ParamData (scalar):                  bounds must be (lb, ub)
- scalar Param component:              bounds must be (lb, ub)
- indexed Param:                       one of:
    * single (lb, ub)       -> broadcast to all indices
    * dict {index: (lb,ub)} -> per-index mapping
    * sequence aligned to `param.keys()` order, each (lb,ub)

Distribution
------------
- `dist="uniform"` or `"normal"` (Gaussian CLIPPED to [lb, ub]).
- If you need per-parameter distributions/means/stds, extend as needed.

Notes
-----
- We DO NOT re-optimize per scenario; we evaluate the fixed baseline solution.
- To also flag var bound/domain violations after parameter changes, add a check pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.core.base.param import Param, ParamData
from pyomo.core.base.var import VarData
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.componentuid import ComponentUID
ParamLike = Union["Param", "ParamData"]

# Use your existing generator
from optichat.tools.robust_analysis.scenario_generator import generate_scenarios_from_model

__all__ = ["run_robustness"]

def _collect_var_values(model: pyo.ConcreteModel) -> Dict[str, Dict[Any, float]]:
    """
    Snapshot current values for all active Var components.

    Returns
    -------
    Dict[str, Dict[Any, float]]
        Mapping: <VarName> -> { index_or_None : value }
    """
    out: Dict[str, Dict[Any, float]] = {}
    for vcomp in model.component_objects(pyo.Var, active=True, descend_into=True):
        is_indexed = vcomp.is_indexed()
        vals: Dict[Any, float] = {}
        for v in vcomp.values():
            if not isinstance(v, VarData):
                continue
            val = pyo.value(v, exception=False)
            if val is None:
                val = 0.0
            key = v.index() if is_indexed else None
            vals[key] = float(val)
        out[vcomp.name] = vals
    return out


def _solve_once(
    model: pyo.ConcreteModel,
    solver: str = "gurobi",
    solver_options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[Any, float]]:
    """
    Solve the model once and return a snapshot of variable values.

    - Works for LP/MILP, any mix of continuous/binary/integer vars.
    - If the requested solver isn't available, falls back to: cbc → glpk → highs.

    Raises
    ------
    RuntimeError
        If no supported LP/MIP solver is available.
    """
    sf = pyo.SolverFactory(solver)
    if not sf.available():
        for cand in ("cbc", "glpk", "highs"):
            sf = pyo.SolverFactory(cand)
            if sf.available():
                break
        else:
            raise RuntimeError(f"No LP/MIP solver available among: {solver}, cbc, glpk, highs")

    if solver_options:
        for k, v in solver_options.items():
            sf.options[k] = v

    sf.solve(model, tee=False, load_solutions=True)
    return _collect_var_values(model)


def _check_constraint_feasible(c: Constraint, tol: float) -> bool:
    """
    Check feasibility of a single ConstraintData with tolerance.

    Returns
    -------
    bool
        True iff (lb - tol) <= body <= (ub + tol).
    """
    body = pyo.value(c.body, exception=False)
    lb = pyo.value(c.lower, exception=False) if c.has_lb() else None
    ub = pyo.value(c.upper, exception=False) if c.has_ub() else None
    ok_lb = True if lb is None else (body >= lb - tol)
    ok_ub = True if ub is None else (body <= ub + tol)
    return bool(ok_lb and ok_ub)



# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _comp_name(obj: Any) -> str:
    """Stable-ish component name (falls back to ComponentUID)."""
    try:
        return obj.parent_component().name  # type: ignore[attr-defined]
    except Exception:
        return str(ComponentUID(obj))


def _idx_to_str(idx: Any) -> str:
    if isinstance(idx, tuple):
        return ",".join(map(str, idx))
    return str(idx)


def _param_indexed_col_name(pname: str, idx: Any) -> str:
    """Canonical Option-B column name: name[i] or name[i,j]."""
    return f"{pname}[{_idx_to_str(idx)}]"


def _save_df(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suf = out.suffix.lower()
    if suf == ".csv":
        df.to_csv(out, index=False)
    elif suf == ".json":
        df.to_json(out, orient="records", indent=2)
    elif suf == ".parquet":
        df.to_parquet(out, index=False)
    else:
        raise ValueError("Output must end with .csv, .json, or .parquet")
    return out


def _collect_var_values(model: pyo.ConcreteModel) -> Dict[str, Dict[Any, float]]:
    """
    Snapshot current Var values: {VarName: {index_or_None: value}}.
    """
    out: Dict[str, Dict[Any, float]] = {}
    for vcomp in model.component_objects(pyo.Var, active=True, descend_into=True):
        is_indexed = vcomp.is_indexed()
        vals: Dict[Any, float] = {}
        for v in vcomp.values():
            if not isinstance(v, VarData):
                continue
            val = pyo.value(v, exception=False)
            if val is None:
                val = 0.0
            key = v.index() if is_indexed else None
            vals[key] = float(val)
        out[vcomp.name] = vals
    return out


def _apply_var_values(model: pyo.ConcreteModel, var_values: Mapping[str, Mapping[Any, float]]) -> None:
    """
    Restore Var values from snapshot (no re-solve).
    """
    for vcomp in model.component_objects(pyo.Var, active=True, descend_into=True):
        mapping = var_values.get(vcomp.name)
        if not mapping:
            continue
        is_indexed = vcomp.is_indexed()
        for v in vcomp.values():
            key = v.index() if is_indexed else None
            if key not in mapping:
                continue
            val = mapping[key]
            try:
                v.set_value(val)
            except TypeError:
                try:
                    v.set_value(val, skip_validation=True)
                except TypeError:
                    v.value = val


def _evaluate_objective(model: pyo.ConcreteModel) -> Optional[float]:
    """
    Evaluate the first active Objective expression; None if absent.
    """
    try:
        if hasattr(model, "obj"):
            return float(pyo.value(model.obj.expr))
        for obj in model.component_data_objects(pyo.Objective, active=True, descend_into=True):
            return float(pyo.value(obj.expr))
    except Exception:
        pass
    return None


def _constraint_key(c: Constraint) -> str:
    """
    Readable, stable key for a ConstraintData (e.g., myCon[i,j]).
    """
    comp = c.parent_component()
    base = comp.name
    if comp.is_indexed():
        idx = c.index()
        return _param_indexed_col_name(base, idx)
    return base


# ---------- Option-B: setter that supports scalar & indexed Params ---------- #

def _set_params_from_row(
    uncertain_params: Sequence[Union[Param, ParamData]],
    row: pd.Series,
) -> None:
    for p in uncertain_params:
        if isinstance(p, ParamData):
            # primary expected column (bare name for scalar ParamData)
            bare = _comp_name(p)
            # secondary: bracketed fallback, in case a generator produced it that way
            try:
                idx = p.index()
            except Exception:
                idx = None
            bracketed = None if (idx is None or (isinstance(idx, tuple) and len(idx) == 0)) else \
                _param_indexed_col_name(p.parent_component().name, idx)

            col = None
            if bare in row.index:
                col = bare
            elif bracketed and bracketed in row.index:
                col = bracketed

            if col is None:
                tried = [bare] + ([bracketed] if bracketed else [])
                raise KeyError(f"Scenario missing parameter column for ParamData; tried {tried}.")

            p.set_value(float(row[col]))
            continue

        if isinstance(p, Param):
            pname = p.parent_component().name if hasattr(p, "parent_component") else p.name
            if p.is_indexed():
                for k in p.keys():
                    col = _param_indexed_col_name(pname, k)
                    if col not in row.index:
                        raise KeyError(f"Scenario missing parameter column '{col}'.")
                    p[k].set_value(float(row[col]))
            else:
                pdata = next(iter(p.values()))
                default_col = _comp_name(pdata)
                col = default_col if default_col in row.index else pname
                if col not in row.index:
                    raise KeyError(
                        f"Scenario missing parameter column for scalar Param '{pname}'. "
                        f"Tried '{default_col}' and '{pname}'."
                    )
                try:
                    p.set_value(float(row[col]))
                except Exception:
                    p.value = float(row[col])
            continue

        raise TypeError(f"Unsupported param type for '{_comp_name(p)}': {type(p)}")


# --------- Internal: make generator inputs & reconcile column names ---------- #

def _expand_params_for_sampling(
    uncertain_params: Sequence[Union[Param, ParamData]],
    bounds: Sequence[Union[Tuple[float, float], Dict[Any, Tuple[float, float]], Sequence[Tuple[float, float]]]],
) -> Tuple[List[ParamData], List[Tuple[float, float]], List[str], List[str]]:
    """
    Build a FLAT list of ParamData and aligned bounds for sampling.

    Returns
    -------
    flat_params : list[ParamData]
    flat_bounds : list[(lb, ub)]
    desired_names : list[str]
        Canonical Option-B names we want in the scenario table (name or name[i], name[i,j]).
    generator_base_names : list[str]
        Names the scenario generator will initially produce (component names)
        BEFORE its duplicate disambiguation. We use these to predict the
        generator's unique column names and build a rename map.

    Notes
    -----
    The scenario generator (by default) infers names from parent component names,
    so multiple ParamData from the same component collide and are fixed by adding
    suffixes like '__2'. We precompute that disambiguation to (a) pass correct
    per-param distribution settings and (b) rename back to Option-B names.
    """
    if len(uncertain_params) != len(bounds):
        raise ValueError("`uncertain_params` and `bounds` must have the same length.")

    flat_params: List[ParamData] = []
    flat_bounds: List[Tuple[float, float]] = []
    desired_names: List[str] = []
    generator_base_names: List[str] = []

    for param_entry, b in zip(uncertain_params, bounds):
        if isinstance(param_entry, ParamData):
            if not (isinstance(b, tuple) and len(b) == 2):
                raise ValueError("Bounds for ParamData must be a single (lb, ub).")
            flat_params.append(param_entry)
            flat_bounds.append((float(b[0]), float(b[1])))

            pd_comp = param_entry.parent_component()
            comp_name = pd_comp.name
            # >> FIX: only use bracket form if index is not None/empty
            try:
                idx = param_entry.index()
            except Exception:
                idx = None
            if idx is None or (isinstance(idx, tuple) and len(idx) == 0):
                desired_names.append(comp_name)                 # e.g., 'rhs_eq'
            else:
                desired_names.append(_param_indexed_col_name(comp_name, idx))  # e.g., 'rhs[i]'
            generator_base_names.append(comp_name)
            continue

        if isinstance(param_entry, Param):
            if param_entry.is_indexed():
                keys = list(param_entry.keys())
                # normalize bounds b
                if isinstance(b, tuple) and len(b) == 2:
                    b_list = [b] * len(keys)
                elif isinstance(b, dict):
                    b_list = [b[k] for k in keys]
                elif isinstance(b, (list, tuple)) and len(b) == len(keys) and all(
                    isinstance(x, tuple) and len(x) == 2 for x in b
                ):
                    b_list = list(b)
                else:
                    raise ValueError(
                        f"Bounds for indexed Param '{param_entry.name}' must be "
                        f"(lb, ub), or dict{{index:(lb,ub)}}, or a sequence aligned to keys()."
                    )

                for k, (lb, ub) in zip(keys, b_list):
                    pd = param_entry[k]
                    flat_params.append(pd)
                    flat_bounds.append((float(lb), float(ub)))
                    desired_names.append(_param_indexed_col_name(param_entry.name, k))
                    generator_base_names.append(param_entry.name)
            else:
                # scalar Param component
                if not (isinstance(b, tuple) and len(b) == 2):
                    raise ValueError(f"Bounds for scalar Param '{param_entry.name}' must be a single (lb, ub).")
                pd = next(iter(param_entry.values()))
                flat_params.append(pd)
                flat_bounds.append((float(b[0]), float(b[1])))
                desired_names.append(param_entry.name)
                generator_base_names.append(param_entry.name)
            continue

        raise TypeError(f"Unsupported uncertain param type: {type(param_entry)}")

    return flat_params, flat_bounds, desired_names, generator_base_names


def _disambiguate_like_generator(basenames: List[str]) -> List[str]:
    """
    Reproduce the scenario generator's uniqueness policy:
    if duplicates, suffix subsequent occurrences with '__k' (1-based k>1).
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for nm in basenames:
        c = seen.get(nm, 0)
        out.append(nm if c == 0 else f"{nm}__{c+1}")
        seen[nm] = c + 1
    return out


# --------------------------------------------------------------------------- #
# Public: One-call API                                                        #
# --------------------------------------------------------------------------- #

def run_robustness(
    *,
    model: pyo.ConcreteModel,
    uncertain_params: Sequence[ParamLike],
    bounds: List,
    n_scenarios: int,
    dist: str = "uniform",
    seed: Optional[int] = None,
    out_path: Optional[str] = "robust_results.csv",
    tol: float = 1e-6,
    solver: str = "gurobi",
) -> pd.DataFrame:
    """
    Solve once, evaluate across sampled scenarios (Option B: scalar & indexed Params).

    Parameters
    ----------
    model : Callable[[], (model, uncertain_params, bounds, xi_set)]
        Zero-argument factory.
    n_scenarios : int
        Number of scenarios to sample.
    dist : {"uniform", "normal"}, default "uniform"
        Global distribution for ALL entries (normal is clipped to [lb, ub]).
    seed : int | None, default None
        RNG seed.
    out_path : str, default "robust_results.csv"
        Results path (.csv | .json | .parquet). Scenarios saved next to it as
        `out_path + ".scenarios.csv"`.
    tol : float, default 1e-6
        Constraint feasibility tolerance.
    solver : str, default "gurobi"
        Primary solver; fallbacks: cbc → glpk → highs.

    Returns
    -------
    pandas.DataFrame
        Columns:
          scenario_id,
          <Option-B param columns...>,
          objective,
          <constraint_1>, <constraint_2>, ...
    """
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")

    dist = (dist or "uniform").strip().lower()
    if dist not in ("uniform", "normal"):
        raise ValueError("dist must be 'uniform' or 'normal'.")

    # 1) Build model & unpack uncertainty
    model_obj = model  # noqa: F841

    # 2) Build FLAT ParamData + bounds for sampling, and compute both:
    #    - desired Option-B names (name[index]) for final table
    #    - generator unique names, to pass per-param dist and later rename
    flat_params, flat_bounds, desired_names, gen_basenames = _expand_params_for_sampling(
        uncertain_params, bounds
    )
    gen_unames = _disambiguate_like_generator(gen_basenames)

    # per-param dist mapping uses the generator's unique names
    per_param_dist = {nm: dist for nm in gen_unames}

    # 3) Generate scenarios using your generator
    scenarios = generate_scenarios_from_model(
        uncertain_params=flat_params,      # ParamData only
        bounds=flat_bounds,                # 1:1 with flat_params
        n=n_scenarios,
        seed=seed,
        per_param_dist=per_param_dist,     # drive uniform/normal
        per_param_normal=None,             # defaults if normal
    )

    # The generator's column order is ["scenario_id"] + gen_unames.
    # Build a rename map to canonical Option-B names.
    rename_map = {old: new for old, new in zip([c for c in scenarios.columns if c != "scenario_id"], desired_names)}
    scenarios = scenarios.rename(columns=rename_map)

    # Save the exact scenarios used (with Option-B column names)
    scenarios_out = Path(out_path).with_suffix(Path(out_path).suffix + ".scenarios.csv")
    scenarios.to_csv(scenarios_out, index=False)

    # 4) Collect baseline variable values (model already solved before arriving here)
    stored_vars = _collect_var_values(model_obj)

    # 5) Constraints roster
    con_list: List[Constraint] = list(
        model_obj.component_data_objects(pyo.Constraint, active=True, descend_into=True)
    )
    con_cols: List[str] = [_constraint_key(c) for c in con_list]

    # 6) Evaluate stored solution per scenario (no re-solve)
    param_cols = [c for c in scenarios.columns if c != "scenario_id"]
    rows: List[Dict[str, Any]] = []
    for _, srow in scenarios.iterrows():
        # Set params from Option-B columns
        _set_params_from_row(uncertain_params, srow)
        _apply_var_values(model_obj, stored_vars)
        obj_val = _evaluate_objective(model_obj)

        out_row: Dict[str, Any] = {"scenario_id": int(srow["scenario_id"])}
        # Echo all param columns (already Option-B names)
        for nm in param_cols:
            out_row[nm] = float(srow[nm]) if nm != "scenario_id" else int(srow[nm])
        out_row["objective"] = obj_val
        # Constraint feasibility flags
        for c, cname in zip(con_list, con_cols):
            out_row[cname] = 0 if _check_constraint_feasible(c, tol=tol) else 1

        rows.append(out_row)

    results = pd.DataFrame(rows)

    # 7) Persist & return
    _save_df(results, out_path)
    print(f"[OK] Wrote {len(scenarios)} scenarios -> {scenarios_out.name}")
    print(f"[OK] Wrote results -> {Path(out_path).name} (shape {results.shape[0]} x {results.shape[1]})")
    return results
