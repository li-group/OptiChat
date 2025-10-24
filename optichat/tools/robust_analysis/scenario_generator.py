#!/usr/bin/env python3
# scenario_generator.py
"""
Scenario generator tailored to a Pyomo model factory function.

Expected model function signature (as in your example):
    model, uncertain_params, bounds, xi_set = simple_lp_model()

- `uncertain_params`: list[ParamData or Param] (scalars or arrays; here: scalars)
- `bounds`: list[tuple[lb, ub]] in the SAME ORDER as `uncertain_params`

Features
--------
- Uniform or Normal (clipped) sampling per parameter.
- Deterministic seeding for reproducibility.
- Clean DataFrame output with stable parameter names.
- Optional save to CSV / JSON / Parquet + sidecar .meta.json.
- CLI wrapper with dynamic model import.

Quick Start (Library)
---------------------
from scenario_generator import generate_scenarios_from_model, save_scenarios
from your_module import simple_lp_model

model, uncertain_params, bounds, xi_set = simple_lp_model()

df = generate_scenarios_from_model(
    uncertain_params=uncertain_params,
    bounds=bounds,
    n=1000,
    seed=42,
    per_param_dist={"rhs_eq": "uniform", "rhs_ge": "normal", "rhs_le": "uniform"},
    per_param_normal={"rhs_ge": {"mean": 20.0, "std": 5.0}},  # optional
)

save_scenarios(df, "scenarios.csv", meta={"model": "simple_lp_model"})

Quick Start (CLI)
-----------------
python scenario_generator.py \
  --n 1000 \
  --seed 42 \
  --out scenarios.csv \
  --model "your_module:simple_lp_model" \
  --dist '{"rhs_ge":"normal"}' \
  --normal '{"rhs_ge":{"mean":20,"std":5}}'

Notes
-----
- "Normal" sampling here is clipped to [lb, ub]. This is NOT a truncated normal.
- Names for parameters are inferred from Pyomo components and disambiguated if needed.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pyomo.environ as pyo  # noqa: F401 (import check)
    from pyomo.core.base.param import Param, ParamData
    from pyomo.core.base.componentuid import ComponentUID
except Exception as e:
    raise RuntimeError("Pyomo is required for scenario_generator.py") from e


# ----------------------------- Public Interface ----------------------------- #

__all__ = [
    "generate_scenarios_from_model",
    "save_scenarios",
    "main",
]


# ----------------------------- Internal Helpers ----------------------------- #

ParamLike = Union["Param", "ParamData"]
Bounds = Sequence[Tuple[float, float]]


def _uid_name(obj: Union[ParamLike, str]) -> str:
    """Return a stable-ish identifier for a Pyomo Param/ParamData or pass through a string."""
    if isinstance(obj, str):
        return obj
    return str(ComponentUID(obj))


def _infer_param_names(uncertain_params: Sequence[ParamLike]) -> List[str]:
    """
    Infer column names for each uncertainty object.

    Strategy:
    - Prefer the parent component's `name` (common for scalar Params).
    - Fall back to a ComponentUID string when needed.
    - Ensure uniqueness by suffixing duplicates with `__{k}`.
    """
    names: List[str] = []
    for p in uncertain_params:
        try:
            nm = p.parent_component().name  # type: ignore[attr-defined]
        except Exception:
            nm = _uid_name(p)
        names.append(nm)

    if len(names) != len(set(names)):
        seen: Dict[str, int] = {}
        uniq: List[str] = []
        for nm in names:
            c = seen.get(nm, 0)
            uniq.append(nm if c == 0 else f"{nm}__{c+1}")
            seen[nm] = c + 1
        names = uniq

    return names


def _validate_dimensions(uncertain_params: Sequence[ParamLike], bounds: Bounds, n: int) -> None:
    """Validate lengths and bounds format."""
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if len(uncertain_params) != len(bounds):
        raise ValueError(
            f"`uncertain_params` and `bounds` must have the same length "
            f"(got {len(uncertain_params)} vs {len(bounds)})."
        )

    for i, (lb, ub) in enumerate(bounds):
        if not (isinstance(lb, (int, float)) and isinstance(ub, (int, float))):
            raise ValueError(f"bounds[{i}] must be numeric (got {lb}, {ub}).")
        if lb >= ub:
            raise ValueError(f"bounds[{i}] requires lb < ub (got lb={lb}, ub={ub}).")


def _validate_per_param_keys(names: Sequence[str], per_param_dist: Dict[str, str], per_param_normal: Dict[str, Dict[str, float]]) -> None:
    """
    Ensure user-provided per-parameter configs only reference known names.
    Fails fast on any typo to avoid silent misconfiguration.
    """
    valid = set(names)
    bad_dist = [k for k in per_param_dist.keys() if k not in valid]
    bad_norm = [k for k in per_param_normal.keys() if k not in valid]
    if bad_dist:
        raise KeyError(f"Unknown parameter names in `per_param_dist`: {bad_dist}. Valid: {sorted(valid)}")
    if bad_norm:
        raise KeyError(f"Unknown parameter names in `per_param_normal`: {bad_norm}. Valid: {sorted(valid)}")


def _sample_uniform(lb: float, ub: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sampling on [lb, ub]."""
    return rng.uniform(lb, ub, size=n)


def _sample_normal_clipped(lb: float, ub: float, mean: float, std: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample from Normal(mean, std) and clip to [lb, ub].

    NOTE: Clipping != truncated normal. For robustness analyses where the box
    is normative and tails are not critical, this is commonly acceptable.
    """
    x = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(x, lb, ub)


# ----------------------------- Core API ----------------------------- #

def generate_scenarios_from_model(
    *,
    uncertain_params: Sequence[ParamLike],
    bounds: Bounds,
    n: int,
    seed: Optional[int] = None,
    per_param_dist: Optional[Dict[str, str]] = None,
    per_param_normal: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Generate i.i.d. scenarios for a given list of uncertain Pyomo params and their box bounds.

    Workflow (at a glance)
    ----------------------
    1) Infer stable parameter names from the provided Pyomo Params/ParamData.
    2) For each parameter j with bounds [lb_j, ub_j], draw n samples via either:
         - Uniform(lb_j, ub_j), or
         - Normal(mean_j, std_j) then clip to [lb_j, ub_j].
    3) Return an (n x (1+d)) DataFrame with a leading integer `scenario_id` column.

    Parameters
    ----------
    uncertain_params : Sequence[Param | ParamData]
        The uncertain scalars in the SAME ORDER you pass their bounds.
    bounds : Sequence[Tuple[float, float]]
        Box bounds [(lb_1, ub_1), ..., (lb_d, ub_d)] matching `uncertain_params`.
    n : int
        Number of scenarios to draw.
    seed : Optional[int], default None
        RNG seed for reproducibility. Uses numpy Generator.
    per_param_dist : Optional[Dict[str, str]], default None
        Optional mapping {param_name: "uniform" | "normal"}.
        Defaults to "uniform" for any name not listed.
        Names must match the inferred column names (see return DataFrame).
    per_param_normal : Optional[Dict[str, Dict[str, float]]], default None
        Optional mapping of normal hyperparameters per parameter:
            {"param_name": {"mean": float, "std": float}}
        Missing keys fall back to:
            mean = (lb+ub)/2
            std  = (ub-lb)/6   # ~99.7% of mass within [lb, ub] pre-clipping

    Returns
    -------
    pandas.DataFrame
        Columns: ["scenario_id", "<name_1>", ..., "<name_d>"].

    Raises
    ------
    ValueError
        - If lengths mismatch or bounds invalid.
        - If `n <= 0` or a provided `std <= 0`.
    KeyError
        - If `per_param_dist` / `per_param_normal` reference unknown parameter names.

    Example
    -------
    >>> # df = generate_scenarios_from_model(
    ... #     uncertain_params=[...],
    ... #     bounds=[(0, 10), (5, 25)],
    ... #     n=1000,
    ... #     seed=123,
    ... #     per_param_dist={"rhs_ge": "normal"},
    ... #     per_param_normal={"rhs_ge": {"mean": 20.0, "std": 4.0}},
    ... # )
    """
    _validate_dimensions(uncertain_params, bounds, n)

    names = _infer_param_names(uncertain_params)
    rng = np.random.default_rng(seed)

    per_param_dist = (per_param_dist or {}).copy()
    per_param_normal = (per_param_normal or {}).copy()
    _validate_per_param_keys(names, per_param_dist, per_param_normal)

    d = len(uncertain_params)
    out = np.empty((n, d), dtype=float)

    for j, nm in enumerate(names):
        lb, ub = bounds[j]
        dist = (per_param_dist.get(nm, "uniform") or "uniform").strip().lower()
        if dist not in ("uniform", "normal"):
            raise ValueError(f"Invalid dist for '{nm}': {dist!r} (expected 'uniform' or 'normal').")

        if dist == "uniform":
            out[:, j] = _sample_uniform(lb, ub, n, rng)
        else:
            spec = per_param_normal.get(nm, {})
            mean = float(spec.get("mean", 0.5 * (lb + ub)))
            std = float(spec.get("std", (ub - lb) / 6.0))
            if std <= 0:
                raise ValueError(f"Normal std must be positive for '{nm}' (got {std}).")
            out[:, j] = _sample_normal_clipped(lb, ub, mean, std, n, rng)

    df = pd.DataFrame(out, columns=names)
    df.insert(0, "scenario_id", np.arange(1, n + 1, dtype=int))
    return df


def save_scenarios(
    df: pd.DataFrame,
    out_path: Union[str, Path],
    *,
    meta: Optional[Dict[str, object]] = None,
) -> Path:
    """
    Save the scenario table to CSV/JSON/Parquet and a sidecar `<ext>.meta.json`.

    The sidecar captures basic provenance such as shape, column names, and any
    user-provided metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from `generate_scenarios_from_model(...)`.
    out_path : str | pathlib.Path
        A path ending in one of: .csv | .json | .parquet
    meta : dict, optional
        Arbitrary metadata to include (e.g., model name, seed, config dicts).

    Returns
    -------
    pathlib.Path
        The resolved output file path.

    Raises
    ------
    ValueError
        If the output suffix is not one of the supported formats.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    suf = out.suffix.lower()
    if suf not in (".csv", ".json", ".parquet"):
        raise ValueError("Output must be one of: .csv, .json, .parquet")

    if suf == ".csv":
        df.to_csv(out, index=False)
    elif suf == ".json":
        df.to_json(out, orient="records", indent=2)
    else:
        df.to_parquet(out, index=False)

    payload = {
        "file": str(out.resolve()),
        "n": int(df.shape[0]),
        "d": int(df.shape[1] - 1),  # exclude scenario_id
        "columns": [c for c in df.columns if c != "scenario_id"],
        "note": "Normal sampling is clipped to [lb, ub] (not truncated).",
    }
    if meta:
        payload["extra"] = meta

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out


# ----------------------------- CLI Utilities ----------------------------- #

def _load_model_factory(model_spec: str) -> Callable[[], Tuple[object, List[ParamLike], Bounds, object]]:
    """
    Load a model factory function from a spec like 'package.module:function'.

    The function is expected to return: (model, uncertain_params, bounds, xi_set).

    Parameters
    ----------
    model_spec : str
        Import path in the form 'pkg.mod:function'. If None-like, falls back
        to 'your_module:simple_lp_model' for backward compatibility.

    Returns
    -------
    Callable
        Zero-arg callable that builds and returns (model, uncertain_params, bounds, xi_set).

    Raises
    ------
    ValueError
        If the spec is malformed or the attribute is missing.
    """
    spec = (model_spec or "your_module:simple_lp_model").strip()
    if ":" not in spec:
        raise ValueError(
            f"Invalid --model spec {spec!r}. Expected 'package.module:function'."
        )
    mod_name, func_name = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        raise ValueError(f"Could not import module {mod_name!r}: {e}") from e
    try:
        func = getattr(mod, func_name)
    except AttributeError as e:
        raise ValueError(f"Module {mod_name!r} has no attribute {func_name!r}.") from e
    if not callable(func):
        raise ValueError(f"{spec!r} did not resolve to a callable.")
    return func  # type: ignore[return-value]


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Important:
    - `--model` lets you point to any factory function via 'pkg.module:function'.
      Defaults to 'your_module:simple_lp_model' (the original example).
    - `--dist` / `--normal` expect JSON strings.

    Examples:
    ---------
    --dist '{"rhs_ge": "normal", "rhs_le": "uniform"}'
    --normal '{"rhs_ge": {"mean": 20, "std": 5}}'
    """
    p = argparse.ArgumentParser(
        description="Generate uncertainty scenarios from your Pyomo model function."
    )
    p.add_argument("--n", type=int, required=True, help="Number of scenarios.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p.add_argument("--out", type=str, required=True, help="Output file (.csv|.json|.parquet).")
    p.add_argument(
        "--model",
        type=str,
        default="your_module:simple_lp_model",
        help="Model factory spec as 'package.module:function'. Default: your_module:simple_lp_model",
    )
    p.add_argument(
        "--dist",
        type=str,
        default=None,
        help='JSON mapping param->"uniform"|"normal" (e.g., \'{"rhs_ge":"normal"}\')',
    )
    p.add_argument(
        "--normal",
        type=str,
        default=None,
        help='JSON mapping param->{"mean":m,"std":s} (e.g., \'{"rhs_ge":{"mean":20,"std":5}}\')',
    )
    return p.parse_args(list(argv) if argv is not None else None)


# ----------------------------- CLI Entry ----------------------------- #

def main(argv: Optional[Iterable[str]] = None) -> int:
    """
    CLI entry-point. Loads the model factory, generates scenarios, and saves them.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = _parse_args(argv)

    # Resolve the model factory (module:function)
    model_factory = _load_model_factory(args.model)

    # Build the model and pick up uncertain params and bounds
    # Expected: (model, uncertain_params, bounds, xi_set)
    model, uncertain_params, bounds, xi_set = model_factory()

    # Parse JSON inputs if provided
    per_param_dist = json.loads(args.dist) if args.dist else None
    per_param_normal = json.loads(args.normal) if args.normal else None

    df = generate_scenarios_from_model(
        uncertain_params=uncertain_params,
        bounds=bounds,
        n=args.n,
        seed=args.seed,
        per_param_dist=per_param_dist,
        per_param_normal=per_param_normal,
    )

    save_scenarios(
        df,
        args.out,
        meta={
            "model": args.model,
            "seed": args.seed,
            "dist": per_param_dist,
            "normal": per_param_normal,
        },
    )
    print(f"[OK] Generated {df.shape[0]} scenarios for {df.shape[1]-1} parameters -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
