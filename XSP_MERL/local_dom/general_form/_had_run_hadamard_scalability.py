"""
Sequential scalability benchmark for the Hadamard-Cramer Big-M local-dominance model.

Run from:
    local_dom/general_form/

Per-instance outputs are written by the core solver to exactly:
    <instance_name>/output_data/hadamard_exp/

Aggregate reports are written to:
    <general_form>/hadamard_scalability_summary/
        hadamard_scalability_summary.csv
        hadamard_scalability_summary.json
        hadamard_scalability_summary.md
        hadamard_scalability_grouped.csv
        hadamard_scalability_grouped.json

The runner is sequential so wall-clock/scalability comparisons are not polluted
by multiple Gurobi jobs competing for CPU resources.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from _had_local_dominance_radius_hadamard_bigm import (
    HadamardBigMError,
    run_hadamard_local_dominance,
)

try:
    from local_dominance_radius_unbounded import LocalDominanceData
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import LocalDominanceData. Run this script from local_dom/general_form/."
    ) from exc


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
        f.write("\n")


def _family(name: str) -> str:
    if name == "lands_instance_naive":
        return "naive"
    if "_small_" in name:
        return "small"
    if "_med_" in name:
        return "med"
    if "_hard_" in name:
        return "hard"
    return "other"


def _natural_instance_key(name: str) -> Tuple[Any, ...]:
    if name == "lands_instance_naive":
        return (0,)
    fam = 1 if "_small_" in name else 2 if "_med_" in name else 3 if "_hard_" in name else 9
    nums = []
    for token in name.split("_"):
        try:
            nums.append(int(token))
        except ValueError:
            pass
    return (fam, *nums)


def discover_instances(base_dir: Path, include_naive: bool = False) -> List[str]:
    root = base_dir / "lands_generated_instances"
    names = [p.name for p in root.glob("lands_instance_*") if p.is_dir()]
    if include_naive and (base_dir / "lands_instance_naive").is_dir():
        names.append("lands_instance_naive")
    return sorted(set(names), key=_natural_instance_key)


def _flatten_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    structure = result.get("instance_structure", {})
    raw = result.get("raw_gurobi_model_structure", {})
    sel = result.get("selected_scenario", {})
    perf = result.get("performance", {})
    val = result.get("validation") or {}
    hb_all = result.get("hadamard_bounds", {})
    hb = hb_all.get("hadamard", {})
    inc = hb_all.get("safe_incumbent", {})
    fact = hb_all.get("factorial_reference_not_used", {})
    relax = result.get("continuous_relaxation", {})
    config = result.get("solver_configuration", {})
    util = val.get("bigM_utilization", {}) if isinstance(val, Mapping) else {}

    return {
        "instance_name": result.get("instance_name"),
        "family": _family(str(result.get("instance_name"))),
        "num_scenarios": structure.get("num_scenarios"),
        "n_x": structure.get("n_x_first_stage"),
        "n_y": structure.get("n_y_second_stage"),
        "n_recourse_rows": structure.get("n_recourse_rows"),
        "xi_dim": structure.get("xi_dim"),
        "nnz_T": structure.get("nnz_T"),
        "nnz_W": structure.get("nnz_W"),
        "nnz_H": structure.get("nnz_H"),
        "selected_scenario": sel.get("scenario_name"),
        "initial_cost_gap": sel.get("initial_cost_gap"),
        "safe_radius_U": inc.get("radius_U"),
        "safe_radius_source": inc.get("source"),
        "bound_preprocessing_sec": perf.get("bounds_preprocessing_runtime_sec"),
        "hadamard_M_primal": hb.get("M_primal_uniform_for_y_star_and_row_slack"),
        "hadamard_M_mu": hb.get("M_mu_uniform"),
        "hadamard_M_rc_max": hb.get("M_reduced_cost_max"),
        "log10_M_primal": hb.get("log10_M_primal"),
        "log10_M_mu": hb.get("log10_M_mu"),
        "log10_M_rc_max": hb.get("log10_M_reduced_cost_max"),
        "factorial_over_hadamard_factor_ratio": fact.get("factorial_over_hadamard_factor_ratio"),
        "factorial_M_primal_reference": fact.get("M_primal"),
        "factorial_M_mu_reference": fact.get("M_mu"),
        "raw_model_num_vars": raw.get("num_vars"),
        "raw_model_num_linear_constraints": raw.get("num_linear_constraints"),
        "raw_model_num_binary_vars": raw.get("num_binary_vars"),
        "raw_model_num_sos": raw.get("num_sos"),
        "continuous_relaxation_bound": relax.get("objective"),
        "continuous_relaxation_runtime_sec": relax.get("runtime_sec"),
        "status": perf.get("status"),
        "has_incumbent": perf.get("has_incumbent"),
        "mip_runtime_sec": perf.get("runtime_sec_gurobi"),
        "total_runtime_including_bounds_sec": perf.get("total_runtime_including_bounds_sec"),
        "objective_gamma": perf.get("objective_value"),
        "best_bound": perf.get("best_bound"),
        "mip_gap": perf.get("mip_gap"),
        "node_count": perf.get("node_count"),
        "simplex_iterations": perf.get("simplex_iterations"),
        "work": perf.get("work"),
        "true_gap_at_returned_xi": val.get("true_gap_from_independent_recourse_LPs"),
        "star_KKT_obj_minus_independent_LP": val.get("star_recourse_KKT_minus_independent_LP"),
        "max_y_star_over_M_primal": util.get("max_y_star_over_M_primal"),
        "max_row_slack_over_M_primal": util.get("max_row_slack_over_M_primal"),
        "max_mu_over_M_mu": util.get("max_mu_over_M_mu"),
        "max_reduced_cost_over_component_M": util.get("max_reduced_cost_over_component_M"),
        "time_limit_sec": config.get("time_limit_sec"),
        "target_mip_gap": config.get("target_mip_gap"),
        "threads": config.get("threads"),
        "seed": config.get("seed"),
        "numeric_focus": config.get("numeric_focus"),
    }


def _failure_row(
    instance_name: str,
    base_dir: Path,
    message: str,
    time_limit: float,
    threads: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "instance_name": instance_name,
        "family": _family(instance_name),
        "status": "ERROR",
        "error": message,
        "time_limit_sec": time_limit,
        "threads": threads,
    }
    try:
        d = LocalDominanceData.load(instance_name, base_dir=base_dir)
        row.update(
            {
                "num_scenarios": d.num_scenarios,
                "n_x": d.num_first_stage_vars,
                "n_y": d.num_second_stage_vars,
                "n_recourse_rows": d.num_recourse_constraints,
                "xi_dim": d.xi_dim,
            }
        )
    except Exception:
        pass
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "instance_name", "family", "num_scenarios", "n_x", "n_y",
        "n_recourse_rows", "xi_dim", "nnz_T", "nnz_W", "nnz_H",
        "selected_scenario", "initial_cost_gap", "safe_radius_U",
        "safe_radius_source", "bound_preprocessing_sec",
        "hadamard_M_primal", "hadamard_M_mu", "hadamard_M_rc_max",
        "log10_M_primal", "log10_M_mu", "log10_M_rc_max",
        "factorial_over_hadamard_factor_ratio", "factorial_M_primal_reference",
        "factorial_M_mu_reference", "raw_model_num_vars",
        "raw_model_num_linear_constraints", "raw_model_num_binary_vars",
        "raw_model_num_sos", "continuous_relaxation_bound",
        "continuous_relaxation_runtime_sec", "status", "has_incumbent",
        "mip_runtime_sec", "total_runtime_including_bounds_sec",
        "objective_gamma", "best_bound", "mip_gap", "node_count",
        "simplex_iterations", "work", "true_gap_at_returned_xi",
        "star_KKT_obj_minus_independent_LP", "max_y_star_over_M_primal",
        "max_row_slack_over_M_primal", "max_mu_over_M_mu",
        "max_reduced_cost_over_component_M", "time_limit_sec",
        "target_mip_gap", "threads", "seed", "numeric_focus", "error",
    ]
    extra = sorted({k for row in rows for k in row} - set(preferred))
    fields = [k for k in preferred if any(k in row for row in rows)] + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = _finite_values(values)
    return statistics.median(vals) if vals else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = _finite_values(values)
    return statistics.mean(vals) if vals else None


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("family"), row.get("num_scenarios"), row.get("n_x"),
            row.get("n_y"), row.get("n_recourse_rows"), row.get("xi_dim"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        family, S, nx, ny, r, m = key
        statuses = [str(x.get("status")) for x in items]
        nonoptimal_gaps = [
            x.get("mip_gap") for x in items
            if x.get("status") != "OPTIMAL" and x.get("mip_gap") is not None
        ]
        out.append(
            {
                "family": family,
                "num_scenarios": S,
                "n_x": nx,
                "n_y": ny,
                "n_recourse_rows": r,
                "xi_dim": m,
                "N": len(items),
                "optimal": sum(s == "OPTIMAL" for s in statuses),
                "time_limit": sum(s == "TIME_LIMIT" for s in statuses),
                "errors": sum(s == "ERROR" for s in statuses),
                "median_mip_runtime_sec": _median(x.get("mip_runtime_sec") for x in items),
                "mean_mip_runtime_sec": _mean(x.get("mip_runtime_sec") for x in items),
                "median_total_runtime_sec": _median(x.get("total_runtime_including_bounds_sec") for x in items),
                "median_nodes": _median(x.get("node_count") for x in items),
                "median_nonoptimal_gap": _median(nonoptimal_gaps),
                "median_continuous_relaxation_bound": _median(x.get("continuous_relaxation_bound") for x in items),
                "median_log10_M_primal": _median(x.get("log10_M_primal") for x in items),
                "median_log10_M_mu": _median(x.get("log10_M_mu") for x in items),
                "median_log10_M_rc_max": _median(x.get("log10_M_rc_max") for x in items),
                "median_factorial_over_hadamard_factor_ratio": _median(
                    x.get("factorial_over_hadamard_factor_ratio") for x in items
                ),
                "median_bound_preprocessing_sec": _median(x.get("bound_preprocessing_sec") for x in items),
            }
        )
    return out


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}g}"


def _write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    grouped: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Hadamard Big-M local-dominance scalability report")
    lines.append("")
    lines.append("## Per-instance results")
    lines.append("")
    lines.append("| Instance | S | nx | ny | r | m | binaries | log10 Mp | log10 Mmu | LP relax | Status | MIP runtime (s) | Gamma/incumbent | Best bound | MIP gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for x in rows:
        lines.append(
            "| {instance} | {S} | {nx} | {ny} | {r} | {m} | {bins} | {lmp} | {lmmu} | {lp} | {status} | {rt} | {obj} | {bd} | {gap} |".format(
                instance=x.get("instance_name", ""), S=x.get("num_scenarios", ""),
                nx=x.get("n_x", ""), ny=x.get("n_y", ""), r=x.get("n_recourse_rows", ""),
                m=x.get("xi_dim", ""), bins=x.get("raw_model_num_binary_vars", ""),
                lmp=_fmt(x.get("log10_M_primal")), lmmu=_fmt(x.get("log10_M_mu")),
                lp=_fmt(x.get("continuous_relaxation_bound")), status=x.get("status", ""),
                rt=_fmt(x.get("mip_runtime_sec")), obj=_fmt(x.get("objective_gamma")),
                bd=_fmt(x.get("best_bound")), gap=_fmt(x.get("mip_gap")),
            )
        )

    lines.append("")
    lines.append("## Grouped scalability summary")
    lines.append("")
    lines.append("| Family | S | nx | ny | r | m | N | Optimal | Time limit | Median MIP runtime (s) | Median nodes | Median nonoptimal gap | Median LP relax | Median log10 Mp | Median log10 Mmu |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in grouped:
        lines.append(
            "| {family} | {S} | {nx} | {ny} | {r} | {m} | {N} | {opt} | {tl} | {rt} | {nodes} | {gap} | {lp} | {lmp} | {lmmu} |".format(
                family=g.get("family", ""), S=g.get("num_scenarios", ""), nx=g.get("n_x", ""),
                ny=g.get("n_y", ""), r=g.get("n_recourse_rows", ""), m=g.get("xi_dim", ""),
                N=g.get("N", ""), opt=g.get("optimal", ""), tl=g.get("time_limit", ""),
                rt=_fmt(g.get("median_mip_runtime_sec")), nodes=_fmt(g.get("median_nodes")),
                gap=_fmt(g.get("median_nonoptimal_gap")), lp=_fmt(g.get("median_continuous_relaxation_bound")),
                lmp=_fmt(g.get("median_log10_M_primal")), lmmu=_fmt(g.get("median_log10_M_mu")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_existing(base_dir: Path, instance_name: str) -> Optional[Dict[str, Any]]:
    try:
        d = LocalDominanceData.load(instance_name, base_dir=base_dir)
        path = d.instance_dir / "output_data" / "hadamard_exp" / "hadamard_result.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _persist(summary_dir: Path, rows: List[Dict[str, Any]]) -> None:
    grouped = _group_rows(rows)
    _write_csv(summary_dir / "hadamard_scalability_summary.csv", rows)
    _write_json(summary_dir / "hadamard_scalability_summary.json", {"rows": rows})
    _write_csv(summary_dir / "hadamard_scalability_grouped.csv", grouped)
    _write_json(summary_dir / "hadamard_scalability_grouped.json", {"groups": grouped})
    _write_markdown(summary_dir / "hadamard_scalability_summary.md", rows, grouped)


def run_batch(
    *,
    base_dir: Path,
    instance_names: List[str],
    scenario_policy: str,
    gap_tol: float,
    time_limit: float,
    mip_gap: float,
    threads: int,
    seed: int,
    int_feas_tol: float,
    feasibility_tol: float,
    numeric_focus: int,
    tee: bool,
    rerun_policy: str,
    search_radius_multiplier: float,
    search_growth_factor: float,
    search_bisection_iterations: int,
    search_random_directions: int,
    search_random_seed: int,
    search_max_radius: Optional[float],
    max_finite_M: float,
) -> List[Dict[str, Any]]:
    summary_dir = base_dir / "hadamard_scalability_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i, name in enumerate(instance_names, 1):
        print(f"\n[{i}/{len(instance_names)}] {name}")
        existing = _load_existing(base_dir, name)
        if existing is not None:
            status = str(existing.get("performance", {}).get("status", ""))
            skip = rerun_policy == "missing" or (
                rerun_policy == "nonoptimal" and status == "OPTIMAL"
            )
            if skip:
                print(f"  using existing result ({status})")
                rows.append(_flatten_result(existing))
                _persist(summary_dir, rows)
                continue

        try:
            result = run_hadamard_local_dominance(
                name,
                base_dir=base_dir,
                scenario=None,
                scenario_policy=scenario_policy,
                gap_tol=gap_tol,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                seed=seed,
                int_feas_tol=int_feas_tol,
                feasibility_tol=feasibility_tol,
                numeric_focus=numeric_focus,
                tee=tee,
                write_model=False,
                compute_lp_relaxation=True,
                radius_multiplier=search_radius_multiplier,
                growth_factor=search_growth_factor,
                bisection_iterations=search_bisection_iterations,
                num_random_directions=search_random_directions,
                random_seed=search_random_seed,
                max_radius=search_max_radius,
                max_finite_M=max_finite_M,
            )
            row = _flatten_result(result)
            print(
                f"  status={row.get('status')} mip_runtime={row.get('mip_runtime_sec')} "
                f"gap={row.get('mip_gap')} log10(Mp)={row.get('log10_M_primal')}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            row = _failure_row(name, base_dir, repr(exc), time_limit, threads)
            try:
                d = LocalDominanceData.load(name, base_dir=base_dir)
                err = d.instance_dir / "output_data" / "hadamard_exp" / "hadamard_error.json"
                _write_json(err, row)
            except Exception:
                pass

        rows.append(row)
        _persist(summary_dir, rows)

    return rows


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Hadamard Big-M local-dominance scalability suite sequentially."
    )
    p.add_argument("--base-dir", default=None, help="Defaults to directory containing this script.")
    p.add_argument("--include-naive", action="store_true")
    p.add_argument("--instances", nargs="*", default=None)
    p.add_argument(
        "--scenario-policy",
        choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive",
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--int-feas-tol", type=float, default=1e-8)
    p.add_argument("--feasibility-tol", type=float, default=1e-8)
    p.add_argument("--numeric-focus", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--tee", action="store_true", help="Show each Gurobi log on the console.")
    p.add_argument(
        "--rerun-policy",
        choices=["missing", "nonoptimal", "all"],
        default="missing",
        help="missing: reuse any existing result; nonoptimal: rerun non-optimal only; all: rerun all.",
    )
    p.add_argument("--search-radius-multiplier", type=float, default=10.0)
    p.add_argument("--search-growth-factor", type=float, default=2.0)
    p.add_argument("--search-bisection-iterations", type=int, default=40)
    p.add_argument("--search-random-directions", type=int, default=20)
    p.add_argument("--search-random-seed", type=int, default=12345)
    p.add_argument("--search-max-radius", type=float, default=None)
    p.add_argument("--max-finite-M", type=float, default=1e90)
    return p


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    args = _parser().parse_args(argv)
    base_dir = (
        Path(args.base_dir).expanduser().resolve()
        if args.base_dir is not None
        else Path(__file__).resolve().parent
    )
    names = args.instances or discover_instances(base_dir, include_naive=args.include_naive)
    if not names:
        raise RuntimeError("No LANDS instances found.")

    rows = run_batch(
        base_dir=base_dir,
        instance_names=list(names),
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        int_feas_tol=args.int_feas_tol,
        feasibility_tol=args.feasibility_tol,
        numeric_focus=args.numeric_focus,
        tee=args.tee,
        rerun_policy=args.rerun_policy,
        search_radius_multiplier=args.search_radius_multiplier,
        search_growth_factor=args.search_growth_factor,
        search_bisection_iterations=args.search_bisection_iterations,
        search_random_directions=args.search_random_directions,
        search_random_seed=args.search_random_seed,
        search_max_radius=args.search_max_radius,
        max_finite_M=args.max_finite_M,
    )

    summary = base_dir / "hadamard_scalability_summary" / "hadamard_scalability_summary.md"
    print(f"\nScalability report: {summary}")
    return rows


if __name__ == "__main__":
    main()
