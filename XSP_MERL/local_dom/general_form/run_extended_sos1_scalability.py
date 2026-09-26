"""
Sequential scalability benchmark for the extended/symmetric Gurobi SOS1 model.

Default behavior intentionally runs only SMALL and MED LandS instances.  Add
--include-hard to include the generated hard instances.  The naive instance is
optional via --include-naive.  An explicit --instances list always takes
precedence over the family filter.

Per-instance outputs:
    <instance>/output_data/sos1_extended_exp/

Aggregate outputs:
    <base_dir>/extended_sos1_scalability_summary/
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from local_dominance_radius_extended_sos1 import (
    ExtendedSOS1LocalDominanceError,
    run_extended_sos1_local_dominance,
)

try:
    from local_dominance_radius_unbounded import LocalDominanceData
except Exception as exc:
    raise RuntimeError(
        "Could not import LocalDominanceData from the existing architecture."
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
        return (0, 0, 0, 0)
    rank = {"small": 1, "med": 2, "hard": 3}.get(_family(name), 9)
    nums: List[int] = []
    for p in name.split("_"):
        try:
            nums.append(int(p))
        except ValueError:
            pass
    return (rank, *nums)


def discover_instances(
    base_dir: Path,
    *,
    include_naive: bool = False,
    include_hard: bool = False,
) -> List[str]:
    root = base_dir / "lands_generated_instances"
    names: List[str] = []
    for p in root.glob("lands_instance_*"):
        if not p.is_dir():
            continue
        fam = _family(p.name)
        if fam == "hard" and not include_hard:
            continue
        if fam in {"small", "med", "hard"}:
            names.append(p.name)
    if include_naive and (base_dir / "lands_instance_naive").is_dir():
        names.append("lands_instance_naive")
    return sorted(set(names), key=_natural_instance_key)


def _flatten_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    structure = result.get("instance_structure", {})
    raw = result.get("raw_gurobi_model_structure", {})
    sel = result.get("selected_scenario", {})
    perf = result.get("performance", {})
    validation = result.get("validation") or {}
    config = result.get("solver_configuration", {})
    ev_val = validation.get("EV") or {}
    star_val = validation.get("star") or {}

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
        "violation_margin": result.get("formulation", {}).get("violation_margin"),
        "raw_model_num_vars": raw.get("num_vars"),
        "raw_model_num_linear_constraints": raw.get("num_linear_constraints"),
        "raw_model_num_sos": raw.get("num_sos"),
        "raw_model_num_binary_vars": raw.get("num_binary_vars"),
        "status": perf.get("status"),
        "has_incumbent": perf.get("has_incumbent"),
        "runtime_sec": perf.get("runtime_sec_gurobi"),
        "objective_gamma": perf.get("objective_value"),
        "best_bound": perf.get("best_bound"),
        "mip_gap": perf.get("mip_gap"),
        "node_count": perf.get("node_count"),
        "simplex_iterations": perf.get("simplex_iterations"),
        "barrier_iterations": perf.get("barrier_iterations"),
        "true_gap_at_returned_xi": validation.get("true_gap_from_independent_recourse_LPs"),
        "true_margin_constraint_residual": validation.get("true_margin_constraint_residual"),
        "EV_max_row_sos_pair_min_abs": ev_val.get("max_row_sos_pair_min_abs"),
        "EV_max_variable_sos_pair_min_abs": ev_val.get("max_variable_sos_pair_min_abs"),
        "EV_KKT_minus_independent_LP": ev_val.get("KKT_minus_independent_LP"),
        "star_max_row_sos_pair_min_abs": star_val.get("max_row_sos_pair_min_abs"),
        "star_max_variable_sos_pair_min_abs": star_val.get("max_variable_sos_pair_min_abs"),
        "star_KKT_minus_independent_LP": star_val.get("KKT_minus_independent_LP"),
        "time_limit_sec": config.get("time_limit_sec"),
        "target_mip_gap": config.get("target_mip_gap"),
        "threads": config.get("threads"),
        "seed": config.get("seed"),
        "PreSOS1BigM": config.get("PreSOS1BigM"),
    }


def _failure_row(
    instance_name: str,
    base_dir: Path,
    message: str,
    time_limit: float,
    threads: int,
    violation_margin: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "instance_name": instance_name,
        "family": _family(instance_name),
        "status": "ERROR",
        "error": message,
        "time_limit_sec": time_limit,
        "threads": threads,
        "violation_margin": violation_margin,
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
        "selected_scenario", "initial_cost_gap", "violation_margin",
        "raw_model_num_vars", "raw_model_num_linear_constraints",
        "raw_model_num_sos", "raw_model_num_binary_vars", "status",
        "has_incumbent", "runtime_sec", "objective_gamma", "best_bound",
        "mip_gap", "node_count", "simplex_iterations", "barrier_iterations",
        "true_gap_at_returned_xi", "true_margin_constraint_residual",
        "EV_max_row_sos_pair_min_abs", "EV_max_variable_sos_pair_min_abs",
        "EV_KKT_minus_independent_LP", "star_max_row_sos_pair_min_abs",
        "star_max_variable_sos_pair_min_abs", "star_KKT_minus_independent_LP",
        "time_limit_sec", "target_mip_gap", "threads", "seed", "PreSOS1BigM",
        "error",
    ]
    extra = sorted({k for row in rows for k in row.keys()} - set(preferred))
    fields = [k for k in preferred if any(k in row for row in rows)] + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _median(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("family"), row.get("num_scenarios"), row.get("n_x"),
            row.get("n_y"), row.get("n_recourse_rows"), row.get("xi_dim"),
            row.get("raw_model_num_vars"), row.get("raw_model_num_sos"),
            row.get("violation_margin"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        family, S, nx, ny, nr, m, nvars, nsos, eps = key
        statuses = [str(r.get("status")) for r in members]
        optimal = sum(s == "OPTIMAL" for s in statuses)
        timeouts = sum(s == "TIME_LIMIT" for s in statuses)
        incumbents = sum(bool(r.get("has_incumbent")) for r in members)
        nonopt_gaps = [
            r.get("mip_gap") for r in members
            if r.get("status") != "OPTIMAL" and r.get("mip_gap") is not None
        ]
        out.append(
            {
                "family": family,
                "num_scenarios": S,
                "n_x": nx,
                "n_y": ny,
                "n_recourse_rows": nr,
                "xi_dim": m,
                "raw_model_num_vars": nvars,
                "raw_model_num_sos": nsos,
                "violation_margin": eps,
                "num_instances": len(members),
                "num_optimal": optimal,
                "num_time_limit": timeouts,
                "num_with_incumbent": incumbents,
                "optimal_fraction": optimal / len(members) if members else None,
                "median_runtime_sec": _median(r.get("runtime_sec") for r in members),
                "mean_runtime_sec": _mean(r.get("runtime_sec") for r in members),
                "max_runtime_sec": max(
                    [float(r["runtime_sec"]) for r in members if r.get("runtime_sec") is not None],
                    default=None,
                ),
                "median_node_count": _median(r.get("node_count") for r in members),
                "median_objective_gamma": _median(r.get("objective_gamma") for r in members),
                "median_nonoptimal_mip_gap": _median(nonopt_gaps),
            }
        )
    return out


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _write_markdown(path: Path, rows: List[Dict[str, Any]], grouped: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Extended SOS1 local-dominance scalability report\n\n")
        f.write(
            "This report uses KKT+SOS1 optimality on both the EV and x* recourse LPs. "
            "Native SOS1 is retained with `PreSOS1BigM=0`.\n\n"
        )
        f.write("## Per-instance results\n\n")
        f.write("| Instance | S | nx | ny | r | m | SOS1 | eps | Status | Runtime (s) | Gamma/incumbent | Best bound | MIP gap | True gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.get('instance_name','')} | {_fmt(r.get('num_scenarios'))} | {_fmt(r.get('n_x'))} | "
                f"{_fmt(r.get('n_y'))} | {_fmt(r.get('n_recourse_rows'))} | {_fmt(r.get('xi_dim'))} | "
                f"{_fmt(r.get('raw_model_num_sos'))} | {_fmt(r.get('violation_margin'))} | {r.get('status','')} | "
                f"{_fmt(r.get('runtime_sec'))} | {_fmt(r.get('objective_gamma'))} | {_fmt(r.get('best_bound'))} | "
                f"{_fmt(r.get('mip_gap'))} | {_fmt(r.get('true_gap_at_returned_xi'))} |\n"
            )

        f.write("\n## Grouped scalability summary\n\n")
        f.write("| Family | S | nx | ny | r | m | Vars | SOS1 | eps | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median nonoptimal gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for g in grouped:
            f.write(
                f"| {g.get('family','')} | {_fmt(g.get('num_scenarios'))} | {_fmt(g.get('n_x'))} | "
                f"{_fmt(g.get('n_y'))} | {_fmt(g.get('n_recourse_rows'))} | {_fmt(g.get('xi_dim'))} | "
                f"{_fmt(g.get('raw_model_num_vars'))} | {_fmt(g.get('raw_model_num_sos'))} | "
                f"{_fmt(g.get('violation_margin'))} | {_fmt(g.get('num_instances'))} | {_fmt(g.get('num_optimal'))} | "
                f"{_fmt(g.get('num_time_limit'))} | {_fmt(g.get('median_runtime_sec'))} | {_fmt(g.get('median_node_count'))} | "
                f"{_fmt(g.get('median_nonoptimal_mip_gap'))} |\n"
            )


def _load_existing_canonical(base_dir: Path, instance_name: str) -> Optional[Dict[str, Any]]:
    try:
        data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        path = data.instance_dir / "output_data" / "sos1_extended_exp" / "extended_sos1_result.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _existing_matches_margin(result: Mapping[str, Any], violation_margin: float) -> bool:
    existing = result.get("formulation", {}).get("violation_margin", 0.0)
    try:
        return abs(float(existing) - float(violation_margin)) <= 1e-15
    except Exception:
        return False


def _persist(summary_dir: Path, rows: List[Dict[str, Any]]) -> None:
    grouped = _group_rows(rows)
    _write_csv(summary_dir / "extended_sos1_scalability_summary.csv", rows)
    _write_json(summary_dir / "extended_sos1_scalability_summary.json", {"rows": rows})
    _write_csv(summary_dir / "extended_sos1_scalability_grouped.csv", grouped)
    _write_json(summary_dir / "extended_sos1_scalability_grouped.json", {"groups": grouped})
    _write_markdown(summary_dir / "extended_sos1_scalability_summary.md", rows, grouped)


def run_batch(
    *,
    base_dir: Path,
    instance_names: List[str],
    scenario_policy: str,
    gap_tol: float,
    violation_margin: float,
    time_limit: float,
    mip_gap: float,
    threads: int,
    seed: int,
    int_feas_tol: float,
    feasibility_tol: float,
    tee: bool,
    rerun_policy: str,
) -> List[Dict[str, Any]]:
    summary_dir = base_dir / "extended_sos1_scalability_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i, name in enumerate(instance_names, start=1):
        print(f"\n[{i}/{len(instance_names)}] {name}")

        existing = _load_existing_canonical(base_dir, name)
        if existing is not None and _existing_matches_margin(existing, violation_margin):
            existing_status = str(existing.get("performance", {}).get("status", ""))
            should_skip = (
                rerun_policy == "missing"
                or (rerun_policy == "nonoptimal" and existing_status == "OPTIMAL")
            )
            if should_skip:
                print(f"  using existing result ({existing_status})")
                rows.append(_flatten_result(existing))
                _persist(summary_dir, rows)
                continue

        try:
            result = run_extended_sos1_local_dominance(
                name,
                base_dir=base_dir,
                scenario=None,
                scenario_policy=scenario_policy,
                gap_tol=gap_tol,
                violation_margin=violation_margin,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                seed=seed,
                int_feas_tol=int_feas_tol,
                feasibility_tol=feasibility_tol,
                tee=tee,
                keep_native_sos1=True,
                write_model=False,
            )
            row = _flatten_result(result)
            print(
                f"  status={row.get('status')} runtime={row.get('runtime_sec')} "
                f"gap={row.get('mip_gap')}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            row = _failure_row(name, base_dir, repr(exc), time_limit, threads, violation_margin)
            try:
                d = LocalDominanceData.load(name, base_dir=base_dir)
                err_path = d.instance_dir / "output_data" / "sos1_extended_exp" / "extended_sos1_error.json"
                _write_json(err_path, row)
            except Exception:
                pass

        rows.append(row)
        _persist(summary_dir, rows)

    return rows


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the extended/symmetric SOS1 local-dominance scalability suite sequentially."
    )
    p.add_argument("--base-dir", default=None, help="Defaults to directory containing this script.")
    p.add_argument("--include-naive", action="store_true", help="Also run lands_instance_naive.")
    p.add_argument(
        "--include-hard",
        action="store_true",
        help="Also include hard LandS instances. By default only small+med are discovered.",
    )
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional explicit instance names. Explicit names may include hard instances even without --include-hard.",
    )
    p.add_argument(
        "--scenario-policy",
        choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive",
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument(
        "--violation-margin",
        type=float,
        default=0.0,
        help="epsilon >= 0 in Delta(xi) <= -epsilon. Default 0 reproduces the closure/tie boundary.",
    )
    p.add_argument("--time-limit", type=float, default=300.0, help="Per-instance seconds. Default 300.")
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--int-feas-tol", type=float, default=1e-8)
    p.add_argument("--feasibility-tol", type=float, default=1e-8)
    p.add_argument("--tee", action="store_true", help="Show each Gurobi log on console as well as in files.")
    p.add_argument(
        "--rerun-policy",
        choices=["all", "missing", "nonoptimal"],
        default="missing",
        help=(
            "all=re-run everything; missing=skip any instance with a matching extended_sos1_result.json; "
            "nonoptimal=re-run only existing non-OPTIMAL results."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else Path(__file__).resolve().parent

    if args.instances:
        names = list(args.instances)
        if args.include_naive and "lands_instance_naive" not in names:
            names.append("lands_instance_naive")
    else:
        names = discover_instances(
            base_dir,
            include_naive=args.include_naive,
            include_hard=args.include_hard,
        )
    names = sorted(set(names), key=_natural_instance_key)

    if not names:
        raise SystemExit("No instances found.")

    print(f"Base directory:   {base_dir}")
    print(f"Instances:        {len(names)}")
    print(f"Hard included:    {args.include_hard or any(_family(n) == 'hard' for n in names)}")
    print(f"Time limit:       {args.time_limit} s per instance")
    print(f"Threads:          {args.threads}")
    print(f"Scenario rule:    {args.scenario_policy}")
    print(f"Violation margin: {args.violation_margin}")
    print("PreSOS1BigM:      0 (native SOS1 retained; no automatic big-M reformulation)")

    t0 = time.perf_counter()
    rows = run_batch(
        base_dir=base_dir,
        instance_names=names,
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
        violation_margin=args.violation_margin,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        int_feas_tol=args.int_feas_tol,
        feasibility_tol=args.feasibility_tol,
        tee=args.tee,
        rerun_policy=args.rerun_policy,
    )
    elapsed = time.perf_counter() - t0
    optimal = sum(str(r.get("status")) == "OPTIMAL" for r in rows)
    timed = sum(str(r.get("status")) == "TIME_LIMIT" for r in rows)
    print("\n=== Batch complete ===")
    print(f"Total wall time: {elapsed:.1f} s")
    print(f"Optimal:         {optimal}/{len(rows)}")
    print(f"Time limit:      {timed}/{len(rows)}")
    print(f"Reports:         {base_dir / 'extended_sos1_scalability_summary'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
