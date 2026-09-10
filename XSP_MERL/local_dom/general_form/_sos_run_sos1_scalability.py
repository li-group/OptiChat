"""
Sequential scalability benchmark for the Gurobi SOS1 local-dominance model.

Run this file from local_dom/general_form/ after copying it beside
local_dominance_radius_sos1.py.

Per-instance outputs are always written by the core solver to:
    <instance>/output_data/sos1_exp/

Aggregate reports are written to:
    <base_dir>/sos1_scalability_summary/
        sos1_scalability_summary.csv
        sos1_scalability_summary.json
        sos1_scalability_summary.md
        sos1_scalability_grouped.csv
        sos1_scalability_grouped.json

The runner is intentionally sequential.  Parallel instance solves would make
Gurobi thread usage and wall-clock comparisons much harder to interpret.
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

# # from XSP_MERL.local_dom.general_form.local_dominance_radius_sos1 import (
# from local_dom.general_form.local_dominance_radius_sos1 import (
#     SOS1LocalDominanceError,
#     run_sos1_local_dominance,
# )

# try:
#     from local_dominance_radius_unbounded import LocalDominanceData
# except Exception as exc:  # pragma: no cover
#     raise RuntimeError("Could not import LocalDominanceData from the existing architecture.") from exc

from local_dominance_radius_sos1 import (
    SOS1LocalDominanceError,
    run_sos1_local_dominance,
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
    return "other"


def _natural_instance_key(name: str) -> Tuple[Any, ...]:
    if name == "lands_instance_naive":
        return (0, 0, 0, 0)
    family_rank = 1 if "_small_" in name else 2 if "_med_" in name else 9
    parts = name.split("_")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            pass
    return (family_rank, *nums)


def discover_instances(base_dir: Path, include_naive: bool = False) -> List[str]:
    names = [
        p.name
        for p in (base_dir / "lands_generated_instances").glob("lands_instance_*")
        if p.is_dir()
    ]
    if include_naive and (base_dir / "lands_instance_naive").is_dir():
        names.append("lands_instance_naive")
    return sorted(set(names), key=_natural_instance_key)


def _flatten_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    structure = result.get("instance_structure", {})
    raw = result.get("raw_gurobi_model_structure", {})
    sel = result.get("selected_scenario", {})
    perf = result.get("performance", {})
    validation = result.get("validation") or {}
    solution = result.get("solution") or {}
    config = result.get("solver_configuration", {})

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
        "max_row_sos_pair_min_abs": validation.get("max_row_sos_pair_min_abs"),
        "max_variable_sos_pair_min_abs": validation.get("max_variable_sos_pair_min_abs"),
        "star_KKT_obj_minus_independent_LP": validation.get("star_recourse_KKT_minus_independent_LP"),
        "time_limit_sec": config.get("time_limit_sec"),
        "target_mip_gap": config.get("target_mip_gap"),
        "threads": config.get("threads"),
        "seed": config.get("seed"),
        "PreSOS1BigM": config.get("PreSOS1BigM"),
    }


def _failure_row(instance_name: str, base_dir: Path, message: str, time_limit: float, threads: int) -> Dict[str, Any]:
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
        "selected_scenario", "initial_cost_gap", "raw_model_num_vars",
        "raw_model_num_linear_constraints", "raw_model_num_sos",
        "raw_model_num_binary_vars", "status", "has_incumbent", "runtime_sec",
        "objective_gamma", "best_bound", "mip_gap", "node_count",
        "simplex_iterations", "barrier_iterations", "true_gap_at_returned_xi",
        "max_row_sos_pair_min_abs", "max_variable_sos_pair_min_abs",
        "star_KKT_obj_minus_independent_LP", "time_limit_sec", "target_mip_gap",
        "threads", "seed", "PreSOS1BigM", "error",
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
            row.get("family"),
            row.get("num_scenarios"),
            row.get("n_x"),
            row.get("n_y"),
            row.get("n_recourse_rows"),
            row.get("xi_dim"),
            row.get("raw_model_num_vars"),
            row.get("raw_model_num_sos"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        family, S, nx, ny, nr, m, nvars, nsos = key
        statuses = [str(r.get("status")) for r in members]
        optimal = sum(s == "OPTIMAL" for s in statuses)
        timeouts = sum(s == "TIME_LIMIT" for s in statuses)
        incumbents = sum(bool(r.get("has_incumbent")) for r in members)
        nonopt_gaps = [
            r.get("mip_gap")
            for r in members
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


def _write_markdown(path: Path, rows: List[Dict[str, Any]], grouped: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SOS1 local-dominance scalability report\n\n")
        f.write("## Per-instance results\n\n")
        f.write("| Instance | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Gamma/incumbent | Best bound | MIP gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
        for r in rows:
            def fmt(v: Any, digits: int = 4) -> str:
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.{digits}g}"
                return str(v)
            f.write(
                f"| {r.get('instance_name','')} | {fmt(r.get('num_scenarios'))} | {fmt(r.get('n_x'))} | "
                f"{fmt(r.get('n_y'))} | {fmt(r.get('n_recourse_rows'))} | {fmt(r.get('xi_dim'))} | "
                f"{fmt(r.get('raw_model_num_sos'))} | {r.get('status','')} | {fmt(r.get('runtime_sec'))} | "
                f"{fmt(r.get('objective_gamma'))} | {fmt(r.get('best_bound'))} | {fmt(r.get('mip_gap'))} |\n"
            )

        f.write("\n## Grouped scalability summary\n\n")
        f.write("| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median nonoptimal gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for g in grouped:
            def fmt(v: Any, digits: int = 4) -> str:
                if v is None:
                    return ""
                if isinstance(v, float):
                    return f"{v:.{digits}g}"
                return str(v)
            f.write(
                f"| {g.get('family','')} | {fmt(g.get('num_scenarios'))} | {fmt(g.get('n_x'))} | "
                f"{fmt(g.get('n_y'))} | {fmt(g.get('n_recourse_rows'))} | {fmt(g.get('xi_dim'))} | "
                f"{fmt(g.get('raw_model_num_vars'))} | {fmt(g.get('raw_model_num_sos'))} | "
                f"{fmt(g.get('num_instances'))} | {fmt(g.get('num_optimal'))} | {fmt(g.get('num_time_limit'))} | "
                f"{fmt(g.get('median_runtime_sec'))} | {fmt(g.get('median_node_count'))} | "
                f"{fmt(g.get('median_nonoptimal_mip_gap'))} |\n"
            )


def _load_existing_canonical(base_dir: Path, instance_name: str) -> Optional[Dict[str, Any]]:
    try:
        data = LocalDominanceData.load(instance_name, base_dir=base_dir)
        path = data.instance_dir / "output_data" / "sos1_exp" / "sos1_result.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


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
    tee: bool,
    rerun_policy: str,
) -> List[Dict[str, Any]]:
    summary_dir = base_dir / "sos1_scalability_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for i, name in enumerate(instance_names, start=1):
        print(f"\n[{i}/{len(instance_names)}] {name}")

        existing = _load_existing_canonical(base_dir, name)
        if existing is not None:
            existing_status = str(existing.get("performance", {}).get("status", ""))
            should_skip = (
                rerun_policy == "missing"
                or (rerun_policy == "nonoptimal" and existing_status == "OPTIMAL")
            )
            if should_skip:
                print(f"  using existing result ({existing_status})")
                rows.append(_flatten_result(existing))
                grouped = _group_rows(rows)
                _write_csv(summary_dir / "sos1_scalability_summary.csv", rows)
                _write_json(summary_dir / "sos1_scalability_summary.json", {"rows": rows})
                _write_csv(summary_dir / "sos1_scalability_grouped.csv", grouped)
                _write_json(summary_dir / "sos1_scalability_grouped.json", {"groups": grouped})
                _write_markdown(summary_dir / "sos1_scalability_summary.md", rows, grouped)
                continue

        try:
            result = run_sos1_local_dominance(
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
            row = _failure_row(name, base_dir, repr(exc), time_limit, threads)
            try:
                d = LocalDominanceData.load(name, base_dir=base_dir)
                err_path = d.instance_dir / "output_data" / "sos1_exp" / "sos1_error.json"
                _write_json(err_path, row)
            except Exception:
                pass

        rows.append(row)

        # Persist after every instance so a long benchmark can be interrupted
        # without losing the completed experiment table.
        grouped = _group_rows(rows)
        _write_csv(summary_dir / "sos1_scalability_summary.csv", rows)
        _write_json(summary_dir / "sos1_scalability_summary.json", {"rows": rows})
        _write_csv(summary_dir / "sos1_scalability_grouped.csv", grouped)
        _write_json(summary_dir / "sos1_scalability_grouped.json", {"groups": grouped})
        _write_markdown(summary_dir / "sos1_scalability_summary.md", rows, grouped)

    return rows


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the SOS1 local-dominance scalability suite sequentially.")
    p.add_argument("--base-dir", default=None, help="Defaults to directory containing this script.")
    p.add_argument("--include-naive", action="store_true", help="Also run lands_instance_naive.")
    p.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional explicit instance names. If omitted, all lands_generated_instances are used.",
    )
    p.add_argument(
        "--scenario-policy",
        choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive",
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument("--time-limit", type=float, default=300.0, help="Per-instance seconds. Default 300.")
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--int-feas-tol", type=float, default=1e-8)
    p.add_argument("--feasibility-tol", type=float, default=1e-8)
    p.add_argument("--tee", action="store_true", help="Show each Gurobi log on the console as well as in files.")
    p.add_argument(
        "--rerun-policy",
        choices=["all", "missing", "nonoptimal"],
        default="missing",
        help=(
            "all=re-run everything; missing=skip any instance with sos1_result.json; "
            "nonoptimal=re-run only existing non-OPTIMAL results."
        ),
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else Path(__file__).resolve().parent

    names = args.instances or discover_instances(base_dir, include_naive=args.include_naive)
    if args.instances and args.include_naive and "lands_instance_naive" not in names:
        names = [*names, "lands_instance_naive"]
    names = sorted(set(names), key=_natural_instance_key)

    if not names:
        raise SystemExit("No instances found.")

    print(f"Base directory: {base_dir}")
    print(f"Instances:      {len(names)}")
    print(f"Time limit:     {args.time_limit} s per instance")
    print(f"Threads:        {args.threads}")
    print(f"Scenario rule:  {args.scenario_policy}")
    print("PreSOS1BigM:    0 (native SOS1 retained; no automatic big-M reformulation)")

    t0 = time.perf_counter()
    rows = run_batch(
        base_dir=base_dir,
        instance_names=names,
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
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
    print(f"Reports:         {base_dir / 'sos1_scalability_summary'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
