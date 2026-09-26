"""Shared sequential scalability runner for the four norm/formulation experiments.

Default generated-instance sweep: all SMALL + MEDIUM LandS instances.
Use ``--include-hard`` to add the HARD family and ``--include-naive`` to add
``lands_instance_naive``.  ``--instances`` always permits an explicit manual
subset, including hard instances even without ``--include-hard``.

A summary is regenerated after every processed instance so interrupted runs
retain usable results.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import re
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from .local_dominance_sos1_norm_common import (
        ExperimentConfig,
        NormSOS1LocalDominanceError,
        run_configured_experiment,
    )
    from .local_dominance_radius_unbounded import LocalDominanceData
except Exception:
    from local_dominance_sos1_norm_common import (
        ExperimentConfig,
        NormSOS1LocalDominanceError,
        run_configured_experiment,
    )
    from local_dominance_radius_unbounded import LocalDominanceData


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
        json.dump(_json_safe(payload), f, indent=2, sort_keys=False)
        f.write("\n")


def _family(name: str) -> str:
    if name == "lands_instance_naive":
        return "naive"
    m = re.match(r"^lands_instance_(small|med|hard)_", name)
    return m.group(1) if m else "other"


def _natural_instance_key(name: str) -> Tuple[Any, ...]:
    fam_order = {"naive": -1, "small": 0, "med": 1, "hard": 2, "other": 9}
    family = _family(name)
    nums = tuple(int(x) for x in re.findall(r"\d+", name))
    return (fam_order.get(family, 9),) + nums + (name,)


def discover_instances(
    base_dir: Path,
    *,
    include_naive: bool = False,
    include_hard: bool = False,
) -> List[str]:
    generated_root = base_dir / "lands_generated_instances"
    names = []
    if generated_root.exists():
        for p in generated_root.glob("lands_instance_*"):
            if not p.is_dir():
                continue
            fam = _family(p.name)
            if fam == "hard" and not include_hard:
                continue
            if fam in {"small", "med", "hard"}:
                names.append(p.name)
    if include_naive and (base_dir / "lands_instance_naive").exists():
        names.append("lands_instance_naive")
    return sorted(set(names), key=_natural_instance_key)


def _flatten_result(result: Mapping[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    struct = result.get("instance_structure", {}) or {}
    raw = result.get("raw_gurobi_model_structure", {}) or {}
    perf = result.get("performance", {}) or {}
    sel = result.get("selected_scenario", {}) or {}
    val = result.get("validation", {}) or {}
    solver = result.get("solver_configuration", {}) or {}

    ev_kkt = val.get("EV_KKT", {}) or {}
    ev_cert = val.get("EV_primal_certificate", {}) or {}
    star_kkt = val.get("star_KKT", {}) or {}

    return {
        "instance_name": result.get("instance_name"),
        "family": _family(str(result.get("instance_name", ""))),
        "norm": config.norm_kind,
        "formulation": config.formulation_kind,
        "num_scenarios": struct.get("num_scenarios"),
        "n_x": struct.get("n_x_first_stage"),
        "n_y": struct.get("n_y_second_stage"),
        "n_recourse_rows": struct.get("n_recourse_rows"),
        "xi_dim": struct.get("xi_dim"),
        "nnz_T": struct.get("nnz_T"),
        "nnz_W": struct.get("nnz_W"),
        "nnz_H": struct.get("nnz_H"),
        "selected_scenario": sel.get("scenario_name"),
        "initial_cost_gap": sel.get("initial_cost_gap"),
        "violation_margin": (result.get("formulation", {}) or {}).get("violation_margin"),
        "raw_model_num_vars": raw.get("num_vars"),
        "raw_model_num_linear_constraints": raw.get("num_linear_constraints"),
        "raw_model_num_quadratic_constraints": raw.get("num_quadratic_constraints"),
        "raw_model_num_sos": raw.get("num_sos"),
        "raw_model_num_binary_vars": raw.get("num_binary_vars"),
        "raw_model_num_quadratic_objective_nonzeros": raw.get("num_quadratic_objective_nonzeros"),
        "status": perf.get("status"),
        "has_incumbent": perf.get("has_incumbent"),
        "runtime_sec": perf.get("runtime_sec_gurobi"),
        "radius_incumbent": perf.get("radius_incumbent"),
        "best_bound_radius": perf.get("best_bound_radius"),
        "objective_raw": perf.get("objective_raw"),
        "best_bound_raw": perf.get("best_bound_raw"),
        "mip_gap": perf.get("mip_gap"),
        "node_count": perf.get("node_count"),
        "simplex_iterations": perf.get("simplex_iterations"),
        "barrier_iterations": perf.get("barrier_iterations"),
        "true_gap_at_returned_xi": val.get("true_gap_from_independent_recourse_LPs"),
        "true_margin_constraint_residual": val.get("true_margin_constraint_residual"),
        "EV_KKT_minus_independent_LP": ev_kkt.get("KKT_minus_independent_LP"),
        "EV_certificate_minus_independent_LP": ev_cert.get("certificate_minus_independent_LP"),
        "star_KKT_minus_independent_LP": star_kkt.get("KKT_minus_independent_LP"),
        "EV_max_row_sos_pair_min_abs": ev_kkt.get("max_row_sos_pair_min_abs"),
        "EV_max_variable_sos_pair_min_abs": ev_kkt.get("max_variable_sos_pair_min_abs"),
        "star_max_row_sos_pair_min_abs": star_kkt.get("max_row_sos_pair_min_abs"),
        "star_max_variable_sos_pair_min_abs": star_kkt.get("max_variable_sos_pair_min_abs"),
        "time_limit_sec": solver.get("time_limit_sec"),
        "target_mip_gap": solver.get("target_mip_gap"),
        "threads": solver.get("threads"),
        "seed": solver.get("seed"),
        "PreSOS1BigM": solver.get("PreSOS1BigM"),
    }


def _failure_row(
    config: ExperimentConfig,
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
        "norm": config.norm_kind,
        "formulation": config.formulation_kind,
        "status": "ERROR",
        "error": message,
        "time_limit_sec": time_limit,
        "threads": threads,
        "violation_margin": violation_margin,
    }
    try:
        d = LocalDominanceData.load(instance_name, base_dir=base_dir)
        row.update({
            "num_scenarios": d.num_scenarios,
            "n_x": d.num_first_stage_vars,
            "n_y": d.num_second_stage_vars,
            "n_recourse_rows": d.num_recourse_constraints,
            "xi_dim": d.xi_dim,
        })
    except Exception:
        pass
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "instance_name", "family", "norm", "formulation", "num_scenarios",
        "n_x", "n_y", "n_recourse_rows", "xi_dim", "nnz_T", "nnz_W", "nnz_H",
        "selected_scenario", "initial_cost_gap", "violation_margin",
        "raw_model_num_vars", "raw_model_num_linear_constraints",
        "raw_model_num_quadratic_constraints", "raw_model_num_sos",
        "raw_model_num_binary_vars", "raw_model_num_quadratic_objective_nonzeros",
        "status", "has_incumbent", "runtime_sec", "radius_incumbent",
        "best_bound_radius", "objective_raw", "best_bound_raw", "mip_gap",
        "node_count", "simplex_iterations", "barrier_iterations",
        "true_gap_at_returned_xi", "true_margin_constraint_residual",
        "EV_KKT_minus_independent_LP", "EV_certificate_minus_independent_LP",
        "star_KKT_minus_independent_LP", "EV_max_row_sos_pair_min_abs",
        "EV_max_variable_sos_pair_min_abs", "star_max_row_sos_pair_min_abs",
        "star_max_variable_sos_pair_min_abs", "time_limit_sec", "target_mip_gap",
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
            row.get("family"), row.get("num_scenarios"), row.get("n_x"),
            row.get("n_y"), row.get("n_recourse_rows"), row.get("xi_dim"),
            row.get("raw_model_num_vars"), row.get("raw_model_num_sos"),
            row.get("norm"), row.get("formulation"), row.get("violation_margin"),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        fam, S, nx, ny, nr, m, nvars, nsos, norm, formulation, eps = key
        statuses = [str(r.get("status")) for r in members]
        optimal = sum(s == "OPTIMAL" for s in statuses)
        timeouts = sum(s == "TIME_LIMIT" for s in statuses)
        incumbents = sum(bool(r.get("has_incumbent")) for r in members)
        nonopt_gaps = [
            r.get("mip_gap") for r in members
            if r.get("status") != "OPTIMAL" and r.get("mip_gap") is not None
        ]
        runtimes = [float(r["runtime_sec"]) for r in members if r.get("runtime_sec") is not None]
        out.append({
            "family": fam,
            "norm": norm,
            "formulation": formulation,
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
            "max_runtime_sec": max(runtimes, default=None),
            "median_node_count": _median(r.get("node_count") for r in members),
            "median_radius_incumbent": _median(r.get("radius_incumbent") for r in members),
            "median_nonoptimal_mip_gap": _median(nonopt_gaps),
        })
    return out


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def _write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    grouped: List[Dict[str, Any]],
    config: ExperimentConfig,
    violation_margin: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {config.summary_stem} scalability summary\n\n")
        f.write(f"- Norm: **{'L2' if config.norm_kind == '2' else 'L-infinity'}**\n")
        f.write(f"- Formulation: **{config.formulation_kind}**\n")
        f.write("- Complementarity: native Gurobi SOS1, with `PreSOS1BigM=0` by default.\n")
        f.write(f"- Violation margin: `Delta(xi) <= -{violation_margin:g}`.\n")
        if config.norm_kind == "2":
            f.write("- Solver objective is squared Euclidean distance; reported radius is its square root.\n")
        else:
            f.write("- Solver objective is the L-infinity radius `rho`.\n")
        f.write("\n## Per-instance results\n\n")
        f.write("| Instance | Family | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Radius | Radius bound | MIP gap | True gap |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r.get('instance_name','')} | {r.get('family','')} | {_fmt(r.get('num_scenarios'))} | "
                f"{_fmt(r.get('n_x'))} | {_fmt(r.get('n_y'))} | {_fmt(r.get('n_recourse_rows'))} | "
                f"{_fmt(r.get('xi_dim'))} | {_fmt(r.get('raw_model_num_sos'))} | {r.get('status','')} | "
                f"{_fmt(r.get('runtime_sec'))} | {_fmt(r.get('radius_incumbent'))} | "
                f"{_fmt(r.get('best_bound_radius'))} | {_fmt(r.get('mip_gap'))} | "
                f"{_fmt(r.get('true_gap_at_returned_xi'))} |\n"
            )

        f.write("\n## Grouped scalability summary\n\n")
        f.write("| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median radius | Median nonoptimal gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for g in grouped:
            f.write(
                f"| {g.get('family','')} | {_fmt(g.get('num_scenarios'))} | {_fmt(g.get('n_x'))} | "
                f"{_fmt(g.get('n_y'))} | {_fmt(g.get('n_recourse_rows'))} | {_fmt(g.get('xi_dim'))} | "
                f"{_fmt(g.get('raw_model_num_vars'))} | {_fmt(g.get('raw_model_num_sos'))} | "
                f"{_fmt(g.get('num_instances'))} | {_fmt(g.get('num_optimal'))} | {_fmt(g.get('num_time_limit'))} | "
                f"{_fmt(g.get('median_runtime_sec'))} | {_fmt(g.get('median_node_count'))} | "
                f"{_fmt(g.get('median_radius_incumbent'))} | {_fmt(g.get('median_nonoptimal_mip_gap'))} |\n"
            )


def _summary_paths(base_dir: Path, config: ExperimentConfig) -> Dict[str, Path]:
    summary_dir = base_dir / config.summary_stem
    stem = f"{config.summary_stem}_scalability"
    return {
        "dir": summary_dir,
        "csv": summary_dir / f"{stem}_summary.csv",
        "json": summary_dir / f"{stem}_summary.json",
        "md": summary_dir / f"{stem}_summary.md",
        "grouped_csv": summary_dir / f"{stem}_grouped.csv",
        "grouped_json": summary_dir / f"{stem}_grouped.json",
    }


def _persist(
    base_dir: Path,
    config: ExperimentConfig,
    rows: List[Dict[str, Any]],
    violation_margin: float,
) -> None:
    grouped = _group_rows(rows)
    paths = _summary_paths(base_dir, config)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _write_csv(paths["csv"], rows)
    _write_json(paths["json"], {"rows": rows})
    _write_csv(paths["grouped_csv"], grouped)
    _write_json(paths["grouped_json"], {"groups": grouped})
    _write_markdown(paths["md"], rows, grouped, config, violation_margin)


def _canonical_path(base_dir: Path, config: ExperimentConfig, instance_name: str) -> Optional[Path]:
    try:
        d = LocalDominanceData.load(instance_name, base_dir=base_dir)
        return d.instance_dir / "output_data" / config.output_subdir / config.canonical_filename
    except Exception:
        return None


def _load_existing(base_dir: Path, config: ExperimentConfig, instance_name: str) -> Optional[Dict[str, Any]]:
    path = _canonical_path(base_dir, config, instance_name)
    if path is not None and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None



def _existing_matches_margin(result: Mapping[str, Any], violation_margin: float) -> bool:
    try:
        stored = (result.get("formulation", {}) or {}).get("violation_margin", 0.0)
        return abs(float(stored) - float(violation_margin)) <= 1e-15
    except Exception:
        return False

def run_batch(
    config: ExperimentConfig,
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
    rerun_policy: str,
    tee: bool,
    keep_native_sos1: bool,
    write_model: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx, instance_name in enumerate(instance_names, start=1):
        print(f"\n[{idx}/{len(instance_names)}] {config.summary_stem}: {instance_name}")
        existing = _load_existing(base_dir, config, instance_name)
        if existing is not None and _existing_matches_margin(existing, violation_margin):
            status = str((existing.get("performance", {}) or {}).get("status", ""))
            skip = (
                rerun_policy == "missing"
                or (rerun_policy == "nonoptimal" and status == "OPTIMAL")
            )
            if skip:
                row = _flatten_result(existing, config)
                row["violation_margin"] = violation_margin
                rows.append(row)
                print(f"  reuse existing result ({status or 'unknown status'})")
                _persist(base_dir, config, rows, violation_margin)
                continue

        try:
            result = run_configured_experiment(
                config,
                instance_name,
                base_dir=base_dir,
                scenario_policy=scenario_policy,
                gap_tol=gap_tol,
                violation_margin=violation_margin,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                seed=seed,
                tee=tee,
                keep_native_sos1=keep_native_sos1,
                write_model=write_model,
            )
            row = _flatten_result(result, config)
            row["violation_margin"] = violation_margin
            rows.append(row)
            print(
                f"  status={row.get('status')} runtime={row.get('runtime_sec')} "
                f"radius={row.get('radius_incumbent')} gap={row.get('mip_gap')}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            rows.append(
                _failure_row(
                    config, instance_name, base_dir, str(exc),
                    time_limit, threads, violation_margin,
                )
            )

        # Persist after every instance so a long run can be interrupted safely.
        _persist(base_dir, config, rows, violation_margin)

    return rows


def build_scalability_parser(config: ExperimentConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Scalability sweep for {config.display_name}.")
    p.add_argument(
        "--base-dir", default=None,
        help="Directory containing lands_generated_instances/ and optionally lands_instance_naive/. Defaults to this file's directory."
    )
    p.add_argument("--include-naive", action="store_true", help="Also run lands_instance_naive.")
    p.add_argument(
        "--include-hard", action="store_true",
        help="Include generated hard instances. Default sweep intentionally stops at medium."
    )
    p.add_argument(
        "--instances", nargs="+", default=None,
        help="Explicit instance names. This overrides size filtering and may include any hard instance."
    )
    p.add_argument(
        "--scenario-policy", choices=["first_positive", "max_gap", "min_positive"],
        default="first_positive"
    )
    p.add_argument("--gap-tol", type=float, default=1e-8)
    p.add_argument("--violation-margin", type=float, default=0.0)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--mip-gap", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--rerun-policy", choices=["missing", "nonoptimal", "all"], default="missing",
        help="missing=reuse any existing result; nonoptimal=rerun existing non-optimal results; all=rerun everything."
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--write-model", action="store_true")
    p.add_argument("--allow-sos1-reformulation", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true",
        help="List the instances that would be run and exit without requiring Gurobi."
    )
    return p


def scalability_main(config: ExperimentConfig, argv: Optional[Sequence[str]] = None) -> int:
    args = build_scalability_parser(config).parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else Path(__file__).resolve().parent

    if args.instances:
        names = list(dict.fromkeys(args.instances))
        if args.include_naive and "lands_instance_naive" not in names:
            names.append("lands_instance_naive")
        names = sorted(names, key=_natural_instance_key)
    else:
        names = discover_instances(
            base_dir,
            include_naive=args.include_naive,
            include_hard=args.include_hard,
        )

    if not names:
        raise SystemExit("No instances found. Check --base-dir or provide --instances explicitly.")

    print(f"Experiment: {config.summary_stem}")
    print(f"Base dir:   {base_dir}")
    print(f"Instances:  {len(names)}")
    for name in names:
        print(f"  - {name}")

    if args.dry_run:
        return 0

    run_batch(
        config,
        base_dir=base_dir,
        instance_names=names,
        scenario_policy=args.scenario_policy,
        gap_tol=args.gap_tol,
        violation_margin=args.violation_margin,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        seed=args.seed,
        rerun_policy=args.rerun_policy,
        tee=not args.quiet,
        keep_native_sos1=not args.allow_sos1_reformulation,
        write_model=args.write_model,
    )
    paths = _summary_paths(base_dir, config)
    print(f"\nSummary: {paths['md']}")
    return 0
