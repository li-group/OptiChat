from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, shutil, tempfile, subprocess

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.core.expr.visitor import identify_mutable_parameters
from pyomo.core.expr.calculus.derivatives import differentiate

# New-architecture helpers (must exist in your project)
from .shortcut_functions import load_model, solve_model, add_dual_suffix


# =========================
# Sensitivity Analysis (legacy-detailed)
# =========================

def sensitivity_analysis(
    version: str,
    models_dictionary: Dict[str, Any],
    queried_components: List[Dict[str, Any]],
    solver_name: str = "gurobi",                      # parity; not used directly
    solver_options: Optional[Dict[str, Any]] = None,  # parity
    tee: bool = False,
) -> str:
    """
    Brief: Perform parameter-level LP sensitivity on queried RHS parameters using derivatives and duals.

    Operations:
      1) Load model; ensure dual suffix; solve (registry updated).
      2) For each queried parameter/index, compute coef = -d(body)/dp + d(lower)/dp + d(upper)/dp.
      3) Multiply coef by constraint dual; aggregate per queried parameter.

    Returns:
      "Feedback from internal tools:\\n..." (plain text).
    """
    model = load_model(version, models_dictionary)

    # Optional LP gate if registry stores this metadata
    model_type = models_dictionary.get(version, {}).get("model type")
    if (model_type is not None) and (model_type != "LP"):
        msg = (
            "Error: The model is not a linear programming model. "
            "Internal tools do not support sensitivity analysis on other types of models."
        )
        return "Feedback from internal tools: \n" + msg

    # Ensure suffixes; solve to populate duals
    model = add_dual_suffix(model)
    if model.find_component("rc") is None:
        model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    models_dictionary = solve_model(model, version, models_dictionary)
    info = models_dictionary.get(version, {}).get("obj", {})
    status = str(info.get("sol_status", "unknown")).lower()
    if "infeasible" in status:
        msg = "Error: The model is infeasible. Sensitivity analysis cannot be performed on an infeasible model."
        return "Feedback from internal tools: \n" + msg

    # Utilities
    def get_component_type(name: str) -> str:
        comp = model.find_component(name)
        if comp is None:
            return "unknown"
        if isinstance(comp, pyo.Param):
            return "parameters"
        if isinstance(comp, pyo.Var):
            return "variables"
        if isinstance(comp, pyo.Constraint):
            return "constraints"
        if isinstance(comp, pyo.Expression):
            return "expressions"
        return "unknown"

    def locate_param_in_constraints(param_name: str, idx) -> List[Dict[str, Any]]:
        """Find constraints influenced by p, with coef = -d(body)/dp + d(lower)/dp + d(upper)/dp."""
        implicated: List[Dict[str, Any]] = []
        param_comp = getattr(model, param_name)

        # EXACT legacy semantics for scalar vs indexed and None:
        if param_comp.is_indexed():
            if idx is None:
                raise IndexError(
                    "Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect."
                )
            param_inst = param_comp[idx]
        else:
            # scalar Param: accept None or () only
            if idx not in (None, ()):
                raise IndexError(
                    "Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect."
                )
            param_inst = param_comp

        target_name = str(param_inst)

        for con_comp in model.component_objects(pyo.Constraint, active=True):
            for con_idx in con_comp.index_set():
                con_i = con_comp[con_idx]
                try:
                    params_in_expr = list(identify_mutable_parameters(con_i.expr))
                except Exception:
                    params_in_expr = []
                if not any(getattr(p, "name", "") == target_name for p in params_in_expr):
                    continue

                d_body = d_lower = d_upper = 0.0
                try:
                    if con_i.body is not None:
                        db = differentiate(con_i.body, wrt=param_inst, mode="reverse_symbolic")
                        d_body = float(pyo.value(db))
                except Exception:
                    d_body = 0.0
                try:
                    if con_i.lower is not None:
                        dl = differentiate(con_i.lower, wrt=param_inst, mode="reverse_symbolic")
                        d_lower = float(pyo.value(dl))
                except Exception:
                    d_lower = 0.0
                try:
                    if con_i.upper is not None:
                        du = differentiate(con_i.upper, wrt=param_inst, mode="reverse_symbolic")
                        d_upper = float(pyo.value(du))
                except Exception:
                    d_upper = 0.0

                coef = -d_body + d_lower + d_upper
                if abs(coef) > 1e-12:
                    implicated.append({
                        "const_name": con_comp.name,
                        "const_indexes": con_idx,
                        "coefficient": coef,
                        "from_body": abs(d_body) > 1e-12,
                        "from_bounds": (abs(d_lower) + abs(d_upper)) > 1e-12,
                    })
        return implicated

    # Build param-constraint pairs (legacy behavior)
    param_const_pairs: List[Dict[str, Any]] = []
    for component in (queried_components or []):
        param_name = component.get("component_name")
        param_indexes = component.get("component_indexes")

        ctype = get_component_type(param_name)
        if ctype != "parameters":
            wrong = ctype if ctype != "unknown" else "unknown type"
            msg = (
                f"Error: {param_name} is not a parameter in the model but a {wrong}. "
                "Please confirm with the user and ask them to provide a valid parameter for sensitivity analysis."
            )
            return "Feedback from internal tools: \n" + msg

        param_obj = getattr(model, param_name)

        def add_entry_for_index(ix):
            consts = locate_param_in_constraints(param_name, ix)
            rhs_like = (len(consts) > 0) and all((c["from_bounds"] and not c["from_body"]) for c in consts)
            if not rhs_like:
                msg = (
                    f"Error: {param_name} is not a RHS parameter in the model. "
                    "Please confirm with the user and ask them to provide a valid RHS parameter for sensitivity analysis, "
                    "or, if they are particularly interested in this parameter, they must specify a modification extent "
                    "(e.g., a 5% increase) to directly assess the impact of this modification."
                )
                raise ValueError(msg)
            param_const_pairs.append({
                "param_name": param_name,
                "param_indexes": ix,
                "consts": consts
            })

        try:
            if isinstance(param_indexes, tuple):
                try:
                    _ = param_obj[param_indexes]
                except Exception:
                    raise IndexError(
                        "Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect."
                    )
                if any(isinstance(x, slice) for x in param_indexes):
                    for model_param_i in param_obj[param_indexes]:
                        add_entry_for_index(model_param_i.index())
                else:
                    add_entry_for_index(param_indexes)
            elif isinstance(param_indexes, slice):
                for model_param_i in param_obj[param_indexes]:
                    add_entry_for_index(model_param_i.index())
            elif isinstance(param_indexes, (int, str)) or param_indexes is None:
                add_entry_for_index(param_indexes)
            else:
                raise IndexError("Error: Unsupported index type for parameter selection.")
        except ValueError as ve:
            return "Feedback from internal tools: \n" + str(ve)
        except IndexError as ie:
            return "Feedback from internal tools: \n" + str(ie)

    # Ensure duals exist (should already)
    if model.find_component("dual") is None:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        models_dictionary = solve_model(model, version, models_dictionary)

    # Attach dual impacts
    for param_const_pair in param_const_pairs:
        for const in param_const_pair["consts"]:
            const_name = const["const_name"]
            const_indexes = const["const_indexes"]
            con_comp = getattr(model, const_name)
            con_i = con_comp[const_indexes]
            dual_val = model.dual.get(con_i, None)
            coef = const["coefficient"]
            const["dual_value"] = None if dual_val is None else float(dual_val) * float(coef)

    # Legacy-style feedback
    lines: List[str] = []
    lines.append("The sensitivity analysis results are as follows: ")
    for param_const_pair in param_const_pairs:
        pname = param_const_pair["param_name"]
        pidx = param_const_pair["param_indexes"]
        idx_text = f" at {pidx}" if pidx is not None else ""
        total = 0.0
        for const in param_const_pair["consts"]:
            dv = const.get("dual_value")
            if dv is not None:
                total += dv
        if total > 1e-5 or total < -1e-5:
            lines.append(f"when a small positive perturbation is made to {pname}{idx_text}, the optimal objective value will change by {total} unit")
        else:
            lines.append(f"when a small positive perturbation is made to {pname}{idx_text}, the optimal objective value will not change ")
    lines.append("Please explain these results to the user. ")
    return "Feedback from internal tools: \n" + "\n".join(lines)


# =============================================
# Infeasibility Diagnosis and Restoration (legacy robustness + new iterative loop)
# =============================================

def write_lp_with_symbolic_names(model: pyo.ConcreteModel, lp_path: str) -> None:
    """
    Brief: Write LP with symbolic labels so IIS lines match Pyomo component names.

    Operations:
      1) model.write(lp_path, io_options={'symbolic_solver_labels': True}).

    Returns:
      None
    """
    model.write(lp_path, io_options={"symbolic_solver_labels": True})


def run_gurobi_cli_iis(lp_path: str, workdir: Optional[str] = None) -> Optional[str]:
    """
    Brief: Compute IIS via Gurobi CLI with DualReductions=0 (robust for INF_OR_UNBD).

    Operations:
      1) Call: gurobi_cl DualReductions=0 IIS=1 <lp_path>.
      2) Detect the IIS .ilp file next to lp_path. Return its path if found.

    Returns:
      Path to generated .ilp, or None if CLI unavailable or fails.
    """
    exe = shutil.which("gurobi_cl")
    if exe is None:
        return None
    wd = workdir or os.path.dirname(os.path.abspath(lp_path)) or "."
    try:
        cmd = [exe, "DualReductions=0", "IIS=1", os.path.abspath(lp_path)]
        subprocess.run(cmd, cwd=wd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        base, _ = os.path.splitext(os.path.basename(lp_path))
        candidate = os.path.join(wd, f"{base}.ilp")
        return candidate if os.path.exists(candidate) else None
    except Exception:
        return None


def iis2json(lp_like_path: str) -> Dict[str, List[str]]:
    """
    Brief: Convert an IIS LP/ILP file into a JSON-friendly list of implicated constraints.

    Operations:
      1) Read the 'Subject To' section.
      2) Collect labels 'name:' and de-duplicate in order.

    Returns:
      {"constraints": [str, ...]}
    """
    txt = open(lp_like_path, "r", encoding="utf-8", errors="replace").read()
    m = re.search(r"Subject To(.*?)(Bounds|Binaries|Binary|Generals|General|End)", txt, flags=re.S | re.I)
    block = m.group(1) if m else txt

    names: List[str] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("\\"):
            continue
        mm = re.match(r"([A-Za-z_][A-Za-z0-9_\[\],\.\-]*)\s*:", line)
        if mm:
            names.append(mm.group(1))

    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return {"constraints": ordered}


def infeasibility_diagnosis(
    version: str,
    models_dictionary: Dict[str, Any],
    solver_name: str = "gurobi",
    solver_options: Optional[Dict[str, Any]] = None,  # parity
    tee: bool = False,
    save_iis_dir: Optional[str] = None,
) -> str:
    """
    Brief: Diagnose infeasibility using robust legacy flow (prefer Gurobi CLI IIS with DualReductions=0; fallback to Pyomo IIS), update registry, and return a text summary.

    Operations:
      1) Load model; solve (registry updated).
      2) If infeasible or INF_OR_UNBD, write symbolic LP; run gurobi_cl DualReductions=0 IIS=1.
         Fallback to pyomo.contrib.iis.write_iis on absence/failure.
      3) Parse IIS; append to registry['iis_history']; optionally save artifact.

    Returns:
      "Feedback from internal tools:\\n..." (plain text).
    """
    model = load_model(version, models_dictionary)
    models_dictionary = solve_model(model, version, models_dictionary)
    info = models_dictionary.get(version, {}).get("obj", {})
    status = str(info.get("sol_status", "unknown")).lower()
    objval = info.get("value", "unknown")

    feasible_like = ("optimal" in status) or ("feasible" in status and "infeasible" not in status)
    if feasible_like:
        return "Feedback from internal tools: \nModel is feasible; no IIS needed."

    with tempfile.TemporaryDirectory() as td:
        lp_path = os.path.join(td, "model.lp")
        write_lp_with_symbolic_names(model, lp_path)

        # Prefers robust CLI IIS; fallback to Pyomo IIS
        iis_path = run_gurobi_cli_iis(lp_path, workdir=td)
        if iis_path is None or not os.path.exists(iis_path):
            iis_path = os.path.join(td, "fallback.iis.ilp")
            try:
                write_iis(model, iis_path, solver=solver_name)
            except Exception as e:
                append_iis_history(
                    version,
                    models_dictionary,
                    {
                        "supported": False,
                        "summary": f"IIS could not be generated: {e}",
                        "constraints": [],
                        "artifact_path": None,
                        "solve": {"status": status, "objective_value": objval},
                    },
                )
                return "Feedback from internal tools: \n" + f"IIS could not be generated: {e}"

        parsed = iis2json(iis_path)
        constraints = parsed.get("constraints", [])

        final_artifact = None
        if save_iis_dir:
            try:
                os.makedirs(save_iis_dir, exist_ok=True)
                final_artifact = os.path.join(save_iis_dir, "iis.ilp")
                shutil.copyfile(iis_path, final_artifact)
            except Exception:
                final_artifact = None

    iis_record = {
        "supported": True,
        "summary": f"IIS includes {len(constraints)} constraint(s).",
        "constraints": constraints,
        "artifact_path": final_artifact,
        "solve": {"status": status, "objective_value": objval},
    }
    append_iis_history(version, models_dictionary, iis_record)

    if constraints:
        first = constraints[0]
        lines = [
            f"IIS includes {len(constraints)} constraint(s).",
            f"First recommendation: add slack to '{first}'.",
        ]
        if final_artifact:
            lines.append(f"IIS artifact saved at: {final_artifact}")
        return "Feedback from internal tools: \n" + "\n".join(lines)
    else:
        return "Feedback from internal tools: \nNo constraints parsed from IIS artifact."


def unique_component_name(model: pyo.ConcreteModel, base: str) -> str:
    """
    Brief: Produce a unique component name to avoid collisions across iterations.

    Operations:
      1) If 'base' exists, append _2, _3, ... until unique.

    Returns:
      Unique name string.
    """
    if not hasattr(model, base):
        return base
    k = 2
    while hasattr(model, f"{base}_{k}"):
        k += 1
    return f"{base}_{k}"


def feasibility_restoration(
    model: pyo.ConcreteModel,
    version: str,
    models_dictionary: Dict[str, Any],
    recommendation: Dict[str, Any],
    slack_penalty: float = 1e6,
) -> str:
    """
    Brief: Apply one restoration recommendation by adding a penalized slack; update registry and return text.

    Operations:
      1) Deactivate target constraint; add relaxed copy with nonnegative slack.
      2) Add penalty to objective (sign handles min/max).
      3) Append to registry['repairs_applied']; re-solve.

    Returns:
      "Feedback from internal tools:\\n..." (plain text).
    """
    try:
        obj = next(model.component_data_objects(pyo.Objective, active=True))
    except StopIteration:
        return "Feedback from internal tools: \nNo active objective to penalize."

    is_min = (obj.sense == pyo.minimize)
    penalty_sign = 1.0 if is_min else -1.0

    if recommendation.get("type") != "constraint_slack":
        return "Feedback from internal tools: \nUnsupported recommendation type."

    con_map = {c.name: c for c in model.component_data_objects(pyo.Constraint, active=True)}
    tname = recommendation.get("target")
    if tname not in con_map:
        return "Feedback from internal tools: \nConstraint not found: " + str(tname)

    c = con_map[tname]
    safe = str(tname).replace("[", "_").replace("]", "").replace(",", "_").replace(" ", "_")

    created = []
    if c.equality():
        s_pos = pyo.Var(domain=pyo.NonNegativeReals)
        s_neg = pyo.Var(domain=pyo.NonNegativeReals)
        name_spos = unique_component_name(model, f"fr_spos_{safe}")
        name_sneg = unique_component_name(model, f"fr_sneg_{safe}")
        model.add_component(name_spos, s_pos)
        model.add_component(name_sneg, s_neg)
        new_con = pyo.Constraint(expr=(c.body == pyo.value(c.lower) + s_pos - s_neg))
        name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
        model.add_component(name_rel, new_con)
        obj.set_value(obj.expr + penalty_sign * slack_penalty * (s_pos + s_neg))
        created = [name_spos, name_sneg, name_rel]
    elif c.has_ub():
        s = pyo.Var(domain=pyo.NonNegativeReals)
        name_s = unique_component_name(model, f"fr_s_{safe}")
        model.add_component(name_s, s)
        new_con = pyo.Constraint(expr=(c.body <= pyo.value(c.upper) + s))
        name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
        model.add_component(name_rel, new_con)
        obj.set_value(obj.expr + penalty_sign * slack_penalty * s)
        created = [name_s, name_rel]
    elif c.has_lb():
        s = pyo.Var(domain=pyo.NonNegativeReals)
        name_s = unique_component_name(model, f"fr_s_{safe}")
        model.add_component(name_s, s)
        new_con = pyo.Constraint(expr=(c.body >= pyo.value(c.lower) - s))
        name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
        model.add_component(name_rel, new_con)
        obj.set_value(obj.expr + penalty_sign * slack_penalty * s)
        created = [name_s, name_rel]
    else:
        return "Feedback from internal tools: \nConstraint has no bound to relax."

    c.deactivate()

    entry = {"type": "constraint_slack", "target": tname, "created": created}
    append_repairs_applied(version, models_dictionary, entry)

    models_dictionary = solve_model(model, version, models_dictionary)

    return "Feedback from internal tools: \n" + f"Applied restoration: added penalized slack to '{tname}'. Created: {', '.join(created)}."


def iterative_feasibility_restoration(
    version: str,
    models_dictionary: Dict[str, Any],
    solver_name: str = "gurobi",                      # parity
    solver_options: Optional[Dict[str, Any]] = None,  # parity
    tee: bool = False,
    max_iterations: int = 10,
    slack_penalty: float = 1e6,
    save_iis_dir: Optional[str] = None,
) -> str:
    """
    Brief: Iteratively diagnose infeasibility, apply the first recommended restoration, and repeat.

    Operations:
      1) For each iteration: solve → IIS (prefer gurobi_cl DualReductions=0; fallback Pyomo IIS) → apply first recommendation → continue.
      2) Logs every IIS to registry['iis_history'] and every fix to registry['repairs_applied'].

    Returns:
      "Feedback from internal tools:\\n..." (plain text summary).
    """
    model = load_model(version, models_dictionary)

    iteration_summaries: List[str] = []
    for it in range(1, max_iterations + 1):
        models_dictionary = solve_model(model, version, models_dictionary)
        info = models_dictionary.get(version, {}).get("obj", {})
        status = str(info.get("sol_status", "unknown")).lower()
        objval = info.get("value", "unknown")

        feasible_like = ("optimal" in status) or ("feasible" in status and "infeasible" not in status)
        if feasible_like:
            iteration_summaries.append(f"Iteration {it}: Model is feasible. Objective={objval}")
            header = "Iterative restoration summary:"
            return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

        with tempfile.TemporaryDirectory() as td:
            lp_path = os.path.join(td, f"iter_{it}.lp")
            write_lp_with_symbolic_names(model, lp_path)
            iis_path = run_gurobi_cli_iis(lp_path, workdir=td)
            if iis_path is None or not os.path.exists(iis_path):
                iis_path = os.path.join(td, f"iis_iter_{it}.ilp")
                try:
                    write_iis(model, iis_path, solver=solver_name)
                except Exception as e:
                    iis_record = {
                        "supported": False,
                        "summary": f"Iteration {it}: IIS could not be generated: {e}",
                        "constraints": [],
                        "artifact_path": None,
                        "solve": {"status": status, "objective_value": objval},
                        "iteration": it,
                    }
                    append_iis_history(version, models_dictionary, iis_record)
                    iteration_summaries.append(iis_record["summary"])
                    header = "Iterative restoration summary:"
                    return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

            parsed = iis2json(iis_path)
            constraints = parsed.get("constraints", [])
            artifact_copy = None
            if save_iis_dir:
                try:
                    os.makedirs(save_iis_dir, exist_ok=True)
                    artifact_copy = os.path.join(save_iis_dir, f"iis_iter_{it}.ilp")
                    shutil.copyfile(iis_path, artifact_copy)
                except Exception:
                    artifact_copy = None

            iis_record = {
                "supported": True,
                "summary": f"Iteration {it}: IIS has {len(constraints)} constraint(s).",
                "constraints": constraints,
                "artifact_path": artifact_copy,
                "solve": {"status": status, "objective_value": objval},
                "iteration": it,
            }
            append_iis_history(version, models_dictionary, iis_record)

        if not constraints:
            iteration_summaries.append(f"Iteration {it}: No IIS recommendations were produced.")
            header = "Iterative restoration summary:"
            return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

        first = {"type": "constraint_slack", "target": constraints[0]}
        fr_msg = feasibility_restoration(
            model=model,
            version=version,
            models_dictionary=models_dictionary,
            recommendation=first,
            slack_penalty=slack_penalty,
        )
        iteration_summaries.append(f"Iteration {it}: Applied restoration on '{constraints[0]}'. {fr_msg.splitlines()[-1]}")

    models_dictionary = solve_model(model, version, models_dictionary)
    info = models_dictionary.get(version, {}).get("obj", {})
    status = str(info.get("sol_status", "unknown"))
    header = "Iterative restoration summary:"
    iteration_summaries.append("Maximum iterations reached without achieving feasibility.")
    iteration_summaries.append(f"Last status: {status}")
    return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)


# ======================
# Registry helpers
# ======================

def append_iis_history(version: str, models_dictionary: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Add an IIS record to the registry entry for this version."""
    entry = models_dictionary.get(version, {})
    history = entry.get("iis_history", [])
    history.append(record)
    entry["iis_history"] = history
    models_dictionary[version] = entry


def append_repairs_applied(version: str, models_dictionary: Dict[str, Any], record: Dict[str, Any]) -> None:
    """Add a restoration record to the registry entry for this version."""
    entry = models_dictionary.get(version, {})
    repairs = entry.get("repairs_applied", [])
    repairs.append(record)
    entry["repairs_applied"] = repairs
    models_dictionary[version] = entry
