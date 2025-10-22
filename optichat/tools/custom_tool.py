from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re, shutil, tempfile, subprocess
from loguru import logger

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from google.adk.tools.tool_context import ToolContext
from optichat.tools.shortcut_functions import load_model, solve_model
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS 


# =========================
# Helpers
# =========================

def write_lp_with_symbolic_names(model: pyo.ConcreteModel, lp_path: str) -> None:
    """
    Brief: Write an LP file with symbolic labels so IIS entries match Pyomo names.

    Operations:
      1) model.write(lp_path, io_options={'symbolic_solver_labels': True})
    Returns:
      None
    """
    model.write(lp_path, io_options={"symbolic_solver_labels": True})


def run_gurobi_cli_iis(lp_path: str, workdir: Optional[str] = None) -> Optional[str]:
    """
    Brief: Request IIS via Gurobi CLI with DualReductions=0 (robust for INF_OR_UNBD).

    Operations:
      1) Call: gurobi_cl DualReductions=0 IIS=1 <lp_path>
      2) Return the generated .ilp path if found, else None
    Returns:
      str | None
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
    Brief: Parse the 'Subject To' section of an LP/ILP file and list implicated constraints.

    Operations:
      1) Read text
      2) Extract labels '<name>:' within 'Subject To' block
      3) Deduplicate in order
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


def unique_component_name(model: pyo.ConcreteModel, base: str) -> str:
    """
    Brief: Generate a unique component name under the model.

    Operations:
      1) If 'base' exists, append _2, _3, ... until unique
    Returns:
      str
    """
    if not hasattr(model, base):
        return base
    k = 2
    while hasattr(model, f"{base}_{k}"):
        k += 1
    return f"{base}_{k}"


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


# Infeasibility Diagnosis

def infeasibility_diagnosis(
    version: str,
    tool_context: ToolContext
) -> str:
    """
    infeasibility_diagnosis is a tool that performs infeasibility diagnosis on a specified model version

    Args:
        version (str): Model version to perform infeasibility diagnosis on. 
    Returns:
        Dict[str, str]: a dictionary with two keys: "status" and "result"
        "status": "success" or "error"
        "result": a report about the Irreducible Infeasible Subsystem (IIS) that represent the minimal set of constraints causing infeasibility, 
        and corresponding recommendations for feasibility restoration.
    """
    # TODO: add these keys to constants when more options are considered
    solver_name = tool_context.state.get("SOLVER_NAME", "gurobi")
    solver_options = tool_context.state.get("SOLVER_OPTIONS", None)
    tee = bool(tool_context.state.get("SOLVE_TEE", False))
    save_iis_dir = tool_context.state.get("IIS_SAVE_DIR", os.path.join("tmp", "iis", version))
    os.makedirs(save_iis_dir, exist_ok=True)

    # solve if not solved yet
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    if models_dictionary.get(version, {}).get("obj", {}).get("sol_status", "unknown") == "unknown":
        model = load_model(version, models_dictionary)
        models_dictionary = solve_model(model, version, models_dictionary)
        tool_context.state[MODELS_DICTIONARY] = models_dictionary
    
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    model = load_model(version, models_dictionary)
    info = models_dictionary.get(version, {}).get("obj", {})
    status = info.get("sol_status", "unknown")
    objval = info.get("value", "unknown")
    # stop if NOT infeasible
    if status not in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        return {"status": "success", "result": "Model is NOT infeasible; infeasibility diagnosis terminated directly."}
    # Produce IIS (robust path, then fallback)
    with tempfile.TemporaryDirectory() as td:
        lp_path = os.path.join(td, "model.lp")
        write_lp_with_symbolic_names(model, lp_path)

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
                tool_context.state[MODELS_DICTIONARY] = models_dictionary
                logger.error(f"write_iis failed: {e}")
                return {"status": "error", 
                        "result": f"Model is infeasible, but write_iis (internal function) failed: {e}"}

        parsed = iis2json(iis_path)
        constraints = parsed.get("constraints", [])

        final_artifact = None
        if save_iis_dir:
            try:
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

    tool_context.state[MODELS_DICTIONARY] = models_dictionary

    if constraints:
        lines = [
            f"IIS includes {len(constraints)} constraint(s): ", 
        ] + constraints
        return {"status": "success", "result": "\n".join(lines)}
    else:
        logger.warning("No constraints parsed from IIS artifact.")
        return {"status": "error", "result": "\nNo constraints parsed from IIS artifact. iis2json might be problematic."}
# Feasibility Restoration

def feasibility_restoration(
    version: str,
    recommendation: Dict[str, Any],
    slack_penalty: float,
    tool_context: ToolContext,
) -> str:
    """
    Brief: Apply a single IIS-based restoration by adding penalized slack to the target constraint; update registry and re-solve.

    Operations:
      1) Identify the active objective and compute penalty sign (min/max).
      2) Locate target constraint by name; deactivate it and add a relaxed copy with nonnegative slack.
      3) Add penalty term to the objective; append restoration record; re-solve and persist registry.

    Returns:
      "Feedback from internal tools:\\n..." (plain text).
    """
    if tool_context is None:
        return "Feedback from internal tools: \nMissing tool_context."

    state = tool_context.state
    try:
        models_dictionary = state["MODELS_DICTIONARY"]
    except KeyError:
        return "Feedback from internal tools: \nMODELS_DICTIONARY not found in tool_context.state."

    # Read solver configuration from state
    solver_name = state.get("SOLVER_NAME", "gurobi")
    solver_options = state.get("SOLVER_OPTIONS", None)
    tee = bool(state.get("SOLVE_TEE", False))

    # Load the current, live model instance from the registry
    model = load_model(version, models_dictionary)

    # Active objective
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

    # Re-solve (same model instance) and persist registry
    models_dictionary = solve_model(
        model, version, models_dictionary,
        solver_name=solver_name, solver_options=solver_options, tee=tee
    )
    state["MODELS_DICTIONARY"] = models_dictionary

    return "Feedback from internal tools: \n" + f"Applied restoration: added penalized slack to '{tname}'. Created: {', '.join(created)}."


# Iterative Infeasibility Restoration

def iterative_feasibility_restoration(
    version: str,
    max_iterations: int,
    slack_penalty: float,
    tool_context: ToolContext
) -> str:
    """
    Brief: Iteratively diagnose infeasibility, apply the first IIS-based restoration, and repeat until feasible or capped.

    Operations:
      1) Read registry and config from application state.
      2) Loop: solve → IIS (robust CLI first, fallback Pyomo IIS) → record IIS → apply first recommendation (penalized slack) → continue.
      3) Persist registry after each step and return a per-iteration summary string.

    Returns:
      "Feedback from internal tools:\\n..." (plain text summary).
    """
    if tool_context is None:
        return "Feedback from internal tools: \nMissing tool_context."

    state = tool_context.state
    try:
        models_dictionary = state["MODELS_DICTIONARY"]
    except KeyError:
        return "Feedback from internal tools: \nMODELS_DICTIONARY not found in tool_context.state."

    solver_name = state.get("SOLVER_NAME", "gurobi")
    solver_options = state.get("SOLVER_OPTIONS", None)
    tee = bool(state.get("SOLVE_TEE", False))
    save_iis_dir = state.get("IIS_SAVE_DIR", os.path.join("tmp", "iis", version))

    # Ensure save dir (best-effort)
    try:
        os.makedirs(save_iis_dir, exist_ok=True)
    except Exception:
        save_iis_dir = None

    # Load model once; modifications (slacks) are applied to this same instance
    model = load_model(version, models_dictionary)

    iteration_summaries: List[str] = []
    for it in range(1, max_iterations + 1):
        # Solve & update registry
        models_dictionary = solve_model(
            model, version, models_dictionary,
            solver_name=solver_name, solver_options=solver_options, tee=tee
        )
        info = models_dictionary.get(version, {}).get("obj", {})
        status = str(info.get("sol_status", "unknown")).lower()
        objval = info.get("value", "unknown")

        feasible_like = ("optimal" in status) or ("feasible" in status and "infeasible" not in status)
        if feasible_like:
            iteration_summaries.append(f"Iteration {it}: Model is feasible. Objective={objval}")
            state["MODELS_DICTIONARY"] = models_dictionary
            header = "Iterative restoration summary:"
            return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

        # IIS (robust path, then fallback)
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
                    state["MODELS_DICTIONARY"] = models_dictionary
                    header = "Iterative restoration summary:"
                    return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

            parsed = iis2json(iis_path)
            constraints = parsed.get("constraints", [])
            artifact_copy = None
            if save_iis_dir:
                try:
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
            state["MODELS_DICTIONARY"] = models_dictionary
            header = "Iterative restoration summary:"
            return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

        # Apply first recommendation using the same in-memory model
        first = {"type": "constraint_slack", "target": constraints[0]}
        fr_msg = feasibility_restoration(
            version=version,
            recommendation=first,
            slack_penalty=slack_penalty,
            tool_context=tool_context,
        )

        # Keep last line of FR message for compact summary
        iteration_summaries.append(
            f"Iteration {it}: Applied restoration on '{constraints[0]}'. {fr_msg.splitlines()[-1]}"
        )

    # Loop exhausted → last attempt and summary
    models_dictionary = solve_model(
        model, version, models_dictionary,
        solver_name=solver_name, solver_options=solver_options, tee=tee
    )
    info = models_dictionary.get(version, {}).get("obj", {})
    status = str(info.get("sol_status", "unknown"))
    iteration_summaries.append("Maximum iterations reached without achieving feasibility.")
    iteration_summaries.append(f"Last status: {status}")
    state["MODELS_DICTIONARY"] = models_dictionary

    header = "Iterative restoration summary:"
    return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)
