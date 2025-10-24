from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, shutil, tempfile, subprocess
from loguru import logger

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

from google.adk.tools.tool_context import ToolContext
from optichat.tools.shortcut_functions import load_model, solve_model, parse_uncertainty_from_state
from optichat.tools.extract_tool import unique_component_name
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS 

from optichat.tools.ldr_explain import core as ldr_core
from optichat.tools.ldr_explain import extractor as ldr_extractor


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
    Extract the constraint names between 'Subject To' section and the next section.

    Operations:
      1) Read text
      2) Extract labels '<name>:' within 'Subject To' block
      3) Deduplicate in order
    Returns:
      {"constraints": [str, ...]}
    """
    txt = open(lp_like_path, "r", encoding="utf-8", errors="replace").read()

    capture = False
    block_lines: List[str] = []
    for raw_line in txt.splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()

        if not capture:
            if lower.startswith("subject to"):
                capture = True
                idx = lower.find("subject to")
                remainder = raw_line[idx + len("subject to"):].strip()
                if remainder:
                    block_lines.append(remainder)
            continue

        if lower.startswith(("bounds", "binaries", "binary", "generals", "general", "end")):
            break
        block_lines.append(raw_line)

    if not block_lines:
        block_lines = txt.splitlines()

    names: List[str] = []
    for raw_line in block_lines:
        line = raw_line.strip()
        if not line or line.startswith("\\"):
            continue

        # Capture everything before the first ':'; IIS writers use that portion as the label.
        if ":" not in line:
            continue
        candidate = line.split(":", 1)[0].strip()
        if not candidate:
            continue

        # Gurobi may quote names with single/double quotes; remove them for consistency.
        candidate = candidate.strip("'\"")
        candidate = candidate.replace("(", "[").replace(")", "]")
        names.append(candidate)

    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return {"constraints": ordered}


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

    # Checking if the mdoel verison exists
    if version not in tool_context.state.get(MODEL_VERSIONS, []):
        return {"status": "error", "result": f"Model version '{version}' not found in tool_context.state. Specify the right version"}
    
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
    

# Linear Decision Rule Functions
    
def ldr_model_generator(
    version: str,
    uncertain_params: Optional[List[str]] = None,
    bounds: Optional[Dict[str, tuple]] = None,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """
    Build primal/dual LDRs for a FEASIBLE base model `version`.
    - No autosolve; we only use cached status and bail if not feasible.
    - If `uncertain_params` / `bounds` missing, parse them from the latest user message in state.
    - Stores derived models as {version}__ldr_primal / {version}__ldr_dual and attaches a compact summary to the base.
    """

    state = tool_context.state
    md = state[MODELS_DICTIONARY].copy()
    if md.get(version, {}).get("obj", {}).get("sol_status", "unknown") in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        return {
            "status": "error",
            "result": f"Base model version '{version}' is not feasible; LDR generation aborted."
        }

    base_model = load_model(version, md)
    print(base_model)

    # ==== Uncertainty spec (from args or user message) ====
    # if uncertain_params is None or bounds is None:
    #     up_auto, b_auto = parse_uncertainty_from_state(state)
    #     if uncertain_params is None:
    #         uncertain_params = up_auto
    #     if bounds is None:
    #         bounds = b_auto

    # if not uncertain_params or not bounds:
    #     return {
    #         "status": "error",
    #         "result": "Missing 'uncertain_params' and/or 'bounds'. "
    #                   "Pass them as tool args or include a JSON block / inline spec in your message."
    #     }

    uncertain_params = ["demand[1,1]", "demand[2,1]"]
    bounds = [(12, 18), (10, 20)]

    # ==== LDR core entrypoint check ====
    Core = getattr(ldr_core, "LDRPrimalDualCore", ldr_core)
    target = getattr(Core, "build_extract_solve_both", None)
    if target is None or not callable(target):
        raise RuntimeError("LDR core is missing 'build_extract_solve_both'.")

    # Decide whether to pass 'param_box' or 'bounds' without try/except
    code = getattr(target, "__code__", None)
    varnames = set(code.co_varnames) if code is not None else set()
    use_param_box = "param_box" in varnames
    use_bounds_kw = "bounds" in varnames

    kwargs = {"base_model": base_model, "uncertain_params": uncertain_params, "param_box": bounds, "xi_set": pyo.RangeSet(1, len(uncertain_params) +1),
              "bounds": bounds, "return_models": True, "tee": False}
    # if use_param_box:
    #     kwargs["param_box"] = bounds
    # elif use_bounds_kw:
    #     kwargs["bounds"] = bounds
    # else:
    #     raise RuntimeError("LDR core 'build_extract_solve_both' expects 'param_box' or 'bounds' keyword.")

    res = target(**kwargs)

    # ==== Unpack result (dict / tuple / attribute) ====
    primal_model = dual_model = None
    obj_primal = obj_dual = None

    primal_model = res.get("primal_ldr") 
    dual_model   = res.get("dual_ldr")  
    obj_primal   = res.get("primal_obj")
    obj_dual     = res.get("dual_obj")
    gap          = res.get("gap")
    primal_model_status = res.get("primal_status").get("termination")
    dual_model_status = res.get("dual_status").get("termination")

    if primal_model is None or dual_model is None:
        raise RuntimeError("LDR core did not return recognized 'primal_model' and 'dual_model'.")

    # --- persist minimal LDR entries ---
    p_ver = f"{version}__ldr_primal"
    d_ver = f"{version}__ldr_dual"

    md[p_ver] = {
        "model": primal_model,
        "parent_version": version,
        "role": "ldr_primal",
        "sol_status": "optimal",
        "is_ldr": True,
    }
    md[d_ver] = {
        "model": dual_model,
        "parent_version": version,
        "role": "ldr_dual",
        "sol_status": "optimal",
        "is_ldr": True,
    }

    # state[MODELS_DICTIONARY] = md

    msg = (
        f"LDR generated for '{version}'. "
        f"Primal obj={obj_primal}, Dual obj={obj_dual}, "
        f"Gap(abs)={gap}. "
    )
    return {"status": "success", "result": msg}

def ldr_expression_generator(
    version: str,
    variable: Optional[str] = None,
    side: Optional[str] = None,   # 'primal' | 'dual' | None
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """
    Return the LDR expression text for `variable`.
    - `version` may be the base version OR an LDR-derived version (…__ldr_primal / …__ldr_dual).
    - If `side` omitted, defaults to 'primal'. If version suffix implies side, that wins.
    - Uses tools/ldr_explainer/extractor.py if available; otherwise falls back to Pyomo string.
    """
    state = tool_context.state
    md: Dict[str, Any] = state.get(MODELS_DICTIONARY, {})
    entry = md.get(version)

    # Map base version to LDR version if needed
    if entry is None and version in md and md[version].get("is_ldr") is not True:
        lsum = md[version].get("ldr", {}).get("summary", {})
        target_ver = lsum.get("primal_version")
        if (side or "").lower() == "dual":
            target_ver = lsum.get("dual_version") or target_ver
        version = target_ver or version
        entry = md.get(version)

    if entry is None:
        return {"status": "error", "result": f"Version '{version}' not found in the registry."}
    if not variable:
        return {"status": "error", "result": "Variable name was not provided."}

    # Decide side
    side_txt = (side or "primal").lower()
    if version.endswith("__ldr_primal"):
        side_txt = "primal"
    if version.endswith("__ldr_dual"):
        side_txt = "dual"

    model = entry["model"]

    # Preferred extractor function name order
    candidates = (
        "ldr_expression",
        "get_ldr_expression",
        "expression_for",
        "generate_expression",
    )
    fn = None
    for name in candidates:
        cand = getattr(ldr_extractor, name, None)
        if callable(cand):
            fn = cand
            break
    if fn is None:
        raise RuntimeError("LDR extractor has no suitable expression function.")

    # Call extractor in a single, explicit way (no try/except)
    # Expected signature: (model=..., var_name=..., side=...)
    if "var_name" in getattr(fn, "__code__", None).co_varnames:
        expr_text = fn(model=model, var_name=variable, side=side_txt)
    elif "variable" in getattr(fn, "__code__", None).co_varnames:
        expr_text = fn(model=model, variable=variable, side=side_txt)
    else:
        raise RuntimeError("LDR extractor expression function must accept 'var_name' or 'variable'.")

    # Fallback: if extractor returns None/empty, produce a generic Pyomo representation
    if not expr_text:
        base = variable.split("[", 1)[0]
        comp = getattr(model, base)  # will raise AttributeError if missing — as desired
        expr_text = str(comp)

    return {
        "status": "success",
        "result": f"LDR {side_txt} expression for {variable} (version={version}):\n{expr_text}",
        "data": {"version": version, "side": side_txt, "variable": variable, "expression": str(expr_text)},
    }

# Feasibility Restoration

# def feasibility_restoration(
#     version: str,
#     recommendation: Dict[str, Any],
#     slack_penalty: float,
#     tool_context: ToolContext,
# ) -> str:
#     """
#     Brief: Apply a single IIS-based restoration by adding penalized slack to the target constraint; update registry and re-solve.

#     Operations:
#       1) Identify the active objective and compute penalty sign (min/max).
#       2) Locate target constraint by name; deactivate it and add a relaxed copy with nonnegative slack.
#       3) Add penalty term to the objective; append restoration record; re-solve and persist registry.

#     Returns:
#       "Feedback from internal tools:\\n..." (plain text).
#     """
#     if tool_context is None:
#         return "Feedback from internal tools: \nMissing tool_context."

#     state = tool_context.state
#     try:
#         models_dictionary = state["MODELS_DICTIONARY"]
#     except KeyError:
#         return "Feedback from internal tools: \nMODELS_DICTIONARY not found in tool_context.state."

#     # Read solver configuration from state
#     solver_name = state.get("SOLVER_NAME", "gurobi")
#     solver_options = state.get("SOLVER_OPTIONS", None)
#     tee = bool(state.get("SOLVE_TEE", False))

#     # Load the current, live model instance from the registry
#     model = load_model(version, models_dictionary)

#     # Active objective
#     try:
#         obj = next(model.component_data_objects(pyo.Objective, active=True))
#     except StopIteration:
#         return "Feedback from internal tools: \nNo active objective to penalize."

#     is_min = (obj.sense == pyo.minimize)
#     penalty_sign = 1.0 if is_min else -1.0

#     if recommendation.get("type") != "constraint_slack":
#         return "Feedback from internal tools: \nUnsupported recommendation type."

#     con_map = {c.name: c for c in model.component_data_objects(pyo.Constraint, active=True)}
#     tname = recommendation.get("target")
#     if tname not in con_map:
#         return "Feedback from internal tools: \nConstraint not found: " + str(tname)

#     c = con_map[tname]
#     safe = str(tname).replace("[", "_").replace("]", "").replace(",", "_").replace(" ", "_")

#     created = []
#     if c.equality():
#         s_pos = pyo.Var(domain=pyo.NonNegativeReals)
#         s_neg = pyo.Var(domain=pyo.NonNegativeReals)
#         name_spos = unique_component_name(model, f"fr_spos_{safe}")
#         name_sneg = unique_component_name(model, f"fr_sneg_{safe}")
#         model.add_component(name_spos, s_pos)
#         model.add_component(name_sneg, s_neg)
#         new_con = pyo.Constraint(expr=(c.body == pyo.value(c.lower) + s_pos - s_neg))
#         name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
#         model.add_component(name_rel, new_con)
#         obj.set_value(obj.expr + penalty_sign * slack_penalty * (s_pos + s_neg))
#         created = [name_spos, name_sneg, name_rel]
#     elif c.has_ub():
#         s = pyo.Var(domain=pyo.NonNegativeReals)
#         name_s = unique_component_name(model, f"fr_s_{safe}")
#         model.add_component(name_s, s)
#         new_con = pyo.Constraint(expr=(c.body <= pyo.value(c.upper) + s))
#         name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
#         model.add_component(name_rel, new_con)
#         obj.set_value(obj.expr + penalty_sign * slack_penalty * s)
#         created = [name_s, name_rel]
#     elif c.has_lb():
#         s = pyo.Var(domain=pyo.NonNegativeReals)
#         name_s = unique_component_name(model, f"fr_s_{safe}")
#         model.add_component(name_s, s)
#         new_con = pyo.Constraint(expr=(c.body >= pyo.value(c.lower) - s))
#         name_rel = unique_component_name(model, f"fr_relaxed_{safe}")
#         model.add_component(name_rel, new_con)
#         obj.set_value(obj.expr + penalty_sign * slack_penalty * s)
#         created = [name_s, name_rel]
#     else:
#         return "Feedback from internal tools: \nConstraint has no bound to relax."

#     c.deactivate()

#     entry = {"type": "constraint_slack", "target": tname, "created": created}
#     append_repairs_applied(version, models_dictionary, entry)

#     # Re-solve (same model instance) and persist registry
#     models_dictionary = solve_model(
#         model, version, models_dictionary,
#         solver_name=solver_name, solver_options=solver_options, tee=tee
#     )
#     state["MODELS_DICTIONARY"] = models_dictionary

#     return "Feedback from internal tools: \n" + f"Applied restoration: added penalized slack to '{tname}'. Created: {', '.join(created)}."


# Iterative Infeasibility Restoration

# def iterative_feasibility_restoration(
#     version: str,
#     max_iterations: int,
#     slack_penalty: float,
#     tool_context: ToolContext
# ) -> str:
#     """
#     Brief: Iteratively diagnose infeasibility, apply the first IIS-based restoration, and repeat until feasible or capped.

#     Operations:
#       1) Read registry and config from application state.
#       2) Loop: solve → IIS (robust CLI first, fallback Pyomo IIS) → record IIS → apply first recommendation (penalized slack) → continue.
#       3) Persist registry after each step and return a per-iteration summary string.

#     Returns:
#       "Feedback from internal tools:\\n..." (plain text summary).
#     """
#     if tool_context is None:
#         return "Feedback from internal tools: \nMissing tool_context."

#     state = tool_context.state
#     try:
#         models_dictionary = state["MODELS_DICTIONARY"]
#     except KeyError:
#         return "Feedback from internal tools: \nMODELS_DICTIONARY not found in tool_context.state."

#     solver_name = state.get("SOLVER_NAME", "gurobi")
#     solver_options = state.get("SOLVER_OPTIONS", None)
#     tee = bool(state.get("SOLVE_TEE", False))
#     save_iis_dir = state.get("IIS_SAVE_DIR", os.path.join("tmp", "iis", version))

#     # Ensure save dir (best-effort)
#     try:
#         os.makedirs(save_iis_dir, exist_ok=True)
#     except Exception:
#         save_iis_dir = None

#     # Load model once; modifications (slacks) are applied to this same instance
#     model = load_model(version, models_dictionary)

#     iteration_summaries: List[str] = []
#     for it in range(1, max_iterations + 1):
#         # Solve & update registry
#         models_dictionary = solve_model(
#             model, version, models_dictionary,
#             solver_name=solver_name, solver_options=solver_options, tee=tee
#         )
#         info = models_dictionary.get(version, {}).get("obj", {})
#         status = str(info.get("sol_status", "unknown")).lower()
#         objval = info.get("value", "unknown")

#         feasible_like = ("optimal" in status) or ("feasible" in status and "infeasible" not in status)
#         if feasible_like:
#             iteration_summaries.append(f"Iteration {it}: Model is feasible. Objective={objval}")
#             state["MODELS_DICTIONARY"] = models_dictionary
#             header = "Iterative restoration summary:"
#             return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

#         # IIS (robust path, then fallback)
#         with tempfile.TemporaryDirectory() as td:
#             lp_path = os.path.join(td, f"iter_{it}.lp")
#             write_lp_with_symbolic_names(model, lp_path)

#             iis_path = run_gurobi_cli_iis(lp_path, workdir=td)
#             if iis_path is None or not os.path.exists(iis_path):
#                 iis_path = os.path.join(td, f"iis_iter_{it}.ilp")
#                 try:
#                     write_iis(model, iis_path, solver=solver_name)
#                 except Exception as e:
#                     iis_record = {
#                         "supported": False,
#                         "summary": f"Iteration {it}: IIS could not be generated: {e}",
#                         "constraints": [],
#                         "artifact_path": None,
#                         "solve": {"status": status, "objective_value": objval},
#                         "iteration": it,
#                     }
#                     append_iis_history(version, models_dictionary, iis_record)
#                     iteration_summaries.append(iis_record["summary"])
#                     state["MODELS_DICTIONARY"] = models_dictionary
#                     header = "Iterative restoration summary:"
#                     return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

#             parsed = iis2json(iis_path)
#             constraints = parsed.get("constraints", [])
#             artifact_copy = None
#             if save_iis_dir:
#                 try:
#                     artifact_copy = os.path.join(save_iis_dir, f"iis_iter_{it}.ilp")
#                     shutil.copyfile(iis_path, artifact_copy)
#                 except Exception:
#                     artifact_copy = None

#             iis_record = {
#                 "supported": True,
#                 "summary": f"Iteration {it}: IIS has {len(constraints)} constraint(s).",
#                 "constraints": constraints,
#                 "artifact_path": artifact_copy,
#                 "solve": {"status": status, "objective_value": objval},
#                 "iteration": it,
#             }
#             append_iis_history(version, models_dictionary, iis_record)

#         if not constraints:
#             iteration_summaries.append(f"Iteration {it}: No IIS recommendations were produced.")
#             state["MODELS_DICTIONARY"] = models_dictionary
#             header = "Iterative restoration summary:"
#             return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)

#         # Apply first recommendation using the same in-memory model
#         first = {"type": "constraint_slack", "target": constraints[0]}
#         fr_msg = feasibility_restoration(
#             version=version,
#             recommendation=first,
#             slack_penalty=slack_penalty,
#             tool_context=tool_context,
#         )

#         # Keep last line of FR message for compact summary
#         iteration_summaries.append(
#             f"Iteration {it}: Applied restoration on '{constraints[0]}'. {fr_msg.splitlines()[-1]}"
#         )

#     # Loop exhausted → last attempt and summary
#     models_dictionary = solve_model(
#         model, version, models_dictionary,
#         solver_name=solver_name, solver_options=solver_options, tee=tee
#     )
#     info = models_dictionary.get(version, {}).get("obj", {})
#     status = str(info.get("sol_status", "unknown"))
#     iteration_summaries.append("Maximum iterations reached without achieving feasibility.")
#     iteration_summaries.append(f"Last status: {status}")
#     state["MODELS_DICTIONARY"] = models_dictionary

#     header = "Iterative restoration summary:"
#     return "Feedback from internal tools: \n" + "\n".join([header] + iteration_summaries)
