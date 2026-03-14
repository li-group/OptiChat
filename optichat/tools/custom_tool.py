from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, re, json, time, shutil, tempfile, subprocess
from loguru import logger

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.core.base.param import ParamData
from pyomo.core.base.componentuid import ComponentUID

from google.adk.tools.tool_context import ToolContext
from optichat.tools.shortcut_functions import load_model, solve_model, _parse_uncertainty_from_state, relax_constraint_and_penalize_violation, add_dual_suffix
from optichat.tools.extract_tool import unique_component_name
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS, TMP_ROBUST_FOLDER

#LDR
from optichat.tools.ldr_explain import core as ldr_core
from optichat.tools.ldr_explain import extractor as ldr_extractor

#Robust Analysis
from optichat.tools.robust_analysis import scenario_generator as robust_scenarios
from optichat.tools.robust_analysis import robustness_analysis as robust_core




# =========================
# Helpers
# =========================

def _json_safe(obj: Any) -> Any:
    """
    Convert pandas/numpy outputs to plain-JSON types for the LLM.
    - DataFrame -> {"schema":{"columns":[...], "rows":N}, "records":[{...}, ...]}
    - numpy scalars -> builtins via .item()
    - numpy arrays / Series -> .tolist()
    - containers -> recurse
    - else -> str(obj)
    """
    # pandas.DataFrame (duck-typed)
    if hasattr(obj, "to_dict") and hasattr(obj, "columns") and hasattr(obj, "shape"):
        records = obj.to_dict(orient="records")
        records = [_json_safe(r) for r in records]
        cols = [str(c) for c in list(obj.columns)]
        return {"schema": {"columns": cols, "rows": int(obj.shape[0])}, "records": records}

    # numpy scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return obj.item()
        except Exception:
            pass

    # numpy array / pandas Series
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return obj.tolist()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_json_safe(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


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


def extract_slack_values_from_model(model: pyo.ConcreteModel, relaxed_constraints: List[str]) -> Dict:
    """
    Extract slack variable values from a model relaxed with relax_constraint_and_penalize_violation.
    
    Args:
        model: The relaxed Pyomo model
        relaxed_constraints: List of original constraint names that were relaxed
    
    Returns:
        Dict mapping (constraint_name, 'ub'/'lb') tuples to slack values
    """
    slack_values = {}
    
    # Debug: Log all variable names in the model
    all_var_names = [var.name for var in model.component_objects(pyo.Var, active=True)]
    logger.debug(f"All variables in model: {all_var_names[:20]}...")  # First 20 for brevity
    slack_var_names = [v for v in all_var_names if 'slack' in v.lower()]
    logger.info(f"Found {len(slack_var_names)} variables with 'slack' in name: {slack_var_names[:10]}")
    
    # Iterate through all variables in the model to find slack variables
    # This approach handles the fact that unique_component_name() may add suffixes
    for var_container in model.component_objects(pyo.Var, active=True):
        var_name = var_container.name.strip("'\"")  # Strip quotes from name
        
        # Check if this is a slack variable (starts with uslack_ or lslack_)
        if var_name.startswith("uslack_") or var_name.startswith("lslack_"):
            slack_type = 'ub' if var_name.startswith("uslack_") else 'lb'
            
            # Extract the constraint name (remove prefix and any unique suffix added by unique_component_name)
            # The constraint name is between the prefix and any potential numeric suffix
            prefix = f"{'uslack_' if slack_type == 'ub' else 'lslack_'}"
            constraint_name_candidate = var_name[len(prefix):]
            
            # Check if this matches any of the relaxed constraints
            # Handle cases where unique_component_name added _2, _3, etc.
            matched_constraint = None
            for relax_constraint in relaxed_constraints:
                if constraint_name_candidate == relax_constraint or \
                   constraint_name_candidate.startswith(relax_constraint + "_"):
                    matched_constraint = relax_constraint
                    break
            
            if matched_constraint:
                try:
                    slack_val = pyo.value(var_container)
                    if slack_val is not None and slack_val > 1e-10:
                        slack_values[(matched_constraint, slack_type)] = slack_val
                        logger.info(f"Extracted slack: {matched_constraint} ({slack_type}) = {slack_val:.4f}")
                except:
                    pass
    
    return slack_values


# Infeasibility Diagnosis

def infeasibility_diagnosis(
    version: str,
    tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Perform multi-stage infeasibility diagnosis.
    
    Returns a dictionary with diagnosis results including any relaxed constraints.
    """
    from optichat.tools.shortcut_functions import load_model  # Import at function start to avoid scoping issues
    
    # --- MULTI-STAGE INFEASIBILITY DIAGNOSIS ---
    logger.info(f"Starting Multi-Stage Infeasibility Diagnosis for version: {version}")
    
    # Read config
    solver_name = tool_context.state.get("SOLVER_NAME", "gurobi")
    solver_options = tool_context.state.get("SOLVER_OPTIONS", None)
    tee = bool(tool_context.state.get("SOLVE_TEE", False))
    save_iis_dir = tool_context.state.get("IIS_SAVE_DIR", os.path.join("tmp", "iis", version))
    os.makedirs(save_iis_dir, exist_ok=True)

    # Load model
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    try:
        model = load_model(version, models_dictionary)
    except KeyError:
        logger.warning(f"Infeasibility Diagnosis failed: Model version '{version}' not found.")
        return f"Error: Model version '{version}' not found. Please verify the version name."   

    # --- ROUND 1: Initial IIS ---
    logger.info("--- Round 1: Initial IIS ---")
    
    with tempfile.TemporaryDirectory() as td:
        lp_path = os.path.join(td, "model_r1.lp")
        write_lp_with_symbolic_names(model, lp_path)
        iis_path = run_gurobi_cli_iis(lp_path, workdir=td)
        
        if not iis_path:
             try:
                 iis_path = os.path.join(td, "fallback.iis.ilp")
                 write_iis(model, iis_path, solver=solver_name)
             except Exception as e:
                 logger.error(f"Failed to generate IIS: {e}")
                 # Record failure
                 append_iis_history(version, models_dictionary, {
                     "supported": False, "summary": f"Round 1 IIS failed: {e}", "constraints": [], "artifact_path": None, "solve": {"status": "infeasible"}
                 })
                 tool_context.state[MODELS_DICTIONARY] = models_dictionary
                 return {"status": "error", "result": f"Failed to generate IIS: {e}"}

        parsed = iis2json(iis_path)
        r1_constraints = parsed.get("constraints", [])
        
        # Save Artifact
        final_artifact = None
        if save_iis_dir:
            try:
                final_artifact = os.path.join(save_iis_dir, "iis_round1.ilp")
                shutil.copyfile(iis_path, final_artifact)
            except Exception:
                final_artifact = None
        
        # Record History
        append_iis_history(version, models_dictionary, {
            "supported": True,
            "summary": f"Round 1 IIS: {len(r1_constraints)} constraints.",
            "constraints": r1_constraints,
            "artifact_path": final_artifact,
            "solve": {"status": "infeasible"},
            "round": 1
        })
        tool_context.state[MODELS_DICTIONARY] = models_dictionary
    
    if not r1_constraints:
        return {"status": "error", "result": "IIS generation returned no constraints."}
        
    logger.info(f"Round 1 IIS found {len(r1_constraints)} constraints.")
    
    # --- 3-Round Iterative Relaxation Workflow ---
    logger.info("--- Starting 3-Round Iterative Relaxation ---")
    
    # Helper to find all indices of a constraint component
    def get_all_constraint_indices(model, base_name):
        c = model.find_component(base_name)
        if c is None: return []
        if c.is_indexed():
            return [c[idx].name for idx in c]
        else:
            return [c.name]

    # Helper to get base name
    def get_base_name(c_name):
        return c_name.split("[")[0]

    # Helper to build a map of "normalized" name -> actual Pyomo name
    def build_constraint_map(model):
        c_map = {}
        for c in model.component_data_objects(pyo.Constraint, active=True):
            raw_name = c.name
            # UNIFY hyphens and underscores: treat them as the same character
            norm_name = raw_name.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace(" ", "").replace("-", "_").replace(",", "_")
            c_map[norm_name] = raw_name
            c_map[raw_name] = raw_name
        return c_map

    def find_real_name(c_map, iis_name):
        # Normalize the IIS name
        norm_iis = iis_name.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace(" ", "").replace("-", "_").replace(",", "_")
        return c_map.get(norm_iis)

    # --- ROUND 1: Relax Initial IIS ---
    logger.info("--- Round 1: Relaxing Initial IIS ---")
    
    version_r1 = f"{version}_relaxed_r1"
    model_r1 = model.clone()
    c_map_r1 = build_constraint_map(model_r1)
    
    for c_name_iis in r1_constraints:
        real_name = find_real_name(c_map_r1, c_name_iis)
        if real_name:
            try:
                relax_constraint_and_penalize_violation(real_name, 10.0, model_r1)
            except Exception as e:
                logger.warning(f"Round 1: Failed to relax {real_name}: {e}")
        else:
            logger.warning(f"Round 1: Could not find model constraint for IIS entry '{c_name_iis}'")
            
    # Solve Round 1
    models_dictionary = solve_model(model_r1, version_r1, models_dictionary, tool_context, description="Round 1 Relaxation (IIS)")
    status_r1 = models_dictionary[version_r1].get("obj", {}).get("sol_status", "unknown")
    logger.info(f"Round 1 Status: {status_r1}")
    
    if status_r1 in [TerminationCondition.optimal, TerminationCondition.feasible, "optimal", "feasible"]:
        obj_val = models_dictionary[version_r1].get("obj", {}).get("value", "N/A")
        
        # Reload the solved model to access slack values
        model_r1_solved = load_model(version_r1, models_dictionary)
        
        # Extract slack values from the relaxed model
        slack_values = extract_slack_values_from_model(model_r1_solved, r1_constraints)
        logger.info(f"Round 1: Extracted {len(slack_values)} slack values")
        
        # Extract unique constraint names that were actually violated (non-zero slacks)
        violated_constraints = sorted(set(key[0] for key in slack_values.keys()))
        
        # Prepare return dictionary
        diagnosis_result = {
            "status": "success",
            "result": f"Infeasibility resolved in Round 1.\n"
                      f"Relaxed Model Version: '{version_r1}'\n"
                      f"Status: {status_r1}\n"
                      f"Objective Value: {obj_val}\n"
                      f"Violated Constraints: {violated_constraints}",
            "relaxed_version": version_r1,
            "final_status": str(status_r1),
            "slack_values": slack_values,
            "violated_constraints": violated_constraints
        }
        
        # Save diagnosis report to file
        report_path = save_diagnosis_report(version, diagnosis_result, round_num=1)
        if report_path:
            diagnosis_result["diagnosis_report_file"] = report_path
        
        return diagnosis_result

    # --- ROUND 2: Pattern Matching / New IIS ---
    logger.info("--- Round 2: Analysis & Relaxation ---")
    
    # Generate IIS for Round 1 model
    r2_constraints = []
    with tempfile.TemporaryDirectory() as td_r2:
        lp_path_r2 = os.path.join(td_r2, "model_r2.lp")
        write_lp_with_symbolic_names(model_r1, lp_path_r2)
        iis_path_r2 = run_gurobi_cli_iis(lp_path_r2, workdir=td_r2)
        
        if not iis_path_r2:
             try:
                 iis_path_r2 = os.path.join(td_r2, "fallback_r2.iis.ilp")
                 write_iis(model_r1, iis_path_r2, solver=solver_name)
             except Exception:
                 iis_path_r2 = None
        
        if iis_path_r2:
            parsed_r2 = iis2json(iis_path_r2)
            r2_constraints = parsed_r2.get("constraints", [])
            
            # Save Round 2 Artifact
            if save_iis_dir:
                try:
                    final_artifact_r2 = os.path.join(save_iis_dir, "iis_round2.ilp")
                    shutil.copyfile(iis_path_r2, final_artifact_r2)
                except Exception:
                    pass
            
    logger.info(f"Round 2 IIS found {len(r2_constraints)} constraints.")
    
    if not r2_constraints:
        logger.warning("Round 2 IIS generation failed or empty. Proceeding to Round 3.")
    else:
        def get_real_base(c_name, c_map):
            real = find_real_name(c_map, c_name)
            if real:
                return get_base_name(real)
            return None

        r1_bases = set()
        for c in r1_constraints:
            b = get_real_base(c, c_map_r1)
            if b: r1_bases.add(b)
            
        r2_bases = set()
        for c in r2_constraints:
            # r2_constraints come from model_r1's IIS, so c_map_r1 is valid
            b = get_real_base(c, c_map_r1) 
            if b: r2_bases.add(b)
            
        common_bases = r1_bases.intersection(r2_bases)
        
        version_r2 = f"{version}_relaxed_r2"
        model_r2 = model_r1.clone() # Start from Round 1 model
        c_map_r2 = build_constraint_map(model_r2)
        
        constraints_to_relax_r2 = []
        
        if common_bases:
            logger.info(f"Round 2: Similarity detected in {common_bases}. Relaxing ALL instances of these types.")
            for base in common_bases:
                indices = get_all_constraint_indices(model_r2, base)
                constraints_to_relax_r2.extend(indices)
        
        for c_iis in r2_constraints:
            real_name = find_real_name(c_map_r2, c_iis)
            if real_name:
                base = get_base_name(real_name)
                if base not in common_bases:
                    constraints_to_relax_r2.append(real_name)
            
        # Relax constraints
        for c_name in constraints_to_relax_r2:
             try:
                relax_constraint_and_penalize_violation(c_name, 10.0, model_r2)
             except Exception as e:
                logger.warning(f"Round 2: Failed to relax {c_name}: {e}")
                
        # Solve Round 2
        models_dictionary = solve_model(model_r2, version_r2, models_dictionary, tool_context, description="Round 2 Relaxation")
        status_r2 = models_dictionary[version_r2].get("obj", {}).get("sol_status", "unknown")
        logger.info(f"Round 2 Status: {status_r2}")
        
        if status_r2 in [TerminationCondition.optimal, TerminationCondition.feasible, "optimal", "feasible"]:
            obj_val = models_dictionary[version_r2].get("obj", {}).get("value", "N/A")
            
            # Reload the solved model to access slack values
            model_r2_solved = load_model(version_r2, models_dictionary)
            
            # Extract slack values from the relaxed model
            slack_values = extract_slack_values_from_model(model_r2_solved, constraints_to_relax_r2)
            logger.info(f"Round 2: Extracted {len(slack_values)} slack values")
            
            # Extract unique constraint names that were actually violated (non-zero slacks)
            violated_constraints = sorted(set(key[0] for key in slack_values.keys()))
            
            # Prepare return dictionary
            diagnosis_result = {
                "status": "success",
                "result": f"Infeasibility resolved in Round 2.\n"
                          f"Relaxed Model Version: '{version_r2}'\n"
                          f"Status: {status_r2}\n"
                          f"Objective Value: {obj_val}\n"
                          f"Violated Constraints: {violated_constraints}",
                "relaxed_version": version_r2,
                "final_status": str(status_r2),
                "slack_values": slack_values,
                "violated_constraints": violated_constraints
            }
            
            # Save diagnosis report to file
            report_path = save_diagnosis_report(version, diagnosis_result, round_num=2)
            if report_path:
                diagnosis_result["diagnosis_report_file"] = report_path
            
            return diagnosis_result

    # --- ROUND 3: Elastic Heuristic ---
    logger.info("--- Round 3: Elastic Heuristic (Final Attempt) ---")
    
    from optichat.tools.elastic_diagnoser import ElasticInfeasibilityDiagnoser
        
    model_elastic = load_model(version, tool_context.state[MODELS_DICTIONARY]) 
    diagnoser = ElasticInfeasibilityDiagnoser(model_elastic)
    
    diagnoser.elasticize_model()
    diagnoser.solve_phase_1()
    safety_set = diagnoser.run_heuristic_2()
    
    version_r3 = f"{version}_relaxed_elastic"
    is_feasible, obj_val_r3, slack_values = diagnoser.verify_feasibility(safety_set)
    model_r3 = diagnoser.model

    # Save the model - solve_model will extract all component info including elastic_slacks
    # Note: Don't pre-populate models_dictionary to avoid version renaming issues
    models_dictionary = solve_model(model_r3, version_r3, models_dictionary, tool_context, description="Round 3 Relaxation (Elastic)")

    # Get the actual version name (may have been renamed if duplicate existed)
    # solve_model adds the version to models_dictionary, so we need to find it
    actual_version = version_r3
    if version_r3 not in models_dictionary:
        # Find the renamed version (e.g., version_r3_2, version_r3_3, etc.)
        for key in models_dictionary.keys():
            if key.startswith(version_r3):
                actual_version = key
                break

    status_r3 = models_dictionary.get(actual_version, {}).get("obj", {}).get("sol_status", "unknown")

    # slack_values already obtained from verify_feasibility on line 521
    # Format the slack values for display
    slack_display_lines = []
    for idx, slack_val in slack_values.items():
        if slack_val > 1e-10:  # Only show non-zero slacks
            slack_display_lines.append(f"  {idx}: {slack_val:.6f}")
    slack_display = "\n".join(slack_display_lines) if slack_display_lines else "  No significant violations"
    
    # Extract unique constraint names that were actually violated (non-zero slacks)
    # Handle both tuple keys (constraint_name, index, type) and simple keys
    violated_constraints = set()
    for key, slack_val in slack_values.items():
        if slack_val > 1e-10:
            # Extract constraint name from the key tuple
            if isinstance(key, tuple) and len(key) >= 1:
                violated_constraints.add(key[0])
            else:
                violated_constraints.add(str(key))
    violated_constraints = sorted(violated_constraints)
    
    # Prepare return dictionary
    diagnosis_result = {
        "status": "success",
        "result": f"Infeasibility resolved in Round 3 (Elastic Heuristic).\n"
                  f"Relaxed Model Version: '{actual_version}'\n"
                  f"Status: {status_r3}\n"
                  f"Objective Value: {obj_val_r3}\n"
                  f"Violated Constraints: {violated_constraints}\n\n"
                  f"Slack Values (Violation Magnitudes):\n{slack_display}",
        "relaxed_version": actual_version,
        "final_status": str(status_r3),
        "slack_values": slack_values,
        "violated_constraints": violated_constraints
    }
    
    # Save diagnosis report to file
    report_path = save_diagnosis_report(version, diagnosis_result, round_num=3)
    if report_path:
        diagnosis_result["diagnosis_report_file"] = report_path
    
    return diagnosis_result
    

def save_diagnosis_report(version: str, diagnosis_result: Dict[str, Any], round_num: int = 3) -> str:
    """
    Save infeasibility diagnosis results to a JSON file.
    
    Args:
        version: Original model name (source model)
        diagnosis_result: Diagnosis dictionary from infeasibility_diagnosis()
        round_num: Which round resolved the issue (1, 2, or 3)
    
    Returns:
        Path to the saved report file
    """
    import os
    import json
    
    # Create output directory
    save_dir = os.path.join(os.getcwd(), "tmp", "inf_detail")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename (JSON format)
    filename = f"{version}_inf_detail.json"
    filepath = os.path.join(save_dir, filename)
    
    # Extract information from diagnosis result
    status = diagnosis_result.get("status", "unknown")
    relaxed_version = diagnosis_result.get("relaxed_version", "N/A")
    final_status = diagnosis_result.get("final_status", "unknown")
    slack_values = diagnosis_result.get("slack_values", {})
    violated_constraints = diagnosis_result.get("violated_constraints", [])
    result_text = diagnosis_result.get("result", "")
    
    # Determine round description
    round_descriptions = {
        1: "Round 1 (IIS Relaxation)",
        2: "Round 2 (Pattern Matching)",
        3: "Round 3 (Elastic Heuristic)"
    }
    round_desc = round_descriptions.get(round_num, f"Round {round_num}")
    
    # Get objective value from result text if available
    obj_val = "N/A"
    if "Objective Value:" in result_text:
        try:
            obj_val_str = result_text.split("Objective Value:")[1].split("\n")[0].strip()
            # Try to convert to float if possible
            try:
                obj_val = float(obj_val_str)
            except:
                obj_val = obj_val_str
        except:
            pass
    
    # Convert slack_values to JSON-serializable format
    # Tuples need to be converted to strings as JSON keys must be strings
    slack_values_json = {}
    for key, value in slack_values.items():
        if isinstance(key, tuple):
            # Convert tuple to string representation
            key_str = str(key)
        else:
            key_str = str(key)
        slack_values_json[key_str] = float(value) if value > 1e-10 else 0.0
    
    # Build the JSON structure
    report_data = {
        "source_model_name": version,
        "relaxed_model_name": relaxed_version,
        "diagnosis_status": status,
        "resolution_round": round_desc,
        "round_number": round_num,
        "solution_status": final_status,
        "objective_value": obj_val,
        "violated_constraints": violated_constraints,
        "slack_values": slack_values_json,
        "full_diagnosis_text": result_text
    }
    
    # Write to JSON file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved infeasibility diagnosis report to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save diagnosis report: {e}")
        return None



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

    # ==== Uncertainty spec (from args or user message) ====
    # if uncertain_params is None or bounds is None:
    #     up_auto, b_auto = _parse_uncertainty_from_state(state)
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


# Robustness Analysis

def _resolve_param(model: pyo.ConcreteModel, name: str):
    """
    Resolve a string like "demand[1,1]" or "cost" into a live Pyomo ParamData or Param object.

    Uses ComponentUID — the Pyomo-native way to resolve component references from strings.
    Handles all index forms (integers, strings, negative numbers, multi-dimensional).

    Raises
    ------
    ValueError
        If the name is not found on the model or is not a Param component.
    """
    cuid = ComponentUID(name)
    comp = cuid.find_component(model)
    if comp is None:
        raise ValueError(f"'{name}' not found on the model.")
    if not isinstance(comp, (pyo.Param, ParamData)):
        raise ValueError(f"'{name}' is not a Param.")
    return comp


def _compute_feasibility_threshold(
    robust_df,
    uncertain_param_names: List[str],
    con_cols_in_df: List[str],
) -> Dict[str, Any]:
    """
    Estimate per-parameter feasibility thresholds from scenario data.

    For each uncertain parameter, compare the distribution of values across
    feasible vs infeasible scenarios to estimate where the transition occurs.

    Strategy (univariate per parameter column):
      - Separate scenario rows into feasible / infeasible based on constraint flags.
      - Compare value distributions in the two groups.
      - If infeasible scenarios cluster at higher values → "upper" threshold
        (crossing above estimated_threshold increases infeasibility risk).
      - If infeasible scenarios cluster at lower values → "lower" threshold.
      - Report the overlap zone as the "transition zone": the value range where
        both feasible and infeasible scenarios co-exist.

    Returns {} if all scenarios share the same feasibility status (nothing to compare).
    """
    if not con_cols_in_df:
        return {}

    con_df = robust_df[con_cols_in_df]
    feasible_mask = (con_df == 0).all(axis=1)

    if feasible_mask.all() or (~feasible_mask).all():
        return {}   # no mixed feasibility — nothing to threshold

    results = {}
    for pname in uncertain_param_names:
        base_name = pname.split("[")[0]

        # Find matching scenario columns for this parameter
        if "[" in pname:
            matching_cols = [c for c in robust_df.columns if c == pname]
        else:
            matching_cols = [c for c in robust_df.columns
                             if c == pname or c.startswith(base_name + "[")]

        if not matching_cols:
            continue

        col_thresholds = {}
        for col in matching_cols:
            feas_vals = robust_df.loc[feasible_mask, col].dropna()
            infeas_vals = robust_df.loc[~feasible_mask, col].dropna()

            if len(feas_vals) == 0 or len(infeas_vals) == 0:
                continue

            feas_min, feas_max = float(feas_vals.min()), float(feas_vals.max())
            infeas_min, infeas_max = float(infeas_vals.min()), float(infeas_vals.max())
            feas_mean = float(feas_vals.mean())
            infeas_mean = float(infeas_vals.mean())

            # Direction: which extreme drives infeasibility
            if infeas_mean > feas_mean:
                direction = "upper"
                threshold = round(infeas_min, 4)
                note = f"above ~{threshold:.4g}: higher infeasibility risk"
            else:
                direction = "lower"
                threshold = round(infeas_max, 4)
                note = f"below ~{threshold:.4g}: higher infeasibility risk"

            # Transition zone: overlap range where both feasible and infeasible exist
            overlap_lo = max(feas_min, infeas_min)
            overlap_hi = min(feas_max, infeas_max)
            transition_zone = (
                [round(overlap_lo, 4), round(overlap_hi, 4)]
                if overlap_lo <= overlap_hi else None
            )

            col_thresholds[col] = {
                "feasible": {
                    "count": int(len(feas_vals)),
                    "min": round(feas_min, 4),
                    "max": round(feas_max, 4),
                    "mean": round(feas_mean, 4),
                },
                "infeasible": {
                    "count": int(len(infeas_vals)),
                    "min": round(infeas_min, 4),
                    "max": round(infeas_max, 4),
                    "mean": round(infeas_mean, 4),
                },
                "estimated_threshold": threshold,
                "direction": direction,
                "note": note,
                "transition_zone": transition_zone,
            }

        if col_thresholds:
            results[pname] = col_thresholds

    return results


def _pre_analyze_params(
    model_data: dict,
    uncertain_param_names: List[str],
) -> Dict[str, Dict]:
    """
    Use the pre-computed JSON model data to identify, for each uncertain param:
      - Which active constraints it appears in (via expression string search)
      - Whether those constraints are binding (from is_binding field)
      - Whether it appears in the objective (from obj.params_in list)
    """
    obj_params_in = set(model_data.get("obj", {}).get("params_in", []))
    results = {}

    for param_name in uncertain_param_names:
        base_name = param_name.split("[")[0]
        pattern = re.compile(r'\b' + re.escape(base_name) + r'(?:\[|\b)')

        binding_keys, nonbinding_keys = [], []

        for key, comp in model_data.items():
            if not isinstance(comp, dict):
                continue
            if comp.get("component_type") != "constraint":
                continue
            if not pattern.search(comp.get("expression", "")):
                continue
            is_binding = comp.get("is_binding", "unknown")
            if is_binding is True:
                binding_keys.append(key)
            elif is_binding is False:
                nonbinding_keys.append(key)

        results[param_name] = {
            "binding_constraint_keys": binding_keys,
            "nonbinding_constraint_keys": nonbinding_keys,
            "in_objective": base_name in obj_params_in,
        }

    return results

# =========================
# Slack-Based Robustness Analysis
# =========================

def _compute_slack_analysis(
    model: pyo.ConcreteModel,
    uncertain_params: list,
    uncertain_param_names: List[str],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    For each uncertain parameter, compute how much it can move before hitting a constraint
    boundary, evaluated at the current fixed solution (no re-solve needed).

    Method
    ------
    For every active constraint, perturb the param once and measure changes in both the
    constraint body AND its bounds:

      body_sens = ∂(body)/∂param
      ub_sens   = ∂(upper)/∂param   (0 if param not in upper bound)
      lb_sens   = ∂(lower)/∂param   (0 if param not in lower bound)

    Slack sensitivity for each side:
      d(ub_slack)/d(param) = ub_sens - body_sens   (upper side:  ub - body)
      d(lb_slack)/d(param) = body_sens - lb_sens   (lower side:  body - lb)

    If slack_sens < 0: increasing param tightens this side →
        max_allowable_increase = slack / |slack_sens|
    If slack_sens > 0: decreasing param tightens this side →
        max_allowable_decrease = slack / slack_sens

    This correctly handles parameters that appear only in a bound (e.g. cap in sum_x ≤ cap)
    where body_sens = 0 but ub_sens = 1, so slack_sens = 1 > 0 → reports max_decrease.
    """
    results: Dict[str, Any] = {}

    # Expand IndexedParams into individual ParamData entries so pyo.value() and
    # set_value() always operate on a single scalar ParamData.
    expanded: List[tuple] = []
    for pname, param in zip(uncertain_param_names, uncertain_params):
        if isinstance(param, pyo.Param) and param.is_indexed():
            for k in param.keys():
                idx_str = ",".join(map(str, k)) if isinstance(k, tuple) else str(k)
                expanded.append((f"{pname}[{idx_str}]", param[k]))
        elif isinstance(param, pyo.Param):
            # Scalar (non-indexed) Param component
            expanded.append((pname, next(iter(param.values()))))
        else:
            # Already a ParamData
            expanded.append((pname, param))

    for pname, param in expanded:
        current_value = pyo.value(param, exception=False)
        if current_value is None:
            results[pname] = {"error": "Cannot read current value of parameter."}
            continue
        current_value = float(current_value)
        h = max(abs(current_value) * 1e-5, 1e-7)

        binding_constraints = []
        non_binding_constraints = []

        for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
            body_before = pyo.value(con.body, exception=False)
            if body_before is None:
                continue

            ub_before = pyo.value(con.upper, exception=False) if con.has_ub() else None
            lb_before = pyo.value(con.lower, exception=False) if con.has_lb() else None

            # Single perturbation: read body and bounds after
            try:
                param.set_value(current_value + h)
                body_after = pyo.value(con.body, exception=False)
                ub_after = pyo.value(con.upper, exception=False) if con.has_ub() else None
                lb_after = pyo.value(con.lower, exception=False) if con.has_lb() else None
            except Exception:
                body_after = ub_after = lb_after = None
            finally:
                param.set_value(current_value)

            if body_after is None:
                continue

            body_sens = (body_after - body_before) / h
            ub_sens = ((ub_after - ub_before) / h) if (ub_after is not None and ub_before is not None) else 0.0
            lb_sens = ((lb_after - lb_before) / h) if (lb_after is not None and lb_before is not None) else 0.0

            # Build constraint key
            comp = con.parent_component()
            if comp.is_indexed():
                idx = con.index()
                idx_str = ",".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
                con_key = f"{comp.name}[{idx_str}]"
            else:
                con_key = comp.name

            # Evaluate each active bound side
            sides = []
            if con.has_ub() and ub_before is not None:
                slack = float(ub_before) - float(body_before)
                slack_sens = ub_sens - body_sens   # d(ub - body)/d(param)
                sides.append((slack, slack_sens, "upper"))
            if con.has_lb() and lb_before is not None:
                slack = float(body_before) - float(lb_before)
                slack_sens = body_sens - lb_sens   # d(body - lb)/d(param)
                sides.append((slack, slack_sens, "lower"))

            for slack, slack_sens, bound_type in sides:
                if abs(slack_sens) < 1e-10:
                    continue  # param does not affect this constraint's slack

                if slack <= tol:
                    binding_constraints.append({
                        "constraint": con_key,
                        "slack": 0.0,
                        "bound_type": bound_type,
                    })
                elif slack_sens < 0:
                    # Increasing param tightens → max allowable increase
                    max_change = slack / abs(slack_sens)
                    room_pct = (max_change / abs(current_value) * 100.0) if abs(current_value) > 1e-10 else float("inf")
                    non_binding_constraints.append({
                        "constraint": con_key,
                        "slack": round(slack, 6),
                        "slack_sensitivity": round(slack_sens, 6),
                        "direction": "increase",
                        "max_allowable_change": round(max_change, 6),
                        "room_pct": round(room_pct, 2),
                        "bound_type": bound_type,
                    })
                else:
                    # Decreasing param tightens → max allowable decrease
                    max_change = slack / slack_sens
                    room_pct = (max_change / abs(current_value) * 100.0) if abs(current_value) > 1e-10 else float("inf")
                    non_binding_constraints.append({
                        "constraint": con_key,
                        "slack": round(slack, 6),
                        "slack_sensitivity": round(slack_sens, 6),
                        "direction": "decrease",
                        "max_allowable_change": round(max_change, 6),
                        "room_pct": round(room_pct, 2),
                        "bound_type": bound_type,
                    })

        tightest = (
            min(non_binding_constraints, key=lambda c: c["max_allowable_change"])
            if non_binding_constraints else None
        )

        results[pname] = {
            "current_value": current_value,
            "binding_constraints": binding_constraints,
            "non_binding_constraints": non_binding_constraints,
            "has_any_binding": len(binding_constraints) > 0,
            "overall_max_change": tightest["max_allowable_change"] if tightest else None,
            "overall_room_pct": tightest["room_pct"] if tightest else None,
            "overall_direction": tightest["direction"] if tightest else None,
            "tightest_constraint": tightest["constraint"] if tightest else None,
        }

    return results


def robustness_analysis(
    version: str,
    uncertain_param_names: List[str],
    tool_context: ToolContext = None,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """
    Slack-based robustness analysis on a FEASIBLE base model `version`.

    Instead of sampling random scenarios, this computes exactly how much each uncertain
    parameter can increase before violating a constraint, given the current fixed solution.

    For a constraint Ax ≤ b with slack = b - Ax:
      - If binding (slack = 0): parameter is already at the limit, no room.
      - If not binding: max_allowable_increase = slack / sensitivity
                        room_pct = max_allowable_increase / current_param_value * 100%
    where sensitivity = ∂(constraint body)/∂(parameter), estimated numerically.

    Parameters
    ----------
    version : str
        The base model version name.
    uncertain_param_names : list of str
        Exact parameter names, e.g. ["demand[1,1]", "cost"].
    tol : float
        Tolerance for classifying a constraint as binding (default 1e-6).

    Returns
    -------
    dict
        JSON-safe payload with status, pre-analysis, shadow prices, and slack results.
    """
    state = tool_context.state
    md = state[MODELS_DICTIONARY].copy()

    # Feasibility guard
    status_in_obj = md.get(version, {}).get("obj", {}).get("sol_status", "unknown")
    if (
        status_in_obj in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]
        or (isinstance(status_in_obj, str) and status_in_obj.lower() in {"infeasible", "infeasibleorunbounded"})
    ):
        return {
            "status": "error",
            "result": f"Base model version '{version}' is not feasible; robustness analysis aborted.",
        }

    base_model = load_model(version, md)

    # Resolve string param names → Pyomo objects
    try:
        uncertain_params = [_resolve_param(base_model, name) for name in uncertain_param_names]
    except ValueError as e:
        return {"status": "error", "result": str(e)}

    # Mutability check (required for numerical perturbation)
    for name, param in zip(uncertain_param_names, uncertain_params):
        comp = param.parent_component() if isinstance(param, ParamData) else param
        if not getattr(comp, "_mutable", False):
            return {
                "status": "error",
                "result": (
                    f"Parameter '{name}' is not mutable. "
                    "Slack-based robustness analysis requires mutable=True parameters. "
                    "Redefine the parameter with mutable=True in the model source."
                ),
            }

    # Pre-analysis: identify binding/non-binding constraints from cached model data
    pre_analysis = _pre_analyze_params(md.get(version, {}), uncertain_param_names)

    pre_lines = []
    for pname, info in pre_analysis.items():
        pre_lines.append(f"\nParameter '{pname}':")
        if info["binding_constraint_keys"]:
            shown = info["binding_constraint_keys"][:5]
            suffix = f" (+{len(info['binding_constraint_keys'])-5} more)" if len(info["binding_constraint_keys"]) > 5 else ""
            pre_lines.append(f"  Binding constraints: {', '.join(shown)}{suffix}")
        if info["nonbinding_constraint_keys"]:
            shown = info["nonbinding_constraint_keys"][:5]
            suffix = f" (+{len(info['nonbinding_constraint_keys'])-5} more)" if len(info["nonbinding_constraint_keys"]) > 5 else ""
            pre_lines.append(f"  Non-binding constraints: {', '.join(shown)}{suffix}")
        if not info["binding_constraint_keys"] and not info["nonbinding_constraint_keys"]:
            pre_lines.append("  Not found in any active constraint.")
        pre_lines.append(f"  In objective: {info['in_objective']}")
    pre_analysis_text = "\n".join(pre_lines) or "No pre-analysis available."

    # Slack-based analysis
    slack_results = _compute_slack_analysis(base_model, uncertain_params, uncertain_param_names, tol)

    # Format for LLM
    slack_lines = []
    for pname, info in slack_results.items():
        if "error" in info:
            slack_lines.append(f"\nParameter '{pname}': ERROR — {info['error']}")
            continue

        slack_lines.append(f"\nParameter '{pname}' (current value = {info['current_value']:.6g}):")

        if info["binding_constraints"]:
            slack_lines.append("  BINDING constraints (already at limit, no room):")
            for c in info["binding_constraints"]:
                slack_lines.append(f"    - {c['constraint']}: slack = 0 ({c['bound_type']} bound)")

        if info["non_binding_constraints"]:
            slack_lines.append("  Non-binding constraints (room available):")
            for c in sorted(info["non_binding_constraints"], key=lambda x: x["max_allowable_change"]):
                verb = "increase" if c["direction"] == "increase" else "decrease"
                slack_lines.append(
                    f"    - {c['constraint']}: slack = {c['slack']:.4g}, "
                    f"param can {verb} by up to {c['max_allowable_change']:.4g} "
                    f"({c['room_pct']:.1f}% of current value)"
                )

        if info["tightest_constraint"]:
            verb = "increase" if info["overall_direction"] == "increase" else "decrease"
            slack_lines.append(
                f"  => Overall: param can {verb} by at most {info['overall_max_change']:.4g} "
                f"({info['overall_room_pct']:.1f}% room), "
                f"tightest constraint: {info['tightest_constraint']}"
            )
        elif not info["binding_constraints"] and not info["non_binding_constraints"]:
            slack_lines.append("  No active constraints found for this parameter.")

    slack_text = "\n".join(slack_lines) or "No slack analysis results."

    return {
        "status": "success",
        "result": (
            f"Slack-based robustness analysis completed for '{version}'.\n"
            f"Uncertain params: {uncertain_param_names}\n\n"
            f"=== Pre-Analysis ===\n{pre_analysis_text}\n\n"
            f"=== Slack-Based Robustness Analysis ===\n{slack_text}"
        ),
        "data": _json_safe(slack_results),
    }

# ================================
# Legacy Robustness Analysis
# ================================
# def robustness_analysis(
#     version: str,
#     uncertain_param_names: List[str],
#     bounds: List,
#     tool_context: ToolContext = None,
#     n_scenarios: int = 40,
#     dist: str = "uniform",
#     bounds_mode: str = "absolute",
#     delta_operation: str = "+-",
# ) -> Dict[str, Any]:
#     """
#     Generate scenarios and run robustness analysis on a FEASIBLE base model `version`.
# 
#     Parameters
#     ----------
#     version : str
#         The base model version name.
#     uncertain_param_names : list of str
#         Exact parameter names as strings, e.g. ["demand[1,1]", "demand[2,1]"] or ["cost"].
#     bounds : list
#         In "absolute" mode (default): list of [lb, ub] pairs, e.g. [[12, 18], [10, 20]].
#         In "delta" mode: list of single delta values, e.g. [10, 5].
#     n_scenarios : int
#         Number of random scenarios to SAMPLE from the uncertainty range (default 40).
#         IMPORTANT: This is NOT the number of perturbation magnitudes.
#         "±10 perturbation" means the sampling RANGE is [current-10, current+10].
#         n_scenarios controls how many random draws are taken from that range.
#         Use at least 40 for a meaningful stress test; 100+ for publication-quality analysis.
#         Never pass n_scenarios=2 just because the user said "±10" — ± defines the range,
#         not the sample count.
#     dist : str
#         Distribution to use: "uniform" (default) or "normal".
#     bounds_mode : str
#         "absolute" (default): bounds are [lb, ub] pairs.
#         "delta": bounds are perturbations applied to current param values via delta_operation.
#     delta_operation : str
#         Only used when bounds_mode="delta". How to compute [lb, ub] from the current value:
#           "+-"  symmetric additive:    [current - delta, current + delta]
#           "+"   upper additive:        [current, current + delta]
#           "-"   lower additive:        [current - delta, current]
#           "*"   multiplicative (frac): [current * (1 - delta), current * (1 + delta)]
# 
#     Returns
#     -------
#     dict
#         JSON-safe payload with status, result summary, and data (DataFrame records).
#     """
#     # --- Input validation ---
#     if len(uncertain_param_names) != len(bounds):
#         return {
#             "status": "error",
#             "result": (
#                 f"'uncertain_param_names' has {len(uncertain_param_names)} entries but "
#                 f"'bounds' has {len(bounds)}. They must be the same length."
#             ),
#         }
# 
#     state = tool_context.state
#     md = state[MODELS_DICTIONARY].copy()
# 
#     # Feasibility guard
#     status_in_obj = md.get(version, {}).get("obj", {}).get("sol_status", "unknown")
#     if (
#         status_in_obj in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]
#         or (isinstance(status_in_obj, str) and status_in_obj.lower() in {"infeasible", "infeasibleorunbounded"})
#     ):
#         return {
#             "status": "error",
#             "result": f"Base model version '{version}' is not feasible; robustness analysis aborted.",
#         }
# 
#     base_model = load_model(version, md)
# 
#     # --- Resolve string param names -> Pyomo objects ---
#     try:
#         uncertain_params = [_resolve_param(base_model, name) for name in uncertain_param_names]
#     except ValueError as e:
#         return {"status": "error", "result": str(e)}
# 
#     # --- Check mutability (immutable Params return raw Python values, not ParamData) ---
#     for name, param in zip(uncertain_param_names, uncertain_params):
#         comp = param.parent_component() if isinstance(param, ParamData) else param
#         if not getattr(comp, "_mutable", False):
#             return {
#                 "status": "error",
#                 "result": (
#                     f"Parameter '{name}' is not mutable. "
#                     "Robustness analysis requires mutable=True parameters. "
#                     "Redefine the parameter with mutable=True in the model source."
#                 ),
#             }
# 
#     # --- Output paths ---
#     robust_out_dir = os.path.join(os.getcwd(), TMP_ROBUST_FOLDER)
#     os.makedirs(robust_out_dir, exist_ok=True)
#     safe_version = version.replace("/", "_").replace("\\", "_")
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     csv_out_path = os.path.join(robust_out_dir, f"{safe_version}_{timestamp}_robust_results.csv")
#     json_out_path = os.path.join(robust_out_dir, f"{safe_version}_{timestamp}_robust_report.json")
# 
#     # --- Pre-analysis: identify binding/non-binding constraints per uncertain param ---
#     pre_analysis = _pre_analyze_params(md.get(version, {}), uncertain_param_names)
# 
#     pre_lines = []
#     for pname, info in pre_analysis.items():
#         pre_lines.append(f"\nParameter '{pname}':")
#         if info["binding_constraint_keys"]:
#             shown = info["binding_constraint_keys"][:5]
#             suffix = f" (+{len(info['binding_constraint_keys'])-5} more)" if len(info["binding_constraint_keys"]) > 5 else ""
#             pre_lines.append(f"  Binding constraints: {', '.join(shown)}{suffix}")
#         if info["nonbinding_constraint_keys"]:
#             shown = info["nonbinding_constraint_keys"][:5]
#             suffix = f" (+{len(info['nonbinding_constraint_keys'])-5} more)" if len(info["nonbinding_constraint_keys"]) > 5 else ""
#             pre_lines.append(f"  Non-binding constraints: {', '.join(shown)}{suffix}")
#         if not info["binding_constraint_keys"] and not info["nonbinding_constraint_keys"]:
#             pre_lines.append("  Not found in any active constraint.")
#         pre_lines.append(f"  In objective: {info['in_objective']}")
#     pre_analysis_text = "\n".join(pre_lines) or "No pre-analysis available."
# 
#     # --- Dual (shadow price) analysis on binding constraints ---
#     has_binding = any(info["binding_constraint_keys"] for info in pre_analysis.values())
#     dual_summary = ""
#     dual_values_by_param: Dict[str, Dict] = {}
# 
#     if has_binding:
#         base_model = add_dual_suffix(base_model)
#         if hasattr(base_model, "dual"):
#             SolverFactory("gurobi").solve(base_model, tee=False, load_solutions=True)
#             all_duals: Dict[str, Any] = {}
#             for con_data, dual_val in base_model.dual.items():
#                 try:
#                     all_duals[pyo.name(con_data)] = dual_val
#                 except Exception:
#                     pass
# 
#             dual_lines = []
#             for pname, info in pre_analysis.items():
#                 if not info["binding_constraint_keys"]:
#                     continue
#                 param_duals: Dict[str, Any] = {}
#                 dual_lines.append(f"\nParameter '{pname}' — shadow prices on binding constraints:")
#                 for con_key in info["binding_constraint_keys"]:
#                     val = all_duals.get(con_key)
#                     param_duals[con_key] = val
#                     label = f"{val:.6g}" if val is not None else "(not available)"
#                     dual_lines.append(f"  {con_key}: {label}")
#                 dual_values_by_param[pname] = param_duals
#             dual_summary = "\n".join(dual_lines) or "No dual values populated."
#         else:
#             dual_summary = "Dual analysis skipped (model has integer/binary variables)."
# 
#     # --- Delta mode: convert perturbations to absolute [lb, ub] using current param values ---
#     if bounds_mode == "delta":
#         _valid_ops = ("+-", "+", "-", "*")
#         if delta_operation not in _valid_ops:
#             return {"status": "error", "result": f"Unknown delta_operation '{delta_operation}'. Use one of {_valid_ops}."}
# 
#         def _delta_to_bounds(current: float, delta: float, op: str):
#             if op == "+-":
#                 return current - delta, current + delta
#             elif op == "+":
#                 return current, current + delta
#             elif op == "-":
#                 return current - delta, current
#             elif op == "*":
#                 return current * (1 - delta), current * (1 + delta)
# 
#         abs_bounds = []
#         for name, param, d in zip(uncertain_param_names, uncertain_params, bounds):
#             delta = float(d[0]) if isinstance(d, (list, tuple)) else float(d)
#             if isinstance(param, ParamData):
#                 # Single indexed entry — one scalar value
#                 current = pyo.value(param)
#                 if current is None:
#                     return {"status": "error", "result": f"Cannot read current value of '{name}' for delta mode."}
#                 lb, ub = _delta_to_bounds(current, delta, delta_operation)
#                 abs_bounds.append([lb, ub])
#             else:
#                 # Whole IndexedParam — compute per-index bounds as a dict
#                 bounds_dict = {}
#                 for k in param.keys():
#                     current = pyo.value(param[k])
#                     if current is None:
#                         return {"status": "error", "result": f"Cannot read current value of '{name}[{k}]' for delta mode."}
#                     bounds_dict[k] = _delta_to_bounds(current, delta, delta_operation)
#                 abs_bounds.append(bounds_dict)
#         bounds = abs_bounds
# 
#     # --- Convert bounds to tuples; dicts (per-index) are passed through as-is ---
#     bounds_tuples = [b if isinstance(b, dict) else (float(b[0]), float(b[1])) for b in bounds]
# 
#     # --- Robustness analysis ---
#     if not hasattr(robust_core, "run_robustness"):
#         raise RuntimeError(
#             "robust_analysis.robustness_analysis has no supported entrypoint. "
#             "Missing run_robustness function."
#         )
# 
#     robust_function = getattr(robust_core, "run_robustness")
#     robust_df = robust_function(
#         model=base_model,
#         uncertain_params=uncertain_params,
#         bounds=bounds_tuples,
#         n_scenarios=n_scenarios,
#         dist=dist,
#         out_path=csv_out_path,
#     )
# 
#     # --- Pre-compute violation summary so the LLM doesn't have to crunch raw data ---
#     # Constraint columns are everything after 'objective' in the result DataFrame.
#     try:
#         obj_idx = list(robust_df.columns).index("objective")
#         con_cols_in_df = list(robust_df.columns[obj_idx + 1:])
#     except ValueError:
#         con_cols_in_df = []
# 
#     if con_cols_in_df:
#         con_df = robust_df[con_cols_in_df]
#         feasible_mask = (con_df == 0).all(axis=1)
#         n_feasible = int(feasible_mask.sum())
#         n_infeasible = n_scenarios - n_feasible
#         violation_counts = con_df.sum().astype(int)
#         violated_cons = violation_counts[violation_counts > 0]
#         if len(violated_cons) > 0:
#             viol_lines = "\n".join(f"  - {c}: violated in {v} scenario(s)" for c, v in violated_cons.items())
#             viol_summary = f"{len(violated_cons)} constraint(s) had violations:\n{viol_lines}"
#         else:
#             viol_summary = "No constraint violations detected across all scenarios."
#         feasibility_summary = (
#             f"{n_feasible}/{n_scenarios} scenarios feasible, {n_infeasible} infeasible.\n"
#             f"{viol_summary}"
#         )
#     else:
#         feasibility_summary = "No constraint columns found in results."
# 
#     # --- Threshold analysis (only when there is a mix of feasible / infeasible scenarios) ---
#     threshold_analysis = _compute_feasibility_threshold(
#         robust_df, uncertain_param_names, con_cols_in_df
#     )
# 
#     if threshold_analysis:
#         th_lines = []
#         for pname, col_thresholds in threshold_analysis.items():
#             for col, info in col_thresholds.items():
#                 f, inf_ = info["feasible"], info["infeasible"]
#                 th_lines.append(f"\nParameter '{col}':")
#                 th_lines.append(
#                     f"  Feasible   ({f['count']} scenarios): "
#                     f"[{f['min']:.4g}, {f['max']:.4g}], mean {f['mean']:.4g}"
#                 )
#                 th_lines.append(
#                     f"  Infeasible ({inf_['count']} scenarios): "
#                     f"[{inf_['min']:.4g}, {inf_['max']:.4g}], mean {inf_['mean']:.4g}"
#                 )
#                 if info["transition_zone"]:
#                     th_lines.append(
#                         f"  Transition zone: [{info['transition_zone'][0]:.4g}, "
#                         f"{info['transition_zone'][1]:.4g}]"
#                     )
#                 th_lines.append(f"  Estimated threshold: {info['note']}")
#         threshold_text = "\n".join(th_lines)
#     elif con_cols_in_df:
#         if n_infeasible == 0:
#             threshold_text = "All scenarios were feasible — system appears robust within the given bounds."
#         else:
#             threshold_text = "All scenarios were infeasible — current solution is already beyond the feasibility boundary."
#     else:
#         threshold_text = "No constraint data available for threshold analysis."
# 
#     # --- JSON-safe return ---
#     js = _json_safe(robust_df)
# 
#     # --- Save structured JSON report ---
#     report = {
#         "version": version,
#         "dist": dist,
#         "n_scenarios": n_scenarios,
#         "uncertain_params": uncertain_param_names,
#         "pre_analysis": {
#             pname: {
#                 "binding_constraints": info["binding_constraint_keys"],
#                 "nonbinding_constraints": info["nonbinding_constraint_keys"],
#                 "in_objective": info["in_objective"],
#                 "shadow_prices": dual_values_by_param.get(pname, {}),
#             }
#             for pname, info in pre_analysis.items()
#         },
#         "robustness": {
#             "n_feasible": n_feasible if con_cols_in_df else None,
#             "n_infeasible": n_infeasible if con_cols_in_df else None,
#             "feasibility_rate": round(n_feasible / n_scenarios, 4) if con_cols_in_df else None,
#             "violated_constraints": {
#                 c: int(v) for c, v in violated_cons.items()
#             } if con_cols_in_df and len(violated_cons) > 0 else {},
#         },
#         "threshold_analysis": {
#             pname: {
#                 col: {
#                     "feasible_count": info["feasible"]["count"],
#                     "feasible_range": [info["feasible"]["min"], info["feasible"]["max"]],
#                     "infeasible_count": info["infeasible"]["count"],
#                     "infeasible_range": [info["infeasible"]["min"], info["infeasible"]["max"]],
#                     "estimated_threshold": info["estimated_threshold"],
#                     "direction": info["direction"],
#                     "note": info["note"],
#                     "transition_zone": info["transition_zone"],
#                 }
#                 for col, info in col_thresholds.items()
#             }
#             for pname, col_thresholds in threshold_analysis.items()
#         },
#         "output_files": {
#             "results_csv": csv_out_path,
#             "scenarios_csv": csv_out_path + ".scenarios.csv",
#             "report_json": json_out_path,
#         },
#     }
#     with open(json_out_path, "w") as f:
#         json.dump(report, f, indent=2, default=str)
# 
#     return {
#         "status": "success",
#         "result": (
#             f"Robustness analysis ({dist}, {n_scenarios} scenarios) completed for '{version}'.\n"
#             f"Uncertain params: {uncertain_param_names}\n\n"
#             f"=== Pre-Analysis ===\n{pre_analysis_text}\n\n"
#             f"=== Dual (Shadow Price) Analysis ===\n{dual_summary or 'No binding constraints found.'}\n\n"
#             f"=== Scenario Feasibility Summary ===\n{feasibility_summary}\n\n"
#             f"=== Feasibility Threshold Analysis ===\n{threshold_text}\n\n"
#             f"Output saved to: {robust_out_dir}"
#         ),
#         "data": js,
#     }