from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os, shutil, tempfile, subprocess
from loguru import logger

import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.core.base.param import ParamData
from pyomo.core.base.componentuid import ComponentUID

from google.adk.tools.tool_context import ToolContext
from optichat.tools.shortcut_functions import load_model, solve_model, parse_uncertainty_from_state, relax_constraint_and_penalize_violation
from optichat.tools.extract_tool import unique_component_name
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS 

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


def robustness_analysis(
    version: str,
    uncertain_param_names: List[str],
    bounds: List[List[float]],
    tool_context: ToolContext = None,
    n_scenarios: int = 10,
    dist: str = "uniform",
) -> Dict[str, Any]:
    """
    Generate scenarios and run robustness analysis on a FEASIBLE base model `version`.

    Parameters
    ----------
    version : str
        The base model version name.
    uncertain_param_names : list of str
        Exact parameter names as strings, e.g. ["demand[1,1]", "demand[2,1]"] or ["cost"].
    bounds : list of [lb, ub]
        Lower/upper bounds aligned 1:1 with uncertain_param_names, e.g. [[12, 18], [10, 20]].
    n_scenarios : int
        Number of scenarios to sample (default 10).
    dist : str
        Distribution to use: "uniform" (default) or "normal".

    Returns
    -------
    dict
        JSON-safe payload with status, result summary, and data (DataFrame records).
    """
    # --- Input validation ---
    if len(uncertain_param_names) != len(bounds):
        return {
            "status": "error",
            "result": (
                f"'uncertain_param_names' has {len(uncertain_param_names)} entries but "
                f"'bounds' has {len(bounds)}. They must be the same length."
            ),
        }

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

    # --- Resolve string param names -> Pyomo objects ---
    try:
        uncertain_params = [_resolve_param(base_model, name) for name in uncertain_param_names]
    except ValueError as e:
        return {"status": "error", "result": str(e)}

    # --- Convert bounds List[List[float]] -> List[Tuple[float, float]] ---
    bounds_tuples: List[Tuple[float, float]] = [(float(b[0]), float(b[1])) for b in bounds]

    # --- Robustness analysis ---
    if not hasattr(robust_core, "run_robustness"):
        raise RuntimeError(
            "robust_analysis.robustness_analysis has no supported entrypoint. "
            "Missing run_robustness function."
        )

    robust_function = getattr(robust_core, "run_robustness")
    robust_df = robust_function(
        model=base_model,
        uncertain_params=uncertain_params,
        bounds=bounds_tuples,
        n_scenarios=n_scenarios,
        dist=dist,
    )

    # --- JSON-safe return ---
    js = _json_safe(robust_df)
    rows = 0
    try:
        rows = int(js.get("schema", {}).get("rows", 0))
    except Exception:
        pass

    return {
        "status": "success",
        "result": (
            f"Robustness analysis ({dist}, {n_scenarios} scenarios) completed for '{version}'. "
            f"Rows: {rows}. Uncertain params: {uncertain_param_names}."
        ),
        "data": js,
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
