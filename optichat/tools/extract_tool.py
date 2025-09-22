import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import cloudpickle
import os
from optichat.config.constants import TMP_MODEL_OBJECT_FOLDER


def extract_expressions_from_lp(lp_local_file_path: str):
    """
    Process the .lp file given by lp_local_file_path,
    get all expressions (constraints and objective) from the LP file.
    
    **This function works for modelling languages that load and write .lp files, 
    not very applicable to Pyomo (as far as I know)
    Therefore this is NOT USED for now.**

    Args:
    lp_local_file_path(str): path to the local LP file

    Returns:
    dict: {name: {'expression': expression, 'component_type': "objective" or "constraint"}}.
    """
    info = {}
    with open(lp_local_file_path, 'r') as f:
        lp_content = f.read()
    obj_keywords_to_check = ["Minimize", "Maximize"]
    existing_obj_keywords = [k for k in obj_keywords_to_check if k in lp_content]
    if len(existing_obj_keywords) > 1:
        raise ValueError(f"The LP file contains both {obj_keywords_to_check} sections.")
    elif len(existing_obj_keywords) == 1:
        obj_keyword = existing_obj_keywords[0]
    else:
        raise ValueError(f"The LP file does not contain any of {obj_keywords_to_check} section.")
    if "Subject To" not in lp_content:
        raise ValueError("The LP file does not contain 'Subject To' section.")

    objective_start = lp_content.find(obj_keyword)
    objective_end = lp_content.find("Subject To")
    objective_section = lp_content[objective_start:objective_end].strip()
    # Parse the objective expression
    lines = objective_section.split('\n')
    objective_expr = obj_keyword
    for line in lines:
        line = line.strip()
        if line and not line.startswith('\\'):  # Skip comment lines
            if ":" in line:
                # Remove the objective name part (e.g., "__OBJ__:")
                expr_part = line[line.find(":") + 1:].strip()
                if expr_part:
                    objective_expr += " " + expr_part
            elif line.startswith(('+', '-')) or any(char.isdigit() for char in line):
                # This is a continuation line with coefficients and variables
                objective_expr += " " + line
    # TODO: address the special characters parsed by .lp file (check .lp file's parser and parse it back)
    parsed_objective_expr = objective_expr.replace("@2D", "-")
    info["obj"] = {"expression": parsed_objective_expr.strip(), "component_type": "objective"}

    # Extract constraint expressions
    sections = lp_content.split("Subject To")[1]
    # TODO: currently NOT handle constraints in "Bounds", "General", "Binary", "End" sections
    next_sections = ["Bounds", "General", "Binary", "End"]
    for section in next_sections:
        if section in sections:
            constraints_section = sections.split(section)[0]
            break
    else:
        constraints_section = sections
    lines = constraints_section.strip().split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if this line starts with a constraint name followed by a colon
        if ":" in line:
            constraint_name = line.split(":")[0].strip()
            # Extract the expression part (after the colon)
            expression = line[line.find(":") + 1:].strip()
            # Continue reading lines until the next constraint or end of section is found
            j = i + 1
            while j < len(lines) and ":" not in lines[j] and lines[j].strip():
                expression += " " + lines[j].strip()
                j += 1
            # TODO: address the special characters parsed by .lp file
            parsed_constraint_name = constraint_name.replace("@2D", "-")
            parsed_constraint_expr = expression.replace("@2D", "-")
            info[parsed_constraint_name] = {"expression": parsed_constraint_expr.strip(),
                                            "component_type": "constraint"}
            # update i to j to skip the lines that have been processed
            i = j - 1
        i += 1
    return info



def extract_model_param(model, termination_condition):
    """
    Extract (mutable) parameters from a Pyomo model.
    Information includes name, component_type, value, TODO:is_RHS?
    """
    param_info = {}
    for param in model.component_objects(pe.Param, active=True):
        if param.mutable:
            for idx in param:
                try:
                    v = param[idx].value
                except Exception as e:
                    raise ValueError(f"Error accessing value of parameter {pe.name(param)} with index {idx}: {e}")
                param_info[pe.name(param[idx])] = {"component_type": "parameter",
                                                   "value": param[idx].value}
    return param_info


def extract_model_var(model, termination_condition):
    """
    Extract variables from a Pyomo model.
    Information includes name, component_type, solution.
    """
    var_info = {}
    for var in model.component_objects(pe.Var, active=True):
        for idx in var:
            try:
                v = var[idx].value
            except Exception as e:
                raise ValueError(f"Error accessing value of variable {pe.name(var)} with index {idx}: {e}")
            var_info[pe.name(var[idx])] = {"component_type": "variable",
                                           "solution": var[idx].value}
    return var_info


def extract_model_constraint(model, termination_condition):
    """
    Extract constraints from a Pyomo model.
    Information includes name, component_type, expression, TODO: is_binding implementation here needs verification
    """
    eps = 1e-5
    constraint_info = {}
    for constraint in model.component_objects(pe.Constraint, active=True):
        for idx in constraint:
            try:
                v = constraint[idx].expr
            except Exception as e:
                raise ValueError(f"Error accessing expression of constraint {pe.name(constraint)} with index {idx}: {e}")
            try:
                lslack = constraint[idx].lslack()
                uslack = constraint[idx].uslack()
            except Exception as e:
                raise ValueError(f"Error accessing slack of constraint {pe.name(constraint)} with index {idx}: {e}")
            if str(termination_condition) == 'optimal':
                if abs(constraint[idx].lslack()) < eps or abs(constraint[idx].uslack()) < eps:
                    is_binding = True
                else:
                    is_binding = False
            else:
                is_binding = "unknown"
            constraint_info[pe.name(constraint[idx])] = {"component_type": "constraint",
                                                         "expression": str(constraint[idx].expr), 
                                                         "is_binding": is_binding}
    return constraint_info


def extract_model_objective(model, termination_condition):
    """
    Extract objectives from a Pyomo model.
    Information includes name, component_type, expression (with sense: min/max), value. 
    """
    objective_info = {}
    objectives = list(model.component_objects(pe.Objective, active=True))
    if len(objectives) > 1:
        raise ValueError("The model has multiple objectives, which is not supported.")
    else:
        obj = objectives[0]
        if obj.sense == pe.minimize:
            obj_sense = "MINIMIZE: "
        elif obj.sense == pe.maximize:
            obj_sense = "MAXIMIZE: "
        else:
            raise ValueError(f"Unknown objective sense {obj.sense} for objective {pe.name(obj)}")
        objective_info["obj"] = {"component_type": "objective",
                                 "expression": obj_sense + str(obj.expr),
                                 "sol_status": str(termination_condition), 
                                 "value": pe.value(obj) if str(termination_condition) == 'optimal' else "unknown"}
    return objective_info


def extract_model_info(model, termination_condition='unknown'):
    # parameters
    param_info = extract_model_param(model, termination_condition)
    # variables
    var_info = extract_model_var(model, termination_condition)
    # constraints
    constraint_info = extract_model_constraint(model, termination_condition)
    # objective
    objective_info = extract_model_objective(model, termination_condition)
    # combine all info
    info = {**param_info, **var_info, **constraint_info, **objective_info}
    return info


def restore_model_object(file_path):
    """
    use cloudpickle to restore a Pyomo model object from a file.
    Note that file_name without suffix. .pkl is returned as well.
    """
    with open(file_path, mode='rb') as file:
        model = cloudpickle.load(file)
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return model, file_name


def save_model_object(model, file_name):
    """
    use cloudpickle to save a Pyomo model object to a file.
    Note that file_name is without suffix .pkl
    """
    folder_name = os.path.join(os.getcwd(), TMP_MODEL_OBJECT_FOLDER)
    os.makedirs(folder_name, exist_ok=True)
    local_path_to_object = os.path.join(folder_name, f"{file_name}.pkl")

    with open(local_path_to_object, mode='wb') as file:
        cloudpickle.dump(model, file)
    return local_path_to_object


def auto_extract_function_docs(module_path: str) -> str:
    """
    Automatically extracts function documentation from docstrings
    for all functions in the specified module path.
    Returns formatted strings that can be directly used in prompts.

    Args:
        module_path (str): Path to the module to extract docs
                           (e.g., 'optichat.tools.shortcut_functions').

    Returns:
        str: Formatted documentation strings extracted from docstrings.
    """
    try:
        # Import the module dynamically
        import importlib
        module = importlib.import_module(module_path)
        import inspect
        docs = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            # Skip private functions and utility functions
            if name.startswith("_") or name in ["auto_extract_function_docs"]:
                continue
            # Skip functions not defined in the target module
            if obj.__module__ != module.__name__:
                continue
            # Get docstrings
            docstring = inspect.getdoc(obj)
            if docstring:
                docs.append(docstring)
        return "\n".join(docs)
    except ImportError as e:
        return f"Error importing module {module_path}: {e}"

