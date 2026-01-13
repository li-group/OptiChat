from loguru import logger
import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from optichat.tools.extract_tool import extract_model_info, restore_model_object, save_model_object, unique_component_name
from optichat.tools.metadata_store import (load_metadata, save_metadata, save_model_data,
                                           add_model_to_metadata)
from typing import Any, Dict, List, Tuple
import re, json
from optichat.config.constants import USER_QUERY

def load_model(version: str, models_dictionary: dict):
    """
    ```model = load_model(version: str, models_dictionary: dict)```
    loads a model associated with a given version
    """
    info = models_dictionary[version]
    local_path_to_object = info["local_path_to_object"]
    sol_status = info["obj"].get('sol_status', 'unknown')
    objval = info["obj"].get('value', 'unknown')

    model, file_name = restore_model_object(local_path_to_object)
    
    # remove dual suffix if exists so that it won't interfere with newly added constraints
    if hasattr(model, 'dual'):
        model.del_component(model.dual)

    print(f"Model, in version of {version}, is loaded.")
    print(f"{version} Model status: {sol_status}")
    print(f"{version} Model optimal objective value: {objval}")
    return model


def add_dual_suffix(model: pe.ConcreteModel):
    """
    ```model_with_dual_suffix = add_dual_suffix(model: pe.ConcreteModel)```
    adds dual suffix to the model so that the resulting model will include dual solution after being solved
    """
    if hasattr(model, "dual"):
        print("Model already has dual suffix. Original model is returned.")
    else:
        # simple safeguard to ensure model is LP
        for var in model.component_objects(pe.Var, active=True):
            for idx in var:
                if var[idx].is_binary():
                    print(("Model has binary variables. "
                    "Dual suffix can only be added to LP models. "
                    "Original model is returned."))
                    return model
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    return model 


def solve_model(model: pe.ConcreteModel, version: str, models_dictionary: dict, tool_context=None, description=None):
    """
    ```new_models_dictionary = solve_model(model, version: str, models_dictionary: dict, tool_context=None, description=None)```
    solves the model and update it in the models_dictionary by labelling it with the given version.

    Args:
        model: Pyomo ConcreteModel to solve
        version: Version name for this model
        models_dictionary: Runtime cache of model data
        tool_context: Optional ToolContext for state management
        description: Optional human-readable description of this model variant (RECOMMENDED)
                    Example: "What-if analysis: increased demand[3,1] by 10 to test capacity constraints"

    Example usage:
        ```python
        # Define description before solving
        description = "What-if analysis: increased demand[3,1] by 10 to test peak season capacity"
        model = load_model('supply_chain_model', models_dictionary)
        model.demand[3,1] += 10
        models_dictionary = solve_model(model, 'supply_chain_model__demand_3_1_plus10',
                                       models_dictionary, tool_context, description)
        ```
    """
    # Ensure version name uniqueness (safeguard against duplicate names)
    original_version = version
    if version in models_dictionary:
        counter = 2
        base_version = version
        while version in models_dictionary:
            version = f"{base_version}_{counter}"
            counter += 1
        print(f"⚠️  Version name conflict! Using '{version}' instead of '{base_version}'")

    time_limit_seconds = 180
    print(f"Solving model with time limit of {time_limit_seconds} seconds...")
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = time_limit_seconds
    results = solver.solve(model, tee=False)
    info = extract_model_info(model, results.solver.termination_condition)

    sol_status = info["obj"].get('sol_status', 'unknown')
    objval = info["obj"].get('value', 'unknown')

    if sol_status != "optimal":
        print(f"Model, in version of {version}, is NOT solved.")
    else:
        print(f"Model, in version of {version}, is solved.")
    print(f"{version} Model status: {sol_status}")
    print(f"{version} Model optimal objective value: {objval}")

    local_path_to_object = save_model_object(model, version)
    info.update({
        "local_path_to_object": local_path_to_object,
    })
    print(f"Model, in version of {version}, is updated in the models_dictionary.")

    # Infer base model from version name
    # Format: <base_model>__<param_name>_<index>_<operation><value>
    base_model = None

    if "__" in version:
        # This is a modified model
        parts = version.split("__")
        base_model = parts[0]
    else:
        # This might be an initial model or a simple variant
        # Check if any existing models could be the base
        for existing_version in models_dictionary.keys():
            if existing_version in version and existing_version != version:
                base_model = existing_version
                break

    # Add to runtime cache
    models_dictionary.update({version: info})

    # Save model data to individual file
    try:
        save_model_data(version, info)
        logger.info(f"Saved model data for {version}")
    except Exception as e:
        logger.error(f"Failed to save model data for {version}: {e}")

    # Update metadata (with expert-provided description)
    try:
        metadata = load_metadata()
        metadata = add_model_to_metadata(
            metadata=metadata,
            version_name=version,
            model_info=info,
            base_model=base_model,
            description=description  # Expert-provided description
        )
        save_metadata(metadata)
        logger.info(f"Updated metadata for {version} with description: {description}")
    except Exception as e:
        logger.error(f"Failed to update metadata for {version}: {e}")
        # Continue execution even if save fails

    return models_dictionary


def relax_constraint_and_penalize_violation(constraint_name: str, 
                                            penalty_coef: float | int, 
                                            model: pe.ConcreteModel):
    """
    ```relaxed_model = relax_constraint_and_penalize_violation(constraint_name: str, penalty_coef: float | int, model)```
    relaxes a constraint in the model by adding slacks and penalizes the violation in the objective in place, returns the relaxed model.
    """
    obj = next(model.component_data_objects(pe.Objective, active=True))
    is_min = (obj.sense == pe.minimize)
    penalty_sign = 1.0 if is_min else -1.0
    constraint = model.find_component(constraint_name)
    if constraint:
        if constraint.equality:
            us = pe.Var(domain=pe.NonNegativeReals)
            model.add_component(unique_component_name(model, f"uslack_{constraint_name}"), us)
            ls = pe.Var(domain=pe.NonNegativeReals)
            model.add_component(unique_component_name(model, f"lslack_{constraint_name}"), ls)
            eqcon = pe.Constraint(expr=(constraint.body == pe.value(constraint.lower) + us - ls))
            model.add_component(unique_component_name(model, f"relaxed_{constraint_name}"), eqcon)
            obj.set_value(expr=obj.expr + penalty_sign * penalty_coef * (us + ls))
        elif constraint.has_ub():
            us = pe.Var(domain=pe.NonNegativeReals)
            model.add_component(unique_component_name(model, f"uslack_{constraint_name}"), us)
            ucon = pe.Constraint(expr=(constraint.body <= pe.value(constraint.upper) + us))
            model.add_component(unique_component_name(model, f"relaxed_{constraint_name}"), ucon)
            obj.set_value(expr=obj.expr + penalty_sign * penalty_coef * us)
        elif constraint.has_lb():
            ls = pe.Var(domain=pe.NonNegativeReals)
            model.add_component(unique_component_name(model, f"lslack_{constraint_name}"), ls)
            lcon = pe.Constraint(expr=(constraint.body >= pe.value(constraint.lower) - ls))
            model.add_component(unique_component_name(model, f"relaxed_{constraint_name}"), lcon)
            obj.set_value(expr=obj.expr + penalty_sign * penalty_coef * ls)
        else:
            raise Exception("Constraint has no bounds. No changes made.")
        # deactivated constraint can still be found by model.find_component, delete it to avoid confusion
        if constraint.is_indexed():
             model.del_component(constraint)
        else:
             parent = constraint.parent_component()
             if parent is constraint:
                 # ScalarConstraint
                 model.del_component(constraint)
             else:
                 del parent[constraint.index()]
        
        print(f"Constraint {constraint_name} is relaxed with slacks.")
        print(f"Constraint {constraint_name} violation is penalized in the objective with coefficient {penalty_coef}.")
    else: 
        print(f"Constraint {constraint_name} not found in the model. No changes made.")
    return model


# Parse user query and return the uncertain parameters and it's bounds
def parse_uncertainty_from_state(state: Dict[str, Any]) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """
    Parse uncertainty specification from the most recent user message in `state`.
    Priority:
      1) A fenced JSON code block with keys:
         {"uncertain_params":[...], "bounds":{"p":[lo,hi], ...}}
      2) Inline fallback, e.g.:
         "uncertain: p,q  bounds: p[0,10]; q[-5,5]"   or   "p in [0,10]"
    Returns (uncertain_params, bounds) where bounds[k] = (lo, hi) as floats.
    If nothing is found, returns ([], {}).
    """
    text = (state.get(USER_QUERY) or "").strip()
    if not text:
        return [], {}

    # 1) JSON block (preferred)
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if m:
        try:
            blob = json.loads(m.group(1))
            up = blob.get("uncertain_params") or blob.get("uncertain") or []
            bd = blob.get("bounds") or {}
            up = [str(u) for u in up]
            bounds = {k: (float(v[0]), float(v[1])) for k, v in bd.items()}
            if up or bounds:
                return up, bounds
        except Exception:
            pass

    # 2) Inline fallback
    up: List[str] = []
    bdict: Dict[str, Tuple[float, float]] = {}

    mup = re.search(r"uncertain(?:\s*params)?\s*:\s*([A-Za-z0-9_,\s]+)", text, re.IGNORECASE)
    if mup:
        up = [u.strip() for u in mup.group(1).split(",") if u.strip()]

    # patterns like  p[0,10]  |  p in [0,10]  |  demand[0,1e3]
    for name, lo, hi in re.findall(
        r"([A-Za-z_]\w*)\s*(?:in)?\s*\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]",
        text,
    ):
        bdict[name] = (float(lo), float(hi))

    return up, bdict



# def fix_variable(variable_name: str, value_to_fix: float | int, model: pe.ConcreteModel):
#     """
#     ```fix_variable(variable_name: str, value_to_fix: float | int, model)```
#     fixes a variable in the model to a given value in place, returns nothing.
#     """
#     var = model.find_component(variable_name)
#     var.fix(value_to_fix)


# def unfix_variable(variable_name: str, model: pe.ConcreteModel):
#     """
#     ```unfix_variable(variable_name: str, model)```
#     unfixes a variable in the model in place, returns nothing.
#     """
#     var = model.find_component(variable_name)
#     var.unfix()


# def add_constraint(constraint_name: str, expression: str, model: pe.ConcreteModel):
#     """
#     ```add_constraint(constraint_name: str, expression: str, model)```
#     adds a constraint to the model in place, returns nothing.
# TODO: the most difficult part
# reconstruct pyomo expression from string
# - model.find_component(component_name) can get the actual pyomo component
# - need a way to rearrange the components into a valid pyomo expression from expression string
# need a way to parse indexed expression into pyomo rule function
#     """
#     pass


# def deactivate_constraint(constraint_name: str, model: pe.ConcreteModel):
#     """
#     ```deactivate_constraint(constraint_name: str, model)```
#     deactivates a constraint in the model in place, returns nothing.
#     """
#     constraint = model.find_component(constraint_name)
#     constraint.deactivate()