from loguru import logger
import pyomo.environ as pe
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from optichat.tools.extract_tool import extract_model_info, restore_model_object, save_model_object


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

    print(f"Model, in version of {version}, is loaded.")
    print(f"{version} Model status: {sol_status}")
    print(f"{version} Model optimal objective value: {objval}")
    return model


def solve_model(model: pe.ConcreteModel, version: str, models_dictionary: dict):
    """
    ```new_models_dictionary = solve_model(model, version: str, models_dictionary: dict)```
    solves the model and update it in the models_dictionary by labelling it with the given version.
    """
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

    models_dictionary.update({version: info})
    return models_dictionary


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
# TODO: 
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