from typing import Dict, Optional, Union, List
import random
import copy

import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_mutable_parameters, replace_expressions, clone_expression
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.opt import SolverFactory, TerminationCondition, SolverStatus

from extractor import pyomo2json
# from get_code_from_markdown import *
from contextlib import redirect_stdout
import sys
import time

def fnArgsDecoder(queried_components):
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in ["none", "null"]:
                    queried_component[key] = None
                elif value.lower() in ["__all__"]:
                    queried_component[key] = slice(None)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["none", "null", "None", "Null"]:
                        value[i] = None
                    elif value_i in ["__all__"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)
    return queried_components


def old_fnArgsDecoder(queried_components):
    for queried_component in queried_components:
        for key, value in queried_component.items():
            if isinstance(value, str):
                if value.lower() in ["none", "null", "slice(none)", "slice(null)", "slice('none')"]:
                    queried_component[key] = None if "slice" not in value else slice(None)
                elif "slice(None)" in value and value != "slice(None)":
                    queried_component[key] = eval(value)
                # if value in ["None", "null"]:
                #     queried_component[key] = None
                # elif value in ["slice(None)", "slice(null)", "slice('None')"]:
                #     queried_component[key] = slice(None)
                # elif value != 'slice(None)' and 'slice(None)' in value:
                #     # in case that llm should have returned a tuple ('slice(None)', 'slice(None)', "specific_index")
                #     # but returned a string "('slice(None)', 'slice(None)', "specific_index")"
                #     queried_component[key] = eval(value)

            elif isinstance(value, tuple):
                value = list(value)
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                queried_component[key] = tuple(value)

            elif isinstance(value, list):
                for i, value_i in enumerate(value):
                    if value_i in ["None", "null"]:
                        value[i] = None
                    elif value_i in ["slice(None)", "slice(null)", "slice('None')"]:
                        value[i] = slice(None)
                    else:
                        value[i] = value_i
                queried_component[key] = tuple(value)
    return queried_components


def get_component_type(name, m):
    TYPES = ['parameters', 'variables', 'sets', 'constraints', 'objective']
    return next((c_type for c_type in TYPES if name in m["components"][c_type]), None)


def get_new_model_name(queried_model):
    prefix, number = queried_model.rsplit('_', 1)
    incremented_number = int(number) + 1
    new_model_name = f"{prefix}_{incremented_number}"
    return new_model_name


def syntax_guidance(queried_function: str,
                    queried_components: List[str],
                    queried_model: str,
                    models_dict):

    FUNCTIONS = ['feasibility_restoration', 'sensitivity_analysis', 'components_retrival', 'evaluate_modification',
                 'external_tools']
    assert queried_function in FUNCTIONS, f"Function {queried_function} is not recognized."
    if queried_function == 'external_tools':
        return "external_tools", "none"

    model_dict = models_dict[queried_model]
    function_syntax = "function to call: " + queried_function + "\n\n"  #
    queried_model_syntax = "queried_model: " + queried_model + "\n\n"  #

    def get_index_guidance(pattern):
        """
        provide index guidance in terms of an indexed pattern,
        supplementary is 'evaluate_modification' or None
        """

        if isinstance(pattern, tuple):
            mode = 'multiple'
            tuple_guidance = f"""
Return a tuple with dimensions to be {len(pattern)}.
You need to fill in the tuple with the specific indexes provided by the user, 
and the rest of the indexes that are not specified should be "__all__" in string.

- Must return a tuple with {len(pattern)} elements
- "__all__" is placeholder that represents all indexes in a dimension that user didn't specify
- "__all__" can be inserted into the tuple multiple times if there are multiple dimensions that user didn't specify
Example: If the dimension of an indexed component is 3, 
the first index is specified as 4, the second and third indexes are not specified by users,
then return the tuple (4, "__all__", "__all__")

- Must be careful with the order of indexes in the tuple by inspecting code
Example: If the dimension of an indexed component is 2,
the first dimension represents time, the second dimension represents location,
the user specifies the location to be "NY", the time is not specified by users,
then return the tuple ("__all__", "NY")
"""
            return tuple_guidance, mode

        elif isinstance(pattern, int) or isinstance(pattern, str):
            mode = 'single'
            primitive_guidance = f"""
When specific index provided, fill in the type of {type(pattern)}
If no specific index provided, return "__all__" in string

- "__all__" is placeholder that represents all indexes that user didn't specify
"""
            return primitive_guidance, mode

    def get_complex_guidance(flag):
        example = [{'component_name': 'dem', 'component_indexes': 'a'},
                   {'component_name': 'dem', 'component_indexes': 'c'}]
        cs = f"""
If the user provides multiple indexes that belong to the same dimension, 
add every specified index separately in the queried_components.
Example: help me change demand of a and c.

If the user provides indexes that are prohibited from being changed,
add every permitted index separately in the queried_components.
Example: help me change demand, but please note that the demand of b cannot be changed.

dem is a 1-dim parameter that means demand, and dem is indexed by a, b, c, then, 
queried_components: {example}"""
        if flag:
            return cs
        else:
            return ""

    def get_supplementary_guidance(fn):
        if fn == 'evaluate_modification':
            supplementary_guidance = f"""
When no specific modification extent provided, always return operation: "!" and delta: 0

Otherwise, choose one of the following operations: "+", "-", "*", "/", and fill in the delta value.
Demonstrations:
change it to 5: operation: "=", delta: 13;
increase it to 5: operation: "=", delta: 5;
increase it by 5: operation: "+", delta: 5;
increase it by 5%: operation: "*", delta: 1.05;
decrease it to 5: operation: "=", delta: 5;
decrease it by 5: operation: "-", delta: 5;
decrease it by 5%: operation: "*", delta: 0.95;
discount it by 5%: operation: "*", delta: 0.95;
have 5 more units: operation: "+", delta: 5;
have 5 less units: operation: "-", delta: 5;

Make sure the delta value is consistent with the positivity/negativity of the parameters being modified."""
            return supplementary_guidance
        else:
            return ""

    need_complex_syntax = False
    ref = []
    syntax_mode = []
    for component_name in queried_components:
        component_type = get_component_type(component_name, model_dict)
        if model_dict["components"][component_type][component_name]["is_indexed"]:
            # component is indexed
            need_complex_syntax = True
            component_index_set = model_dict["components"][component_type][component_name]["index_set"]
            component_pattern = random.choice(list(component_index_set))
            situation, mode_i = get_index_guidance(component_pattern)
        else:
            # component is not indexed
            situation = "always return null"
            mode_i = 'none'
        ref.append({"component_name": component_name, "component_indexes": situation})
        syntax_mode.append(mode_i)
    queried_component_syntax = f"queried_components: {ref} \n\n"  #
    complex_syntax = get_complex_guidance(need_complex_syntax)  #
    supplementary = get_supplementary_guidance(queried_function)  #
    syntax_output = function_syntax + queried_model_syntax + queried_component_syntax + complex_syntax + supplementary

    syntax_mode = set(syntax_mode)
    if len(syntax_mode) > 1:
        syntax_mode = "all"
    else:
        syntax_mode = next(iter(syntax_mode))
    return syntax_output, syntax_mode


def feasibility_restoration(queried_components: List[Dict], queried_model: str, models_dict):
    queried_model_dict = models_dict[queried_model]

    if queried_model_dict['model status'] not in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        return "The model is not infeasible. No need to restore feasibility. Please confirm with the user."

    model = queried_model_dict['model class'].clone()
    # define slack parameters
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        print(f'param_indexes: {param_indexes}')

        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == 'parameters':
            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

            if queried_model_dict['components']['parameters'][param_name]['is_RHS']:
                # First, add slacks to all indexes and fix all of them as 0
                exec(
                    "model.slack_pos_" + param_name + "=pe.Var(model." + param_name + ".index_set(), within=pe.NonNegativeReals)")
                exec(
                    "model.slack_neg_" + param_name + "=pe.Var(model." + param_name + ".index_set(), within=pe.NonNegativeReals)")
                model_slack_pos_param = eval("model.slack_pos_" + param_name)
                model_slack_neg_param = eval("model.slack_neg_" + param_name)
                model_slack_pos_param.fix(0)
                model_slack_neg_param.fix(0)
                # Second, unfix the slacks for the specific indexes provided in the query
                model_slack_pos_param[param_indexes].unfix()
                model_slack_neg_param[param_indexes].unfix()

            else:
                feedback = f"""
Feedback from internal tools:
Warning. {param_name} is not a RHS parameter in the model.
This parameter is LHS parameter. 
Changing LHS parameter for feasibility restoration without specifying modification extent 
can extend solving time and risk terminating the optimization process prematurely before finding an optimal solution. 
Users need to try other parameters for feasibility restoration, 
or specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification, if they are particularly interested in this parameter."""
                return feedback
        else:
            wrong_component_type = component_type
            feedback = f"""
Feedback from internal tools:
Error. {param_name} is not a parameter in the model but a {wrong_component_type}.
Users need to provide a valid parameter for feasibility restoration."""
            return feedback

    # generate replacements
    iis_param = []
    replacements_list = []
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        for idx in eval("model." + param_name + ".index_set()"):
            model_param = eval("model." + param_name)
            iis_param.append((param_name, idx))  ###
            expr_param = model_param[idx]
            slack_var_pos = eval("model.slack_pos_" + param_name)[idx]
            slack_var_neg = eval("model.slack_neg_" + param_name)[idx]
            replacements = {id(expr_param): expr_param + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)
    # replace constraints
    original_consts = []
    for consts_name, consts in model.component_map(pe.Constraint).items():
        original_consts.append(consts)
    model.slack_iis_constraints = pe.ConstraintList()
    for consts in original_consts:
        for const_idx in consts.index_set():
            const = consts[const_idx]
            new_expr = clone_expression(const.expr)
            for replacements in replacements_list:
                new_expr = replace_expressions(new_expr, replacements)
            model.slack_iis_constraints.add(new_expr)
            const.deactivate()
    # replace objective
    objectives = model.component_objects(pe.Objective, active=True)
    for obj in objectives:
        obj.deactivate()
    # minimize the 1-norm of the slacks that are added
    new_obj = 0
    for p, idx in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p)[idx]
        slack_var_neg = eval("model.slack_neg_" + p)[idx]
        new_obj += slack_var_pos + slack_var_neg
    model.slack_obj = pe.Objective(expr=new_obj, sense=pe.minimize)
    # solve the model
    opt = SolverFactory('gurobi')
    opt.options['nonConvex'] = 2
    opt.options['TimeLimit'] = 300  # 5min time limit
    results = opt.solve(model, tee=True)
    # construct technical feedback
    termination_condition = results.solver.termination_condition
    #new_model_dict = pyomo2json(model, termination_condition=termination_condition)
    new_model_dict = copy.deepcopy(queried_model_dict)
    feedback = f"The following changes are made to {queried_model}: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    new_model_name = get_new_model_name(queried_model)

    if termination_condition == TerminationCondition.maxTimeLimit:
        for p, idx in iis_param:
            feedback = feedback + f"attempt to change {p} at {idx}; \n"
            description = description + f"attempt to change {p}{idx}; \n"
        feedback = feedback + f"\n\nThe model cannot be solved due to time limit."
        description = description[:7] + f", which cannot be solved due to time limit." + description[7:]
        new_model_dict["model description"] = description
        new_model_dict["model status"] = TerminationCondition.maxTimeLimit
        models_dict[new_model_name] = new_model_dict
    elif termination_condition == TerminationCondition.optimal:
        for p, idx in iis_param:
            slack_var_pos = eval("model.slack_pos_" + p)[idx].value
            slack_var_neg = eval("model.slack_neg_" + p)[idx].value
            idx = "" if idx is None else f" at {idx}"
            if slack_var_pos > 1e-5:
                feedback = feedback + f"change {p}{idx} by +{slack_var_pos} unit; \n"
                description = description + f"change {p}{idx} by +{slack_var_pos} unit; \n"
            elif slack_var_neg > 1e-5:
                feedback = feedback + f"change {p}{idx} by -{slack_var_neg} unit; \n"
                description = description + f"change {p}{idx} by -{slack_var_neg} unit; \n"
        feedback = feedback + f"\n\nThe model now becomes feasible. "
        feedback = feedback + f"\n\nHelp the user analyze why the feasibility can be restored by these changes. Let user know this new model will be referred to as {new_model_name}."
        description = description[:7] + f", which becomes feasible" + description[7:]
        new_model_dict["model description"] = description
        new_model_dict["model status"] = TerminationCondition.optimal
        models_dict[new_model_name] = new_model_dict
    else:
        feedback = f"The model remains infeasible after only changing the following: \n"
        for p, idx in iis_param:
            idx = "" if idx is None else f" at {idx}"
            feedback = feedback + f"{p}{idx}; \n"
        description = feedback
        models_dict[queried_model]["model description"] = description
        feedback = feedback + f"\n\nThis is determined by the nature of the model, rather than an error of internal tools. Help the user analyze why the feasibility is not restored."

    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def sensitivity_analysis(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()

    if queried_model_dict['model status'] in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
        feedback = "Error: The model is infeasible. Sensitivity analysis cannot be performed on an infeasible model."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback
    if queried_model_dict['model type'] != 'LP':
        feedback = "Error: The model is not a linear programming model. Internal tools do not support sensitivity analysis on other types of models."
        feedback = "Feedback from internal tools: \n" + feedback
        return feedback

    def locate_param(param_name, idx, model=model):
        in_consts = []
        param_name_idx = str(eval("model." + param_name)[idx])
        for const_name in queried_model_dict['components']["parameters"][param_name]["cons_in"]:
            model_const = eval("model." + const_name)
            for con_idx in model_const.index_set():
                con_i = model_const[con_idx]
                expr_params = identify_mutable_parameters(con_i.expr)
                for expr_param in expr_params:
                    if expr_param.name == param_name_idx:
                        coef_body = - differentiate(con_i.body, wrt=expr_param, mode='reverse_symbolic')
                        coef_lower = differentiate(con_i.lower, wrt=expr_param, mode='reverse_symbolic')
                        coef_upper = differentiate(con_i.upper, wrt=expr_param, mode='reverse_symbolic')
                        coef = coef_body + coef_lower + coef_upper
                        in_consts.append({"const_name": const_name,
                                          "const_indexes": con_idx,
                                          "coefficient": coef})
                        break
        return in_consts

    param_const_pairs = []
    for component in queried_components:
        param_name = component['component_name']
        param_indexes = component['component_indexes']
        print(f'param_indexes: {param_indexes}')
        component_type = get_component_type(param_name, queried_model_dict)
        if component_type == 'parameters':

            if isinstance(param_indexes, tuple):
                eval_param = eval(f"model.{param_name}")
                if len(eval_param[param_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

            if queried_model_dict['components']['parameters'][param_name]['is_RHS']:
                if isinstance(param_indexes, tuple):
                    if slice(None) in param_indexes:
                        for model_param_i in eval("model." + param_name)[param_indexes]:
                            model_param_i_indexes = model_param_i.index()
                            param_const_pair = {"param_name": param_name,
                                                "param_indexes": model_param_i_indexes,
                                                "consts": locate_param(param_name, model_param_i_indexes)}
                            param_const_pairs.append(param_const_pair)
                    else:
                        param_const_pair = {"param_name": param_name,
                                            "param_indexes": param_indexes,
                                            "consts": locate_param(param_name, param_indexes)}
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, slice):
                    for model_param_i in eval("model." + param_name)[param_indexes]:
                        model_param_i_indexes = model_param_i.index()
                        param_const_pair = {"param_name": param_name,
                                            "param_indexes": model_param_i_indexes,
                                            "consts": locate_param(param_name, model_param_i_indexes)}
                        param_const_pairs.append(param_const_pair)
                elif isinstance(param_indexes, int) or isinstance(param_indexes, str):
                    param_const_pair = {"param_name": param_name,
                                        "param_indexes": param_indexes,
                                        "consts": locate_param(param_name, param_indexes)}
                    param_const_pairs.append(param_const_pair)
                elif param_indexes == None:
                    param_const_pair = {"param_name": param_name,
                                        "param_indexes": param_indexes,
                                        "consts": locate_param(param_name, param_indexes)}
                    param_const_pairs.append(param_const_pair)

            else:
                feedback = f"Error: {param_name} is not a RHS parameter in the model. "
                feedback += """
Please confirm with the user and ask them to provide a valid RHS parameter for sensitivity analysis, 
or if they are particularly interested in these parameters, they must specify a modification extent (e.g., a 5% increase) to directly assess the impact of this modification."""
                feedback = "Feedback from internal tools: \n" + feedback
                return feedback

        else:
            wrong_component_type = component_type
            feedback = f"Error: {param_name} is not a parameter in the model but a {wrong_component_type}. "
            feedback += """Please confirm with the user and ask them to provide a valid parameter for sensitivity analysis."""
            feedback = "Feedback from internal tools: \n" + feedback
            return feedback

    # duals = []
    print(f' Does this model have model.dual? {model.find_component("dual") is None}')
    if model.find_component('dual') is None:
        model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
        opt = SolverFactory("gurobi")
        results = opt.solve(model, tee=True)
        termination_condition = results.solver.termination_condition
        # update the models_dict
        models_dict[queried_model]["model class"] = model

    for param_const_pair in param_const_pairs:
        for const in param_const_pair['consts']:
            const_name = const['const_name']
            const_indexes = const['const_indexes']
            const_coef = const['coefficient']
            model_const = eval("model." + const_name)
            model_const_i = model_const[const_indexes]
            const['dual_value'] = const_coef * model.dual[model_const_i]
            #print(f'const_name: {const_name}, const_indexes: {const_indexes}, const_coef: {const_coef}, dual_value: {const["dual_value"]}')
            #break

    # construct feedback
    feedback = "The sensitivity analysis results are as follows: \n"
    for param_const_pair in param_const_pairs:
        param_name = param_const_pair['param_name']
        param_indexes = param_const_pair['param_indexes']
        param_indexes = f" at {param_indexes}" if param_indexes != None else ""
        feedback = feedback + f"when a small positive perturbation is made to {param_name}{param_indexes}, "
        total_value = 0
        for const in param_const_pair['consts']:
            #print(f'retrieving dual value of {const["const_name"]} at {const["const_indexes"]}')
            dual_value = const['dual_value']
            #print(f'the dual value of {const["const_name"]} at {const["const_indexes"]} is {dual_value}')
            total_value += dual_value

        if total_value > 1e-5:
            feedback = feedback + f"the optimal objective value will change by {total_value} unit\n"
        elif total_value < -1e-5:
            feedback = feedback + f"the optimal objective value will change by {total_value} unit\n"
        else:
            feedback = feedback + f"the optimal objective value will not change \n"

    feedback += "Please explain these results to the user. \n"
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def components_retrival(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()
    feedback = f"In the {queried_model}, "
    for component in queried_components:
        component_name = component['component_name']
        component_indexes = component['component_indexes']
        print(f'component_indexes: {component_indexes}')
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            # supposed to retrieve multiple components
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    feedback = feedback + f"{component_name} at {str(model_component_i_indexes)} is "
                    component_retrieval = ""
                    if component_name in queried_model_dict["components"]["parameters"]:
                        component_retrieval = str(model_component_i.value)
                    elif component_name in queried_model_dict["components"]["variables"]:
                        component_retrieval = str(model_component_i.value)
                    elif component_name in queried_model_dict["components"]["sets"]:
                        component_retrieval = str(model_component_i.data())
                    elif component_name in queried_model_dict["components"]["constraints"]:
                        component_retrieval = str(model_component_i.expr)
                    elif component_name in queried_model_dict["components"]["objective"]:
                        component_retrieval = str(model_component_i())
                    feedback = feedback + f"{component_retrieval}.\n"

            else:
                # supposed to retrieve one component
                feedback = feedback + f"{component_name} at {str(component_indexes)} is "
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component[component_indexes].value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component[component_indexes].data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component[component_indexes].expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component[component_indexes]())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, slice):
            # supposed to retrieve multiple components
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                feedback = feedback + f"{component_name} at {str(model_component_i_indexes)} is "
                component_retrieval = ""
                if component_name in queried_model_dict["components"]["parameters"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["variables"]:
                    component_retrieval = str(model_component_i.value)
                elif component_name in queried_model_dict["components"]["sets"]:
                    component_retrieval = str(model_component_i.data())
                elif component_name in queried_model_dict["components"]["constraints"]:
                    component_retrieval = str(model_component_i.expr)
                elif component_name in queried_model_dict["components"]["objective"]:
                    component_retrieval = str(model_component_i())
                feedback = feedback + f"{component_retrieval}.\n"

        elif isinstance(component_indexes, int) or isinstance(component_indexes, str):
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} at {str(component_indexes)} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component[component_indexes].value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component[component_indexes].data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component[component_indexes].expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component[component_indexes]())
            feedback = feedback + f"{component_retrieval}.\n"

        elif component_indexes == None:
            # supposed to retrieve one component
            feedback = feedback + f"{component_name} is "
            component_retrieval = ""
            if component_name in queried_model_dict["components"]["parameters"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["variables"]:
                component_retrieval = str(model_component.value)
            elif component_name in queried_model_dict["components"]["sets"]:
                component_retrieval = str(model_component.data())
            elif component_name in queried_model_dict["components"]["constraints"]:
                component_retrieval = str(model_component.expr)
            elif component_name in queried_model_dict["components"]["objective"]:
                component_retrieval = str(model_component())
            feedback = feedback + f"{component_retrieval}.\n"

    feedback += "Please describe the information using their physical meanings to the user. \n"
    feedback = "Feedback from internal tools: \n" + feedback
    return feedback


def evaluate_modification(queried_components: List[Dict], queried_model, models_dict):
    queried_model_dict = models_dict[queried_model]
    model = queried_model_dict['model class'].clone()
    for obj_name, obj in model.component_map(pe.Objective).items():
        original_obj_value = queried_model_dict['components']['objective'][obj_name]['optimal_value']
    feedback = f"In the {queried_model}, the following modifications are made: \n"
    description = f"a model with the following changes to {queried_model}: \n"
    for component in queried_components:
        component_name = component['component_name']
        component_indexes = component['component_indexes']
        component_operation = component['operation']

        if component_operation == "!":
            return ("Error: The evaluate_modification function requires a specific modification extent. "
                    "Debug suggestion: distribute this task to operator again and ask them to use sensitivity_analysis function instead.")

        component_delta = str(component['delta'])
        print(f'component_indexes: {component_indexes}')
        print(f'component_operation: {component_operation}')
        print(f'component_delta: {component_delta}')
        model_component = eval("model." + component_name)

        if isinstance(component_indexes, tuple):
            if slice(None) in component_indexes:

                if len(model_component[component_indexes].index()) <= 0:
                    raise IndexError(f"Error: Indexes are not valid. This usually happens when the order of indexes in the tuple is incorrect.")

                for model_component_i in model_component[component_indexes]:
                    model_component_i_indexes = model_component_i.index()
                    if component_name in queried_model_dict["components"]["parameters"]:
                        value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                        value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                        model_component[model_component_i_indexes].set_value(value_after_modification)
                        changed_or_fixed = " is changed to "
                    elif component_name in queried_model_dict["components"]["variables"]:
                        value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                        value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                        model_component[model_component_i_indexes].fix(value_after_modification)
                        changed_or_fixed = " is fixed to "
                    feedback += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                    description += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

            else:
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[component_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[component_indexes].set_value(value_after_modification)
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[component_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[component_indexes].fix(value_after_modification)
                    changed_or_fixed = " is fixed to "
                feedback += f"{component_name} at {str(component_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                description += f"{component_name} at {str(component_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

        elif isinstance(component_indexes, slice):
            for model_component_i in model_component[component_indexes]:
                model_component_i_indexes = model_component_i.index()
                if component_name in queried_model_dict["components"]["parameters"]:
                    value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[model_component_i_indexes].set_value(value_after_modification)
                    changed_or_fixed = " is changed to "
                elif component_name in queried_model_dict["components"]["variables"]:
                    value_for_modification = eval("model." + component_name)[model_component_i_indexes].value
                    value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                    model_component[model_component_i_indexes].fix(value_after_modification)
                    changed_or_fixed = " is fixed to "
                feedback += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"
                description += f"{component_name} at {str(model_component_i_indexes)}" + changed_or_fixed + str(value_after_modification) + ".\n"

        elif isinstance(component_indexes, int) or isinstance(component_indexes, str) or component_indexes == None:
            if component_name in queried_model_dict["components"]["parameters"]:
                value_for_modification = eval("model." + component_name)[component_indexes].value
                value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                model_component[component_indexes].set_value(value_after_modification)
                changed_or_fixed = " is changed to "
            elif component_name in queried_model_dict["components"]["variables"]:
                value_for_modification = eval("model." + component_name)[component_indexes].value
                value_after_modification = eval(component_delta) if component_operation == "=" else eval(str(value_for_modification) + component_operation + component_delta)
                model_component[component_indexes].fix(value_after_modification)
                changed_or_fixed = " is fixed to "
            idx = "" if component_indexes == None else f" at {str(component_indexes)}"
            feedback += f"{component_name}{idx}" + changed_or_fixed + str(value_after_modification) + ".\n"
            description += f"{component_name}{idx}" + changed_or_fixed + str(value_after_modification) + ".\n"

        # resolve the model
        opt = SolverFactory('gurobi')
        opt.options['nonConvex'] = 2
        opt.options['TimeLimit'] = 300  # 5min time limit
        results = opt.solve(model, tee=True)
        # construct technical feedback
        termination_condition = results.solver.termination_condition
        new_model_dict = pyomo2json(model, termination_condition=termination_condition)

        new_model_name = get_new_model_name(queried_model)

        if termination_condition == TerminationCondition.maxTimeLimit:
            feedback += f"\n\nThe model is not solved due to time limit, and the new model will be referred to as {new_model_name}."
            feedback += f"The best objective value found so far is {results.Problem[0]['Upper bound']}.\n"
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications."
            description = description[:7] + f", which is not solved due to time limit" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict
        elif termination_condition == TerminationCondition.optimal:
            feedback += f"\n\nThe model now is feasible, and the new model will be referred to as {new_model_name}."
            feedback += f"The optimal objective value found is {results.Problem[0]['Lower bound']}.\n"
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications. "
            description = description[:7] + f", which is feasible" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict
        else:
            feedback += f"\n\nThe model now is infeasible, and the new model will be referred to as {new_model_name}."
            feedback = feedback + f"\n\nHelp the user analyze the influence of these modifications."
            description = description[:7] + f", which is infeasible" + description[7:]
            new_model_dict["model description"] = description
            models_dict[new_model_name] = new_model_dict

        reminder = f"Reminder: the status of old model, {queried_model}, was "
        if queried_model_dict['model status'] == TerminationCondition.maxTimeLimit:
            reminder += f"is not solved due to time limit with best objective value found as {original_obj_value}."
        elif queried_model_dict['model status'] == TerminationCondition.optimal:
            reminder += f"solved with optimal objective value found as {original_obj_value}."
        elif queried_model_dict['model status'] in [TerminationCondition.infeasible, TerminationCondition.infeasibleOrUnbounded]:
            reminder += f"infeasible."

        feedback += reminder
        feedback = "Feedback from internal tools: \n" + feedback

    return feedback


