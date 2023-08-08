# Gurobi
import typing
import os
import sys
import importlib
import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
from pyomo.core.expr.visitor import identify_mutable_parameters, replace_expressions, clone_expression
# GPT
import openai
import tiktoken
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']



def get_completion_standalone(prompt, gpt_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def load_model(pyomo_file):
    original_dir = os.getcwd()
    directory_path = os.path.dirname(pyomo_file)
    filename_wo_extension = os.path.splitext(os.path.basename(pyomo_file))[0]
    sys.path.append(directory_path)

    module = importlib.import_module(filename_wo_extension)
    model = module.model  # access the pyomo model (remember to name your model as 'model' eg. model = RTN)
    print(f'Model {pyomo_file} loaded')
    ilp_name = write_iis(model, filename_wo_extension + ".ilp", solver="gurobi")
    ilp_path = os.path.abspath(filename_wo_extension + ".ilp")
    return model, ilp_path


def extract_component(model, pyomo_file):
    const_list = []
    param_list = []
    var_list = []
    for const in model.component_objects(pe.Constraint):
        const_list.append(str(const))
    for param in model.component_objects(pe.Param):
        param_list.append(str(param))
    for var in model.component_objects(pe.Var):
        var_list.append(str(var))
    with open(pyomo_file, 'r') as file:
        PYOMO_CODE = file.read()
    file.close()
    return const_list, param_list, var_list, PYOMO_CODE


def extract_summary(var_list, param_list, const_list, PYOMO_CODE, gpt_model):
    prompt = f"""Here is an optimization model written in Pyomo, which is delimited by triple backticks. 
    Your task is to 
    (1): use plain English to describe the objective funtion of this model. \n\n
    (2): We identified that it includes variables: {var_list}, please output a table and each row is in a style of 
    - <Name of the variable> | <physical meaning of the variable>. \n\n
    (3) We identified that it includes parameters: {param_list}, please output a table and each row is in a style of
    - <Name of the parameter> | <physical meaning of the parameter>. \n\n
    (4) We identified that it includes constraints: {const_list} please output a table and each row is in a style of 
    - <Name of the constraint> | <physical meaning of the constraint>. 
    You need to cover the physical meaning of each term in the constraint expression and give a detailed explanation. \n\n
    (5) Identify the parameters that have product with variables in constraints. 
    For example, suppose "a" is a parameter and "b" is a variable, if a*b is in the constraint, then a is the parameter that 
    has product with variables in constraints.
    
    Pyomo Model Code: ```{PYOMO_CODE}```"""
    summary_response = get_completion_standalone(prompt, gpt_model)
    return summary_response


def add_eg(summary, gpt_model):
    prompt = f"""I will give you a decription of an optimization model with parameters, variables, constraints and objective. 
    First introduce this model to the user using the following four steps. However, DO NOT write bullets 1-4\
        make it more sounds like coherent paragraph:
                                      1. Try to guess what the problem is about and who is using is model for deciding 
                                      what problem.\
                                        give a high level summary, e.g. "An oil\
                                        producer has developed an optimization to determine where to drill the wells".
                                        "A travel planner is determining the best way to visit n cities".explain what data is available to the decision maker\
                                            make decisions in plain English. Avoid using bullet points!
                                      Try to make it smooth like a story. 
                                        for example you could say "You are given a number of cities and the distance between any two
                                            cities." for a TSP problem. You can say "You are given n item with different values and
                                                weights to be filled in a knapsack who capacity is known"
                                      2. explain what decisions are to be made in plain English. Avoid using bullet points!
                                      Try to make it smooth like a story. \
                                        for example, you could say "you would like to decide the sequence to visit all the n cities." for the TSP 
                                        problem.
                                        you could say "you would like to decide the items to be filled in the knapsack" for the knapsack problem. 
                                    3. explain what constraints the decisions have to satisfy in plain English
                                        for example you could say "the weights of all the items in the knapsack have to be less than or 
                                        equal to the knapsack capacity"
                                    4. explain the objective function in plain English
                                        you could say "given these decisions, we would like to find the shortest path" for the TSP problem.
                                        "given these decisions and constraints, we would like to find the items to be filled in the knapsack that 
                                        have the total largest values"               
    Model Description: ```{summary}```"""
    summary_response = get_completion_standalone(prompt, gpt_model)
    return summary_response


def read_iis(ilp_file, model):
    with open(ilp_file, 'r') as file:
        ilp_string = file.read()
    file.close()
    ilp_lines = ilp_string.split("\n")
    constr_names = []
    for iis_line in ilp_lines:
        if ":" in iis_line:
            constr_name = iis_line.split(":")[0].split("(")[0]
            if constr_name not in constr_names:
                constr_names.append(constr_name)

    iis_dict = {}
    param_names = []
    for const_name in constr_names:
        iis_dict.update({const_name: []})
        consts = eval('model.' + const_name)
        for const_idx in consts:
            const = consts[const_idx]
            expr_parameters = identify_mutable_parameters(const.expr)
            for p in expr_parameters:
                p_name = p.name.split("[")[0]
                param_names.append(p_name)

                if p_name not in iis_dict[const_name]:
                    iis_dict[const_name].append(p_name)

    param_names = list(set(param_names))
    return constr_names, param_names, iis_dict


def param_in_const(iis_dict):
    text_list = []
    for key, values in iis_dict.items():
        if values:
            if len(values) == 1:
                text_list.append(f"{key} constraint only contains {values[0]} parameter")
            else:
                objects = ', '.join(values[:-1]) + f" and {values[-1]}"
                text_list.append(f"{key} constraint contains {objects} parameters")
        else:
            text_list.append(f"{key} constraint contains no parameter")

    final_text = ', '.join(text_list) + '.\n'
    return final_text


def infer_infeasibility(const_names, param_names, summary, gpt_model):
    prompt = f"""Optimization experts are troubleshooting an infeasible optimization model. 
    They found that {', '.join(const_names)} constraints are in the Irreducible infeasible set.
    and that  {', '.join(param_names)} are the parameters involved in the Irreducible infeasible set.
    To understand what the parameters and the constraints mean, Here's the  Model Summary \
        in a Markdown Table ```{summary}```\
    Now, given these information, your job is to do the following steps. Try to use plain
    english! DO NOT show "A-C", show the answers in three papagraphs:
    A. Tell the user something like "The following constraints are causing the model to be infeasible". 
    Then provide the list constraints ( {', '.join(const_names)}) and their physical meaning in an itemized list.
    You can refer to the Model Summary I gave you to get the meaning of the constraints. Avoid using any
    symbols of the constraints, use natural language. For example, answer to this step can be 
    "The following constraints are causing the model to be infeasible:
    C1. The mass balance constraints that specify the level of the storage vessel at a given time point\
        is equal to the 
    C2. The storage level should be less than its maximum capacity.
    "
    B. Tell the user all the parameters, {', '.join(param_names)} \
        involved in the constraints and their physical meaning in an itemized list. 
        You can refer to the Model Summary I gave you to get the meaning of the parameters.\
             Avoid using any symbols of the parameters.  For example, answer to this step can be 
             "The following input data are involved in the constraints:
             P1. The molecular weight of a molecule A
             P2. the demand of customers 
             P3. the storage capacity"
    C. Tell the user they might want to change some data involved in {', '.join(param_names)} to make the model feasible, 
       but skip the parameters that have product with another variable in the constraints.\
       For this step, you should provide the user with an recommendation. To decide which parameters to recommend
        there is a rule of thumb you should consider:\
        In general, recommend parameters that can be easily change in the physical world. 
            For example, if I have the molecular weight of a molecule and the demand of customers in the parameters, 
            you should only recommend the demand of the customers to be changed because the molecular weight is a 
            physical property that cannot be changed.\
            
            DO NOT mention that "we don't recommend changing parameters a, b, c,.. etc because they have product with variables." \
            Use an explanation corresponding to the physical meaning of the parameters that makes them a good candidate. \
            An example answer would be
            "Based on my interpretation of your data, you might want to change the demand of the customers and expand 
            your storage capacity to make the model feasible."
            """
    explanation = get_completion_standalone(prompt, gpt_model)
    return explanation


def add_slack(param_names, model):
    """
    use <param_names> to add slack for ALL indices of the parameters
    """
    is_slack_added = {}  # indicator: is slack added to constraints?
    # define slack parameters
    for p in param_names:
        if eval("model." + p + ".is_indexed()"):
            is_slack_added[p] = {}
            for index in eval("model." + p + ".index_set()"):
                is_slack_added[p][index] = False
            exec("model.slack_pos_" + p + "=pe.Var(model." + p + ".index_set(), within=pe.NonNegativeReals)")
            exec("model.slack_neg_" + p + "=pe.Var(model." + p + ".index_set(), within=pe.NonNegativeReals)")

        else:
            is_slack_added[p] = False
            exec("model.slack_pos_" + p + "=pe.Var(within=pe.NonNegativeReals)")
            exec("model.slack_neg_" + p + "=pe.Var(within=pe.NonNegativeReals)")

    return is_slack_added


def generate_replacements(param_names, model):
    iis_param = []
    replacements_list = []
    for p_name in param_names:
        for idx in eval("model." + p_name + ".index_set()"):
            p_index = str(idx).replace("(", "[").replace(")", "]")

            if "[" and "]" in p_index:  # this happens when p is a parameter that has more than one index [idx1, idx2, ]
                p_name_index = p_name + p_index
            elif p_index == 'None':  # this happens when p is a parameter that doesn't have index
                p_name_index = p_name
            else:  # this happens when p is a parameter that has only one index [idx1]
                p_index = str([idx])
                p_name_index = p_name + p_index

            iis_param.append(p_name_index)
            expr_p = eval("model." + p_name_index)
            slack_var_pos = eval("model.slack_pos_" + p_name_index)
            slack_var_neg = eval("model.slack_neg_" + p_name_index)

            replacements = {id(expr_p): expr_p + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)
    return iis_param, replacements_list


def replace_const(replacements_list, model):
    const_list = []
    for const in model.component_objects(pe.Constraint):
        const_list.append(str(const))
    # const_list is a list containing all const_names in the model
    model.slack_iis = pe.ConstraintList()
    # replace each param in each const
    for const_name in const_list:
        consts = eval('model.' + const_name)
        for const_idx in consts:
            const = consts[const_idx]
            new_expr = clone_expression(const.expr)
            for replacements in replacements_list:
                new_expr = replace_expressions(new_expr, replacements)
            model.slack_iis.add(new_expr)
            const.deactivate()


def replace_obj(iis_param, model):
    # deactivate all the existing objectives
    objectives = model.component_objects(pe.Objective, active=True)
    for obj in objectives:
        obj.deactivate()

    # minimize the 1-norm of the slacks that are added
    new_obj = 0
    for p in iis_param:
        # other slack vars outside iis_param have been fixed to 0
        slack_var_pos = eval("model.slack_pos_" + p)
        slack_var_neg = eval("model.slack_neg_" + p)
        new_obj += slack_var_pos + slack_var_neg
    model.slack_obj = pe.Objective(expr=new_obj, sense=pe.minimize)


def resolve(model):
    opt = SolverFactory('gurobi')
    opt.options['nonConvex'] = 2
    opt.options['TimeLimit'] = 300  # 5min time limit
    results = opt.solve(model, tee=True)
    termination_condition = results.solver.termination_condition
    if termination_condition == "maxTimeLimit" and 'Upper bound' in results.Problem[0]:
        termination_condition = 'optimal'    
    return str(termination_condition)


def generate_slack_text(iis_param, model):
    text = "Model becomes feasible after the following change: "
    for p in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p + ".value")
        slack_var_neg = eval("model.slack_neg_" + p + ".value")

        if slack_var_pos > 1e-5:
            text = text + f"increase {p} by {slack_var_pos} unit; "
        elif slack_var_neg > 1e-5:
            text = text + f"decrease {p} by {slack_var_neg} unit; "
    return text


def solve_the_model(param_names: list[str], param_names_aval, model) -> str:
    if all(param_name in param_names_aval for param_name in param_names):
        model_copy = model.clone()
        is_slack_added = add_slack(param_names, model_copy)
        # all_const_in_model = find_const_in_model(model_copy)
        iis_param, replacements_list = generate_replacements(param_names, model_copy)
        replace_const(replacements_list, model_copy)
        replace_obj(iis_param, model_copy)
        termination_condition = resolve(model_copy)
        if termination_condition == 'optimal':
            out_text = generate_slack_text(iis_param, model_copy)
            flag = 'feasible'
        else:
            out_text = f"Changing {param_names} is not sufficient to make this model feasible, \n" \
                       f"Try other potential mutable parameters instead. \n"
            flag = 'infeasible'
    else:
        out_text = f"I can't help you change {param_names} " \
                   f"because they aren't valid mutable parameters in this model. \n"
        flag = 'invalid'
    return out_text, flag



def get_completion_from_messages_withfn(messages, gpt_model):
    functions = [
        {
            "name": "solve_the_model",
            "description": "Given the parameters to be changed, re-solve the model and report the extent of the changes",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A parameter name"
                        },
                        "description": "List of parameter names to be changed in order to re-solve the model"
                    }
                },
                "required": ["param_names"]
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        functions=functions,
        function_call='auto'
    )
    return response


def gpt_function_call(ai_response, param_names_aval, model):
    fn_call = ai_response["choices"][0]["message"]["function_call"]
    fn_name = fn_call["name"]
    arguments = fn_call["arguments"]
    if fn_name == "solve_the_model":
        param_names = eval(arguments).get("param_names")
        return solve_the_model(param_names, param_names_aval, model), fn_name
    else:
        return
