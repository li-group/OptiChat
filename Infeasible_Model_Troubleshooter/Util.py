# Gurobi
import typing
import json
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
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, SequentialChain, ConversationChain

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model="gpt-3.5-turbo-16k"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def get_completion_from_messages(messages, model="gpt-3.5-turbo-16k"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message["content"]


llm = ChatOpenAI(temperature=0.0)


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


def build_QARetriever(pyomo_file):
    loader = TextLoader(file_path=pyomo_file)
    PYOMO_CODE = loader.load()
    embeddings_model = OpenAIEmbeddings()
    docsearch_PYOMO_CODE = Chroma.from_documents(PYOMO_CODE, embeddings_model)
    qa_PYOMO_CODE = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                retriever=docsearch_PYOMO_CODE.as_retriever())
    return qa_PYOMO_CODE


def ask_QARetriever(query_object, qa_retriever):
    if query_object == 'Constraint':
        query = f"what are the {query_object} involved in this model, as well as their physical meaning? \n" \
                f"Output a table and each row is in a style of " \
                f"- <Name of the {query_object}> | <physical meaning of the {query_object}> \n" \
                f"You need to cover the physical meaning of each term in the constraint expression and connect them."
    else:
        query = f"what are the {query_object} involved in this model, as well as their physical meaning? \
            Output a table and each row is in a style of - <Name of the {query_object}> | <physical meaning of the {query_object}>"
    summary = qa_retriever({'query': query})["result"]
    return summary


def read_iis(ilp_file, model):
    command = "grep : " + ilp_file + " | awk -F\":\" '{print $1}'"
    iis_const = os.popen(command).readlines()
    iis_const = [const.strip().replace("(", "[").replace(")", "]") for const in iis_const]
    const_names = [const.split("[")[0] for const in iis_const]
    # keep the unique constaint types
    const_names = list(set(const_names))

    iis_dict = {}
    param_names = []
    for const_name in const_names:
        consts = eval('model.' + const_name)
        for const_idx in consts:
            const = consts[const_idx]
            expr_parameters = identify_mutable_parameters(const.expr)
            for p in expr_parameters:
                p_name = p.name.split("[")[0]
                param_names.append(p_name)

                if p_name in iis_dict.keys():
                    if const_name not in iis_dict[p_name]:
                        iis_dict[p_name].append(const_name)
                else:
                    iis_dict[p_name] = [const_name]

    param_names = list(set(param_names))
    return const_names, param_names, iis_dict


def infer_infeasibility(const_names, summary_const, summary_param, summary_var):
    prompt = f"""Optimization experts are troubleshooting an infeasible optimization model.
    They found that {', '.join(const_names)} constraints are contradictory and lead to infeasibility. 

    Your task is to identify the most probable conflicts among the constraints based on the model summary delimited by
    triple backticks. \

    Summary: ```{summary_const}\n\n{summary_param}\n\n{summary_var}```"""

    # prompt = f"""Optimization experts are troubleshooting an infeasible optimization model.
    # They found that {', '.join(const_names)} constraints are contradictory and lead to infeasibility. 

    # Your task is to identify the most probable conflicts among the constraints based on the model summary delimited by
    # triple backticks. \

    # Summary: ```{summary_const}```"""

    explanation = get_completion(prompt)
    return explanation


def simple_chatbot(user_text, messages):
    messages.append({'role': 'user', 'content': f"{user_text}"})
    response = get_completion_from_messages(messages)
    messages.append({'role': 'assistant', 'content': f"{response}"})
    return response


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
    print("generating replacements...", param_names)
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
    results = opt.solve(model, tee=True)
    return str(results.solver.termination_condition)


def generate_slack_text(iis_param, model):
    text = "The model becomes feasible after the following change: "
    print(iis_param)
    for p in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p + ".value")
        slack_var_neg = eval("model.slack_neg_" + p + ".value")

        if slack_var_pos > 1e-5:
            text = text + f"increase {p} by {slack_var_pos} unit; "
        elif slack_var_neg < -1e-5:
            text = text + f"decrease {p} by {slack_var_neg} unit; "
    return text


def explain_slack(summary, infeasibility_report, description):
    # todo not useful now
    prompt = f""""The model information is summarized below: ```{summary}```. \n\n
    {infeasibility_report}. \n
    In order to enable feasibility, optimization experts change the values of parameters in this model. 
    {description}. 

    Your task is to explain to inexperienced undergraduates the step-by-step process by which the changes make this model feasible.
    """
    explanation = get_completion(prompt)
    return explanation


def solve_the_model(param_names: list[str], model) -> str:
    print("solving the model...", param_names)
    model_copy = model.clone()  ####todo change it back to self.model if you need to use in gui_v4.py
    is_slack_added = add_slack(param_names, model_copy)
    # all_const_in_model = find_const_in_model(model_copy)
    iis_param, replacements_list = generate_replacements(param_names, model_copy)
    replace_const(replacements_list, model_copy)
    replace_obj(iis_param, model_copy)
    termination_condition = resolve(model_copy)
    if termination_condition == 'optimal':
        out_text = generate_slack_text(iis_param, model_copy)
    else:
        out_text = f"Changing {param_names} is not sufficient to make this model feasible, \n" \
                   f"Try other potential mutable parameters instead. \n"
    return out_text


def get_completion_from_messages_withfn(messages, model="gpt-3.5-turbo-16k"):
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
        model=model,
        messages=messages,
        functions=functions,
        function_call='auto'
    )
    return response


def gpt_function_call(ai_response, model):
    fn_call = ai_response["choices"][0]["message"]["function_call"]
    fn_name = fn_call["name"]
    arguments = fn_call["arguments"]
    if fn_name == "solve_the_model":
        param_names = eval(arguments).get("param_names")
        return solve_the_model(param_names, model), fn_name
    else:
        return

# pyomo_file = "/Users/chen4433/Documents/MATLAB/macro_v1.py"
# model, ilp_path = load_model(pyomo_file)

#
# pyomo_file = "/Users/chen4433/Documents/GPTandOPT/Infeasible_Model_Troubleshooter/macro_v1.py"
# pyomo_file = "/Users/chen4433/Documents/GPTandOPT/Infeasible_Model_Troubleshooter/RTN_v1.py"
# model, ilp_path = load_model(pyomo_file)
# const_names, param_names, iis_dict = read_iis(ilp_path, model)
# example_user_input = "can you please change the Xmax parameter ?"
# example_messages = [{"role": "user", "content": example_user_input}]
#
# response = get_completion_from_messages_withfn(example_messages)
# a,b = gpt_function_call(response, model)
# print(a)

# pyomo_file = '/Users/chen4433/Documents/GPTandOPT/'
# modelllll, _ = load_model(pyomo_file)
#
# ### try ###
# #it seems, gpt-3.5-turbo can't have function_call (set auto or nudge or none or specify a fn)
# # if function is called, response should have a key called "function_call" but no "content"
# # if function is not called, response should have a key called "content" but no "function_call"
# example_user_input = "can you please change the parameter Pi and Xmax and solve the model again?"
# # example_user_input = "Hi nice to meet you"
# example_messages = [{"role": "user", "content": example_user_input}]
#
# response = get_completion_from_messages_withfn(example_messages)
#
# out_txt = gpt_function_call(response, modelllll)







