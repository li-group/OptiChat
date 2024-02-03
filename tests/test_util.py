import os, sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from src.Util import *
from src.GUI import *

def test_openai_api_key():
    assert "OPENAI_API_KEY" in os.environ, "You need to set the OPENAI_API_KEY environment variable"

def test_openai_api_key_valid():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    try:
        response = get_completion_standalone("hello", "gpt-3.5-turbo")
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"API error OPENAI_API_KEY is invalid: {e}")

def test_openai_api_key_valid_gpt4():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    try:
        response = get_completion_standalone("hello", "gpt-4")
        assert isinstance(response, str)
    except Exception as e:
        pytest.fail(f"API error OPENAI_API_KEY is invalid for gpt-4: {e}")


from pyomo.environ import *
import os
import json

def test_load_model():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)
    assert ilp_path != ""
    assert isinstance(model, ConcreteModel)

    file_name = "./Pyomo_Model_Lib/feasibleproblems/bid_v0.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)
    assert ilp_path == ""
    assert isinstance(model, ConcreteModel)

def test_extract_components():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    const_list, param_list, var_list, PYOMO_CODE = extract_component(model, abs_path)

    true_const_list = ['subset_purchase', 'oneonly', 'maxpl', 'minpl', 'costdef', 'demand']
    true_param_list = ['setup', 'price', 'qmin', 'qmax', 'req']
    true_var_list = ['c', 'pl', 'plb']
    with open(abs_path, "r") as f:
        true_pyomo_code = f.read()
    
    for const in true_const_list:
        assert const in const_list
    
    for par in true_param_list:
        assert par in param_list

    for var in true_var_list:
        assert var in var_list
    
    assert true_pyomo_code == PYOMO_CODE

def test_get_parameters_n_indices():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    model_info = get_parameters_n_indices(model)
    
    idxes = [('a', 1), ('b', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 1), ('d', 1), ('e', 1), ('e', 2)]
    
    true_info = {
        'setup': {
            'is_indexed': True,
            'index_dim': 2,
            'index_set': idxes
        },
        'price': {
            'is_indexed': True,
            'index_dim': 2,
            'index_set': idxes
        },
        'qmin': {
            'is_indexed': True,
            'index_dim': 2,
            'index_set': idxes
        },
        'qmax': {
            'is_indexed': True,
            'index_dim': 2,
            'index_set': idxes
        },
        'req': {
            'is_indexed': False,
            'index_dim': 0,
            'index_set': [None]
        }
    }

    assert model_info == true_info

def test_get_constraints_n_indices():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    model_info = get_constraints_n_indices(model)

    idxes = [('a', 1), ('b', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 1), ('d', 1), ('e', 1), ('e', 2)]
    
    true_info = {
        'demand': {
            'is_indexed': False, 
            'index_dim': 0, 
            'index_set': [None]
        },
        'costdef': {
            'is_indexed': False, 
            'index_dim': 0, 
            'index_set': [None]
        }, 
        'minpl': {
            'is_indexed': True, 
            'index_dim': 2, 
            'index_set': idxes
        }, 
        'maxpl': {
            'is_indexed': True, 
            'index_dim': 2, 
            'index_set': idxes
        }, 
        'oneonly': {
        'is_indexed': True, 
        'index_dim': 1, 
        'index_set': ['a', 'b', 'c', 'd', 'e']
        }, 
        'subset_purchase': {
            'is_indexed': False, 
            'index_dim': 0, 
            'index_set': [None]
        }
    }
    assert model_info == true_info

def test_get_constraints_n_parameters():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)
    
    model_info = get_constraints_n_parameters(model)

    true_info = {
        'demand': ['req'], 
        'costdef': ['price', 'setup'], 
        'minpl': ['qmin'], 
        'maxpl': ['qmax'], 
        'oneonly': [None], 
        'subset_purchase': [None]
    }

    print(model_info, true_info)
    assert model_info == true_info

def test_get_completion_for_index():
    file_name = "./Pyomo_Model_Lib/bid_v4.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    _, _, _, PYOMO_CODE = extract_component(model, abs_path)
    model_info = get_parameters_n_indices(model)

    objs = [
{
    'input': {'user_prompt': 'How can I adjust the price for the vendor "a" in the 1st segment?'},
    'output': { 'index': [{'parameter': 'price', 'indices': [['a', 1]]}] }
}
,
{
    'input': {'user_prompt': 'What if I want to change the maximum quantity for the 1st vendor and 1st segment?'},
    'output': { 'index': [{'parameter': 'qmax', 'indices': [['a', 1]]}] }
}
,
{
    'input': {'user_prompt': 'Can I alter the setup cost for "b" in the 3rd segment?'},
    'output': { 'index': [{'parameter': 'setup', 'indices': [['b', 3]]}] }
}
,
{
    'input': {'user_prompt': 'What is the process of modifying the minimum quantity from the 3rd vendor for 1st segment?'},
    'output': { 'index': [{'parameter': 'qmin', 'indices': [['c', 1]]}] }
}
,
{
    'input': {'user_prompt': 'I want to modify the price for "d" in the first segment.' },
    'output': { 'index': [{'parameter': 'price', 'indices': [['d', 1]]}] }
}
,
{
    'input': {'user_prompt': 'Is it possible to change the maximum quantity for the 4th vendor in the 1st segment?'},
    'output': { 'index': [{'parameter': 'qmax', 'indices': [['d', 1]]}] }
}
,
{
    'input': {'user_prompt': 'How do I change the setup cost for "e" in the second segment?'},
    'output': { 'index': [{'parameter': 'setup', 'indices': [['e', 2]]}] }
}
,
{
    'input': {'user_prompt': 'Is it possible to adjust the minimum quantity for the 5th vendor and 2nd segment sector?'},
    'output': { 'index': [{'parameter': 'qmin', 'indices': [['e', 2]]}] }
}
,
{
    'input': {'user_prompt': 'I want to alter the price for "b" in the fourth segment.' },
    'output': { 'index': [{'parameter': 'price', 'indices': [['b', 4]]}] }
}
,
# {
#     'input': {'user_prompt': 'Can I change the requirements?'},
#     'output': { 'index': [{'parameter': 'req', 'indices': [None]}] }
# }
]

    for obj in objs:
        content = obj['input']['user_prompt']
        chatbot_messages = {'role': 'user', 'content': content}
        response = get_completion_for_index(chatbot_messages, model_info,PYOMO_CODE, 'gpt-4')
        fn_call = response.choices[0].message.function_call
        fn_name = fn_call.name
        arguments = fn_call.arguments

        assert fn_name == 'get_index'

        args = json.loads(arguments)

        true_response = obj['output']

        assert json.dumps(args, sort_keys=True) == json.dumps(true_response, sort_keys=True)

def test_get_completion_for_index_sensitivity():
    file_name = "./Pyomo_Model_Lib/feasibleproblems/bid_v0.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    _, _, _, PYOMO_CODE = extract_component(model, abs_path)
    model_constraint_info = get_constraints_n_indices(model)
    model_constraint_param_info = get_constraints_n_parameters(model)

    args = """
        [
            {
                "input": {
                    "user_prompt": "What happens if the minimum purchase level for vendor C increases?"
                },
                "output": {
                    "index": [{"constraint": "minpl", "indices": [["c", 1]]}]
                }
            },
            {
                "input": {
                    "user_prompt": "How will the setup cost affect my total cost?"
                },
                "output": {
                    "index": [{"constraint": "costdef", "indices": [null]}]
                }
            },
            {
                "input": {
                    "user_prompt": "Will changing the price for vendor B in the 3rd segment affect my total cost?"
                },
                "output": {
                    "index": [{"constraint": "costdef", "indices": [null]}]
                }
            },
            {
                "input": {
                    "user_prompt": "If vendor D increases their maximum purchase level, how will it affect my purchases?"
                },
                "output": {
                    "index": [{"constraint": "maxpl", "indices": [["d", 1]]}]
                }
            },
            {
                "input": {
                    "user_prompt": "How does the minimum purchase level for vendor E in the 2nd segment affect my purchases?"
                },
                "output": {
                    "index": [{"constraint": "minpl", "indices": [["e", 2]]}]
                }
            },
            {
                "input": {
                    "user_prompt": "Will purchasing from vendor A affect my total cost?"
                },
                "output": {
                    "index": [{"constraint": "costdef", "indices": [null]}]
                }
            },
            {
                "input": {
                    "user_prompt": "How does the maximum purchase level for vendor A affect my purchases?"
                },
                "output": {
                    "index": [{"constraint": "maxpl", "indices": [["a", 1]]}]
                }
            }
        ]
    """

    objs = json.loads(args)

    for obj in objs:
        content = obj['input']['user_prompt']
        chatbot_messages = {
            'role': 'user',
            'content': content
        }
        response = get_completion_for_index_sensitivity(chatbot_messages, model_constraint_info, model_constraint_param_info, PYOMO_CODE, 'gpt-4')

        # print(response)

        fn_call = response.choices[0].message.function_call
        fn_name = fn_call.name
        arguments = fn_call.arguments

        assert fn_name == 'sensitivity_analysis'

        args = json.loads(arguments)

        true_response = obj['output']
        print(content, json.dumps(args, sort_keys=True), json.dumps(true_response, sort_keys=True))

        assert json.dumps(args, sort_keys=True) == json.dumps(true_response, sort_keys=True)

def test_add_slack():
    file_name = "./Pyomo_Model_Lib/feasibleproblems/bid_v0.py"
    abs_path = os.path.abspath(file_name)
    model, ilp_path = load_model(abs_path)

    is_slack_added = add_slack(['setup', 'qmax'], model)

    idxes = [('a', 1), ('b', 1), ('b', 2), ('b', 3), ('b', 4), ('c', 1), ('d', 1), ('e', 1), ('e', 2)]

    for p in ['setup', 'qmax']:
        for idx in idxes:
            assert is_slack_added[p][idx] == False

def test_generate_replacements():
    test_add_slack()

def test_gurobi_solve():
    model = ConcreteModel()
    model.x = Var(within=NonNegativeReals)
    model.obj = Objective(expr=model.x, sense=minimize)
    solver = SolverFactory('gurobi')
    try:
        results = solver.solve(model)
    except Exception as e:
        pytest.fail(f"Gurobi not installed properly {e}")

    assert results.solver.termination_condition == TerminationCondition.optimal, "Gurobi solver failed to solve the model to optimality"

def test_classify_question():
    objs = [{'input': {'question': {'role': 'user',
    'content': "Why can't I find a solution to my production planning problem even though I've listed out all my constraints? Are any of them conflicting with each other?"},
   'gpt_model': 'curie'},
  'output': '1'},
 {'input': {'question': {'role': 'user',
    'content': "I've set up an optimal staffing schedule to minimize costs, but the solver says there's no feasible solution. Can we figure out which staffing requirement is causing the issue?"},
   'gpt_model': 'curie'},
  'output': '1'},
 {'input': {'question': {'role': 'user',
    'content': "I'm trying to optimize the routing for my delivery trucks, but it's not working out. Could there be any route or time constraints that are impossible to meet together?"},
   'gpt_model': 'curie'},
  'output': '1'},
 {'input': {'question': {'role': 'user',
    'content': "My inventory optimization model was working fine last month. This month, I can't get a solution. What might have changed in the demand or supply constraints that's causing this?"},
   'gpt_model': 'curie'},
  'output': '1'},
 {'input': {'question': {'role': 'user',
    'content': "I've modeled a diet plan to minimize costs while meeting all nutrient requirements. However, I can't find a feasible diet. Are there any nutrient requirements that are contradicting each other or impossible to meet with the given food items?"},
   'gpt_model': 'curie'},
  'output': '1'},
 {'input': {'question': {'role': 'user',
    'content': 'If we can get an extra hour of machine time each day, how much more profit can we expect? Is it worth paying overtime for the workers?'},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': 'How much more would our transportation costs increase if fuel prices went up by 10%? Should we consider negotiating long-term fuel contracts now?'},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': "Suppose there's a slight decrease in the yield of our main crop due to unexpected weather changes. How would this affect our yearly revenue? Should we consider diversifying our crops next season?"},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': 'If we allocate an additional $10,000 to our marketing budget, how much more revenue can we expect? Is it a better return on investment than, say, investing in product development?'},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': 'How would extending our customer service hours by two hours every day affect our monthly operating costs? And if we did extend the hours, would it significantly improve our customer satisfaction ratings?'},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': 'What are the constraints that are causing my model to be not feasible?'},
   'gpt_model': 'curie'},
  'output': '3'},
 {'input': {'question': {'role': 'user',
    'content': 'What physical quantities are making the model infeasible?'},
   'gpt_model': 'curie'},
  'output': '3'},
 {'input': {'question': {'role': 'user',
    'content': 'What are the parameters that I need to change to make the model feasible?'},
   'gpt_model': 'curie'},
  'output': '3'},
 {'input': {'question': {'role': 'user',
    'content': 'What are the indexed parameters present in the model?'},
   'gpt_model': 'curie'},
  'output': '4'},
 {'input': {'question': {'role': 'user',
    'content': "If ('a', 1) a valid index for demand?"},
   'gpt_model': 'curie'},
  'output': '4'},
 {'input': {'question': {'role': 'user',
    'content': 'What are the indices of the parameter `ship_class`?'},
   'gpt_model': 'curie'},
  'output': '4'},
 {'input': {'question': {'role': 'user',
    'content': 'How many different kinds of ships are there? What are their capacities?'},
   'gpt_model': 'curie'},
  'output': '4'},
 {'input': {'question': {'role': 'user',
    'content': 'What are the different kinds of ships we have?'},
   'gpt_model': 'curie'},
  'output': '4'},
 {'input': {'question': {'role': 'user',
    'content': 'How is the fuel efficiency of our fleet impacting the overall costs?'},
   'gpt_model': 'curie'},
  'output': '2'},
 {'input': {'question': {'role': 'user',
    'content': "I'm confused about the constraint 'max_supply' in our production model. Could you explain its role?"},
   'gpt_model': 'curie'},
  'output': '4'}]
    objs += [{'input': {'question': {'role': 'user', 'content': "Why can't I find a solution to my production planning problem even though I've listed out all my constraints? Are any of them conflicting with each other?"}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': "I've set up an optimal staffing schedule to minimize costs, but the solver says there's no feasible solution. Can we figure out which staffing requirement is causing the issue?"}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': "I'm trying to optimize the routing for my delivery trucks, but it's not working out. Could there be any route or time constraints that are impossible to meet together?"}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': "My inventory optimization model was working fine last month. This month, I can't get a solution. What might have changed in the demand or supply constraints that's causing this?"}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': "I've modeled a diet plan to minimize costs while meeting all nutrient requirements. However, I can't find a feasible diet. Are there any nutrient requirements that are contradicting each other or impossible to meet with the given food items?"}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': 'If we can get an extra hour of machine time each day, how much more profit can we expect? Is it worth paying overtime for the workers?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'How much more would our transportation costs increase if fuel prices went up by 10%? Should we consider negotiating long-term fuel contracts now?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': "Suppose there's a slight decrease in the yield of our main crop due to unexpected weather changes. How would this affect our yearly revenue? Should we consider diversifying our crops next season?"}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'If we allocate an additional $10,000 to our marketing budget, how much more revenue can we expect? Is it a better return on investment than, say, investing in product development?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'How would extending our customer service hours by two hours every day affect our monthly operating costs? And if we did extend the hours, would it significantly improve our customer satisfaction ratings?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'What are the constraints that are causing my model to be not feasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What physical quantities are making the model infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What are the parameters that I need to change to make the model feasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What are the indexed parameters present in the model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': "If ('a', 1) a valid index for demand?"}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'What are the indices of the parameter `ship_class`?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'How many different kinds of ships are there? What are their capacities?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'What are the different kinds of ships we have?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'What constraints are conflicting in my production optimization model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': 'Which staffing requirements make my work schedule optimization model infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'Which route constraints make my delivery truck routing model infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What changes in inventory constraints made the model infeasible this month?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What nutrient requirements make my diet plan infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'How would an increase in machine time affect profit?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'How does an increase in fuel price affect transportation cost?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'What impact does a decrease in crop yield have on yearly revenue?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'What is the expected increase in revenue on extending the marketing budget?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'How does extending customer service hours affect operating costs and customer satisfaction?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'Which constraints make my model infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What parameters make the model infeasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'How can I change parameters to make the model feasible?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What are the indices for a particular parameter in my model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'Is a specific index valid for demand in my model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'What indices are associated with the parameter `ship_class` in my model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'How many kinds of ships are there in my model and what are their respective capacities?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'In my model, how many types of ships do we have?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}, {'input': {'question': {'role': 'user', 'content': 'What are the potential reasons for infeasibility in my supply chain optimization model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '1'}, {'input': {'question': {'role': 'user', 'content': 'What would be the impact of reducing the processing time in the scheduling problem?'}, 'gpt_model': 'text-davinci-002'}, 'output': '2'}, {'input': {'question': {'role': 'user', 'content': 'Which constraints are impacting the feasibility of my assembly line scheduling problem?'}, 'gpt_model': 'text-davinci-002'}, 'output': '3'}, {'input': {'question': {'role': 'user', 'content': 'What are the parameter indices associated with job schedules in my model?'}, 'gpt_model': 'text-davinci-002'}, 'output': '4'}]

    for obj in objs:
        question = obj['input']['question']
        gpt_model = "gpt-4"
        output = obj['output']
        prediction = classify_question(question, gpt_model)
        print(prediction, output)
        if prediction != output:
            print(question)
        assert prediction == output