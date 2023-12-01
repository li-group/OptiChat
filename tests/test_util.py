import os, sys
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pytest
from src.Util import *

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
    'content': 'How is the fuel efficiency of our fleet impacting the overall costs? Is there a way to optimize it?'},
   'gpt_model': 'curie'},
  'output': '1'},
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