feasibility_restoration_fn_description = """
Use when: The model is infeasible and you need to find out the minimal change to specific [component name] for restoring feasibility.
Example: “How much should we adjust the [component name] to make the model feasible”
Example: "I believe changing [component name] is practical, by how much do I need to change in order to make the model feasible"
"""
components_retrival_fn_description = """
Use when: You need to know the current values or expressions of [component name] within the model.
Example: “What are the values of the [component name]”
Example: "How many [component name] are currently available"
"""
sensitivity_analysis_fn_description = """
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **without specifying the extent of changes**.
Example: “How will the optimal profit change with the change in the [component name]”
Example: "How stable is the objective value in response to variations in the [component name]"
Example: "Will the optimal value be greatly affected if we have more [component name]"
"""
evaluate_modification_fn_description = """
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **by specifying the extent of changes**.
Example: “How will the optimal profit change with **a 10% increase** in the [component name]”
Example: "How stable is the objective value in response to the modification that [component name] is **decreased by 20 units**"
Example: "Will the optimal value be greatly affected if we have *two* more [component name]"
"""


def get_prompts(prompt):
    need2describe_prompt = """
Here are the name of {component_type} that need to be described
-----
{component_names}
-----


"""

    model_interpretation_json = {
        "components": {
            "sets": [
                {
                    "name": "The name of the component in sets",
                    "description": "The description of the component",
                }
            ],
            "parameters": [
                {
                    "name": "The name of the component in parameters",
                    "description": "The description of the component",
                }
            ],
            "variables": [
                {
                    "name": "The name of the component in variables",
                    "description": "The description of the component",
                }
            ],
            "constraints": [
                {
                    "name": "The name of the component in constraints",
                    "description": "The description of the component",
                }
            ],
            "objective": [
                {
                    "name": "The name of the component in objective",
                    "description": "The description of the component",
                }
            ]
        }
    }

    model_interpretation_prompt = """
You are an operations research expert and your role is to use PLAIN ENGLISH to interpret an optimization model written in Pyomo. 
The Pyomo code is given below:

-----
{code}
-----


{cat_need2describe_prompt}
Your task is carefully inspect the code and write a description for each of the components. 

Then, generate a json file accordingly with the following format (STICK TO THIS FORMAT!)

{model_interpretation_json}

- description should be either physical meanings, intended use, or any other relevant information about the component.
- Note that I'm going to use python json.loads() function to parse the json file, so please make sure the format is correct (don't add ',' before enclosing '}}' or ']' characters.
- Generate the complete json file and don't omit anything.
- Use 'name' and 'description' as the keys, and provide the name and description of the component as the values.
- Use '```json' and '```' to enclose the json file.

Take a deep breath and solve the problem step by step.
"""

    model_illustration_prompt = """
You are an operations research expert and your role is to introduce an optimization model to non-experts, based on an abstract representation of the model in json format.
The json representation is given below:

-----
{json_representation}
-----

- Start with a brief introduction of the model, what the problem is about, who is using the model, and what the model is trying to achieve.
- Explain what decisions (variables) are to be made
- Explain what data or information (parameters) is already known
- Explain what constraints are imposed on the decisions
- Explain what the objective is, what is being optimized

The explanation must be coherent and easy to understand for the users who are experts in the filed for which this model is built but not in optimization.
"""

    model_inference_prompt = """
You are an operations research expert and your role is to infer why an optimization model is infeasible, based on an abstract representation of the infeasible model in json format.
Particularly, your team has identified the Irreducible Infeasible Subset (IIS) of the model, which is given below:

-----
{iis_info}
-----


To understand what the parameters and the constraints mean, the json representation is given below for your reference:

-----
{json_representation}
-----


- Introduce to the user what constraints are potentially causing the infeasibility, and what parameters are involved in these constraints.
- Explain the relationship between the constraints and the parameters, and infer why the constraints are conflicting with each other.
- Provide inference by analyzing their physical meanings, and AVOID using jargon and symbols as much as possible but the explanation style must be formal. 
- Recommend some parameters that you believe can be adjusted to make the model feasible.
- Parameters recommended for adjustment MUST be changeable physically in practice. For example, molecular weight of a molecule is not changeable in practice.
- Assess the practical implications of the recommendations. For example, increasing the number of workers implies hiring more workers, which incurs additional costs.
"""

    coordinator_prompt = """
You're a coordinator in a team of optimization experts. The goal of the team is to help non-experts analyze an 
optimization problem. Your task is to choose the next expert to work on the problem based on the current situation. 

Here's the list of agents in your team:
-----
{agents}
-----

Considering the conversation, generate a json file with the following format: 
{{ "agent_name": "Name of the agent you want to call next", "task": "The task you want the agent to carry out" }} 

to identify the next agent to work on the problem, and also the task it has to carry out. 
- Only generate the json file, and don't generate any other text.
- DO NOT change the keys of the json file, only change the values. Keys are "agent_name" and "task".
- if you think the problem is solved, generate the json file below:
{{ "agent_name": "Explainer", "task": "DONE" }} 
"""

    explainer_prompt = """
You're an optimization expert who helps your team answer user queries in MARKDOWN format.

- The users are not experts in optimization, but they are experts in the filed for which this model is built.
- Provide a detailed explanation only when you believe the users need more context about optimization to understand your explanation.
- Otherwise, the explanation must be succinct and concise, because users may be distracted by too much information.
- If Operators and Programmers in your team have provided technical feedback, then you need to summarize the feedback because the user cannot see them.
"""

    syntax_reminder_prompt = """
You're an operator working on a pyomo model.
Your task is to identify the following arguments: 
- the component names that the user is interested in,
- the most appropriate function that can answer the user's query, 
- the model that the user is querying.
then call the predefined syntax_guidance function to generate syntax guidance.

----- Instruction to select the most appropriate function -----
you MUST select a function from ```{function_names}```, DO NOT make up your own function.
1. feasibility_restoration:
Use when: The model is infeasible and you need to find out the minimal change to specific [component name] for restoring feasibility.
Example: “How much should we adjust the [component name] to make the model feasible”
Example: "I believe changing [component name] is practical, by how much do I need to change in order to make the model feasible"
[component name] category: parameters. If only constraint name is provided in the query, you need to infer the parameters involved in the constraint.

2. components_retrival:
Use when: You need to know the current values or expressions of [component name] within the model.
Example: “What are the values of the [component name]”
Example: "How many [component name] are currently available"
[component name] category: sets, parameters, variables, constraints, or objectives.

3. sensitivity_analysis:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **without specifying the extent of changes**.
Example: “How will the optimal profit change with the change in the [component name]”
Example: "How stable is the objective value in response to variations in the [component name]"
Example: "Will the optimal value be greatly affected if we have more [component name]"
[component name] category: parameters.

4. evaluate_modification:
Use when: The model is feasible and you want to understand the impact of changing [component name] on the optimal objective value, **by specifying the extent of changes**.
Example: “How will the optimal profit change with **a 10% increase** in the [component name]”
Example: "How stable is the objective value in response to the modification that [component name] is **decreased by 20 units**"
Example: "Will the optimal value be greatly affected if we have *two* more [component name]"
[component name] category: parameters or variables.

5. external_tools:
Use when: User doubts the model's optimal solution and provides a counterexample, and you want to add new constraints to implement the counterexample.
Example: “Why is it not recommended to have [component name] lower than 400 in the optimal solution”
Example: "Why isn’t [component name] and [component name] both used in the optimal scenario"
[component name] category: parameters or variables.
    
----- Instruction to determine the correct component name -----
The [component name] MUST be in a symbolic form, instead of its description.
Use the following dictionary to find the correct [component name] based on its description:
{component_name_meaning_pairs}

----- Instruction to find the queried model -----
In the form of 'model_integer', e.g. 'model_1'
"""

    operator_prompt = """
You're an optimization expert who helps your team to access and interact with optimization models by internal tools.

Your task is to invoke the most appropriate tool correctly based on the user's query and system reminders.
"""

    code_reminder_prompt = """
{source_code}

# OPTICHAT REVISION CODE GOES HERE

from pyomo.environ import SolverFactory, TerminationCondition
solver = SolverFactory('gurobi')
solver.options['TimeLimit'] = 300  # 5min time limit
results = solver.solve(model, tee=False)
print("Solver Status: ", results.solver.status)
print("Termination Condition: ", results.solver.termination_condition)
if results.solver.termination_condition == TerminationCondition.optimal:
    from pyomo.environ import Objective
    for obj_name, obj in model.component_map(Objective).items():
        print('Optimal Objective Value: ', pyo.value(obj))
else:
    print("Model is infeasible or unbounded, no optimal objective value is available.")

# OPTICHAT PRINT CODE GOES HERE

"""

    programmer_prompt = """
You're an optimization expert who helps your team to write pyomo code to answer users questions.
(1) write code snippet to revise the model, only when the user doubts the model's optimal solution and provides a counterexample
(2) write code snippet to print out the information useful for answering the user's question

Output Format:
==========
```python
CODE SNIPPET FOR REVISING THE MODEL
```

```python
CODE SNIPPET FOR PRINTING OUT USEFUL INFORMATION
```
==========

Here are some example questions and their answer codes:
----- EXAMPLE 1 -----
Question: Why is it not recommended to use just one supplier for roastery 2?

Answer Code:
```python
# user is actually interested in the case that only one supplier can supply roastery 2 and does not believe the optimal solution.
model.force_one_supplier = ConstraintList()
model.force_one_supplier.add(sum(model.z[s,'roastery2'] for s in model.suppliers) <= 1)
for s in model.suppliers:
    model.force_one_supplier.add(model.x[s,'roastery2'] <= model.capacity_in_supplier[s] * model.z[s, 'roastery2'])
```

```python
# I print out the new optimal objective value so that you can tell the user how the objective value changes if only one supplier supplies roastery 2.
print('If forcing only one supplier to supply roastery 2, the optimal objective value will become: ', model.obj())
```

----- EXAMPLE 2 -----
Question: Why is it not recommended to have production cost larger than transportation cost in the optimal setting?

Answer Code:
```python
# user does not believe the optimal solution obtained when production cost smaller than transportation cost.
# so we force production cost to be less than transportation cost to see what will happen.
model.counter_example = ConstraintList()
model.counter_example.add(model.production <= model.transportation)
```

```python
# I print out the new optimal objective value so that you can tell the user how the objective value changes.
print('If forcing production cost be smaller than transportation cost, the optimal objective value will become: ', model.obj())
```

- Code reminder has provided you with the source code of the pyomo model
- Your written code will be added to the lines with substring: "# OPTICHAT *** CODE GOES HERE"
So, you don't need to repeat the source code that has already been provided by Code reminder.
- The code for re-solving the model has already been given, 
So you don't need to add it. Solving the model repeatedly can lead to errors.
- Your written code should be accompanied by comments to explain the purpose of the code.
- Evaluator will execute the new code for you and read the execution result.
So, you MUST print out the model information that you believe is necessary for the user's question.
"""

    evaluator_prompt = """
You're an optimization expert who helps your team to review pyomo code,
based on the execution result of the code provided by the programmer.

Is the code bug-free and valid to answer the user's query?
Generate the following json file if you accept the code, and provide your own comment.
{{ "decision": "accept", "comment": "your own comment" }}
Generate the following json file if you reject the code, and provide your own comment.
{{ "decision": "reject", "comment": "your own comment" }}

- Only generate the json file, and don't generate any other text.
- Use 'decision' and 'comment' as the keys, 
- choose 'accept' or 'reject' for the decision, and provide your own comment. 
- Note that infeasibility caused by the new constraints may be acceptable. 
This is because programmers are trying to create a counterfactual example that the user is interested in, and this counterfactual example may be infeasible in nature.
"""

    test_prompt = """
You are a judge who determines if the LLM’s answer passes the test.
Criteria: The data in the execution result should be consistent with the human expert's answer.
Details: LLM may omit some data that human experts collected from other sources, but if it covers the correct objective value correctly, it should pass.

Human Expert Answer: 
{human_expert_answer}

- Return either "Pass" or "Fail."
- No additional comments or explanations.
"""

    if prompt == 'model_interpretation_prompt':
        return model_interpretation_prompt
    elif prompt == 'need2describe_prompt':
        return need2describe_prompt
    elif prompt == 'model_interpretation_json':
        return model_interpretation_json
    elif prompt == 'model_illustration_prompt':
        return model_illustration_prompt
    elif prompt == 'model_inference_prompt':
        return model_inference_prompt
    elif prompt == 'coordinator_prompt':
        return coordinator_prompt
    elif prompt == 'explainer_prompt':
        return explainer_prompt
    elif prompt == 'syntax_reminder_prompt':
        return syntax_reminder_prompt
    elif prompt == 'operator_prompt':
        return operator_prompt
    elif prompt == 'code_reminder_prompt':
        return code_reminder_prompt
    elif prompt == 'programmer_prompt':
        return programmer_prompt
    elif prompt == 'evaluator_prompt':
        return evaluator_prompt
    elif prompt == 'test_prompt':
        return test_prompt



def old_get_fn_json(fn_name):
    fn_json_template = \
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component_name": {"type": "string"},
                                    "component_indexes": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {"type": "string"},
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "oneOf": [
                                                        {"type": "string"},
                                                        {"type": "integer"},
                                                    ]
                                                }
                                            }
                                        ],
                                    },
                                },
                                "required": ["component_name", "component_indexes"]
                            },
                            "description": "List of dictionary of component_name and component_indexes that users are interested in."
                        },
                        "queried_model": {
                            "type": "string",
                            "description": "'model_int' e.g. 'model_1'"
                        },
                    },
                    "required": ["queried_components", "queried_model"]
                }
            }
        }
    fn_delta_json_template = \
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_components": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "operation": {"type": "string",
                                                  "description": "modification operation e.g. * / + - = !"},
                                    "delta": {"type": "number", "description": "The extent of the modification"},
                                    "component_name": {"type": "string"},
                                    "component_indexes": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {"type": "string"},
                                            {"type": "integer"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "oneOf": [
                                                        {"type": "string"},
                                                        {"type": "integer"},
                                                    ]
                                                }
                                            }
                                        ],
                                    },
                                },
                                "required": ["component_name", "component_indexes", "operation", "delta"]
                            },
                            "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                        },
                        "queried_model": {
                            "type": "string",
                            "description": "'model_int' e.g. 'model_1'"
                        },
                    },
                    "required": ["queried_components", "queried_model"]
                }
            }
        }
    # fn_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                     {"type": "null"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes"]
    #                         },
    #                         "description": "List of dictionary of component_name and component_indexes that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    # fn_delta_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "operation": {"type": "string",
    #                                               "description": "modification operation e.g. * / + - = !"},
    #                                 "delta": {"type": "number", "description": "The extent of the modification"},
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                     {"type": "null"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes", "operation", "delta"]
    #                         },
    #                         "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }

    fn_json_template["function"]["name"] = fn_name
    fn_delta_json_template["function"]["name"] = fn_name
    if fn_name == "feasibility_restoration":
        fn_json_template["function"]["description"] += feasibility_restoration_fn_description
    elif fn_name == "sensitivity_analysis":
        fn_json_template["function"]["description"] += sensitivity_analysis_fn_description
    elif fn_name == "components_retrival":
        fn_json_template["function"]["description"] += components_retrival_fn_description
    elif fn_name == "evaluate_modification":
        fn_delta_json_template["function"]["description"] += evaluate_modification_fn_description
        return fn_delta_json_template
    return fn_json_template


def get_fn_json(fn_name, mode):
    if mode == 'multiple':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"},
                                                ]
                                            },
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "type": "array",
                                            "items": {
                                                "oneOf": [
                                                    {"type": "string"},
                                                    {"type": "integer"},
                                                ]
                                            },
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'single':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "integer"},
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "integer"},
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'none':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {"type": "null"},
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {"type": "null"},
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    elif mode == 'all':
        fn_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "null"},
                                                {"type": "string"},
                                                {"type": "integer"},
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "oneOf": [
                                                            {"type": "string"},
                                                            {"type": "integer"},
                                                        ]
                                                    }
                                                }
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes"]
                                },
                                "description": "List of dictionary of component_name and component_indexes that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
        fn_delta_json_template = \
            {
                "type": "function",
                "function": {
                    "name": "",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queried_components": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "operation": {"type": "string",
                                                      "description": "modification operation e.g. * / + - = !"},
                                        "delta": {"type": "number", "description": "The extent of the modification"},
                                        "component_name": {"type": "string"},
                                        "component_indexes": {
                                            "oneOf": [
                                                {"type": "null"},
                                                {"type": "string"},
                                                {"type": "integer"},
                                                {
                                                    "type": "array",
                                                    "items": {
                                                        "oneOf": [
                                                            {"type": "string"},
                                                            {"type": "integer"},
                                                        ]
                                                    }
                                                }
                                            ],
                                        },
                                    },
                                    "required": ["component_name", "component_indexes", "operation", "delta"]
                                },
                                "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
                            },
                            "queried_model": {
                                "type": "string",
                                "description": "'model_int' e.g. 'model_1'"
                            },
                        },
                        "required": ["queried_components", "queried_model"]
                    }
                }
            }
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    # fn_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes"]
    #                         },
    #                         "description": "List of dictionary of component_name and component_indexes that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    # fn_delta_json_template = \
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "",
    #             "description": "",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "queried_components": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "operation": {"type": "string",
    #                                               "description": "modification operation e.g. * / + - = !"},
    #                                 "delta": {"type": "number", "description": "The extent of the modification"},
    #                                 "component_name": {"type": "string"},
    #                                 "component_indexes": {
    #                                     "oneOf": [
    #                                         {"type": "null"},
    #                                         {"type": "string"},
    #                                         {"type": "integer"},
    #                                         {
    #                                             "type": "array",
    #                                             "items": {
    #                                                 "oneOf": [
    #                                                     {"type": "string"},
    #                                                     {"type": "integer"},
    #                                                 ]
    #                                             }
    #                                         }
    #                                     ],
    #                                 },
    #                             },
    #                             "required": ["component_name", "component_indexes", "operation", "delta"]
    #                         },
    #                         "description": "List of dictionary of component_name, component_indexes, modification type and modification extent that users are interested in."
    #                     },
    #                     "queried_model": {
    #                         "type": "string",
    #                         "description": "'model_int' e.g. 'model_1'"
    #                     },
    #                 },
    #                 "required": ["queried_components", "queried_model"]
    #             }
    #         }
    #     }
    fn_json_template["function"]["name"] = fn_name
    fn_delta_json_template["function"]["name"] = fn_name
    if fn_name == "feasibility_restoration":
        fn_json_template["function"]["description"] += feasibility_restoration_fn_description
    elif fn_name == "sensitivity_analysis":
        fn_json_template["function"]["description"] += sensitivity_analysis_fn_description
    elif fn_name == "components_retrival":
        fn_json_template["function"]["description"] += components_retrival_fn_description
    elif fn_name == "evaluate_modification":
        fn_delta_json_template["function"]["description"] += evaluate_modification_fn_description
        return fn_delta_json_template
    return fn_json_template


def get_syntax_guidance_fn_json():
    fn_json_template = \
        {
            "type": "function",
            "function": {
                "name": "syntax_guidance",
                "description": "generate syntax reminder, based on the most appropriate function that can answer the user's query, the component names that the user is interested in, and the model that the user is querying.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queried_function": {"type": "string",
                                             "description": "The name of the function that the user is querying."},
                        "queried_components": {"type": "array",
                                               "items": {"type": "string"},
                                               "description": "List of component names that users are interested in."},
                        "queried_model": {"type": "string",
                                          "description": "'model_integer' e.g. 'model_1'"},
                    },
                    "required": ["queried_function", "queried_components", "queried_model"]
                }
            }
        }

    return fn_json_template


def get_tools(fn_names):
    multiple_tools = []
    single_tools = []
    none_tools = []
    all_tools = []
    for fn_name in fn_names:
        if fn_name != 'external_tools':
            multiple_tools.append(get_fn_json(fn_name, 'multiple'))
            single_tools.append(get_fn_json(fn_name, 'single'))
            none_tools.append(get_fn_json(fn_name, 'none'))
            all_tools.append(get_fn_json(fn_name, 'all'))
    return multiple_tools, single_tools, none_tools, all_tools, 'auto'


def get_syntax_guidance_tool():
    syntax_guidance_fn_json = get_syntax_guidance_fn_json()
    syntax_guidance_tool = [syntax_guidance_fn_json]
    return syntax_guidance_tool

