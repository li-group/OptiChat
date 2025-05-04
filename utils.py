import json
import time
import copy
import typing
import os
import sys
import re
import importlib
# Streamlit
import streamlit as st
# Gurobi
import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
import re
from pyomo.core.expr.visitor import identify_mutable_parameters, replace_expressions, clone_expression
# GPT
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
import tiktoken


from prompts import get_prompts, get_tools, get_syntax_guidance_tool
from agents import Interpreter, Coordinator, Explainer, Engineer


def get_agents(fn_names, client, llm_model_name):
    interpreter = Interpreter(client=client, llm=llm_model_name)
    explainer = Explainer(client=client, llm=llm_model_name)

    multiple_tools, single_tools, none_tools, all_tools, tool_choice = get_tools(fn_names, llm_model_name)
    syntax_guidance_tool = get_syntax_guidance_tool(llm_model_name)
    engineer = Engineer(client=client, llm=llm_model_name,
                        multiple_tools=multiple_tools, single_tools=single_tools,
                        none_tools=none_tools, all_tools=all_tools,
                        tool_choice=tool_choice,
                        syntax_guidance_tool=syntax_guidance_tool,
                        function_names=str(fn_names))
    coordinator = Coordinator(client=client, agents=[explainer, engineer], llm=llm_model_name)
    return interpreter, explainer, engineer, coordinator


def save_team_conversation(team_conversation, filename):
    with open(filename, 'w') as f:
        for message in team_conversation:
            f.write(json.dumps(message) + '\n')


#=========================optichat_workflow_exp=====================================
# multi-agent workflow where the coordinator decides which agent (Engineer or Explainer) should perform a task at each round.

def OptiChat_workflow_exp(args, coordinator, engineer, explainer, messages, models_dict):
    team_conversation = []
    rounds = 0

    # Resets the time counters for each agent’s task-related time
    coordinator.coordination_time = 0
    engineer.syntax_time = 0
    engineer.programing_time = 0
    engineer.evaluation_time = 0
    explainer.explanation_time = 0

    # in current design, if coordinator has assigned the task once,
    # actually there will be no need to call llm to generate the decision again
    while rounds <= coordinator.max_rounds:
        coordinator_start = time.time()
        #coordinator generates a decision based on the current messages and the conversation history.
        decision = coordinator.generate_decision_exp(args, messages, team_conversation) 
        coordinator_end = time.time()
        coordinator.coordination_time += (coordinator_end - coordinator_start)
        
        # If the coordinator fails to generate a valid decision
        if not decision:
            print(f'coordinator failed to generate decision')
            messages.append({"role": "assistant", "content": "LLM failed"})
            return messages, team_conversation

        else: #if cordinator decision is engineer
            if decision["agent_name"] == 'Engineer':
                # unlike explainer, engineer team has already updated the team_conversation and messages in fn below
                # syntax time, programming time, evaluation time are also updated in the fn below
                messages, team_conversation = engineer.generate_report_exp(args,
                                                                           messages, team_conversation, models_dict)
            
            elif decision["agent_name"] == "Explainer": #if cordinator decision is engineer
                explainer_start = time.time()
                explanation = explainer.generate_explanation_exp(args, messages, team_conversation)

                # Ensure explanation_response is always initialized
                explanation_response = "" 
                if args.explanation_stream:
                    with st.chat_message("assistant"):
                        placeholder = st.empty()  # Placeholder for live updating


                        if isinstance(explanation, str):  # Gemini case (full string response)
                            for char in explanation:  
                                explanation_response += char
                                placeholder.markdown(explanation_response)
                                time.sleep(0.01)  
                        else:  # OpenAI case (streaming response)
                            for chunk in st.write_stream(explanation):  
                                chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                                explanation_response += chunk_text
                                placeholder.markdown(explanation_response)
 
                else:
                    explanation_response = explanation  # type=string

                explainer_end = time.time()
                explainer.explanation_time += (explainer_end - explainer_start)


                if explanation_response is None:
                    raise ValueError("Error: explanation_response is None before appending to messages!")

                # Append to conversation history
                team_conversation.append({"agent_name": "Explainer", "agent_response": explanation_response})
                messages.append({"role": "assistant", "content": explanation_response})

                return messages, team_conversation


            else:
                raise ValueError(
                    f"Decision {decision} has an invalid agent name. Please choose from Engineer or Explainer.")

        rounds += 1
        
#===============================================================================================================
