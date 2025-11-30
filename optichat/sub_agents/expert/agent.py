from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent, ParallelAgent, Agent
from optichat.llm import *
from optichat.config.constants import *
from optichat.sub_agents.expert.prompt import get_expert_agent_prompt
from optichat.tools.search_tool import get_model_components
from optichat.tools.python_repl import python_repl_func
from optichat.tools.rag_tool import code_rag, paper_rag
from optichat.tools.callback_tool import (check_is_expert_agent_used, check_expert_agent_runtime,
                                          check_llm_request, check_llm_response, check_tool_usage, check_tool_response)
from optichat.tools.custom_tool import infeasibility_diagnosis, ldr_model_generator, ldr_expression_generator, robustness_analysis

def create_expert_agent(prompt_version=1, tools_version=1):
    expert_agent_prompt = get_expert_agent_prompt(prompt_version)
    if tools_version == 1:
        expert_agent_tools = [get_model_components,
                              python_repl_func,
                              code_rag,
                              paper_rag,
                              infeasibility_diagnosis,
                              ldr_model_generator,
                              ldr_expression_generator,
                              robustness_analysis] # TODO: infeasibility diagnosis, ldr_model_generator and ldr_expression_generator, robustness_analysis under testing
    else:
        raise NotImplementedError(f"Tools version '{tools_version}' is not implemented.")

    expert_agent = Agent(name="expert_agent",
                         model=gpt_5,  # remember to change it back under debugging
                         tools=expert_agent_tools,
                         description=("Optimization & operations research expert that "
                                      "interacts with <models>, <models_code>, <models_paper>"),
                         instruction=expert_agent_prompt,
                         output_key=OUTPUT_KEY_EXPERT_AGENT,
                         before_agent_callback=check_is_expert_agent_used,
                         after_agent_callback=check_expert_agent_runtime,
                         before_model_callback=check_llm_request,
                         after_model_callback=check_llm_response,
                         before_tool_callback=check_tool_usage,
                         after_tool_callback=check_tool_response
                         )
    return expert_agent
