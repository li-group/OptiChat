from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent, ParallelAgent, Agent
from google.adk.tools.agent_tool import AgentTool
from optichat.llm import *
from optichat.config.constants import OUTPUT_KEY_ROOT_AGENT
from optichat.sub_agents.root.prompt import *
from optichat.sub_agents.expert.agent import create_expert_agent
from optichat.sub_agents.illustrator.agent import create_illustrator_agent
from optichat.tools.callback_tool import (initialize_session, check_llm_request,
                                          check_llm_response, handle_illustrator_response)
from loguru import logger


def create_root_agent(workflow="default"):
    if workflow == "default":
        expert_agent = create_expert_agent(prompt_version=1, tools_version=1)
        illustrator_agent = create_illustrator_agent()

        root_agent = Agent(name="root_agent",
                           model=gpt_5_nano,
                           tools=[
                               AgentTool(expert_agent),
                               AgentTool(illustrator_agent)
                           ],
                           description="first point of contact for all user queries",
                           instruction=ROOT_AGENT_PROMPT,
                           output_key=OUTPUT_KEY_ROOT_AGENT,
                           before_agent_callback=initialize_session,
                           before_model_callback=check_llm_request,
                           after_model_callback=check_llm_response,
                           after_tool_callback=handle_illustrator_response)

        return root_agent
    else:
        raise NotImplementedError(f"Workflow '{workflow}' is not implemented.")