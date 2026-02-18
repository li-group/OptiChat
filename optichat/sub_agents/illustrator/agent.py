"""Illustrator agent for automatic model description generation."""

from google.adk.agents import Agent
from optichat.llm import *
from optichat.sub_agents.illustrator.prompt import ILLUSTRATOR_PROMPT
from optichat.tools.illustrator_tool import get_model_info_for_description
from optichat.tools.callback_tool import (check_llm_request, check_llm_response,
                                          check_tool_usage, check_tool_response,
                                          diagnose_if_infeasible, check_illustrator_agent_runtime)


def create_illustrator_agent():
    """
    Create the illustrator agent.

    The illustrator agent generates comprehensive model descriptions from
    model components and source code. It uses GPT-5-mini for cost-effective
    and fast generation of high-quality narratives.

    Returns:
        Agent: Configured illustrator agent
    """
    illustrator_agent = Agent(
        name="illustrator_agent",
        model=gpt_5_mini,  # Fast and cost-effective for text generation
        tools=[get_model_info_for_description],  # Tool to retrieve model info from state
        description=(
            "Generates comprehensive descriptions of optimization models "
            "by explaining the problem context, decision variables, parameters, "
            "constraints, and objectives in accessible language for domain practitioners."
        ),
        instruction=ILLUSTRATOR_PROMPT,
        output_key="MODEL_DESCRIPTION",
        before_agent_callback=diagnose_if_infeasible,  # Run diagnosis if model is infeasible
        after_agent_callback=check_illustrator_agent_runtime,  # Track runtime
        before_model_callback=check_llm_request,
        after_model_callback=check_llm_response,
        before_tool_callback=check_tool_usage,
        after_tool_callback=check_tool_response
    )
    return illustrator_agent
