from loguru import logger
from typing import Dict, Any
from typing import Optional
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool


def rag(request: str, models_code_cfg: Dict, number_of_results: int) -> str:
    """
    TODO: check langchain rag
    """
    return "todo: implement rag"


def code_rag(request: str, number_of_results: int, tool_context: ToolContext) -> str:
    """
    code_rag retrieves code blocks from <models_code> by
    performing semantic similarity search against the submitted request.

    Args:
        request: the information that you want to get from <models_code>, be specific and detailed
        number_of_results: The maximum number of search results to retrieve
                           (No more than 5 each time, so make your query specific and detailed).

    Returns:
        string containing the code blocks that are relevant to the request submitted by you.
    """
    cfg = tool_context.state["cfg"]
    models_code_cfg = cfg.get("models_code", None)
    result = rag(request, models_code_cfg, number_of_results)
    return {"result": result}


def paper_rag(request: str, number_of_results: int, tool_context: ToolContext) -> str:
    """
    paper_rag retrieves paper contents from <models_paper> by
    performing semantic similarity search against the submitted request.

    Args:
        request: the information that you want to get from <models_paper>, be specific and detailed
        number_of_results: The maximum number of search results to retrieve
                           (No more than 5 each time, so make your query specific and detailed).

    Returns:
        string containing the paper contents that are relevant to the request submitted by you.
    """
    cfg = tool_context.state["cfg"]
    models_paper_cfg = cfg.get("models_paper", None)
    result = rag(request, models_paper_cfg, number_of_results)
    return {"result": result}