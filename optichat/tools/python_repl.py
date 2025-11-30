from typing import List, Dict, Any
from typing import Optional
from loguru import logger
from langchain_experimental.utilities import PythonREPL
# from langchain_core.tools import Tool
from google.adk.tools import ToolContext
from optichat.config.constants import *

python_repl = PythonREPL()
# python_repl_tool = Tool(
#     name="python_repl",
#     description="A Python shell. Use this to execute python commands. Input should be a valid python snippet",
#     func=python_repl.run,
# )


def python_repl_func(code_snippet: str, tool_context: ToolContext) -> Dict[str, str]:
    """
    Execute Python code and return the result.
    ONLY interact with <models> in code_snippet.

    Args:
        code_snippet (str): Python code to execute

    Returns:
        Dict[str, str]: a single key named "result" and its value as the execution result
    """
    import optichat.tools.shortcut_functions as shortcut_functions
    for name in dir(shortcut_functions):
        item = getattr(shortcut_functions, name)
        if callable(item) and not name.startswith("_"):
            python_repl.globals[name] = item
            logger.info(f"Injected shortcut function: {name} into REPL")
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    python_repl.globals[MODELS_DICTIONARY.lower()] = models_dictionary
    logger.info(f"Injected models_dictionary into REPL")

    # Inject tool_context so solve_model() can access USER_QUERY
    python_repl.globals['tool_context'] = tool_context
    logger.info(f"Injected tool_context into REPL")

    # execute the code snippet
    logger.info(f"Python code to execute:\n{code_snippet}")
    result = str(python_repl.run(code_snippet))
    # update MODELS_DICTIONARY if a model was modified and solved in REPL
    tool_context.state[MODELS_DICTIONARY] = models_dictionary
    tool_context.state[MODEL_VERSIONS] = list(models_dictionary.keys())
    return {"result": result}




