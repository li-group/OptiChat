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
    import pyomo.environ as pyo

    # Clean up user-defined variables from previous calls to prevent namespace pollution
    # Keep only: builtins, modules, and injected functions/state
    if hasattr(python_repl, 'globals') and python_repl.globals:
        # List of keys to preserve (injected functions, modules, pyomo objects)
        preserved_keys = {
            '__builtins__', 'pyo', 'value', 'Constraint', 'ConstraintList', 'Var',
            'Param', 'Objective', 'ConcreteModel', 'Set', 'Expression', 'minimize',
            'maximize', 'models_dictionary', 'MODEL_VERSIONS', 'tool_context'
        }
        # Add shortcut function names to preserved keys
        for name in dir(shortcut_functions):
            if callable(getattr(shortcut_functions, name)) and not name.startswith("_"):
                preserved_keys.add(name)

        # Remove user-defined variables from previous calls
        keys_to_remove = [k for k in list(python_repl.globals.keys()) if k not in preserved_keys]
        for key in keys_to_remove:
            del python_repl.globals[key]

        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} user variables from previous REPL calls")

    # Pre-inject pyomo module and commonly used objects to handle various import styles
    python_repl.globals['pyo'] = pyo
    python_repl.globals['value'] = pyo.value
    python_repl.globals['Constraint'] = pyo.Constraint
    python_repl.globals['ConstraintList'] = pyo.ConstraintList
    python_repl.globals['Var'] = pyo.Var
    python_repl.globals['Param'] = pyo.Param
    python_repl.globals['Objective'] = pyo.Objective
    python_repl.globals['ConcreteModel'] = pyo.ConcreteModel
    python_repl.globals['Set'] = pyo.Set
    python_repl.globals['Expression'] = pyo.Expression
    python_repl.globals['minimize'] = pyo.minimize
    python_repl.globals['maximize'] = pyo.maximize
    logger.debug(f"Injected pyomo.environ module and common objects into REPL")

    for name in dir(shortcut_functions):
        item = getattr(shortcut_functions, name)
        if callable(item) and not name.startswith("_"):
            python_repl.globals[name] = item
            logger.debug(f"Injected shortcut function: {name} into REPL")
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    python_repl.globals[MODELS_DICTIONARY.lower()] = models_dictionary
    logger.debug(f"Injected models_dictionary into REPL")

    model_versions = tool_context.state.get(MODEL_VERSIONS, [])
    python_repl.globals['MODEL_VERSIONS'] = model_versions
    logger.debug(f"Injected MODEL_VERSIONS into REPL: {model_versions}")

    # Inject tool_context so solve_model() can access USER_QUERY
    python_repl.globals['tool_context'] = tool_context
    logger.debug(f"Injected tool_context into REPL")

    # This ensures user-defined variables persist in the same namespace as injected functions, preventing NameError when accessing variables later
    python_repl.locals = python_repl.globals
    logger.debug(f"Unified REPL locals and globals namespaces")

    # execute the code snippet
    logger.info(f"Python code to execute:\n{code_snippet}")

    # Wrap execution to capture detailed tracebacks on error
    try:
        result = str(python_repl.run(code_snippet))

        # Check if result indicates an error (LangChain REPL returns error strings)
        if "Error" in result or "Traceback" in result:
            logger.error(f"REPL execution error detected:\n{result}")
            # Log current namespace state for debugging
            user_vars = {k: type(v).__name__ for k, v in python_repl.globals.items()
                        if not k.startswith('_') and k not in ['__builtins__']}
            logger.error(f"REPL globals at error time ({len(user_vars)} variables): {list(user_vars.keys())}")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"REPL execution exception:\n{tb_str}")
        result = f"Exception during execution: {type(e).__name__}: {e}\n\nTraceback:\n{tb_str}"

    # Read back from REPL globals to capture any updates made during execution
    updated_models_dictionary = python_repl.globals.get(MODELS_DICTIONARY.lower(), models_dictionary)
    tool_context.state[MODELS_DICTIONARY] = updated_models_dictionary
    tool_context.state[MODEL_VERSIONS] = list(updated_models_dictionary.keys())
    return {"result": result}




