import asyncio
import json
import re
import traceback as tb
from typing import AsyncGenerator

import pyomo.environ as pyo
from langchain_experimental.utilities import PythonREPL
from loguru import logger
from pydantic import PrivateAttr

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS, GENERATOR_OUTPUT
from optichat.sub_agents.generator.prompt import get_coder_prompt
from optichat.config.llm_cfg import GPT_5_CODEX, GPT_5_CODEX_MAX_TOKENS

# Raw model ID for the OpenAI responses API (strip LiteLLM "openai/" prefix)
_CODEX_MODEL_ID = GPT_5_CODEX.removeprefix("openai/")  # "gpt-5-codex"

_PYTHON_REPL_TOOL = {
    "type": "function",
    "name": "python_repl",
    "description": (
        "Execute Python code in the REPL. "
        "pyomo, models_dictionary, solve_model, modify_and_solve, and all other "
        "shortcut functions are pre-injected — do NOT import them. "
        "Returns stdout output from the execution."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python code to execute.",
            }
        },
        "required": ["code"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StateProxy:
    """Minimal tool_context proxy so generated code can access tool_context.state."""
    def __init__(self, state: dict):
        self.state = state


def _get_last_user_text(ctx: InvocationContext) -> str:
    """Return the most recent user-role text from session events (the expert's instruction)."""
    for event in reversed(list(ctx.session.events)):
        if event.content and event.content.role == "user":
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
    return ""


def _inject_state_into_repl(repl: PythonREPL, state: dict) -> None:
    """Inject pyomo objects, shortcut functions, and session state into the REPL."""
    import optichat.tools.shortcut_functions as shortcut_functions

    # Clean stale user variables from prior calls
    if hasattr(repl, "globals") and repl.globals:
        preserved = {
            "__builtins__", "pyo", "value", "Constraint", "ConstraintList",
            "Var", "Param", "Objective", "ConcreteModel", "Set", "Expression",
            "minimize", "maximize", "models_dictionary", "MODEL_VERSIONS",
            "tool_context",
        }
        for name in dir(shortcut_functions):
            if callable(getattr(shortcut_functions, name)) and not name.startswith("_"):
                preserved.add(name)
        stale = [k for k in list(repl.globals.keys()) if k not in preserved]
        for k in stale:
            del repl.globals[k]

    repl.globals["pyo"]            = pyo
    repl.globals["value"]          = pyo.value
    repl.globals["Constraint"]     = pyo.Constraint
    repl.globals["ConstraintList"] = pyo.ConstraintList
    repl.globals["Var"]            = pyo.Var
    repl.globals["Param"]          = pyo.Param
    repl.globals["Objective"]      = pyo.Objective
    repl.globals["ConcreteModel"]  = pyo.ConcreteModel
    repl.globals["Set"]            = pyo.Set
    repl.globals["Expression"]     = pyo.Expression
    repl.globals["minimize"]       = pyo.minimize
    repl.globals["maximize"]       = pyo.maximize

    for name in dir(shortcut_functions):
        item = getattr(shortcut_functions, name)
        if callable(item) and not name.startswith("_"):
            repl.globals[name] = item

    models_dictionary = state.get(MODELS_DICTIONARY, {}).copy()
    repl.globals[MODELS_DICTIONARY.lower()] = models_dictionary
    repl.globals["MODEL_VERSIONS"]          = state.get(MODEL_VERSIONS, [])
    repl.globals["tool_context"]            = _StateProxy(state)
    repl.locals = repl.globals


# ---------------------------------------------------------------------------
# CoderAgent
# ---------------------------------------------------------------------------

class CoderAgent(BaseAgent):
    """
    Single gpt-5-codex agent that writes and executes Pyomo code via the
    OpenAI responses API with python_repl registered as a native tool.

    Agentic loop:
      1. Codex receives the expert's structured instruction.
      2. Codex calls python_repl(code=...) to execute code.
      3. Stdout result is fed back; codex iterates if needed.
      4. Loop exits when codex stops calling tools.
    """

    _repl: PythonREPL = PrivateAttr(default_factory=PythonREPL)

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        import openai

        state = ctx.session.state

        user_instruction = _get_last_user_text(ctx)
        if not user_instruction:
            logger.warning("[CoderAgent] No user instruction found in session events")
            user_instruction = "Generate and execute Python/Pyomo code as instructed."
        logger.info(f"[CoderAgent] user_instruction received ({len(user_instruction)} chars):\n{user_instruction[:500]}")

        # Parse the MODEL: field from the expert's grammar instruction as a hint
        _model_match = re.search(r'(?i)^MODEL\s*:\s*(.+)$', user_instruction, flags=re.MULTILINE)
        model_hint = _model_match.group(1).strip() if _model_match else None

        from optichat.tools.callback_tool import _get_source_code_for_prompt
        model_source_code = _get_source_code_for_prompt(state, model_hint=model_hint)
        system_instruction = get_coder_prompt(model_source_code=model_source_code)

        _inject_state_into_repl(self._repl, state)

        client = openai.AsyncOpenAI()
        execution_log = []

        # Initial call
        response = await client.responses.create(
            model=_CODEX_MODEL_ID,
            instructions=system_instruction,
            input=user_instruction,
            tools=[_PYTHON_REPL_TOOL],
            reasoning={"effort": "low"},
            max_output_tokens=GPT_5_CODEX_MAX_TOKENS,
        )

        # Agentic tool-call loop
        max_iterations = 10
        for iteration in range(1, max_iterations + 1):
            tool_calls = [item for item in response.output if item.type == "function_call"]
            if not tool_calls:
                break

            tool_outputs = []
            for tc in tool_calls:
                args = json.loads(tc.arguments)
                code = args.get("code", "")
                logger.info(f"[CoderAgent] iter={iteration} executing {len(code)} chars…")

                try:
                    repl_result = str(await asyncio.wait_for(
                        asyncio.to_thread(self._repl.run, code),
                        timeout=180.0,
                    ))
                except asyncio.TimeoutError:
                    repl_result = "Execution timed out after 180 s."
                    logger.error(f"[CoderAgent] iter={iteration} timeout")
                except Exception as exc:
                    repl_result = f"Error: {type(exc).__name__}: {exc}\n{tb.format_exc()}"
                    logger.error(f"[CoderAgent] iter={iteration} exception: {exc}")

                execution_log.append(
                    f"--- code (iter {iteration}) ---\n{code}\n--- result ---\n{repl_result}"
                )
                logger.info(f"[CoderAgent] iter={iteration} result: {repl_result[:120]}")

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": repl_result,
                })

            response = await client.responses.create(
                model=_CODEX_MODEL_ID,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=[_PYTHON_REPL_TOOL],
                reasoning={"effort": "low"},
                max_output_tokens=GPT_5_CODEX_MAX_TOKENS,
            )

        # Collect final text from the last response
        final_text_parts = []
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if hasattr(part, "text") and part.text:
                        final_text_parts.append(part.text)

        combined = "\n\n".join(execution_log)
        if final_text_parts:
            combined += "\n\n--- final response ---\n" + "\n".join(final_text_parts)

        logger.info(f"[CoderAgent] Done — {len(combined)} chars, {iteration} iteration(s)")

        updated_models = self._repl.globals.get(MODELS_DICTIONARY.lower(), {})
        final_models = updated_models if updated_models else state.get(MODELS_DICTIONARY, {})

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=combined)],
            ),
            actions=EventActions(state_delta={
                GENERATOR_OUTPUT: combined,
                MODELS_DICTIONARY: final_models,
                MODEL_VERSIONS: list(final_models.keys()),
            }),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_generator_agent() -> CoderAgent:
    """
    Create the generator agent.

    Returns a single CoderAgent named 'generator_agent' that:
      - Calls the OpenAI responses API directly (bypasses LiteLLM/ADK routing)
      - Has python_repl registered as a native tool
      - Writes and executes Pyomo code in an agentic loop
      - Writes GENERATOR_OUTPUT and updated MODELS_DICTIONARY to session state
    """
    return CoderAgent(
        name="generator_agent",
        description=(
            "Pyomo code generator and executor. Receives a structured instruction "
            "from the expert agent, writes Python/Pyomo code, and executes it via "
            "python_repl in an agentic loop. Returns execution log as GENERATOR_OUTPUT."
        ),
    )
