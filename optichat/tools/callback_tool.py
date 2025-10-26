import time
import json
import os
import glob
from xml.parsers.expat import model
import tiktoken
from loguru import logger
from typing import Dict, Any, List
from typing import Optional
from copy import deepcopy
from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.models import LlmResponse, LlmRequest
from optichat.config.constants import (IS_SESSION_INITIALIZED, PERSISTENT_STATES, TEMPORARY_STATES,
                                       CFG, IS_EXPERT_AGENT_USED, EXPERT_AGENT_START_TIME,
                                       MODELS_DICTIONARY, MODEL_VERSIONS, IS_MODELS_DICTIONARY_AVAILABLE,
                                       IS_MODELS_CODE_AVAILABLE, IS_MODELS_PAPER_AVAILABLE, USER_QUERY)
from optichat.tools.extract_tool import restore_model_object, save_model_object, extract_model_info, _solve_model
from optichat.tools.rag_tool import init_paper_rag, init_code_rag


def initialize_session(callback_context: CallbackContext):
    if IS_SESSION_INITIALIZED not in callback_context.state:
        callback_context.state.update(PERSISTENT_STATES)  # which also set IS_SESSION_INITIALIZED as False
        callback_context.state.update(TEMPORARY_STATES)
    user_content = callback_context.user_content
    user_query = callback_context.user_content.parts[0].text
    parts_wo_json = []
    for part in user_content.parts:
        if getattr(part, "inline_data", None) is not None:
            if part.inline_data.mime_type == "application/json":
                if callback_context.state[IS_SESSION_INITIALIZED]:
                    raise NotImplementedError("Session is already initialized, cannot re-initialize with a new cfg. Open a new session instead.")
                else:
                    raw = part.inline_data.data
                    cfg = json.loads(raw.decode("utf-8"))
                    cfg = _init_cfg(cfg)
                    callback_context.state[CFG] = cfg
                    models_dictionary, model_versions = _init_models(cfg)
                    is_model_dictionary_available = len(models_dictionary) > 0
                    callback_context.state[MODELS_DICTIONARY] = models_dictionary
                    callback_context.state[MODEL_VERSIONS] = model_versions
                    callback_context.state[IS_MODELS_DICTIONARY_AVAILABLE] = is_model_dictionary_available
                    is_models_code_available = _init_models_code(cfg)
                    callback_context.state[IS_MODELS_CODE_AVAILABLE] = is_models_code_available
                    is_models_paper_available = _init_models_paper(cfg)
                    callback_context.state[IS_MODELS_PAPER_AVAILABLE] = is_models_paper_available
                    callback_context.state[IS_SESSION_INITIALIZED] = True
            else:
                parts_wo_json.append(part)
        else:
            parts_wo_json.append(part)
    # replace user_content with parts without json part (if a part has json, it cannot be processed)
    callback_context.user_content.parts = parts_wo_json
    # reset temporary states for every query
    callback_context.state.update(TEMPORARY_STATES)
    callback_context.state[USER_QUERY] = user_query
    return None


def _init_models(cfg: dict):
    models_dictionary = {}
    model_versions = []
    if "models" in cfg:
        is_solved = cfg["models"].get("is_solved", False)
        is_lp = cfg["models"].get("is_lp", False)
        for resource_path in cfg["models"].get("local_resources", []):
            model, version = restore_model_object(resource_path)
            model, termination_condition = _solve_model(model, is_lp=is_lp, is_solved=is_solved)
            info = extract_model_info(model, termination_condition=termination_condition)
            local_path_to_object = save_model_object(model, version)
            info.update({"local_path_to_object": local_path_to_object,})
            models_dictionary.update({version: info})
            model_versions.append(version)
    return models_dictionary, model_versions


def _init_models_code(cfg: dict):
    if "models_code" in cfg:
        paths = cfg["models_code"].get("local_resources", [])
        model_name = cfg.get("model_name", "default_model")
        # TODO: temporary solution to use GenericLoader in rag_tool.py, which only supports one path
        init_code_rag(paths[0], model_name)
        is_models_code_available = True
    else:
        logger.debug("No 'models_code' in cfg")
        is_models_code_available = False
    return is_models_code_available


def _init_models_paper(cfg: dict):
    if "models_paper" in cfg:
        paths = cfg["models_paper"].get("local_resources", [])
        model_name = cfg.get("model_name", "default_model")
        init_paper_rag(paths, model_name)
        is_models_paper_available = True 
    else:
        logger.debug("No 'models_paper' in cfg")
        is_models_paper_available = False
    return is_models_paper_available


def _init_cfg(cfg: dict):
    cfg_out = deepcopy(cfg)
    
    required_sections = ["models", "models_code", "models_paper"]
    extension_filters = {
        "models": [".pkl"],
        "models_code": [".py"],
        "models_paper": [".txt", ".pdf"]
    }

    for section_key, section_cfg in cfg.items():
        if section_key in required_sections:
            local_resources = section_cfg.get("local_resources", [])
            allowed_extensions = extension_filters.get(section_key, None)
            # TODO: temporary solution to use init_code_rag() in rag_tool.py
            if section_key == "models_code":
                logger.warning(("For 'models_code', ONLY one path is supported for now, "
                    "which must be a .py file or a wildcard path to indicate a folder. "
                    "No expand_resources() is performed for 'models_code' in cfg."))
                assert len(local_resources) == 1, "ONLY one path for models_code is supported for now."
                assert local_resources[0].endswith(".py") or local_resources[0].endswith("*"), "models_code path must be a .py file or a wildcard path."
            else:
                expanded_resources = _expand_resources(local_resources, allowed_extensions)
                cfg_out[section_key]["local_resources"] = expanded_resources
    return cfg_out


def _expand_resources(local_resources: List[str], allowed_extensions: Optional[List[str]]):
    """
    Expand wildcard * patterns with allowed extensions filtering
    """
    expanded_resources = []
    for resource_path in local_resources:
        if "*" in resource_path:
            matches = glob.glob(resource_path)
        else:
            matches = [resource_path]
        if not matches:
            raise FileNotFoundError(f"No files matched: {resource_path}")
        if allowed_extensions:
            filtered_matches = [
                match for match in matches 
                    if any(match.lower().endswith(ext) for ext in allowed_extensions)
                    ]
        else:
            filtered_matches = matches
        expanded_resources.extend(sorted(filtered_matches))
    return expanded_resources


def check_is_expert_agent_used(callback_context: CallbackContext):
    is_expert_agent_used = callback_context.state.get(IS_EXPERT_AGENT_USED, None)
    if is_expert_agent_used is None:
        raise ValueError("check_is_expert_agent_used: IS_EXPERT_AGENT_USED is not set in the state.")

    if is_expert_agent_used:
        return types.Content(
            parts=[types.Part(text=f"[system message]: Expert agent has already been used. "
                                   f"Expert agent can ONLY be used once per user query. "
                                   f"Explain the last response from expert agent to the user first. ")],
            role="model"
        )
    else:
        callback_context.state[IS_EXPERT_AGENT_USED] = True
        callback_context.state[EXPERT_AGENT_START_TIME] = time.time()
        return None


def check_expert_agent_runtime(callback_context: CallbackContext):
    is_expert_agent_used = callback_context.state.get(IS_EXPERT_AGENT_USED)
    if is_expert_agent_used:
        start_time = callback_context.state.get(EXPERT_AGENT_START_TIME)
        if start_time:
            elapsed_time = time.time() - start_time
            logger.debug(f"*** Expert Agent Runtime: {elapsed_time:.2f} s "
                         f"({elapsed_time/60:.2f} min) ***")
    return None


def check_llm_request(callback_context: CallbackContext, llm_request: LlmRequest):
    agent_name = callback_context.agent_name
    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    original_text = original_instruction.parts[0].text or ""
    logger.info((f"[Callback] Inspecting LLM request from '{agent_name}': "
                 f"{original_text}"))
    return None


def check_llm_response(callback_context: CallbackContext, llm_response: LlmResponse):
    agent_name = callback_context.agent_name
    if llm_response.content and llm_response.content.parts:
        if llm_response.content.parts[0].text:
            original_text = llm_response.content.parts[0].text
            logger.info((f"[Callback] Inspecting LLM response from '{agent_name}': "
                         f"{original_text}"))
        elif llm_response.content.parts[0].function_call:
            logger.info((f"[Callback] Inspecting LLM function call from '{agent_name}': "
                         f"{llm_response.content.parts[0].function_call.name}"))
        else:
            logger.info("[Callback] Inspected LLM response: No text content found.")
    elif llm_response.error_message:
        logger.error((f"[Callback] Inspected LLM response: "
                      f"Contains error '{llm_response.error_message}'. "))
    else:
        logger.warning("[Callback] Inspected LLM response: Empty LlmResponse.")
    return None


def check_tool_usage(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    agent_name = tool_context.agent_name
    tool_name = tool.name

    usage_key = f"{agent_name.upper()}_{tool_name.upper()}_USES"
    if usage_key in tool_context.state:
        uses_left = tool_context.state[usage_key]
        if uses_left <= 0:
            logger.debug(f"Usage key '{usage_key}' has no remaining uses.")
            return {"result": f"\n[system message]: **WARNING** '{tool_name}' tool cannot be used anymore! "}
        else:
            tool_context.state[usage_key] -= 1
            logger.debug(f"Usage key '{usage_key}' decremented. Remaining uses: {tool_context.state[usage_key]}")
    else:
        logger.debug(f"Usage key '{usage_key}' not found in the state. Skipping tool usage check.")
    return None


def check_tool_response(tool: BaseTool,
                        args: Dict[str, Any],
                        tool_context: ToolContext,
                        tool_response: Dict):
    show_first_n_chars = 500
    agent_name = tool_context.agent_name
    tool_name = tool.name
    # AgentTool may return str instead of Dict as tool_response
    result = tool_response.get("result", "")
    max_tokens_key = f"{agent_name.upper()}_{tool_name.upper()}_MAX_TOKENS"
    if max_tokens_key in tool_context.state:
        max_tokens = tool_context.state[max_tokens_key]
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(result)
            token_count = len(tokens)
            if token_count > max_tokens:
                # truncate the result to max_tokens
                truncated_tokens = tokens[:max_tokens]
                truncated_result = encoding.decode(truncated_tokens)
                truncated_result += ("... \n[system message]: **WARNING** "
                                     "Execution result was truncated due to token limit.")
                logger.warning((f"'{tool_name.upper()}' execution (truncated) result: {truncated_result[:show_first_n_chars]}"
                                "\n... (showing only the first {show_first_n_chars} characters)"))
                # return a truncated tool_response dictionary
                truncated_tool_response = deepcopy(tool_response)
                truncated_tool_response["result"] = truncated_result
                return truncated_tool_response
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {e}.")
    else:
        logger.debug(f"max tokens key '{max_tokens_key}' not found in the state. Skipping tool response check.")
    logger.info(f"'{tool_name.upper()}' execution result: {result[:show_first_n_chars]}"
                f"\n... (showing only the first {show_first_n_chars} characters)")
    return None  # Return None to indicate no modification to tool_response

