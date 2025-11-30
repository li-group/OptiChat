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
                                       CFG, EXPERT_AGENT_USES, EXPERT_AGENT_MAX_USES, EXPERT_AGENT_START_TIME,
                                       MODELS_DICTIONARY, MODEL_VERSIONS, HISTORICAL_MODELS_METADATA,
                                       IS_MODELS_DICTIONARY_AVAILABLE,
                                       IS_MODELS_CODE_AVAILABLE, IS_MODELS_PAPER_AVAILABLE,
                                       SYNTHETIC_PAPER_GENERATED, NEED_SYNTHETIC_PAPER,
                                       MODEL_FOR_PAPER_GENERATION, USER_QUERY)
from optichat.tools.extract_tool import restore_model_object, save_model_object, extract_model_info, _solve_model
from optichat.tools.rag_tool import init_paper_rag, init_code_rag
from optichat.tools.metadata_store import (load_metadata, save_metadata, save_model_data,
                                           add_model_to_metadata)


def format_models_metadata_for_prompt(historical_metadata: dict) -> str:
    """
    Format model metadata into human-readable string for prompt display.

    Args:
        historical_metadata: Dictionary of model metadata (tree structure)

    Returns:
        Formatted string with one line per model showing key information

    Example output:
        - supply_chain_model_3: Base model: supply_chain_model_3 (optimal)
        - supply_chain_model_3__demand_3_1_plus10: What-if analysis: increased demand[3,0] by 10. Objective improved to 45.2. (optimal)
    """
    if not historical_metadata:
        return "    No models available"

    formatted_lines = []
    
    # Iterate through dates (sorted reverse)
    for date_key in sorted(historical_metadata.keys(), reverse=True):
        date_models = historical_metadata[date_key]
        
        # Iterate through base models
        for base_name, base_data in date_models.items():
            # Add base model
            description = base_data.get("description", "Base model")
            status = base_data.get("solution_status", "unknown")
            line = f"    - {base_name}: {description} ({status})"
            formatted_lines.append(line)
            
            # Add modified models
            modified_models = base_data.get("modified_models", {})
            for mod_name, mod_data in modified_models.items():
                mod_desc = mod_data.get("description", "No description")
                mod_status = mod_data.get("solution_status", "unknown")
                mod_line = f"    - {mod_name}: {mod_desc} ({mod_status})"
                formatted_lines.append(mod_line)

    return "\n".join(formatted_lines)


def initialize_session(callback_context: CallbackContext):
    if IS_SESSION_INITIALIZED not in callback_context.state:
        callback_context.state.update(PERSISTENT_STATES)  # which also set IS_SESSION_INITIALIZED as False
        callback_context.state.update(TEMPORARY_STATES)
    
    user_content = callback_context.user_content
    # Handle case where user_content.parts might be empty or text is missing
    user_query = ""
    if user_content.parts:
        user_query = user_content.parts[0].text or ""

    parts_wo_json = []
    cfg = {}
    json_found = False

    # Scan for JSON config
    for part in user_content.parts:
        if getattr(part, "inline_data", None) is not None and part.inline_data.mime_type == "application/json":
            if callback_context.state[IS_SESSION_INITIALIZED]:
                raise NotImplementedError("Session is already initialized, cannot re-initialize with a new cfg. Open a new session instead.")
            else:
                raw = part.inline_data.data
                cfg = json.loads(raw.decode("utf-8"))
                json_found = True
        else:
            parts_wo_json.append(part)

    # Perform initialization if not done yet (either with found JSON or empty config)
    if not callback_context.state[IS_SESSION_INITIALIZED]:
        cfg = _init_cfg(cfg)
        callback_context.state[CFG] = cfg
        models_dictionary, current_model_versions, historical_metadata = _init_models(cfg)
        
        # Available if we have loaded models OR historical metadata (lazy loading)
        is_model_dictionary_available = len(models_dictionary) > 0 or len(historical_metadata) > 0
        
        callback_context.state[MODELS_DICTIONARY] = models_dictionary
        # Only show current models (from config) to root agent, not historical ones
        callback_context.state[MODEL_VERSIONS] = current_model_versions
        # Store lightweight metadata for all models (for lazy loading)
        callback_context.state[HISTORICAL_MODELS_METADATA] = historical_metadata
        # Format metadata for prompt display
        formatted_metadata = format_models_metadata_for_prompt(historical_metadata)
        callback_context.state["MODELS_METADATA_FORMATTED"] = formatted_metadata
        callback_context.state[IS_MODELS_DICTIONARY_AVAILABLE] = is_model_dictionary_available

        # Initialize code RAG if provided
        is_models_code_available = _init_models_code(cfg)
        callback_context.state[IS_MODELS_CODE_AVAILABLE] = is_models_code_available

        # Check if models_paper is provided
        is_models_paper_available = _init_models_paper(cfg)

        # If no models_paper provided, check if we need to generate synthetic description
        if not is_models_paper_available and is_model_dictionary_available:
            # Get model name from actual loaded version (not config)
            # This ensures we use the version name extracted from the pickle file
            model_name = current_model_versions[0] if current_model_versions else cfg.get("model_name", "model")

            # Check for cached synthetic paper first
            from optichat.tools.metadata_store import get_cached_synthetic_paper
            cached_paper_path = get_cached_synthetic_paper(model_name)

            if cached_paper_path and os.path.exists(cached_paper_path):
                logger.info(f"Using cached synthetic paper: {cached_paper_path}")
                # Load existing synthetic paper
                try:
                    init_paper_rag([cached_paper_path], model_name)
                    is_models_paper_available = True
                    callback_context.state[SYNTHETIC_PAPER_GENERATED] = True
                    logger.info("✓ Cached synthetic paper loaded for paper_rag queries")
                except Exception as e:
                    logger.error(f"Failed to load cached synthetic paper: {e}")
            else:
                # No cached paper - set flag for root agent to generate it
                # Only if we actually have a current model to generate for
                if current_model_versions:
                    logger.info(f"No synthetic paper for {model_name}. Flagging for generation...")
                    callback_context.state[NEED_SYNTHETIC_PAPER] = True
                    callback_context.state[MODEL_FOR_PAPER_GENERATION] = model_name
                # is_models_paper_available stays False until generation completes

        callback_context.state[IS_MODELS_PAPER_AVAILABLE] = is_models_paper_available

        # Add note to prompt if synthetic paper was generated
        if callback_context.state.get(SYNTHETIC_PAPER_GENERATED, False):
            callback_context.state["SYNTHETIC_PAPER_NOTE"] = (
                "Note: A comprehensive model description was automatically generated "
                "during initialization since no research paper was provided."
            )
        else:
            callback_context.state["SYNTHETIC_PAPER_NOTE"] = ""

        callback_context.state[IS_SESSION_INITIALIZED] = True

    # replace user_content with parts without json part (if a part has json, it cannot be processed)
    callback_context.user_content.parts = parts_wo_json
    # reset temporary states for every query
    callback_context.state.update(TEMPORARY_STATES)
    callback_context.state[USER_QUERY] = user_query
    return None


def _init_models(cfg: dict):
    """
    Initialize models with new architecture:
    - Load lightweight metadata for all models
    - Only load current models (from config) into models_dictionary
    - Historical models loaded lazily on demand

    Returns:
        models_dictionary: Runtime cache with current models only
        current_model_versions: List of current model versions (shown to root agent)
        historical_metadata: Dict of all models metadata (for lazy loading)
    """
    # STEP 1: Load existing metadata (lightweight, always loaded)
    metadata = load_metadata()
    # In new structure, metadata IS the historical metadata (tree structure)
    historical_metadata = metadata

    if historical_metadata:
        logger.info(f"Loaded metadata with {len(historical_metadata)} date entries")
    else:
        logger.info("No existing metadata found, starting fresh")

    # STEP 2: Initialize current models from config
    models_dictionary = {}  # Runtime cache for loaded models
    current_model_versions = []  # Models from config (shown to root agent)

    if "models" in cfg:
        is_solved = cfg["models"].get("is_solved", False)
        is_lp = cfg["models"].get("is_lp", False)

        for resource_path in cfg["models"].get("local_resources", []):
            # Load and solve model
            model, version = restore_model_object(resource_path)
            model, termination_condition = _solve_model(model, is_lp=is_lp, is_solved=is_solved)

            # Extract model information
            info = extract_model_info(model, termination_condition=termination_condition)
            local_path_to_object = save_model_object(model, version)
            info.update({"local_path_to_object": local_path_to_object})

            # Save full model data to individual file
            save_model_data(version, info)

            # Add to runtime cache
            models_dictionary[version] = info
            current_model_versions.append(version)

            # Add/update metadata (base models get automatic description)
            metadata = add_model_to_metadata(
                metadata=metadata,
                version_name=version,
                model_info=info,
                base_model=None,
                description=None  # Will auto-generate "Base model: {version}"
            )

            # Update historical_metadata dict
            historical_metadata = metadata

        logger.info(f"Initialized {len(models_dictionary)} model(s) from config")

    # STEP 3: Save updated metadata
    if metadata:
        save_metadata(metadata)
        logger.info(f"Saved metadata")

    logger.info(f"Current models (shown to root agent): {current_model_versions}")
    
    return models_dictionary, current_model_versions, historical_metadata


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


def _read_models_code_files(cfg: dict) -> Optional[str]:
    """
    Read and concatenate source code from models_code paths in config.

    Args:
        cfg: Configuration dictionary

    Returns:
        Concatenated source code string, or None if no code available
    """
    if "models_code" not in cfg:
        return None

    code_parts = []
    paths = cfg["models_code"].get("local_resources", [])

    for path in paths:
        try:
            # Handle wildcard paths (directories)
            if path.endswith("*"):
                # Expand to all .py files in directory
                import glob as glob_module
                py_files = glob_module.glob(path.replace("*", "*.py"))
                for py_file in py_files:
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            code_parts.append(f"# File: {py_file}\n{f.read()}")
                    except Exception as e:
                        logger.warning(f"Could not read code file {py_file}: {e}")
            else:
                # Single .py file
                with open(path, 'r', encoding='utf-8') as f:
                    code_parts.append(f"# File: {path}\n{f.read()}")

        except Exception as e:
            logger.warning(f"Could not read code from {path}: {e}")

    if not code_parts:
        return None

    return "\n\n".join(code_parts)


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
    expert_agent_uses = callback_context.state.get(EXPERT_AGENT_USES, 0)
    expert_agent_max_uses = callback_context.state.get(EXPERT_AGENT_MAX_USES, 5)

    if expert_agent_uses >= expert_agent_max_uses:
        return types.Content(
            parts=[types.Part(text=f"[system message]: Expert agent has been used {expert_agent_uses} times. "
                                   f"The limit is {expert_agent_max_uses} times per user query. "
                                   f"Please explain the results gathered so far to the user.")],
            role="model"
        )
    else:
        callback_context.state[EXPERT_AGENT_USES] = expert_agent_uses + 1
        # Only set start time on first use
        if expert_agent_uses == 0:
            callback_context.state[EXPERT_AGENT_START_TIME] = time.time()
        return None


def check_expert_agent_runtime(callback_context: CallbackContext):
    expert_agent_uses = callback_context.state.get(EXPERT_AGENT_USES, 0)
    if expert_agent_uses > 0:
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
    if isinstance(tool_response, dict):
        result = tool_response.get("result", "")
    else:
        result = str(tool_response)

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
                # return a truncated tool_response dictionary or string
                if isinstance(tool_response, dict):
                    truncated_tool_response = deepcopy(tool_response)
                    truncated_tool_response["result"] = truncated_result
                    return truncated_tool_response
                else:
                    return truncated_result
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {e}.")
    else:
        logger.debug(f"max tokens key '{max_tokens_key}' not found in the state. Skipping tool response check.")
    logger.info(f"'{tool_name.upper()}' execution result: {result[:show_first_n_chars]}"
                f"\n... (showing only the first {show_first_n_chars} characters)")
    return None  # Return None to indicate no modification to tool_response


def handle_illustrator_response(tool: BaseTool,
                                args: Dict[str, Any],
                                tool_context: ToolContext,
                                tool_response: Dict):
    """
    Handle response from illustrator_agent tool.

    When illustrator completes, save the generated description as a synthetic paper
    and initialize paper_rag with it.

    Args:
        tool: The tool that was executed (should be illustrator_agent)
        args: Arguments passed to the tool
        tool_context: Tool context with state
        tool_response: Response from the tool containing the generated description
    """
    # Only process if this is the illustrator_agent
    if tool.name != "illustrator_agent":
        return None

    logger.info("[ILLUSTRATOR_CALLBACK] Processing illustrator agent response...")

    try:
        # Extract the generated description
        if isinstance(tool_response, dict):
            result = tool_response.get("result", "")
        else:
            result = str(tool_response)
        if not result or len(result.strip()) == 0:
            logger.warning("[ILLUSTRATOR_CALLBACK] Illustrator returned empty description")
            return None

        # Get model name from state
        model_name = tool_context.state.get(MODEL_FOR_PAPER_GENERATION, "unknown_model")
        logger.info(f"[ILLUSTRATOR_CALLBACK] Saving description for model: {model_name}")
        logger.info(f"[ILLUSTRATOR_CALLBACK] Description length: {len(result)} characters")

        # Save as synthetic paper
        from optichat.tools.illustrator_tool import save_synthetic_paper
        from optichat.tools.metadata_store import save_synthetic_paper_path

        paper_path = save_synthetic_paper(description=result, model_name=model_name)

        # Cache path in metadata
        save_synthetic_paper_path(model_name, paper_path)

        # Initialize paper_rag with the synthetic paper
        init_paper_rag([paper_path], model_name)

        # Update state flags
        tool_context.state[NEED_SYNTHETIC_PAPER] = False
        tool_context.state[SYNTHETIC_PAPER_GENERATED] = True
        tool_context.state[IS_MODELS_PAPER_AVAILABLE] = True
        tool_context.state["SYNTHETIC_PAPER_NOTE"] = (
            "Note: A comprehensive model description was automatically generated "
            "during initialization since no research paper was provided."
        )

        logger.info("[ILLUSTRATOR_CALLBACK] ✓ Synthetic paper saved and initialized successfully")

    except Exception as e:
        logger.error(f"[ILLUSTRATOR_CALLBACK] Failed to process illustrator response: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return None  # Return None to indicate no modification to tool_response

