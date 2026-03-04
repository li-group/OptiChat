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
                                       MODEL_FOR_PAPER_GENERATION, USER_QUERY, EXPERT_AGENT_PYTHON_REPL_FUNC_USES,
                                       HAS_INFEASIBILITY_DIAGNOSIS, INFEASIBILITY_DIAGNOSIS,
                                       LLM_CALL_COUNTER, TOOL_CALL_START_TIME, LLM_CALL_START_TIME,
                                       MODEL_COMPONENTS_CACHE, CURRENT_ANALYSIS_TYPE,
                                       OUTPUT_KEY_EXPERT_AGENT, OUTPUT_KEY_ROOT_AGENT)
from optichat.tools import timing_tracker
from optichat.tools.extract_tool import restore_model_object, save_model_object, extract_model_info, _solve_model
from optichat.tools.rag_tool import init_paper_rag, init_code_rag
from optichat.tools.metadata_store import (load_metadata, save_metadata, save_model_data,
                                           add_model_to_metadata)
from optichat.sub_agents.expert.prompt import get_expert_agent_prompt
import re


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


def initialize_session_and_start_root_agent(callback_context: CallbackContext):
    """
    Wrapper callback for root agent before_agent_callback.
    Calls initialize_session first, then starts root agent timing.

    Args:
        callback_context: Callback context with state

    Returns:
        None or Content (if initialization returns early)
    """
    # Call initialize_session
    result = initialize_session(callback_context)
    if result is not None:
        return result

    # Start root agent timing
    timing_tracker.record_agent_start("root_agent")
    return None


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

    # Clear model components cache for new query
    callback_context.state[MODEL_COMPONENTS_CACHE] = {}
    logger.info("🔄 Model components cache cleared for new query")

    # Reset timing tracker for new query
    timing_tracker.reset_tracker()
    logger.info("⏱️  Timing tracker reset for new query")

    return None


def _find_and_copy_source_file(version_name: str, cfg: dict) -> Optional[str]:
    """
    Find the .py source file for this model and copy it to tmp folder.
    
    Strategy:
    1. Check models_code paths in config for {version}.py
    2. If found, copy to tmp/model_objects/{version}.py
    3. Return the tmp path
    
    Args:
        version_name: Model version name (e.g., 'RTN_inf_1')
        cfg: Configuration dictionary
    
    Returns:
        Path to copied .py file in tmp, or None if not found
    """
    import shutil
    
    if "models_code" not in cfg:
        logger.debug(f"No models_code in config, skipping source file copy for {version_name}")
        return None
    
    # Look for source file in config paths
    code_paths = cfg["models_code"].get("local_resources", [])
    source_file = None
    
    for path in code_paths:
        if "*" in path:
            # Handle wildcard paths (e.g., "Infeas/*.py")
            matches = glob.glob(path.replace("*", f"{version_name}"))
            if matches:
                source_file = matches[0]
                logger.debug(f"Found source file via wildcard: {source_file}")
                break
        elif version_name in path and path.endswith(".py"):
            # Direct path match
            source_file = path
            logger.debug(f"Found source file via direct match: {source_file}")
            break
    
    if not source_file or not os.path.exists(source_file):
        logger.warning(f"No .py source file found for {version_name}")
        return None
    
    # Copy to tmp/model_objects
    tmp_dir = os.path.join(os.getcwd(), "tmp/model_objects")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create tmp/inf_detail directory for infeasibility diagnosis reports
    inf_detail_dir = os.path.join(os.getcwd(), "tmp/inf_detail")
    os.makedirs(inf_detail_dir, exist_ok=True)

    # Create tmp/robust directory for robustness analysis outputs
    robust_dir = os.path.join(os.getcwd(), "tmp/robust")
    os.makedirs(robust_dir, exist_ok=True)
    
    dest_path = os.path.join(tmp_dir, f"{version_name}.py")
    
    try:
        shutil.copy2(source_file, dest_path)
        logger.info(f"Copied source file: {source_file} -> {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Failed to copy source file {source_file}: {e}")
        return None

def _load_model_from_py(py_file_path: str, json_data: dict = None):
    """
    Load Pyomo model from .py file with optional JSON data injection.
    This enables the separated .py + .json format for model initialization.
    
    Args:
        py_file_path: Path to the .py file containing model definition
        json_data: Optional dictionary to inject as 'data' in module namespace
    
    Returns:
        Tuple of (model, version_name)
    """
    import importlib.util
    import sys

    # Create module from file.
    # IMPORTANT: do NOT register in sys.modules before pickling.
    # If cloudpickle sees the module in sys.modules it stores a lightweight
    # reference ("module: loaded_model, fn: supply_rule") instead of the actual
    # bytecode. That reference breaks across server restarts because sys.modules
    # is wiped. By keeping the module out of sys.modules, cloudpickle is forced
    # to serialize the full function bytecode into the .pkl, making it truly
    # self-contained and loadable in future sessions without needing the .py file.
    spec = importlib.util.spec_from_file_location("loaded_model", py_file_path)
    loaded_module = importlib.util.module_from_spec(spec)

    # Inject JSON data if provided (same as extractor.py::initial_loading)
    if json_data is not None:
        loaded_module.__dict__["data"] = json_data

    # Execute the module (works without sys.modules registration)
    spec.loader.exec_module(loaded_module)

    # Extract model and version name
    model = loaded_module.model
    version = os.path.splitext(os.path.basename(py_file_path))[0]

    return model, version


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
            try:
                # Check if this is .pkl (backward compat) or .py (new format)
                if resource_path.endswith(".pkl"):
                    # OLD PATH: Load from pickle
                    logger.info(f"Loading model from pickle: {resource_path}")
                    model, version = restore_model_object(resource_path)
                elif resource_path.endswith(".py"):
                    # NEW PATH: Load from .py + .json
                    logger.info(f"Loading model from Python file: {resource_path}")
                    
                    # Load JSON data if provided
                    json_data = None
                    json_data_path = cfg["models"].get("json_data_path")
                    if json_data_path and os.path.exists(json_data_path):
                        logger.info(f"Loading JSON data from: {json_data_path}")
                        with open(json_data_path, 'r') as f:
                            json_data = json.load(f)
                    else:
                        logger.warning(f"No JSON data file found or specified for {resource_path}")
                    
                    # Load model from .py file with optional JSON data
                    model, version = _load_model_from_py(resource_path, json_data)
                else:
                    raise ValueError(f"Unsupported model file format: {resource_path}. Must be .pkl or .py")
                
                # Solve model (same for both formats)
                model, termination_condition = _solve_model(model, is_lp=is_lp, is_solved=is_solved)

                # Extract model information
                info = extract_model_info(model, termination_condition=termination_condition)
                local_path_to_object = save_model_object(model, version)
                info.update({"local_path_to_object": local_path_to_object})

                # NEW: Copy .py source file to tmp if it exists
                source_py_path = _find_and_copy_source_file(version, cfg)
                if source_py_path:
                    info.update({"source_file_path": source_py_path})

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
                    description=None,  # Will auto-generate "Base model: {version}"
                    source_file_path=source_py_path  # NEW: track .py file path
                )

                # Update historical_metadata dict
                historical_metadata = metadata
                
                logger.info(f"✓ Successfully loaded model: {version}")
                
            except Exception as e:
                logger.error(f"Failed to load model from {resource_path}: {e}")
                logger.exception("Full traceback:")
                # Continue to next model instead of failing entire initialization
                continue

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
        "models": [".pkl", ".py"],  # Support both .pkl (old) and .py (new separated format)
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
            # Record agent start in timing tracker
            timing_tracker.record_agent_start("expert_agent")

        # Read analysis type already set by root's after_model_callback (check_llm_response)
        analysis_type = callback_context.state.get(CURRENT_ANALYSIS_TYPE, "RETRIEVAL")
        logger.info(f"Expert agent invoked with analysis type: {analysis_type}")
        return None


def check_expert_agent_runtime(callback_context: CallbackContext):
    expert_agent_uses = callback_context.state.get(EXPERT_AGENT_USES, 0)
    if expert_agent_uses > 0:
        # Record agent end in timing tracker
        timing_tracker.record_agent_end("expert_agent")

        # Legacy timing (kept for backwards compatibility)
        start_time = callback_context.state.get(EXPERT_AGENT_START_TIME)
        if start_time:
            elapsed_time = time.time() - start_time
            logger.debug(f"*** Expert Agent Runtime: {elapsed_time:.2f} s "
                         f"({elapsed_time/60:.2f} min) ***")
    return None


def check_llm_request(callback_context: CallbackContext, llm_request: LlmRequest):
    agent_name = callback_context.agent_name

    # Start timing for LLM call
    callback_context.state[LLM_CALL_START_TIME] = time.time()

    original_instruction = llm_request.config.system_instruction or types.Content(role="system", parts=[])
    # Ensure system_instruction is Content and parts list exists
    if not isinstance(original_instruction, types.Content):
         # Handle case where it might be a string (though config expects Content)
         original_instruction = types.Content(role="system", parts=[types.Part(text=str(original_instruction))])
    if not original_instruction.parts:
        original_instruction.parts.append(types.Part(text="")) # Add an empty part if none exist

    original_text = original_instruction.parts[0].text or ""
    show_first_n_chars = 100
    logger.info((f"[Callback] Inspecting LLM request from '{agent_name}': "
                 f"{original_text[:show_first_n_chars]}"
                 f"\n... (showing only the first {show_first_n_chars} characters)"))
                   
    # Dynamic Prompt Injection for Expert Agent
    if agent_name == "expert_agent":
        analysis_type = callback_context.state.get(CURRENT_ANALYSIS_TYPE, "RETRIEVAL")
        logger.info(f"Injecting Dynamic Prompt for {agent_name} with strategy: {analysis_type}")

        try:
            # Inject component index for ALL expert_agent queries
            model_description = _get_component_index_for_prompt(callback_context.state)

            # Inject diagnosis report for feasibility restoration queries
            diagnosis_report = None
            cached_model_name = None  # Track the model name for diagnosis status
            if analysis_type == "FEASIBILITY_RESTORATION":
                user_query = callback_context.state.get(USER_QUERY, "")
                models_dictionary = callback_context.state.get(MODELS_DICTIONARY, {})

                # Identify which model the user is asking about
                base_model_name = _extract_base_model_from_query(user_query, models_dictionary)

                if base_model_name:
                    logger.info(f"Identified base model for diagnosis report: {base_model_name}")
                    diagnosis_data = _load_infeasibility_diagnosis_report(base_model_name)
                    if diagnosis_data:
                        diagnosis_report = _format_diagnosis_for_prompt(diagnosis_data)
                        cached_model_name = base_model_name  # Store for prompt injection
                        logger.info("✓ Diagnosis report formatted and ready for injection")
                    else:
                        logger.warning(f"No diagnosis report found for {base_model_name}")
                else:
                    logger.warning("Could not identify base model from query for diagnosis report")

            new_prompt_text = get_expert_agent_prompt(
                prompt_version=1,
                analysis_type=analysis_type,
                diagnosis_report=diagnosis_report,
                cached_model_name=cached_model_name,
                model_description=model_description
            )

            # Manually substitute state-based placeholders that ADK would normally
            # inject into the static instruction but won't re-process on our override.
            state = callback_context.state
            new_prompt_text = new_prompt_text.replace(
                "{IS_MODELS_DICTIONARY_AVAILABLE}",
                str(state.get(IS_MODELS_DICTIONARY_AVAILABLE, False))
            )
            new_prompt_text = new_prompt_text.replace(
                "{MODELS_METADATA_FORMATTED}",
                state.get("MODELS_METADATA_FORMATTED", "No models available")
            )
            new_prompt_text = new_prompt_text.replace(
                "{USER_QUERY}",
                state.get(USER_QUERY, "")
            )
            new_prompt_text = new_prompt_text.replace(
                "{EXPERT_AGENT_PYTHON_REPL_FUNC_USES}",
                str(state.get(EXPERT_AGENT_PYTHON_REPL_FUNC_USES, 3))
            )

            llm_request.config.system_instruction = new_prompt_text
            logger.info(f"[ExpertAgent] Full system prompt ({len(new_prompt_text)} chars):\n{new_prompt_text}")
        except Exception as e:
            logger.error(f"Failed to inject dynamic prompt: {e}")

    return None


def _generate_component_index(version: str, models_dictionary: dict) -> str:
    """
    Generate a compact component families index for a model version.
    Groups indexed components by base name, shows type and doc string.
    No values or expressions — just enough to make precise GMC calls.
    """
    from collections import defaultdict

    if version not in models_dictionary:
        return ""

    model_data = models_dictionary[version]
    sol_status = model_data.get("obj", {}).get("sol_status", "unknown")
    obj_value = model_data.get("obj", {}).get("value", "unknown")

    # Group components by base name (strip index brackets)
    families = {}
    for comp_name, comp_data in model_data.items():
        if not isinstance(comp_data, dict):
            continue
        comp_type = comp_data.get("component_type", "")
        if not comp_type:
            continue
        # Extract base name: "demand[1,2]" → "demand"
        base_match = re.match(r'^([^\[]+)', comp_name)
        base_name = base_match.group(1).strip() if base_match else comp_name
        is_indexed = '[' in comp_name
        if base_name not in families:
            families[base_name] = {
                "type": comp_type,
                "doc": comp_data.get("doc", ""),
                "indexed": is_indexed
            }

    # Group by component type
    by_type = defaultdict(list)
    for name, info in sorted(families.items()):
        by_type[info["type"]].append((name, info))

    type_order = ["objective", "parameter", "variable", "constraint"]
    type_labels = {
        "objective": "Objective",
        "parameter": "Parameters",
        "variable": "Variables",
        "constraint": "Constraints"
    }

    lines = [f"MODEL COMPONENTS INDEX: {version} ({sol_status}, obj={obj_value})"]
    for t in type_order:
        if t not in by_type:
            continue
        lines.append(f"\n{type_labels[t]}:")
        for name, info in by_type[t]:
            idx_str = "[indexed]" if info["indexed"] else "[scalar]"
            doc_str = f" — {info['doc']}" if info.get("doc") else ""
            lines.append(f"  {name} {idx_str}{doc_str}")
    lines.append(
        "\nUse component names above as pattern filters in get_model_components "
        "to fetch specific current values (e.g., pattern='demand' to get all demand entries)."
    )
    return "\n".join(lines)


def _strip_paper_metadata(content: str) -> str:
    """Strip header metadata and footer disclaimer from a generated paper file.

    Removes everything up to and including the first '---' line (header block
    with timestamp / author / model name) and everything from the last '---'
    line onward (the auto-generated disclaimer footer).
    """
    lines = content.split('\n')

    # Find first '---' separator → marks end of header
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == '---':
            start_idx = i + 1
            break

    # Find last '---' separator → marks start of footer
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == '---':
            end_idx = i
            break

    return '\n'.join(lines[start_idx:end_idx]).strip()


def _get_component_index_for_prompt(state: dict) -> Optional[str]:
    """
    Return the component index / model description for the model being queried.

    Priority:
    1. Cache (state key COMPONENT_INDEX_{base_model})
    2. Generated paper file at tmp/model_objects/generated_papers/{base_model}_description.txt
    3. Fall back to _generate_component_index (programmatic summary)
    """
    user_query = state.get(USER_QUERY, "")
    models_dict = state.get(MODELS_DICTIONARY, {})

    base_model = _extract_base_model_from_query(user_query, models_dict)
    if not base_model:
        # models_dictionary may not be loaded yet (e.g. frontend model selection before init).
        # Scan the papers directory to find an available description file.
        papers_dir = os.path.join(os.getcwd(), "tmp", "model_objects", "generated_papers")
        paper_files = sorted(glob.glob(os.path.join(papers_dir, "*_description.txt")))
        if paper_files:
            # Pick the most recently modified paper
            paper_files.sort(key=os.path.getmtime, reverse=True)
            chosen = os.path.basename(paper_files[0])          # e.g. "recovery_description.txt"
            base_model = chosen[: chosen.rfind("_description.txt")]
            logger.info(f"Inferred base model from papers directory: {base_model}")
        else:
            logger.warning("Could not identify base model for component index injection")
            return None

    # 1. Check cache
    cache_key = f"COMPONENT_INDEX_{base_model}"
    if cache_key in state:
        logger.info(f"Using cached component index for {base_model}")
        return state[cache_key]

    # 2. Try generated paper file
    paper_path = os.path.join(os.getcwd(), "tmp", "model_objects", "generated_papers",
                              f"{base_model}_description.txt")
    if os.path.exists(paper_path):
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                raw = f.read()
            index_str = _strip_paper_metadata(raw)
            logger.info(f"Loaded component index from paper for {base_model} ({len(index_str)} chars)")
            state[cache_key] = index_str
            return index_str
        except Exception as e:
            logger.warning(f"Failed to read paper file for {base_model}: {e}")

    # 3. Fall back to programmatic generation
    index_str = _generate_component_index(base_model, models_dict)
    if not index_str:
        logger.warning(f"Could not generate component index for {base_model}")
        return None

    state[cache_key] = index_str
    logger.info(f"Generated component index for {base_model} ({len(index_str)} chars)")
    return index_str


def _get_source_code_for_prompt(state: dict, model_hint: str = None) -> Optional[str]:
    """
    Retrieve and format source code for the base model referenced in query.

    model_hint: optional model name (e.g. parsed from grammar instruction MODEL: field)
                used as fallback when USER_QUERY doesn't mention the model name.

    Returns formatted source code string or None if not available.
    """
    from optichat.tools.source_code_utils import (
        get_source_code_for_model,
        format_source_for_prompt
    )

    # 1. Identify base model
    user_query = state.get(USER_QUERY, "")
    models_dict = state.get(MODELS_DICTIONARY, {})
    historical_metadata = state.get(HISTORICAL_MODELS_METADATA, {})

    base_model = _extract_base_model_from_query(user_query, models_dict)
    if not base_model and model_hint:
        # Fall back to the model name from the grammar instruction (MODEL: field)
        base_model = _extract_base_model_from_query(model_hint, models_dict)
        if base_model:
            logger.info(f"Using model hint '{model_hint}' → resolved base model: {base_model}")
    if not base_model and model_hint:
        # models_dict may be empty (e.g. called before session state is fully populated).
        # Use model_hint directly as the model name — it comes from MODEL: <exact_key>
        # in the expert's grammar instruction, so it is reliable.
        base_model = model_hint.strip()
        logger.info(f"models_dict empty; using model_hint directly as base model: {base_model}")
    if not base_model:
        logger.warning("Could not identify base model for source code injection")
        return None
    
    # 2. Check cache
    cache_key = f"SOURCE_CODE_{base_model}"
    if cache_key in state:
        logger.info(f"Using cached source code for {base_model}")
        return state[cache_key]
    
    # 3. Retrieve source code
    source_code = get_source_code_for_model(base_model, models_dict, historical_metadata)
    if not source_code:
        logger.warning(f"No source code found for {base_model}")
        return None
    
    # 4. Get pkl path
    model_info = models_dict.get(base_model, {})
    pkl_path = model_info.get("local_path_to_object", "unknown")
    
    # 5. Format for prompt
    formatted_source = format_source_for_prompt(source_code, base_model, pkl_path, historical_metadata)
    
    # 6. Cache it
    state[cache_key] = formatted_source
    
    logger.info(f"Injecting {len(formatted_source)} chars of source code for {base_model}")
    return formatted_source


def _extract_base_model_from_query(user_query: str, models_dictionary: dict) -> Optional[str]:
    """
    Extract the base model name from user query.
    Prefer base models (no __) over modified versions.
    """
    # Remove analysis tag
    query_clean = re.sub(r'^\[(WHAT_IF|WHY_NOT|FEASIBILITY_RESTORATION)\]\s*', 
                         '', user_query, flags=re.IGNORECASE)
    
    # Look for explicit model mentions (prefer longer matches to avoid partial matches)
    for version in sorted(models_dictionary.keys(), key=lambda v: len(v), reverse=True):
        if version in query_clean and "__" not in version:
            return version
    
    # Default to first base model
    for version in models_dictionary.keys():
        if "__" not in version:
            return version

    return None


def _load_infeasibility_diagnosis_report(model_name: str) -> Optional[dict]:
    """
    Load infeasibility diagnosis report from tmp/inf_detail/{model_name}_inf_detail.json.

    Args:
        model_name: Base model name (e.g., 'diet_inf_1', 'aircraft_inf_1')

    Returns:
        Dictionary with diagnosis data, or None if file doesn't exist
    """
    report_path = os.path.join(os.getcwd(), "tmp/inf_detail", f"{model_name}_inf_detail.json")

    if not os.path.exists(report_path):
        logger.warning(f"No diagnosis report found at: {report_path}")
        return None

    try:
        with open(report_path, 'r') as f:
            diagnosis_data = json.load(f)
        logger.info(f"✓ Loaded diagnosis report for {model_name}")
        return diagnosis_data
    except Exception as e:
        logger.error(f"Failed to load diagnosis report from {report_path}: {e}")
        return None


def _format_diagnosis_for_prompt(diagnosis_data: dict) -> str:
    """
    Format the diagnosis report JSON into a readable prompt section.
    Only works with data from {model}_inf_detail.json.

    Args:
        diagnosis_data: Dictionary loaded from {model}_inf_detail.json

    Returns:
        Formatted string for prompt injection
    """
    if not diagnosis_data:
        return "(No prior diagnosis report available)"

    sections = []
    sections.append("## PRIOR INFEASIBILITY DIAGNOSIS REPORT")
    sections.append("")

    # Add model name
    if "model_name" in diagnosis_data:
        sections.append(f"**Model:** {diagnosis_data['model_name']}")

    # Add diagnosis summary
    if "diagnosis_summary" in diagnosis_data:
        sections.append(f"**Summary:** {diagnosis_data['diagnosis_summary']}")

    # Add conflicting constraints
    if "conflicting_constraints" in diagnosis_data:
        sections.append("\n**Conflicting Constraints:**")
        for constraint in diagnosis_data["conflicting_constraints"]:
            sections.append(f"  - {constraint}")

    # Add recommended relaxations
    if "recommended_relaxations" in diagnosis_data:
        sections.append("\n**Recommended Relaxations:**")
        for relaxation in diagnosis_data["recommended_relaxations"]:
            sections.append(f"  - {relaxation}")

    # Add any elastic analysis results
    if "elastic_analysis" in diagnosis_data:
        sections.append("\n**Elastic Analysis Results:**")
        elastic = diagnosis_data["elastic_analysis"]
        if isinstance(elastic, dict):
            for key, value in elastic.items():
                sections.append(f"  - {key}: {value}")
        else:
            sections.append(f"  {elastic}")

    sections.append("")
    sections.append("**IMPORTANT:** Use this diagnosis to guide your feasibility restoration approach.")
    sections.append("You should NOT call the infeasibility_diagnosis tool again unless the user explicitly requests a fresh diagnosis.")

    return "\n".join(sections)

def check_llm_response(callback_context: CallbackContext, llm_response: LlmResponse):
    agent_name = callback_context.agent_name

    # Calculate LLM call duration
    llm_start_time = callback_context.state.get(LLM_CALL_START_TIME)
    if llm_start_time:
        llm_duration = time.time() - llm_start_time

        # Increment LLM call counter for this agent
        llm_call_counters = callback_context.state.get(LLM_CALL_COUNTER, {})
        call_number = llm_call_counters.get(agent_name, 0) + 1
        llm_call_counters[agent_name] = call_number
        callback_context.state[LLM_CALL_COUNTER] = llm_call_counters

        # Record in timing tracker
        timing_tracker.record_llm_call(agent_name, llm_duration, call_number)

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

    # Strip any explanatory text from root's response when it's calling route_to_expert,
    # so the user never sees root's intermediate reasoning — only the expert's answer appears.
    if agent_name == "root_agent" and llm_response.content and llm_response.content.parts:
        has_route_call = any(
            p.function_call and p.function_call.name == "route_to_expert"
            for p in llm_response.content.parts
        )
        if has_route_call:
            function_call_parts = [p for p in llm_response.content.parts if p.function_call]
            stripped_response = LlmResponse(
                content=types.Content(role="model", parts=function_call_parts)
            )
            logger.info("[Root callback] Stripped text from root response — route_to_expert executes silently")
            return stripped_response

    return None


def route_to_expert(analysis_type: str, tool_context: ToolContext) -> str:
    """Classify the current query and immediately transfer control to expert_agent.

    Call this once for every non-[GENERAL] query instead of writing an
    ANALYSIS_TYPE marker and calling transfer_to_agent separately.

    Args:
        analysis_type: One of RETRIEVAL, SENSITIVITY, WHAT_IF, WHY_NOT,
                       FEASIBILITY_RESTORATION, ROBUSTNESS.
        tool_context: Injected by ADK — do not pass manually.

    Returns:
        Confirmation string (not shown to user; transfer fires immediately).
    """
    valid_types = {"RETRIEVAL", "SENSITIVITY", "WHAT_IF", "WHY_NOT",
                   "FEASIBILITY_RESTORATION", "ROBUSTNESS"}
    analysis_type = analysis_type.strip().upper()
    if analysis_type not in valid_types:
        return (f"Invalid analysis_type '{analysis_type}'. "
                f"Must be one of: {', '.join(sorted(valid_types))}")
    tool_context.state[CURRENT_ANALYSIS_TYPE] = analysis_type
    logger.info(f"[route_to_expert] CURRENT_ANALYSIS_TYPE set to {analysis_type}")
    # Trigger ADK transfer to expert_agent via the function response event actions
    tool_context.actions.transfer_to_agent = "expert_agent"
    return f"Routing to expert_agent with analysis_type={analysis_type}."


def check_tool_usage(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext):
    agent_name = tool_context.agent_name
    tool_name = tool.name

    # Start timing for tool call
    tool_context.state[TOOL_CALL_START_TIME] = time.time()

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
    show_first_n_chars = None
    agent_name = tool_context.agent_name
    tool_name = tool.name

    # Calculate tool call duration
    tool_start_time = tool_context.state.get(TOOL_CALL_START_TIME)
    if tool_start_time:
        tool_duration = time.time() - tool_start_time

        # Record in timing tracker
        # For AgentTool (expert_agent, illustrator_agent), this is delegation time
        if tool_name in ["expert_agent", "illustrator_agent"]:
            timing_tracker.record_delegation(agent_name, tool_name, tool_duration)
        else:
            timing_tracker.record_tool_call(agent_name, tool_name, tool_duration)

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
    
    # Log the result - only show truncation message if actually truncating
    if show_first_n_chars is not None:
        logger.info(f"'{tool_name.upper()}' execution result: {result[:show_first_n_chars]}"
                    f"\n... (showing only the first {show_first_n_chars} characters)")
    else:
        logger.info(f"'{tool_name.upper()}' execution result: {result}")
    
    return None  # Return None to indicate no modification to tool_response

def diagnose_if_infeasible(callback_context: CallbackContext):
    """
    Run infeasibility diagnosis before illustrator generates description.

    This callback is triggered when the illustrator agent starts processing.
    If the model is infeasible, it automatically runs the infeasibility diagnosis
    and stores the results in state for the illustrator to access.

    Args:
        callback_context: Callback context with state

    Returns:
        None (modifies state in-place)
    """
    from pyomo.opt import TerminationCondition

    # Record illustrator agent start
    timing_tracker.record_agent_start("illustrator_agent")

    # Get model information
    model_name = callback_context.state.get(MODEL_FOR_PAPER_GENERATION, "")
    models_dict = callback_context.state.get(MODELS_DICTIONARY, {})

    if not model_name or model_name not in models_dict:
        logger.warning("[DIAGNOSE_IF_INFEASIBLE] No model to diagnose or model not found")
        callback_context.state[HAS_INFEASIBILITY_DIAGNOSIS] = False
        return None

    # Check model status
    model_info = models_dict.get(model_name, {})
    status = model_info.get("obj", {}).get("sol_status", "unknown")

    logger.info(f"[DIAGNOSE_IF_INFEASIBLE] Model '{model_name}' status: {status}")

    # Check if infeasible
    is_infeasible = (
        status == TerminationCondition.infeasible or
        status == TerminationCondition.infeasibleOrUnbounded or
        str(status) == "infeasible" or
        str(status) == "infeasibleOrUnbounded"
    )

    if is_infeasible:
        logger.info(f"[DIAGNOSE_IF_INFEASIBLE] Model '{model_name}' is INFEASIBLE. Running diagnosis...")

        try:
            # Import diagnosis tool
            from optichat.tools.custom_tool import infeasibility_diagnosis

            # Pass callback_context directly - both CallbackContext and ToolContext have .state
            # The diagnosis function only accesses .state, so this works via duck typing
            diagnosis_result = infeasibility_diagnosis(model_name, callback_context)

            # Update state with diagnosis results
            callback_context.state[INFEASIBILITY_DIAGNOSIS] = diagnosis_result
            callback_context.state[HAS_INFEASIBILITY_DIAGNOSIS] = True

            # Note: callback_context.state is already modified in-place by infeasibility_diagnosis
            # so MODELS_DICTIONARY already contains any new relaxed model versions

            logger.info("[DIAGNOSE_IF_INFEASIBLE] ✓ Diagnosis completed successfully")
            logger.info(f"[DIAGNOSE_IF_INFEASIBLE] Diagnosis status: {diagnosis_result.get('status', 'unknown')}")

        except Exception as e:
            logger.error(f"[DIAGNOSE_IF_INFEASIBLE] Failed to run diagnosis: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            # Store error information
            callback_context.state[INFEASIBILITY_DIAGNOSIS] = {
                "status": "error",
                "result": f"Diagnosis failed: {str(e)}"
            }
            callback_context.state[HAS_INFEASIBILITY_DIAGNOSIS] = True
    else:
        logger.info(f"[DIAGNOSE_IF_INFEASIBLE] Model '{model_name}' is FEASIBLE. No diagnosis needed.")
        callback_context.state[HAS_INFEASIBILITY_DIAGNOSIS] = False

    return None


def check_illustrator_agent_runtime(callback_context: CallbackContext):
    """
    Record illustrator agent runtime after agent completes.

    Args:
        callback_context: Callback context with state

    Returns:
        None
    """
    timing_tracker.record_agent_end("illustrator_agent")
    return None


def check_root_agent_start(callback_context: CallbackContext):
    """
    Record root agent start before agent processes request.

    This is called AFTER initialize_session, so tracker is already reset.

    Args:
        callback_context: Callback context with state

    Returns:
        None
    """
    timing_tracker.record_agent_start("root_agent")
    return None


def check_root_agent_runtime(callback_context: CallbackContext):
    """
    Record root agent runtime and print summary after agent completes.

    Args:
        callback_context: Callback context with state

    Returns:
        None
    """
    timing_tracker.record_agent_end("root_agent")

    # End query timing and print summary
    timing_tracker.get_tracker().end_query()
    timing_tracker.print_summary()

    # Optionally save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = "tmp/timing"
    os.makedirs(output_dir, exist_ok=True)
    timing_file = os.path.join(output_dir, f"timing_{timestamp}.json")
    timing_tracker.save_to_file(timing_file)

    # Forward expert's response to root's output key when expert responded directly via transfer
    expert_output = callback_context.state.get(OUTPUT_KEY_EXPERT_AGENT, "")
    if expert_output:
        root_output = callback_context.state.get(OUTPUT_KEY_ROOT_AGENT, "")
        if not root_output:
            callback_context.state[OUTPUT_KEY_ROOT_AGENT] = expert_output
            logger.info("[Root callback] Forwarded expert output to root output key")

    return None


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

