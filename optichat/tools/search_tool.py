from typing import List, Dict, Any, Union
from typing import Optional
import re
import json
import hashlib
from loguru import logger
from google.adk.tools.tool_context import ToolContext
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS, HISTORICAL_MODELS_METADATA, MODEL_COMPONENTS_CACHE, MAX_CACHE_ENTRIES
from optichat.tools.shortcut_functions import load_model, solve_model
from optichat.tools.metadata_store import load_model_data


def wildcard_to_regex(pattern: str) -> str:
    """
    Convert a wildcard pattern to a regex pattern.
    The intuition of this function is to escape all regex special characters, as pyomo's component names usually contain "[", "]", "(", ")", etc.
    Then only rely on '*' and '?' for wildcard matching.

    Args:
        pattern (str): The wildcard pattern to convert.

    Returns:
        str: The corresponding regex pattern.
    """
    # Escape all regex special characters
    regex_pattern = re.escape(pattern)
    # Replace escaped wildcards with regex equivalents
    regex_pattern = regex_pattern.replace(r'\*', '.*').replace(r'\?', '.')
    # Anchor the pattern to match the whole string
    return f'^{regex_pattern}$'


def normalize_component_pattern(pattern: str) -> str:
    """
    Normalize quoted index tokens so both `a['cabbage',*]` and `a[cabbage,*]`
    match the same stored Pyomo component names.

    Quotes are stripped only when they behave as token delimiters, which keeps
    apostrophes inside unquoted words intact.
    """
    if not pattern:
        return pattern

    def previous_non_space(idx: int):
        j = idx - 1
        while j >= 0 and pattern[j].isspace():
            j -= 1
        return pattern[j] if j >= 0 else None

    def next_non_space(idx: int):
        j = idx + 1
        while j < len(pattern) and pattern[j].isspace():
            j += 1
        return pattern[j] if j < len(pattern) else None

    normalized = []
    active_quote = None

    for i, ch in enumerate(pattern):
        if active_quote is not None:
            if ch == active_quote and next_non_space(i) in (None, ",", "]"):
                active_quote = None
                continue
            normalized.append(ch)
            continue

        if ch in {"'", '"'} and previous_non_space(i) in (None, "[", ","):
            if next_non_space(i) is not None:
                active_quote = ch
                continue

        normalized.append(ch)

    return "".join(normalized)


def get_all_versions_from_metadata(metadata: Dict[str, Any]) -> set:
    """Extract all model versions from the tree-based metadata structure."""
    versions = set()
    for date_key, date_models in metadata.items():
        for base_name, base_data in date_models.items():
            versions.add(base_name)
            modified_models = base_data.get("modified_models", {})
            for mod_name in modified_models:
                versions.add(mod_name)
    return versions


def _generate_cache_key(version: List[str],
                        component_type: Union[str, List[str]],
                        pattern: str) -> str:
    """
    Generate deterministic cache key from function parameters.

    Args:
        version: List of model versions
        component_type: Single type or list of types
        pattern: Pattern string

    Returns:
        MD5 hash of normalized parameters
    """
    # Normalize version list (order shouldn't matter)
    version_str = ",".join(sorted(version))

    # Normalize component_type
    if isinstance(component_type, list):
        type_str = ",".join(sorted(component_type))
    else:
        type_str = str(component_type)

    # Combine all parameters into one string
    key_parts = f"{version_str}|{type_str}|{pattern}"

    # Hash to get compact key
    return hashlib.md5(key_parts.encode()).hexdigest()


def get_model_components(version: List[str], component_type: Union[str, List[str]], pattern: str,
                         tool_context: ToolContext) -> Dict[str, str]:
    """
    get_model_components is a robust and efficient searching method for model components.
    This function retrieves information about specified model components that are stored in the session.

    Args:
        version (List[str]): Model version(s) to search. Must be provided. Maximum 3 versions allowed.
        component_type (Union[str, List[str]]): **PRIMARY METHOD** Type(s) of components to match against component.
            Can be single: 'objective', 'variable', 'constraint', 'parameter', 'set', or '' (empty string for all types)
            Or multiple: ['objective', 'variable'] for bulk extraction (efficient for reducing tool calls)
            Or all: '' or 'all'
        pattern (str): **BACKUP METHOD** Naming pattern to match against fully-indexed component names.
            Component names include their index, e.g. "N[Rx1,3]", "X[A,0]", "demand[2,1]".
            Use this for additional filtering when the output by component_type alone is truncated.
            Supports:
                - Wildcard patterns: '*' (matches any characters), '?' (matches a single character).
                  Example: "N[*,3]" matches "N[Rx1,3]", "N[Rx2,3]", "N[Rx3,3]" (all N at second index 3).
                  Example: "*[*,3]" matches all components whose second index is 3.
                  Example: "x_*" matches "x_1", "x_transport", etc.
                  NOTE: "*[3]" only matches single-index components like "Balance[3]", NOT "N[Rx1,3]".
                - Substring matching: Simple text matching (case-insensitive).
                  Example: "transport" matches "transport", "x_transport_A", "transport_cost", etc.
                - Empty string: "" matches all component names.

    Returns:
        Dict[str, str]: a dictionary with two keys: "status" and "result"
        "status": "success" or "error"
        "result": the information about the model components that match the specified version, component_type and pattern
    """
    pattern = normalize_component_pattern(pattern)

    # Generate cache key
    cache_key = _generate_cache_key(version, component_type, pattern)

    # Check cache
    cache = tool_context.state.get(MODEL_COMPONENTS_CACHE, {})
    if cache_key in cache:
        logger.info(f"✅ Cache HIT for {version} {component_type}")
        logger.debug(f"   Cache key: {cache_key}")
        return cache[cache_key]

    logger.info(f"📝 Cache MISS for {version} {component_type}")
    logger.debug(f"   Cache key: {cache_key}")

    # Normalize component_type to list for uniform processing
    if isinstance(component_type, str):
        component_types = [component_type] if component_type and component_type != 'all' else []
    else:
        component_types = component_type

    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    historical_metadata = tool_context.state.get(HISTORICAL_MODELS_METADATA, {})

    # Extract all versions from metadata tree
    all_historical_versions = get_all_versions_from_metadata(historical_metadata)

    versions = version
    is_valid = True
    result = ""
    if len(versions) > 3:
        is_valid = False
        result += "**ERROR** Maximum 3 versions allowed at a time"
    if pattern == "" and len(component_types) == 0:
        is_valid = False
        result += "**ERROR** At least one of [pattern, component_type] must be non-empty"
    valid_component_types = ['objective', 'variable', 'constraint', 'parameter', 'set', '', 'all']
    for ct in component_types:
        if ct not in valid_component_types:
            is_valid = False
            result += (f"**ERROR** component_type '{ct}' must be one of {valid_component_types}\n")
    # Check both runtime cache (models_dictionary) and metadata index (historical_metadata)
    available_versions = list(set(models_dictionary.keys()) | all_historical_versions)
    missing_versions = [v for v in versions if v not in available_versions]
    if missing_versions:
        is_valid = False
        result += (f"**ERROR** available versions are {available_versions}, "
                   f"but got '{missing_versions}'")
        
    # If the input is invalid, return the error message.
    if not is_valid:
        return {"status": "error", "result": result}

    result_dictionary = {}

    # Process each version
    for ver in versions:
        # Check if model data is in runtime cache
        if ver not in models_dictionary:
            # Lazy load from file if it exists in historical metadata
            if ver in all_historical_versions:
                logger.info(f"Lazy loading model data for {ver} from file")
                try:
                    model_data = load_model_data(ver)
                    # Add to runtime cache
                    models_dictionary[ver] = model_data
                    tool_context.state[MODELS_DICTIONARY] = models_dictionary
                    logger.info(f"Successfully loaded model data for {ver}")
                except FileNotFoundError as e:
                    # Model metadata exists but data file missing
                    logger.error(f"Failed to lazy load {ver}: {e}")
                    return {
                        "status": "error",
                        "result": f"Model data file not found for {ver}. Metadata exists but data file is missing."
                    }
            else:
                # Model doesn't exist at all
                return {
                    "status": "error",
                    "result": f"Model {ver} not found in metadata or models_dictionary"
                }

        if models_dictionary[ver]["obj"].get("sol_status", "unknown") == "unknown":
            model = load_model(ver, models_dictionary)
            description = f"Re-solving model {ver} to retrieve component information"
            models_dictionary = solve_model(model, ver, models_dictionary, tool_context, description)
            tool_context.state[MODELS_DICTIONARY] = models_dictionary
            tool_context.state[MODEL_VERSIONS] = list(models_dictionary.keys())

        # Pattern searching logic
        if pattern != "":
            info = models_dictionary[ver]
            pattern_matches = {}
            if '*' in pattern or '?' in pattern:
                regex_pattern = wildcard_to_regex(pattern)
                compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
                for k in info.keys():
                    if compiled_pattern.match(k):
                        pattern_matches[k] = models_dictionary[ver][k]
            else:
                for k in info.keys():
                    if pattern.lower() in k.lower():
                        pattern_matches[k] = models_dictionary[ver][k]
        else:
            pattern_matches = models_dictionary[ver]

        # Type searching logic
        if component_types and 'all' not in component_types:
            type_matches = {}
            for comp_name, comp_data in pattern_matches.items():
                if isinstance(comp_data, dict) and comp_data.get("component_type") in component_types:
                    # Exclude component_type field to save tokens since it's already filtered
                    comp_data_filtered = {k: v for k, v in comp_data.items() if k != "component_type"}
                    type_matches[comp_name] = comp_data_filtered
        else:
            type_matches = pattern_matches

        result_dictionary[ver] = type_matches  # type_matches has already applied both pattern and type filtering

    # Prepare final result
    final_result = {"status": "success" if is_valid else "error",
                    "result": json.dumps(result_dictionary, indent=4) if is_valid else result}

    # Store in cache with size limit — skip caching empty results so that
    # modified models created mid-session can be fetched on a retry.
    has_results = any(bool(v) for v in result_dictionary.values())
    if is_valid and has_results:
        cache[cache_key] = final_result
        if len(cache) > MAX_CACHE_ENTRIES:
            # Keep newest 75%
            items = list(cache.items())
            cache = dict(items[len(items)//4:])
            logger.warning(f"Cache limit reached. Evicted {len(items)//4} entries.")
        tool_context.state[MODEL_COMPONENTS_CACHE] = cache

    return final_result
