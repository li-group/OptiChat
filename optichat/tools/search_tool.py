from typing import List, Dict, Any
from typing import Optional
import re
import json
from loguru import logger
from google.adk.tools.tool_context import ToolContext
from optichat.config.constants import MODELS_DICTIONARY, MODEL_VERSIONS
from optichat.tools.shortcut_functions import load_model, solve_model


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


def get_model_components(version: List[str], component_type: str, pattern: str,
                         tool_context: ToolContext) -> Dict[str, str]:
    """
    get_model_components is a robust and efficient searching method for model components.
    This function retrieves information about specified model components that are stored in the session.

    Args:
        version (List[str]): Model version(s) to search. Must be provided. Maximum 2 versions allowed.
        component_type (str): **PRIMARY METHOD** Type of components to match against component.
            Must be one of: ['objective', 'variable', 'constraint', or '' (empty string for all types)]
        pattern (str): **BACKUP METHOD** Naming pattern to match against component.
            Use this for additional filtering when the output by component_type alone is truncated.
            Supports:
                - Wildcard patterns: '*' (matches any characters), '?' (matches a single character).
                  Example: "x_*" matches "x_1", "x_transport", etc.
                - Substring matching: Simple text matching (case-insensitive).
                  Example: "transport" matches "transport", "x_transport_A", "transport_cost", etc.
                - Empty string: "" matches all component names.

    Returns:
        Dict[str, str]: a dictionary with two keys: "status" and "result"
        "status": "success" or "error"
        "result": the information about the model components that match the specified version, component_type and pattern
    """
    models_dictionary = tool_context.state[MODELS_DICTIONARY].copy()
    versions = version
    is_valid = True
    result = ""
    if len(versions) > 2:
        is_valid = False
        result += "**ERROR** Maximum 2 versions allowed at a time"
    if pattern == "" and component_type == "":
        is_valid = False
        result += "**ERROR** At least one of [pattern, component_type] must be non-empty"
    valid_component_types = ['objective', 'variable', 'constraint', '']
    if component_type not in valid_component_types:
        is_valid = False
        result += (f"**ERROR** component_type must be one of {valid_component_types}, "
                   f"but got '{component_type}'")
    available_versions = list(models_dictionary.keys())
    missing_versions = [v for v in versions if v not in available_versions]
    if missing_versions:
        is_valid = False
        result += (f"**ERROR** available versions are {available_versions}, "
                   f"but got '{missing_versions}'")

    result_dictionary = {}
    # Process each version
    for ver in versions:
        # Pre-processing: solve if not solved yet
        if models_dictionary[ver]["obj"].get("sol_status", "unknown") != "optimal":
            model = load_model(ver, models_dictionary)
            models_dictionary = solve_model(model, ver, models_dictionary)
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
        if component_type != "":
            type_matches = {}
            for comp_name, comp_data in pattern_matches.items():
                if isinstance(comp_data, dict) and comp_data.get("component_type") == component_type:
                    # Exclude component_type field to save tokens since it's already filtered
                    comp_data_filtered = {k: v for k, v in comp_data.items() if k != "component_type"}
                    type_matches[comp_name] = comp_data_filtered
        else:
            type_matches = pattern_matches

        result_dictionary[ver] = type_matches  # type_matches has already applied both pattern and type filtering

    return {"status": "success" if is_valid else "error",
            "result": json.dumps(result_dictionary, indent=4) if is_valid else result}