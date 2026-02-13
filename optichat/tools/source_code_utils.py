"""
Utilities for reading and formatting .py source code files for prompt injection.

This module provides functions to:
1. Retrieve source code for models from tmp folder
2. Clean and format source code for LLM consumption
3. Extract modification history from metadata
"""

import os
from typing import Optional, Dict, Any
from loguru import logger


def get_source_code_for_model(
    version_name: str,
    models_dict: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Optional[str]:
    """
    Retrieve the .py source code for a model version.
    
    For base models: Read from tmp/model_objects/{version}.py
    For modified models: Read base model's .py file (recursively)
    
    Args:
        version_name: Model version name (e.g., 'RTN_inf_1' or 'RTN_inf_1__mu_times2')
        models_dict: Runtime models dictionary (models_dictionary from state)
        metadata: Historical metadata dictionary
    
    Returns:
        Cleaned source code string or None if not found
    """
    # Check if this is a modified model (has __)
    if "__" in version_name:
        # Extract base model name
        base_model = version_name.split("__")[0]
        logger.debug(f"Modified model detected: {version_name}, extracting base: {base_model}")
        return get_source_code_for_model(base_model, models_dict, metadata)
    
    # Get source path from model_info
    model_info = models_dict.get(version_name, {})
    source_path = model_info.get("source_file_path")
    
    if not source_path or not os.path.exists(source_path):
        # Fallback: try tmp/model_objects/{version}.py
        source_path = os.path.join("tmp/model_objects", f"{version_name}.py")
        if not os.path.exists(source_path):
            logger.warning(f"No source file found for {version_name}")
            return None
    
    # Read and clean
    try:
        with open(source_path, 'r') as f:
            source_code = f.read()
        logger.info(f"Read source code from {source_path} ({len(source_code)} chars)")
        return clean_source_code(source_code)
    except Exception as e:
        logger.error(f"Error reading source file {source_path}: {e}")
        return None


def clean_source_code(source: str, max_size_kb: int = 50) -> str:
    """
    Clean source code for prompt injection.
    
    - Strip excessive blank lines
    - Remove long comment blocks (###)
    - Truncate if > max_size_kb
    
    Args:
        source: Raw source code string
        max_size_kb: Maximum size in KB
    
    Returns:
        Cleaned source code string
    """
    lines = source.split('\n')
    cleaned_lines = []
    
    skip_next = False
    for line in lines:
        # Skip long comment dividers like ###########
        if line.strip().startswith('###'):
            continue
        
        # Keep code and meaningful comments
        if line.strip():  # Non-empty line
            cleaned_lines.append(line)
    
    cleaned = '\n'.join(cleaned_lines)
    
    # Check size
    size_kb = len(cleaned) / 1024
    if size_kb > max_size_kb:
        logger.warning(f"Source code is {size_kb:.1f}KB, truncating to {max_size_kb}KB")
        # Truncate to max size
        max_chars = int(max_size_kb * 1024)
        cleaned = cleaned[:max_chars]
        cleaned += "\n\n... [Source code truncated due to size]"
    
    logger.debug(f"Cleaned source code: {len(cleaned)} chars ({len(cleaned)/1024:.1f}KB)")
    return cleaned


def format_source_for_prompt(
    source: str,
    version_name: str,
    pkl_path: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Format source code with header, modification history, and instructions.
    
    For base models: Just show source code with instructions
    For modified models: Show base source + modification chain from metadata
    
    Args:
        source: Cleaned source code string
        version_name: Model version name
        pkl_path: Path to pickle file
        metadata: Historical metadata dictionary
    
    Returns:
        Formatted source code ready for prompt injection
    """
    # Check if this is a modified model
    is_modified = "__" in version_name
    
    if is_modified:
        base_model = version_name.split("__")[0]
        # Get modification history from metadata
        mod_history = _get_modification_history(version_name, metadata)
        
        return f"""
===== BASE MODEL SOURCE: {base_model} =====
Pickle file: {pkl_path}

{source}

===== MODIFICATION HISTORY =====
This model ({version_name}) was created from {base_model} with:
{mod_history}

When writing modification code, reference the base structure above
and account for the previous modifications listed.
=======================================
"""
    else:
        return f"""
===== MODEL SOURCE CODE: {version_name} =====
Pickle file: {pkl_path}
Source file: tmp/model_objects/{version_name}.py

The model was defined with the Python code below.
When writing modification code in python_repl_func:
- Use the EXACT same syntax patterns (ConstraintList vs lambda rules)
- Match the index structures (e.g., mu[i, r, theta] not mu[i])
- Reference component names exactly as defined

{source}
=======================================
"""


def _get_modification_history(version_name: str, metadata: Dict[str, Any]) -> str:
    """
    Extract modification history from metadata for a modified model.
    Uses the description field which already contains modification info.
    
    Args:
        version_name: Modified model version name
        metadata: Historical metadata dictionary
    
    Returns:
        Formatted modification history string with description, timestamp, and REPL code if available
    """
    # Search metadata for this version
    for date_key, date_models in metadata.items():
        for base_name, base_data in date_models.items():
            modified_models = base_data.get("modified_models", {})
            if version_name in modified_models:
                mod_data = modified_models[version_name]
                desc = mod_data.get("description", "No description")
                timestamp = mod_data.get("timestamp", "unknown")
                repl_code = mod_data.get("repl_code")
                
                history = f"- Description: {desc}\n- Timestamp: {timestamp}"
                
                if repl_code:
                    history += f"\n- REPL code:\n```python\n{repl_code}\n```"
                
                return history
    
    logger.warning(f"No modification history found for {version_name}")
    return "No modification history found"
