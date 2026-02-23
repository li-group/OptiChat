"""
Utilities for reading and formatting .py source code files for prompt injection.

This module provides functions to:
1. Retrieve source code for models from tmp folder
2. Clean and format source code for LLM consumption
3. Extract modification history from metadata
"""

import ast
import os
from typing import Optional, Dict, Any
from loguru import logger


def extract_constraint_source(source_code: str) -> str:
    """
    Extract only constraint-related code from a Pyomo model source file.

    Handles all common Pyomo constraint styles:
    - Named rule functions:  def rule(...): ...  +  model.c = Constraint(..., rule=rule)
    - Lambda rules:          model.c = Constraint(..., rule=lambda ...)
    - Direct expr=:          model.c = Constraint(expr=...)
    - ConstraintList:        model.cl = ConstraintList()  +  for-loop with .add(...)

    Returns the extracted lines joined as a string, or the original source if
    AST parsing fails (so the caller always gets something useful).
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        logger.warning("extract_constraint_source: SyntaxError, returning full source")
        return source_code

    lines = source_code.splitlines()
    line_ranges = []

    # ── Pass 1: Constraint(...) assignments ───────────────────
    rule_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        func = call.func if isinstance(call, ast.Call) else None
        is_constraint = func and (
            (isinstance(func, ast.Name) and func.id == 'Constraint') or
            (isinstance(func, ast.Attribute) and func.attr == 'Constraint')
        )
        if not is_constraint:
            continue
        line_ranges.append((node.lineno - 1, node.end_lineno))
        for kw in call.keywords:
            if kw.arg == 'rule' and isinstance(kw.value, ast.Name):
                rule_names.add(kw.value.id)

    # Rule function definitions referenced by Constraint(rule=...)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in rule_names:
            line_ranges.append((node.lineno - 1, node.end_lineno))

    # ── Pass 2: ConstraintList declarations + containing for loops ──
    cl_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        func = call.func if isinstance(call, ast.Call) else None
        is_cl = func and (
            (isinstance(func, ast.Name) and func.id == 'ConstraintList') or
            (isinstance(func, ast.Attribute) and func.attr == 'ConstraintList')
        )
        if not is_cl:
            continue
        line_ranges.append((node.lineno - 1, node.end_lineno))
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                cl_names.add(target.attr)

    def _contains_cl_add(node):
        for child in ast.walk(node):
            if (isinstance(child, ast.Call) and
                    isinstance(child.func, ast.Attribute) and
                    child.func.attr == 'add' and
                    isinstance(child.func.value, ast.Attribute) and
                    child.func.value.attr in cl_names):
                return True
        return False

    # Only scan top-level statements to avoid duplicating nested loops
    for node in tree.body:
        if isinstance(node, ast.For) and _contains_cl_add(node):
            line_ranges.append((node.lineno - 1, node.end_lineno))

    if not line_ranges:
        logger.warning("extract_constraint_source: no constraints found, returning full source")
        return source_code

    # Merge overlapping ranges and extract
    line_ranges = sorted(set(line_ranges))
    merged = []
    for start, end in line_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append([start, end])

    parts = ["\n".join(lines[s:e]) for s, e in merged]
    result = "\n\n".join(parts)
    logger.debug(f"extract_constraint_source: {len(source_code)} → {len(result)} chars")
    return result


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
    
    # Read, extract constraints only, then clean
    try:
        with open(source_path, 'r') as f:
            source_code = f.read()
        logger.info(f"Read source code from {source_path} ({len(source_code)} chars)")
        constraint_source = extract_constraint_source(source_code)
        return clean_source_code(constraint_source)
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
===== MODEL CONSTRAINT DEFINITIONS: {version_name} =====
Pickle file: {pkl_path}
Source file: tmp/model_objects/{version_name}.py

These are the constraint definitions extracted from the model source.
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
