import os
from datetime import datetime
from typing import Dict
from loguru import logger
from google.adk.tools.tool_context import ToolContext

from optichat.config.constants import (
    TMP_MODEL_OBJECT_FOLDER,
    MODELS_DICTIONARY,
    MODEL_FOR_PAPER_GENERATION,
    CFG,
    HAS_INFEASIBILITY_DIAGNOSIS,
    INFEASIBILITY_DIAGNOSIS
)


def extract_sets_from_components(components: Dict) -> Dict[str, set]:
    """
    Extract sets (index domains) from component names.
    
    Args:
        components: Dictionary of model components (parameters, variables, or constraints)
    
    Returns:
        Dictionary mapping set names to their values
        
    Example:
        Input: {"X[A,1]": ..., "X[B,2]": ...}
        Output: {"index_1": {"A", "B"}, "index_2": {1, 2}}
    """
    from collections import defaultdict
    import re
    
    # Track indices by position
    indices_by_position = defaultdict(set)
    
    for comp_name in components.keys():
        # Extract indices from brackets
        match = re.search(r'\[(.*?)\]', comp_name)
        if match:
            indices_str = match.group(1)
            # Split by comma
            indices = [idx.strip() for idx in indices_str.split(',')]
            for pos, idx in enumerate(indices):
                indices_by_position[pos].add(idx)
    
    return dict(indices_by_position)


def detect_constraint_pattern(constraint_name: str, expression: str, index_signature: str = "•") -> tuple:
    """
    Detect the pattern family and abstract pattern for a constraint.
    
    Args:
        constraint_name: Name of the constraint (e.g., "ResourceLB[1]")
        expression: Expression of the constraint
        index_signature: Semantic index names to use (e.g., "resource, time")
    
    Returns:
        Tuple of (family_name, pattern)
        
    Example:
        Input: "ResourceLB[1]", "Xmin[A] <= X[A,1]", "resource, time"
        Output: ("ResourceLB", "Xmin[resource] <= X[resource, time]")
    """
    import re
    
    # Extract family name (everything before the index)
    match = re.match(r'^([^\[]+)', constraint_name)
    family_name = match.group(1) if match else constraint_name
    
    # Create abstract pattern by replacing specific indices with semantic names
    # Replace bracketed expressions with abstract form using index signature
    pattern = re.sub(r'\[[^\]]+\]', f'[{index_signature}]', expression)
    
    return family_name, pattern


def infer_index_names(components: Dict) -> list:
    """
    Infer semantic names for indices based on component naming patterns.
    
    Analyzes the types of values in each index position to guess meaningful names.
    
    Args:
        components: Dictionary of components with indexed names
    
    Returns:
        List of inferred index names (e.g., ["resource", "time"])
    
    Example:
        If we see X[A, 1], X[B, 2], etc.:
        - Position 0 has letters -> "resource" or "item"
        - Position 1 has numbers -> "time" or "index"
    """
    import re
    from collections import defaultdict
    
    # Collect sample values for each position
    indices_by_position = defaultdict(list)
    
    for comp_name in list(components.keys())[:50]:  # Sample first 50
        match = re.search(r'\[(.*?)\]', comp_name)
        if match:
            indices_str = match.group(1)
            indices = [idx.strip() for idx in indices_str.split(',')]
            for pos, idx in enumerate(indices):
                if len(indices_by_position[pos]) < 10:  # Keep sample small
                    indices_by_position[pos].append(idx)
    
    if not indices_by_position:
        return []
    
    # Infer names based on patterns
    inferred_names = []
    for pos in sorted(indices_by_position.keys()):
        samples = indices_by_position[pos]
        
        # Check if mostly numeric
        numeric_count = sum(1 for s in samples if s.isdigit())
        
        if numeric_count > len(samples) * 0.7:
            # Mostly numbers -> likely time or period
            inferred_names.append("time")
        else:
            # Letters or mixed -> likely resource, location, item, etc.
            # Try to infer from common patterns
            if any(name.startswith(('Rx', 'R')) for name in samples):
                inferred_names.append("reactor")
            elif len(samples[0]) <= 2:  # Short codes
                inferred_names.append("resource")
            else:
                inferred_names.append("item")
    
    return inferred_names


# ============================================================================
# Hierarchical Component Formatting (Lightweight Structure)
# ============================================================================

def _extract_components_by_type(model_data: dict, component_type: str) -> dict:
    """
    Extract components of given type from flat model_data dict.

    Args:
        model_data: Flat dictionary with component names as keys
        component_type: Type to filter for (plural: "sets", "parameters", "variables", "constraints", "objective")

    Returns:
        Dictionary of components matching the type
    """
    # Map plural type names to singular component_type values
    type_mapping = {
        "sets": "set",
        "parameters": "parameter",
        "variables": "variable",
        "constraints": "constraint",
        "objective": "objective"  # Already singular
    }

    singular_type = type_mapping.get(component_type, component_type)

    components = {}
    for key, value in model_data.items():
        if isinstance(value, dict) and value.get("component_type") == singular_type:
            components[key] = value
    return components


def _generate_component_description(name: str, data: dict, comp_type: str) -> str:
    """Generate a brief description for a component based on its name and type."""
    # Check if description already exists in data (from Pyomo doc strings)
    if "doc" in data and data["doc"]:
        return data["doc"]

    # Infer description from name and type
    # This is a fallback - ideally descriptions come from Pyomo doc strings
    type_descriptions = {
        "sets": f"Set of indices for {name}",
        "parameters": f"Parameter: {name}",
        "variables": f"Decision variable: {name}",
        "constraints": f"Constraint: {name}",
        "objective": f"Objective function"
    }
    return type_descriptions.get(comp_type, name)


def _extract_objective_sense(obj_data: dict) -> str:
    """Extract MINIMIZE/MAXIMIZE from objective data."""
    expression = obj_data.get("expression", "")
    if expression.startswith("MINIMIZE"):
        return "MINIMIZE"
    elif expression.startswith("MAXIMIZE"):
        return "MAXIMIZE"
    return "UNKNOWN"


def _infer_component_type(name: str, data: dict) -> str:
    """Determine if component is scalar or indexed based on name pattern."""
    # Simple heuristic: if name contains '[', it's indexed
    return "indexed" if "[" in name else "scalar"


def _infer_model_type(model_data: dict) -> str:
    """Infer model type (LP, MILP, NLP) from variable types."""
    # Check for integer/binary variables
    has_integer = False
    has_continuous = False

    for key, value in model_data.items():
        if isinstance(value, dict) and value.get("component_type") == "variable":
            domain = value.get("domain", "").lower()
            if "integer" in domain or "binary" in domain:
                has_integer = True
            else:
                has_continuous = True

    if has_integer:
        return "MILP" if has_continuous else "MIP"
    return "LP"  # Default assumption


def format_hierarchical_components(models_dictionary: dict) -> dict:
    """
    Format model components into a lightweight hierarchical structure with pattern-based grouping.

    Returns a dict structure matching the legacy model_representation format:
    {
        "model_name": str,
        "model_type": str,
        "model_status": str,
        "components": {
            "parameters": {
                "demand": {"count": 24, "description": "...", "example": "demand[1] = 500"},
                "fixed_cost": {"count": 1, "description": "...", "value": 1000}
            },
            "variables": { ... },
            "constraints": { ... },
            "objective": {"obj": {"description": "...", "sense": "MINIMIZE"}}
        }
    }

    This minimal structure preserves Pyomo component hierarchy without verbose expressions.
    Indexed components are grouped by family (base name) with count and examples.
    """
    from collections import defaultdict
    import re

    # Get first model (assuming single model in most cases)
    if not models_dictionary:
        return {}

    model_name = list(models_dictionary.keys())[0]
    model_data = models_dictionary[model_name]

    # Build hierarchical structure
    result = {
        "model_name": model_name,
        "model_type": _infer_model_type(model_data),
        "model_status": "unknown",
        "components": {}
    }

    # Extract and group components by type
    component_groups = {
        "sets": {},
        "parameters": defaultdict(list),
        "variables": defaultdict(list),
        "constraints": defaultdict(list),
        "objective": {}
    }

    # Populate component groups
    for comp_name, comp_data in model_data.items():
        if not isinstance(comp_data, dict):
            continue

        comp_type = comp_data.get("component_type", "unknown")

        # Map singular to plural
        type_map = {
            "parameter": "parameters",
            "variable": "variables",
            "constraint": "constraints",
            "objective": "objective"
        }

        plural_type = type_map.get(comp_type)
        if not plural_type:
            continue

        if plural_type == "objective":
            # Objectives are special - just store directly
            component_groups["objective"][comp_name] = comp_data
        else:
            # Extract family name (part before brackets)
            match = re.match(r'^([^\[]+)', comp_name)
            family = match.group(1) if match else comp_name
            component_groups[plural_type][family].append((comp_name, comp_data))

    # Format each component type
    for comp_type in ["parameters", "variables", "constraints"]:
        result["components"][comp_type] = {}

        for family, items in sorted(component_groups[comp_type].items()):
            count = len(items)

            if count == 1:
                # Scalar component - show directly
                name, data = items[0]
                result["components"][comp_type][name] = {
                    "description": _generate_component_description(name, data, comp_type),
                    "type": "scalar"
                }

                # Add value/solution for context
                if comp_type == "parameters" and "value" in data:
                    result["components"][comp_type][name]["value"] = data["value"]
                elif comp_type == "variables" and "solution" in data:
                    result["components"][comp_type][name]["solution"] = data["solution"]

            else:
                # Indexed family - show grouped with count
                first_name, first_data = items[0]

                # Use family name with index pattern
                display_name = f"{family}[...]"

                result["components"][comp_type][display_name] = {
                    "description": _generate_component_description(family, first_data, comp_type),
                    "type": "indexed",
                    "count": count
                }

                # Add example
                if comp_type == "parameters" and "value" in first_data:
                    result["components"][comp_type][display_name]["example"] = f"{first_name} = {first_data['value']}"
                elif comp_type == "variables" and "solution" in first_data:
                    result["components"][comp_type][display_name]["example"] = f"{first_name} = {first_data['solution']}"

    # Format objectives
    result["components"]["objective"] = {}
    for obj_name, obj_data in component_groups["objective"].items():
        result["components"]["objective"][obj_name] = {
            "description": _generate_component_description(obj_name, obj_data, "objective"),
            "sense": _extract_objective_sense(obj_data)
        }

        # Add objective value for context
        if "value" in obj_data:
            result["components"]["objective"][obj_name]["value"] = obj_data["value"]

    # Sets are not extracted by extract_model_info, so leave empty
    result["components"]["sets"] = {}

    return result


# ============================================================================
# Pattern-Based Summary (Legacy/Verbose Format)
# ============================================================================

def format_pattern_based_summary(models_dictionary: Dict) -> str:
    """
    Format models_dictionary into compact pattern-based markdown.
    
    Groups indexed components by pattern and shows generalized forms instead of
    listing thousands of individual instances.
    
    Args:
        models_dictionary: Full model data from extract_model_info()
    
    Returns:
        Formatted markdown string with pattern-based representation
    """
    from collections import defaultdict
    
    sections = []
    
    # Group components by type
    parameters = {}
    variables = {}
    constraints = {}
    objectives = {}
    
    for comp_name, comp_data in models_dictionary.items():
        if not isinstance(comp_data, dict):
            continue
        comp_type = comp_data.get("component_type", "unknown")
        
        if comp_type == "parameter":
            parameters[comp_name] = comp_data
        elif comp_type == "variable":
            variables[comp_name] = comp_data
        elif comp_type == "constraint":
            constraints[comp_name] = comp_data
        elif comp_type == "objective":
            objectives[comp_name] = comp_data
    
    # Extract sets from all components
    all_components = {**parameters, **variables, **constraints}
    sets_by_position = extract_sets_from_components(all_components)
    
    # Infer semantic index names
    index_names = infer_index_names(all_components)
    index_signature = ", ".join(index_names) if index_names else "•"
    
    # Format Sets section
    if sets_by_position:
        set_lines = ["## Sets\n"]
        for pos, values in sorted(sets_by_position.items()):
            # Sample a few values for display
            sample_values = sorted(list(values))[:10]
            if len(values) > 10:
                values_str = f"{{{', '.join(map(str, sample_values))}, ... ({len(values)} total)}}"
            else:
                values_str = f"{{{', '.join(map(str, sample_values))}}}"
            set_lines.append(f"- **Index position {pos}**: {values_str}")
        sections.append("\n".join(set_lines))
    
    # Format Parameters with pattern grouping
    if parameters:
        param_families = defaultdict(list)
        for name, data in parameters.items():
            # Extract family name (before brackets)
            import re
            match = re.match(r'^([^\[]+)', name)
            family = match.group(1) if match else name
            param_families[family].append((name, data))
        
        param_lines = ["## Parameters\n"]
        for family, items in sorted(param_families.items()):
            count = len(items)
            # Show first example
            first_name, first_data = items[0]
            first_value = first_data.get("value", "unknown")
            
            if count == 1:
                param_lines.append(f"- **{first_name}**: {first_value}")
            else:
                param_lines.append(f"**{family}[{index_signature}]** ({count} instances):")
                param_lines.append(f"  - Example: {first_name} = {first_value}")
        
        sections.append("\n".join(param_lines))
    
    # Format Variables with pattern grouping
    if variables:
        var_families = defaultdict(list)
        for name, data in variables.items():
            import re
            match = re.match(r'^([^\[]+)', name)
            family = match.group(1) if match else name
            var_families[family].append((name, data))
        
        var_lines = ["## Variables\n"]
        for family, items in sorted(var_families.items()):
            count = len(items)
            first_name, first_data = items[0]
            first_sol = first_data.get("solution", "unknown")
            
            if count == 1:
                if first_sol != "unknown":
                    var_lines.append(f"- **{first_name}**: Solution = {first_sol}")
                else:
                    var_lines.append(f"- **{first_name}**: (not yet solved)")
            else:
                # Check if any are solved
                solved_count = sum(1 for _, d in items if d.get("solution", "unknown") != "unknown")
                if solved_count > 0:
                    var_lines.append(f"**{family}[{index_signature}]** ({count} instances, {solved_count} solved):")
                    # Find first solved example
                    for name, data in items:
                        if data.get("solution", "unknown") != "unknown":
                            var_lines.append(f"  - Example: {name} = {data['solution']}")
                            break
                else:
                    var_lines.append(f"**{family}[{index_signature}]** ({count} instances): All currently unsolved")
        
        sections.append("\n".join(var_lines))
    
    # Format Constraints with pattern grouping
    if constraints:
        constraint_families = defaultdict(list)
        for name, data in constraints.items():
            family, pattern = detect_constraint_pattern(name, data.get("expression", ""), index_signature)
            constraint_families[family].append((name, data, pattern))
        
        cons_lines = ["## Constraints\n"]
        for family, items in sorted(constraint_families.items()):
            count = len(items)
            first_name, first_data, first_pattern = items[0]
            first_expr = first_data.get("expression", "")
            
            if count == 1:
                cons_lines.append(f"- **{first_name}**: {first_expr}")
                if first_data.get("is_binding") is True:
                    cons_lines.append(f"  _(Binding)_")
            else:
                cons_lines.append(f"\n**{family}** ({count} instances):")
                cons_lines.append(f"  - Pattern: `{first_pattern}`")
                cons_lines.append(f"  - Example: {first_name}: {first_expr}")
        
        sections.append("\n".join(cons_lines))
    
    # Format Objectives (unchanged from original)
    if objectives:
        obj_lines = ["## Objective\n"]
        for name, data in objectives.items():
            expression = data.get("expression", "")
            value = data.get("value", "unknown")
            sol_status = data.get("sol_status", "unknown")
            
            obj_lines.append(f"- **{name}**: {expression}")
            obj_lines.append(f"  - Objective value: {value}")
            obj_lines.append(f"  - Solution status: {sol_status}")
        
        sections.append("\n".join(obj_lines))
    
    # Summary
    summary = f"""## Model Summary

Total components:
- Parameters: {len(parameters)}
- Variables: {len(variables)}
- Constraints: {len(constraints)}
- Objectives: {len(objectives)}
"""
    sections.insert(0, summary)
    
    return "\n\n".join(sections)

def save_synthetic_paper(description: str, model_name: str) -> str:
    """
    Save generated model description as a .txt file (synthetic paper).

    Creates a timestamped markdown file that can be used by paper_rag.

    Args:
        description: Generated model description (markdown format)
        model_name: Name of the model

    Returns:
        Absolute path to saved synthetic paper file

    Example:
        >>> path = save_synthetic_paper(description, "supply_chain")
        >>> print(path)
        "/path/to/tmp/model_objects/generated_papers/supply_chain_description.txt"
    """
    # Create generated_papers directory
    papers_dir = os.path.join(TMP_MODEL_OBJECT_FOLDER, "generated_papers")
    os.makedirs(papers_dir, exist_ok=True)

    # Create filename
    filename = f"{model_name}_description.txt"
    file_path = os.path.join(papers_dir, filename)

    # Add metadata header
    timestamp = datetime.now().isoformat()
    full_content = f"""# {model_name} - Optimization Model Description

**Generated**: {timestamp}
**Generated by**: OptiChat Illustrator Agent
**Model**: {model_name}

---

{description}

---

*This description was automatically generated by the OptiChat Illustrator Agent.
It provides a comprehensive overview of the optimization model for practitioners
in the relevant domain who may not have formal optimization training.*
"""

    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(full_content)

    logger.info(f"✓ Saved synthetic paper to: {file_path}")
    return os.path.abspath(file_path)


def get_model_info_for_description(request: str, tool_context: ToolContext) -> str:
    """
    Retrieve model components and code from session state for description generation.

    This tool is specifically for the illustrator agent to access the model information
    it needs to generate a comprehensive description.

    Args:
        request: The request string (e.g., "generate"). Required for tool schema generation.
        tool_context: Tool context containing session state

    Returns:
        Formatted string containing model components and optionally source code

    Example:
        >>> info = get_model_info_for_description("generate", context)
        >>> print(info[:100])
        "# Model Components\n\n## Model Summary\n\nTotal components:\n- Parameters: 15\n..."
    """
    try:
        # Retrieve model information from state
        models_dictionary = tool_context.state.get(MODELS_DICTIONARY, {})
        model_name = tool_context.state.get(MODEL_FOR_PAPER_GENERATION, "unknown_model")
        cfg = tool_context.state.get(CFG, {})

        logger.info(f"[ILLUSTRATOR_TOOL] Retrieving model info for: {model_name}")
        
        if not models_dictionary:
            logger.warning("[ILLUSTRATOR_TOOL] No models found in state")
            return "Error: No models available in session state."

        # Get the specific model's components
        model_components = models_dictionary.get(model_name)
        if not model_components:
            # Fallback: if model_name not found, try the first available model
            if len(models_dictionary) > 0:
                first_model = list(models_dictionary.keys())[0]
                logger.warning(f"[ILLUSTRATOR_TOOL] Model '{model_name}' not found. Using '{first_model}' instead.")
                model_name = first_model
                model_components = models_dictionary[first_model]
            else:
                logger.warning(f"[ILLUSTRATOR_TOOL] Model '{model_name}' not found in models_dictionary")
                return f"Error: Model '{model_name}' not found in session state."

        logger.info(f"[ILLUSTRATOR_TOOL] Model has {len(model_components)} components")

        # Format model components using hierarchical structure
        import json
        hierarchical_structure = format_hierarchical_components({model_name: model_components})
        formatted_components = json.dumps(hierarchical_structure, indent=2)

        # Build response with model info
        response_parts = [
            f"# Model Information for '{model_name}'\n",
            "## Model Components (JSON Structure)\n",
            "```json\n",
            formatted_components,
            "\n```"
        ]

        # Add source code if available
        models_code_cfg = cfg.get("models_code", {})
        if models_code_cfg and "local_resources" in models_code_cfg:
            code_files = models_code_cfg["local_resources"]
            logger.info(f"[ILLUSTRATOR_TOOL] Source code files available: {code_files}")
            response_parts.append(f"\n\n## Source Code Files\n")
            response_parts.append(f"Available code files: {', '.join(code_files)}")
            response_parts.append("\n(Code content can be retrieved if needed)")
        else:
            logger.info("[ILLUSTRATOR_TOOL] No source code available")
            response_parts.append("\n\n## Source Code\n(Source code not provided)")

        # Add infeasibility diagnosis if available
        has_diagnosis = tool_context.state.get(HAS_INFEASIBILITY_DIAGNOSIS, False)
        if has_diagnosis:
            diagnosis = tool_context.state.get(INFEASIBILITY_DIAGNOSIS, {})
            logger.info("[ILLUSTRATOR_TOOL] Including infeasibility diagnosis information")

            response_parts.append("\n\n## Infeasibility Diagnosis Results\n")

            # Extract diagnosis details
            status = diagnosis.get("status", "unknown")
            result_text = diagnosis.get("result", "No details available")
            relaxed_version = diagnosis.get("relaxed_version", "N/A")
            final_status = diagnosis.get("final_status", "unknown")
            slack_values = diagnosis.get("slack_values", {})

            response_parts.append(f"**Diagnosis Status**: {status}")
            response_parts.append(f"\n**Relaxed Model Version**: {relaxed_version}")
            response_parts.append(f"\n**Final Status**: {final_status}")
            response_parts.append(f"\n\n**Diagnosis Details**:\n{result_text}")

            # Display slack values (violation magnitudes) if available
            if slack_values:
                response_parts.append(f"\n\n**Constraint Violations (Slack Values)**:")
                response_parts.append(f"\nThese values show HOW MUCH each constraint was violated:")

                # Group slack values by constraint type for better readability
                from collections import defaultdict
                grouped_slacks = defaultdict(list)

                for idx, val in slack_values.items():
                    if val > 1e-6:  # Only show significant violations
                        # Handle both Round 1/2 format: (constraint_name, 'ub'/'lb')
                        # and Round 3 format: (constraint_name, index, type)
                        if isinstance(idx, tuple) and len(idx) >= 2:
                            constraint_name = idx[0]
                            grouped_slacks[constraint_name].append((idx, val))

                # Display grouped violations
                for constraint_name, violations in sorted(grouped_slacks.items()):
                    response_parts.append(f"\n\n- **{constraint_name}**:")
                    for idx, val in sorted(violations, key=lambda x: x[1], reverse=True)[:10]:  # Top 10 per constraint
                        response_parts.append(f"  - {idx}: {val:.4f}")
                    if len(violations) > 10:
                        response_parts.append(f"  - ... and {len(violations) - 10} more violations")

            # If relaxed model exists in models_dictionary, show its info
            if relaxed_version != "N/A" and relaxed_version in tool_context.state.get(MODELS_DICTIONARY, {}):
                relaxed_model_info = tool_context.state[MODELS_DICTIONARY][relaxed_version]
                obj_info = relaxed_model_info.get("obj", {})
                obj_value = obj_info.get("value", "unknown")

                response_parts.append(f"\n\n**Relaxed Model Solution**:")
                response_parts.append(f"\n- Objective Value: {obj_value}")
                response_parts.append(f"\n- Status: {final_status}")
        else:
            logger.info("[ILLUSTRATOR_TOOL] No infeasibility diagnosis needed (model is feasible)")

        result = "\n".join(response_parts)
        logger.info(f"[ILLUSTRATOR_TOOL] Generated model info ({len(result)} characters)")
        return result

    except Exception as e:
        logger.error(f"[ILLUSTRATOR_TOOL] Failed to retrieve model info: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return f"Error retrieving model information: {str(e)}"
