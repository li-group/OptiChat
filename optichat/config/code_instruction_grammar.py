CODE_INSTRUCTION_GRAMMAR = r"""
start         : model_line shortcut_functions_block? changes_block? print_block description_line?

model_line    : "MODEL:" model_name
shortcut_functions_block : "SHORTCUT_FUNCTIONS:" func_item+
func_item     : "  - " cname

changes_block : "CHANGES:" change_item+
change_item   : "  - " /[^\n]+/

print_block   : "PRINT:" print_item+
print_item    : "  - " /[^\n]+/

description_line : "DESCRIPTION:" /[^\n]+/

model_name    : /[a-zA-Z_][a-zA-Z0-9_]*/
cname         : /[a-zA-Z_][a-zA-Z0-9_]*/
"""

# ---------------------------------------------------------------------------
# Reference examples (one per task type)
# ---------------------------------------------------------------------------

EXAMPLE_WHAT_IF = """\
MODEL: supply_chain
SHORTCUT_FUNCTIONS:
  - modify_and_solve
CHANGES:
  - demand[3,1]: increase by 10
PRINT:
  - total cost (objective value)
  - supply[i,j] for all supply nodes i and customers j
DESCRIPTION: What-if demand at customer 3 segment 1 increases by 10 units; evaluate total cost delta and supply flow redistribution."""

EXAMPLE_SENSITIVITY = """\
MODEL: supply_chain
SHORTCUT_FUNCTIONS:
  - add_dual_suffix
CHANGES:
  - add dual suffix to the model
PRINT:
  - dual value of supply_constraint[i] for each supply node i
DESCRIPTION: Sensitivity analysis on supply capacity constraints; add dual suffix and re-solve to compute shadow prices showing marginal cost of one additional supply unit."""

EXAMPLE_WHY_NOT = """\
MODEL: diet
CHANGES:
  - force_beef (new constraint): buy["BEEF"] at least 1 serving
PRINT:
  - total cost (objective value)
  - buy[f] for each food f
DESCRIPTION: Why-not forcing at least 1 serving of BEEF; reveals cost penalty and binding nutritional constraints that prevent BEEF from appearing in the optimal diet solution."""

EXAMPLE_FEASIBILITY = """\
MODEL: diet_inf
CHANGES:
  - calorie_constraint: deactivate
  - calorie_relaxed (new constraint): total calories at least 1800 kcal/day
PRINT:
  - total cost (objective value)
  - buy[f] for each food f
DESCRIPTION: Feasibility restoration: relax calorie lower bound from 2000 to 1800 kcal per day, the minimum slack from IIS analysis, to restore a feasible diet solution."""


def get_grammar_reference() -> str:
    """Return a compact grammar + examples block for inclusion in agent prompts."""
    return (
        "INSTRUCTION GRAMMAR (Lark EBNF):\n"
        f"{CODE_INSTRUCTION_GRAMMAR}\n"
        "EXAMPLES:\n\n"
        f"# WHAT_IF\n{EXAMPLE_WHAT_IF}\n\n"
        f"# SENSITIVITY\n{EXAMPLE_SENSITIVITY}\n\n"
        f"# WHY_NOT\n{EXAMPLE_WHY_NOT}\n\n"
        f"# FEASIBILITY\n{EXAMPLE_FEASIBILITY}\n\n"
    )
