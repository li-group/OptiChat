CODE_INSTRUCTION_GRAMMAR = r"""
start         : model_line shortcut_functions_block? changes_block? print_block description_line?

model_line    : "MODEL:" model_name
shortcut_functions_block : "SHORTCUT_FUNCTIONS:" func_item+
func_item     : "  - " cname

changes_block : "CHANGES:" change_item+
change_item   : "  - " /[^\n]+/

print_block   : "PRINT:" print_item+    ← REQUIRED: always include; tells the generator what to output
print_item    : "  - " /[^\n]+/

description_line : "DESCRIPTION:" /[^\n]+/

model_name    : /[a-zA-Z_][a-zA-Z0-9_]*/
cname         : /[a-zA-Z_][a-zA-Z0-9_]*/
"""

# ---------------------------------------------------------------------------
# PRINT guidance — what to include per task type
# ---------------------------------------------------------------------------

PRINT_GUIDANCE = """\
PRINT BLOCK RULES (apply to every generator_agent call):
  • PRINT is REQUIRED — the generator produces no useful output without it.
  • SENSITIVITY  : "dual value of <constraint>[idx] for each <idx>"
                   Use the constraint that directly limits the queried parameter
                   (see <model_description> Part 2c: Constraint–Parameter mapping, RHS driven by: <param>).
                   Example: if query is about `aa[d]` and Part 2c shows `avail_constraint[d]` is RHS-driven by `aa`,
                   write: "dual value of avail_constraint[d] for each aircraft type d"
  • WHAT_IF      : objective value + values of the decision variables most affected by the change.
  • WHY_NOT      : objective value + forced variable value + competing variables that show the trade-off.
  • FEASIBILITY  : objective value + values of the relaxed/restored variables.
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
  - objective value
DESCRIPTION: Sensitivity analysis on supply capacity; add dual suffix and re-solve to compute shadow prices showing marginal cost of one additional supply unit at each node."""

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
        f"{PRINT_GUIDANCE}\n"
        "EXAMPLES:\n\n"
        f"# WHAT_IF\n{EXAMPLE_WHAT_IF}\n\n"
        f"# SENSITIVITY\n{EXAMPLE_SENSITIVITY}\n\n"
        f"# WHY_NOT\n{EXAMPLE_WHY_NOT}\n\n"
        f"# FEASIBILITY\n{EXAMPLE_FEASIBILITY}\n\n"
    )
