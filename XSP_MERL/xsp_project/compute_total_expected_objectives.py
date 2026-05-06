from pathlib import Path
import csv
import re


# ============================================================
# Input folders
# ============================================================

RESULTS_FIRST_STAGE_COST = "results_first_stage_cost"
RESULTS_RECOURSE_Q = "results_recourse_Q"
RESULTS_XSTAR_XEV = "results_xstar_xev"
OUTPUT_FOLDER = "results_total_objectives"


# ============================================================
# Paths
# ============================================================

project_dir = Path(__file__).resolve().parent

first_stage_cost_path = (
    project_dir
    / RESULTS_FIRST_STAGE_COST
    / "first_stage_cost_summary.csv"
)

scenario_summary_path = (
    project_dir
    / RESULTS_XSTAR_XEV
    / "scenario_summary.csv"
)

q_x_star_path = (
    project_dir
    / RESULTS_RECOURSE_Q
    / "Q_all_scenarios_x_star.csv"
)

q_x_ev_path = (
    project_dir
    / RESULTS_RECOURSE_Q
    / "Q_all_scenarios_x_EV.csv"
)

out_dir = project_dir / OUTPUT_FOLDER
out_dir.mkdir(exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def scenario_sort_key(name):
    """
    Sort SCEN1, SCEN2, ..., SCEN200 numerically.
    """
    match = re.search(r"\d+", name)
    if match:
        return int(match.group())
    return name


def read_first_stage_costs(path):
    """
    Reads:
        solution,first_stage_cost
        x_star,...
        x_EV,...
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run first:\n"
            "  python compute_first_stage_cost.py"
        )

    costs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            solution = row["solution"]
            cost = float(row["first_stage_cost"])
            costs[solution] = cost

    required = {"x_star", "x_EV"}
    missing = required - set(costs.keys())

    if missing:
        raise RuntimeError(f"Missing first-stage costs for: {missing}")

    return costs


def read_probabilities(path):
    """
    Reads:
        scenario,parent,stage,probability,num_modifications
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run first:\n"
            "  python solve_xstar_xev.py"
        )

    probs = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            probs[row["scenario"]] = float(row["probability"])

    return probs


def read_Q_values(path):
    """
    Reads:
        scenario,termination_condition,Q_value,num_fixed_first_stage_vars
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run first:\n"
            "  python compute_recourse_Q.py --x x_star --all\n"
            "  python compute_recourse_Q.py --x x_EV --all"
        )

    q_values = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            scenario = row["scenario"]
            term = row["termination_condition"]
            q_raw = row["Q_value"]

            if term.lower() != "optimal":
                raise RuntimeError(
                    f"{path.name}: scenario {scenario} was not optimal. "
                    f"Termination condition: {term}"
                )

            if q_raw in ["", "None", None]:
                raise RuntimeError(
                    f"{path.name}: missing Q_value for scenario {scenario}"
                )

            q_values[scenario] = float(q_raw)

    return q_values


def compute_total_for_solution(solution_name, cTx, Q_values, probabilities):
    """
    Computes:
        c^T x
        sum_s p_s Q_s(x)
        c^T x + sum_s p_s Q_s(x)

    Also returns scenario-level contribution rows.
    """
    q_scenarios = set(Q_values.keys())
    p_scenarios = set(probabilities.keys())

    if q_scenarios != p_scenarios:
        only_q = sorted(q_scenarios - p_scenarios, key=scenario_sort_key)
        only_p = sorted(p_scenarios - q_scenarios, key=scenario_sort_key)

        raise RuntimeError(
            f"Scenario mismatch for {solution_name}.\n"
            f"Only in Q file: {only_q}\n"
            f"Only in probability file: {only_p}"
        )

    scenario_names = sorted(q_scenarios, key=scenario_sort_key)

    weighted_recourse_sum = 0.0
    unweighted_recourse_sum = 0.0
    rows = []

    for scen in scenario_names:
        p_s = probabilities[scen]
        q_s = Q_values[scen]
        weighted_q = p_s * q_s

        weighted_recourse_sum += weighted_q
        unweighted_recourse_sum += q_s

        rows.append({
            "solution": solution_name,
            "scenario": scen,
            "probability": p_s,
            "Q_s_x": q_s,
            "p_s_Q_s_x": weighted_q,
        })

    total_objective = cTx + weighted_recourse_sum

    summary = {
        "solution": solution_name,
        "first_stage_cost_cTx": cTx,
        "expected_recourse_cost_sum_p_Q": weighted_recourse_sum,
        "total_expected_objective_cTx_plus_sum_p_Q": total_objective,
        "unweighted_sum_Q_for_diagnostics": unweighted_recourse_sum,
        "num_scenarios": len(scenario_names),
        "probability_sum": sum(probabilities[s] for s in scenario_names),
    }

    return summary, rows


def write_rows(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

print("Reading inputs...")

first_stage_costs = read_first_stage_costs(first_stage_cost_path)
probabilities = read_probabilities(scenario_summary_path)

Q_x_star = read_Q_values(q_x_star_path)
Q_x_EV = read_Q_values(q_x_ev_path)

print("Number of scenarios:", len(probabilities))
print("Probability sum:", sum(probabilities.values()))


# ============================================================
# Compute totals
# ============================================================

summary_star, rows_star = compute_total_for_solution(
    solution_name="x_star",
    cTx=first_stage_costs["x_star"],
    Q_values=Q_x_star,
    probabilities=probabilities,
)

summary_ev, rows_ev = compute_total_for_solution(
    solution_name="x_EV",
    cTx=first_stage_costs["x_EV"],
    Q_values=Q_x_EV,
    probabilities=probabilities,
)

summary_rows = [summary_star, summary_ev]
scenario_rows = rows_star + rows_ev


# ============================================================
# Optional: compute VSS diagnostic
# ============================================================

z_SP_from_Q = summary_star["total_expected_objective_cTx_plus_sum_p_Q"]
z_EEV_from_Q = summary_ev["total_expected_objective_cTx_plus_sum_p_Q"]
VSS_from_Q = z_EEV_from_Q - z_SP_from_Q


# ============================================================
# Write outputs
# ============================================================

summary_path = out_dir / "total_expected_objective_summary.csv"
scenario_contrib_path = out_dir / "scenario_recourse_contributions.csv"
txt_summary_path = out_dir / "total_expected_objective_summary.txt"

write_rows(
    summary_path,
    summary_rows,
    fieldnames=[
        "solution",
        "first_stage_cost_cTx",
        "expected_recourse_cost_sum_p_Q",
        "total_expected_objective_cTx_plus_sum_p_Q",
        "unweighted_sum_Q_for_diagnostics",
        "num_scenarios",
        "probability_sum",
    ],
)

write_rows(
    scenario_contrib_path,
    scenario_rows,
    fieldnames=[
        "solution",
        "scenario",
        "probability",
        "Q_s_x",
        "p_s_Q_s_x",
    ],
)

with open(txt_summary_path, "w") as f:
    f.write("Total expected objective summary\n")
    f.write("================================\n\n")

    f.write("For x_star:\n")
    f.write(f"  c^T x_star:                  {summary_star['first_stage_cost_cTx']}\n")
    f.write(f"  sum_s p_s Q_s(x_star):       {summary_star['expected_recourse_cost_sum_p_Q']}\n")
    f.write(f"  c^T x_star + sum_s p_s Q_s:  {z_SP_from_Q}\n\n")

    f.write("For x_EV:\n")
    f.write(f"  c^T x_EV:                    {summary_ev['first_stage_cost_cTx']}\n")
    f.write(f"  sum_s p_s Q_s(x_EV):         {summary_ev['expected_recourse_cost_sum_p_Q']}\n")
    f.write(f"  c^T x_EV + sum_s p_s Q_s:    {z_EEV_from_Q}\n\n")

    f.write("Diagnostic:\n")
    f.write(f"  z_EEV - z_SP:                {VSS_from_Q}\n")


# ============================================================
# Print results
# ============================================================

print("\n============================================================")
print("Total expected objectives")
print("============================================================")

print("\nx_star:")
print("  c^T x_star                 =", summary_star["first_stage_cost_cTx"])
print("  sum_s p_s Q_s(x_star)      =", summary_star["expected_recourse_cost_sum_p_Q"])
print("  total                      =", z_SP_from_Q)

print("\nx_EV:")
print("  c^T x_EV                   =", summary_ev["first_stage_cost_cTx"])
print("  sum_s p_s Q_s(x_EV)        =", summary_ev["expected_recourse_cost_sum_p_Q"])
print("  total                      =", z_EEV_from_Q)

print("\nDiagnostic:")
print("  z_EEV - z_SP               =", VSS_from_Q)

print("\nWrote outputs to:")
print(" ", summary_path)
print(" ", scenario_contrib_path)
print(" ", txt_summary_path)
print("============================================================")