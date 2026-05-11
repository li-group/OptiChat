import random
import pyomo.environ as pyo


# -----------------------------
# Data
# -----------------------------

SEED = 42
N_SCENARIOS = 10

FIRST_STAGE_COST = 1.0
RECOURSE_COST = 5.0   # emergency purchase / shortage cost per unit

random.seed(SEED)

# Include both 0 and 100, then sample 8 more demands from 1 to 99
demands = [0, 100] + random.sample(range(1, 100), N_SCENARIOS - 2)
demands = sorted(demands)

scenarios = list(range(N_SCENARIOS))
prob = {s: 1.0 / N_SCENARIOS for s in scenarios}
demand = {s: demands[s] for s in scenarios}

expected_demand = sum(prob[s] * demand[s] for s in scenarios)

print("Scenarios")
for s in scenarios:
    print(f"scenario {s}: demand = {demand[s]}, probability = {prob[s]}")

print(f"\nExpected demand = {expected_demand:.2f}")

def build_expected_value_model():
    m = pyo.ConcreteModel("ExpectedValueModel")

    m.expected_demand = pyo.Param(initialize=expected_demand)

    # First-stage decision from the expected-value approximation
    m.x = pyo.Var(domain=pyo.NonNegativeReals)

    # Deterministic recourse against expected demand
    m.y = pyo.Var(domain=pyo.NonNegativeReals)

    m.demand_constraint = pyo.Constraint(
        expr=m.x + m.y >= m.expected_demand
    )

    m.obj = pyo.Objective(
        expr=FIRST_STAGE_COST * m.x + RECOURSE_COST * m.y,
        sense=pyo.minimize
    )

    return m


def evaluate_solution(x_value, label):
    rows = []

    first_stage_cost = FIRST_STAGE_COST * x_value

    expected_recourse_cost = 0.0
    expected_total_cost = first_stage_cost

    for s in scenarios:
        shortage = max(demand[s] - x_value, 0.0)
        q_s_x = RECOURSE_COST * shortage
        weighted_q_s_x = prob[s] * q_s_x

        expected_recourse_cost += weighted_q_s_x

        rows.append({
            "scenario": s,
            "demand": demand[s],
            "probability": prob[s],
            "x": x_value,
            "shortage": shortage,
            "Q_s(x)": q_s_x,
            "p_s Q_s(x)": weighted_q_s_x,
        })

    expected_total_cost += expected_recourse_cost

    print("\n" + "=" * 70)
    print(f"Evaluation of {label}")
    print("=" * 70)
    print(f"x = {x_value:.4f}")
    print(f"First-stage cost c*x = {first_stage_cost:.4f}")
    print(f"Expected recourse cost E[Q(x,D)] = {expected_recourse_cost:.4f}")
    print(f"Expected total cost = {expected_total_cost:.4f}")

    print("\nScenario recourse costs")
    print(
        f"{'s':>3} "
        f"{'demand':>8} "
        f"{'prob':>8} "
        f"{'shortage':>10} "
        f"{'Q_s(x)':>12} "
        f"{'p_s Q_s(x)':>14}"
    )

    for row in rows:
        print(
            f"{row['scenario']:>3} "
            f"{row['demand']:>8.2f} "
            f"{row['probability']:>8.2f} "
            f"{row['shortage']:>10.2f} "
            f"{row['Q_s(x)']:>12.2f} "
            f"{row['p_s Q_s(x)']:>14.2f}"
        )

    return {
        "x": x_value,
        "first_stage_cost": first_stage_cost,
        "expected_recourse_cost": expected_recourse_cost,
        "expected_total_cost": expected_total_cost,
        "rows": rows,
    }

solver = pyo.SolverFactory("gurobi")

# -----------------------------
# Solve stochastic extensive form
# -----------------------------

stoch_model = build_stochastic_extensive_form()
stoch_result = solver.solve(stoch_model, tee=False)

x_star = pyo.value(stoch_model.x)
stoch_objective = pyo.value(stoch_model.obj)

print("\n" + "=" * 70)
print("Stochastic extensive-form solution")
print("=" * 70)
print(f"x_star = {x_star:.4f}")
print(f"Stochastic objective value = {stoch_objective:.4f}")

for s in scenarios:
    print(
        f"scenario {s}: "
        f"demand = {demand[s]:.2f}, "
        f"y_star[{s}] = {pyo.value(stoch_model.y[s]):.4f}, "
        f"Q_s(x_star) = {RECOURSE_COST * pyo.value(stoch_model.y[s]):.4f}"
    )


# -----------------------------
# Solve expected-value model
# -----------------------------

ev_model = build_expected_value_model()
ev_result = solver.solve(ev_model, tee=False)

x_ev = pyo.value(ev_model.x)
ev_objective = pyo.value(ev_model.obj)

print("\n" + "=" * 70)
print("Expected-value solution")
print("=" * 70)
print(f"x_EV = {x_ev:.4f}")
print(f"Expected-value model objective = {ev_objective:.4f}")
print(f"EV model y = {pyo.value(ev_model.y):.4f}")


eval_x_star = evaluate_solution(x_star, "stochastic solution x_star")
eval_x_ev = evaluate_solution(x_ev, "expected-value solution x_EV")

# Common stochastic programming metric:
# VSS = value of the stochastic solution
#
# EEV = expected result of using the expected-value solution
# RP  = recourse problem objective, i.e. stochastic optimum
#
# VSS = EEV - RP for minimization

RP = eval_x_star["expected_total_cost"]
EEV = eval_x_ev["expected_total_cost"]
VSS = EEV - RP

print("\n" + "=" * 70)
print("Comparison")
print("=" * 70)
print(f"RP  = stochastic optimum expected cost        = {RP:.4f}")
print(f"EEV = expected cost using x_EV in scenarios   = {EEV:.4f}")
print(f"VSS = EEV - RP                                = {VSS:.4f}")