# iis_demo.py
import itertools
import gurobipy as gp
from gurobipy import GRB


def build_model(selected_constraints=None, quiet=True):
    """
    Build a feasibility model with selected constraints.
    If selected_constraints is None, include all constraints.
    """

    m = gp.Model("iis_dummy")

    if quiet:
        m.Params.OutputFlag = 0

    # Make variables free by default, so all restrictions come from named constraints.
    x = m.addVar(lb=-GRB.INFINITY, name="x")
    y = m.addVar(lb=-GRB.INFINITY, name="y")
    z = m.addVar(lb=-GRB.INFINITY, name="z")

    all_constraints = {
        "x_ge_0": lambda: m.addConstr(x >= 0, name="x_ge_0"),
        "x_le_minus_1": lambda: m.addConstr(x <= -1, name="x_le_minus_1"),

        "y_ge_0": lambda: m.addConstr(y >= 0, name="y_ge_0"),
        "z_ge_0": lambda: m.addConstr(z >= 0, name="z_ge_0"),
        "y_plus_z_le_minus_1": lambda: m.addConstr(y + z <= -1, name="y_plus_z_le_minus_1"),
    }

    if selected_constraints is None:
        selected_constraints = list(all_constraints.keys())

    for name in selected_constraints:
        all_constraints[name]()

    # Dummy objective; we only care about feasibility.
    m.setObjective(0, GRB.MINIMIZE)

    return m


def gurobi_iis_demo():
    m = build_model()

    m.optimize()

    if m.Status != GRB.INFEASIBLE:
        print("Model was not infeasible. Something is wrong.")
        return

    m.computeIIS()

    print("\nIIS returned by Gurobi:")
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"  {c.ConstrName}")

    # Optional: write the IIS to a file you can inspect.
    m.write("iis_output.ilp")
    print("\nWrote IIS model to: iis_output.ilp")


def is_infeasible(constraint_names):
    m = build_model(selected_constraints=constraint_names)
    m.optimize()
    return m.Status == GRB.INFEASIBLE


def brute_force_all_iis():
    """
    This is only for tiny toy examples.
    It enumerates all subsets of constraints and checks which ones are IISs.
    """

    names = [
        "x_ge_0",
        "x_le_minus_1",
        "y_ge_0",
        "z_ge_0",
        "y_plus_z_le_minus_1",
    ]

    all_iis = []

    for r in range(1, len(names) + 1):
        for subset in itertools.combinations(names, r):
            subset = tuple(subset)

            if not is_infeasible(subset):
                continue

            # Check irreducibility:
            # removing any one constraint should make it feasible.
            irreducible = True
            for k in range(len(subset)):
                smaller = subset[:k] + subset[k + 1:]
                if is_infeasible(smaller):
                    irreducible = False
                    break

            if irreducible:
                all_iis.append(subset)

    print("\nAll IISs found by brute force:")
    for subset in all_iis:
        print(f"  size {len(subset)}: {subset}")

    min_size = min(len(s) for s in all_iis)
    print(f"\nMinimum-cardinality IIS size: {min_size}")
    print("Minimum-cardinality IISs:")
    for subset in all_iis:
        if len(subset) == min_size:
            print(f"  {subset}")


if __name__ == "__main__":
    gurobi_iis_demo()
    brute_force_all_iis()

