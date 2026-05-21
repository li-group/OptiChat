import pyomo.environ as pyo

from extractor import LDRExtractor


def build_dummy_model():
    m = pyo.ConcreteModel()

    # -----------------------------
    # Uncertain parameters
    # -----------------------------
    m.d = pyo.Param(initialize=5.0, mutable=True)  # uncertain demand
    m.c = pyo.Param(initialize=2.0, mutable=True)  # uncertain cost coefficient

    # -----------------------------
    # Variables
    # -----------------------------
    m.x = pyo.Var(bounds=(0, None))
    m.y = pyo.Var(bounds=(0, 10))

    # -----------------------------
    # Constraints
    # -----------------------------
    # x + y >= d
    m.demand_con = pyo.Constraint(expr=m.x + m.y >= m.d)

    # x <= 8
    m.x_cap = pyo.Constraint(expr=m.x <= 8)

    # -----------------------------
    # Objective
    # -----------------------------
    # IMPORTANT:
    # Do NOT write (1 + m.c) * m.x.
    # The extractor does not distribute that expression.
    #
    # Write it explicitly as:
    #
    #     x + c*x + 3y
    #
    m.obj = pyo.Objective(
        expr=m.x + m.c * m.x + 3 * m.y,
        sense=pyo.minimize,
    )

    return m


if __name__ == "__main__":
    m = build_dummy_model()

    extractor = LDRExtractor()

    uncertain_params = [m.d, m.c]

    param_box = [
        (4.0, 7.0),  # d in [4, 7]
        (1.0, 3.0),  # c in [1, 3]
    ]

    primal_md = extractor.extract_primal_md(
        base=m,
        uncertain_params=uncertain_params,
        param_box=param_box,
    )

    print("\nVARIABLE IDS")
    print(primal_md.var_ids)

    print("\nVARIABLE BOUNDS")
    for k, v in primal_md.var_bounds.items():
        print(k, ":", v)

    print("\nVARIABLE DOMAINS")
    for k, v in primal_md.var_domain.items():
        print(k, ":", v)

    print("\nUNCERTAIN PARAMETER IDS")
    print(primal_md.uncertain_uids)

    print("\nPARAMETER BOX")
    print(primal_md.param_box)

    print("\nCONSTRAINT ROWS")
    for row in primal_md.constraints:
        print("\nname:", row.name)
        print("index:", row.index)
        print("sense:", row.sense)
        print("const:", row.const)
        print("var_coefs:", row.var_coefs)
        print("param_coefs:", row.param_coefs)

    print("\nOBJECTIVE")
    print("sense:", primal_md.obj_sense)
    print("obj_var_coef:", primal_md.obj_var_coef)
    print("obj_param_coef:", primal_md.obj_param_coef)
    print("obj_offset:", primal_md.obj_offset)