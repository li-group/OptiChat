# Pyomo model for the Aircraft landing problem. I propose an MILP reformulation of the problem discussed in https://www.frontiersin.org/articles/10.3389/ffutr.2022.968957/full. 

import pyomo.environ as pyo
import numpy as np
# Create a ConcreteModel
model = pyo.ConcreteModel()

# Define parameters
# Define sets
model.N_aircraft = pyo.Param(initialize=5, doc="number of aircraft in approach")
model.K_runway = pyo.Param(initialize=2, doc="Number of runway")
model.N = pyo.RangeSet(model.N_aircraft)  # aircraft set
model.K = pyo.RangeSet(model.K_runway)  # Runway Set
model.c_plus = pyo.Param(model.N, initialize={i: 10 * i for i in range(1, model.N_aircraft + 1)},
                         doc="Cost of late landing for each aircraft", mutable=True)
model.S = pyo.Param(model.N, model.N, initialize={(i, j): 5 for i in range(1, model.N_aircraft + 1) for j in
                                                  range(1 + i, model.N_aircraft + 1)},
                    doc="Cost of late landing for each aircraft", mutable=True)
model.E = pyo.Param(model.N, initialize={i: 5 for i in range(1, model.N_aircraft + 1)},
                    doc="earliest allowed arrival times for aircraft", mutable=True)
model.L = pyo.Param(model.N, initialize={i: 1000 for i in range(1, model.N_aircraft + 1)},
                    doc="latest allowed arrival times for aircraft ", mutable=True)
model.T = pyo.Param(model.N, initialize={i: 5 for i in range(1, model.N_aircraft + 1)},
                    doc="Initial scheduled arrival time for aircraft", mutable=True)

model.M = pyo.Param(initialize=20, doc="Big-M formulation, Mbig enough", mutable=True)
# Define binary decision variables
model.w = pyo.Var(model.N, model.N, within=pyo.Binary, doc='1 if aircraft i lands before aircraft j 0 otherwise')
model.x = pyo.Var(model.N, model.K, within=pyo.Binary, doc="1 if aircraft i is allocated to runway k")

# Define arrival time variables
model.t = pyo.Var(model.N, within=pyo.NonNegativeReals, doc="landing time of aircraft i")


# Define objective function
def operating_cost(m):
    return sum(m.c_plus[i] * (m.t[i] - m.T[i]) for i in m.N)


model.operating_cost = pyo.Objective(rule=operating_cost, sense=pyo.minimize,
                                     doc='Total cost is built based on landing delay')


def constraint_arrival_time(m, i):
    return m.t[i] >= m.T[i]


model.arrival_time_constr = pyo.Constraint(model.N, rule=constraint_arrival_time,
                                         doc="Maximu landing time greater than expected landing time, i.e. flight is delayed")


def upper_time_window_constr(m, i):
    return m.t[i] <= m.L[i]


def lower_time_window_constr(m, i):
    return m.E[i] <= m.t[i]


model.upper_time_window_constr = pyo.Constraint(model.N, rule=upper_time_window_constr,
                                                doc="upper time limit of the landing window")
model.lower_time_window_constr = pyo.Constraint(model.N, rule=lower_time_window_constr,
                                                doc="lower time limit of the landing window")


def assignment_constraint_rule(m, i):
    return sum(m.x[i, k] for k in m.K) == 1


model.assignment_constr = pyo.Constraint(model.N, rule=assignment_constraint_rule,
                                         doc="make sure every aircraft is assigned to a runway")


def sequencing_constraint_rule1(m, i, j, k):
    if i < j:
        return m.t[i] + m.S[i, j] - m.t[j] <= m.M * (1 - m.w[i, j]) + m.M * (
                    1 - m.x[i, k]) + m.M * (1 - m.x[j, k])
    else:
        return m.t[i] >= m.T[i]


model.sequencing_constr1 = pyo.Constraint(model.N, model.N, model.K, rule=sequencing_constraint_rule1,
                                          doc="make sure that the minimum separation time Sij has to be ensured between two landing aircraft i and j.active only f i lands before j")


def sequencing_constraint_rule2(m, i, j, k):
    if j > i:
        return m.t[j] + m.S[i, j] - m.t[i] <= m.M * m.w[i, j] + m.M * (
                    1 - m.x[i, k]) + m.M * (1 - m.x[j, k])
    else:
        return m.t[i] >= m.T[i]


model.sequencing_constr2 = pyo.Constraint(model.N, model.N, model.K, rule=sequencing_constraint_rule2,
                                          doc="make sure that the minimum separation time Sij has to be ensured between two landing aircraft j and i. active only f j lands before i")

