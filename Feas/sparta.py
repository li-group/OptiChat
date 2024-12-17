#adapted from sparta.gms : Military Manpower Planning from Wagner (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_sparta.html
from pyomo.environ import *

# Create a concrete model
model = ConcreteModel()

# Sets
model.t = Set(initialize=list(range(1, 11)), doc='time periods (years)')
model.l = Set(initialize=list(range(1, 5)), doc='length of enlistment (years)')

# Parameters
model.infl = Param(model.t, mutable=True, initialize={1: 1.00, 2: 1.05, 3: 1.12, 4: 1.71, 5: 1.80,
                                        6: 1.90, 7: 1.97, 8: 2.10, 9: 2.22, 10: 2.38},
                   doc='inflation index')
model.req = Param(model.t, mutable=True, initialize={1: 5, 2: 6, 3: 7, 4: 6, 5: 4,
                                       6: 9, 7: 8, 8: 8, 9: 6, 10: 4},
                  doc='troop requirement')
model.clen = Param(model.l, mutable=True, initialize={1: 50, 2: 85, 3: 115, 4: 143},
                   doc='cost of service')

# Variables
model.x = Var(model.t, model.l, within=NonNegativeReals, doc='recruits by year and length of enlistment')
model.e = Var(model.t, within=NonNegativeReals, doc='enlisted men')
model.z = Var(within=NonNegativeReals, doc='total cost')

# Objective
def cost_rule(mod):
    return mod.z == sum(mod.infl[i]*mod.clen[j]*mod.x[i, j] for i in mod.t for j in mod.l)
model.cost_def = Constraint(rule=cost_rule, doc='cost definition')

model.cost = Objective(expr=model.z)

# Constraints
def bal_rule(mod, i):
    return mod.e[i] == (mod.e[i-1] if i > 1 else 0) + sum(mod.x[i, j] - (mod.x[i - j, j] if i > j else 0) for j in mod.l)
model.bal = Constraint(model.t, rule=bal_rule, doc='troop balance - stock balance')

def min_req_rule(mod, i):
    return mod.e[i] >= mod.req[i]
model.min_req = Constraint(model.t, rule=min_req_rule)