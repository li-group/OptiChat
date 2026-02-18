#adapted from sparta.gms : Military Manpower Planning from Wagner (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_sparta.html
from pyomo.environ import *
import json

# Create a concrete model
model = ConcreteModel()

# Load JSON data
data = globals().get("data", {})
# with open("sparta_data.json", "r") as file:
#     data = json.load(file)

# Sets
model.t = Set(initialize=data["sets"]["time_periods_in_years"], doc='time periods (years)')
model.l = Set(initialize=data["sets"]["lengths_of_enlistment_in_years"], doc='length of enlistment (years)')

# Parameters
model.infl = Param(model.t, mutable=True, initialize={int(k): v for k, v in data["parameters"]["inflation_index_for_each_year"].items()},
                   doc='inflation index')
model.req = Param(model.t, mutable=True, initialize={int(k): v for k, v in data["parameters"]["troop_requirement_for_each_year"].items()},
                  doc='troop requirement')

model.clen = Param(model.l, mutable=True, initialize={int(k): v for k, v in data["parameters"]["cost_of_service"].items()},
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