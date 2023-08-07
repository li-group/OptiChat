#adapted from magic.gms : Magic Power Scheduling Problem (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_magic.html
from pyomo.environ import *

# Model
model = ConcreteModel()

# Sets
model.t = Set(initialize=['12pm-6am', '6am-9am', '9am-3pm', '3pm-6pm', '6pm-12pm'])  # demand blocks
model.g = Set(initialize=['type-1', 'type-2', 'type-3'])  # generators
pre_t = {'6am-9am': '12pm-6am', '9am-3pm': '6am-9am', '3pm-6pm': '9am-3pm', '6pm-12pm': '3pm-6pm'}
# Parameters
# Demand (1000mw) change the demand 6am-9am to 80
dem_init = {'12pm-6am': 15, '6am-9am': 80, '9am-3pm': 25, '3pm-6pm': 40, '6pm-12pm': 27}
model.dem = Param(model.t, mutable=True, initialize=dem_init)

# Duration (hours)
dur_init = {'12pm-6am': 6, '6am-9am': 3, '9am-3pm': 6, '3pm-6pm': 3, '6pm-12pm': 6}
model.dur = Param(model.t, mutable=True, initialize=dur_init)

# Generation data
gen_data = {
    'type-1': {'min-pow': 0.85, 'max-pow': 2.0, 'cost-min': 1000, 'cost-inc': 2.0, 'start': 2000, 'number': 12},
    'type-2': {'min-pow': 1.25, 'max-pow': 1.75, 'cost-min': 2600, 'cost-inc': 1.3, 'start': 1000, 'number': 10},
    'type-3': {'min-pow': 1.5, 'max-pow': 4.0, 'cost-min': 3000, 'cost-inc': 3.0, 'start': 500, 'number': 5}
}

# Split data parameter into individual parameters
model.minpow = Param(model.g, mutable=True, initialize={g: gen_data[g]['min-pow'] for g in model.g})
model.maxpow = Param(model.g, mutable=True,initialize={g: gen_data[g]['max-pow'] for g in model.g})
model.costmin = Param(model.g, mutable=True,initialize={g: gen_data[g]['cost-min'] for g in model.g})
model.costinc = Param(model.g, mutable=True,initialize={g: gen_data[g]['cost-inc'] for g in model.g})
model.start = Param(model.g, mutable=True,initialize={g: gen_data[g]['start'] for g in model.g})
model.number = Param(model.g, mutable=True, initialize={g: gen_data[g]['number'] for g in model.g})

# Variables
model.x = Var(model.g, model.t, within=NonNegativeReals)  # generator output (1000mw)
model.n = Var(model.g, model.t, within=NonNegativeIntegers)  # number of generators in use
model.s = Var(model.g, model.t, within=NonNegativeReals)  # number of generators started up

# Objective
def cost_rule(model):
    # total operating cost (l)
    return sum(
        model.dur[t] * model.costmin[g] * model.n[g, t] + model.start[g] * model.s[g, t] +
        1000 * model.dur[t] * model.costinc[g] * (model.x[g, t] - model.minpow[g] * model.n[g, t])
        for g in model.g for t in model.t)

model.cost = Objective(rule=cost_rule, sense=minimize)

# Constraints
def pow_rule(model, t):
    # demand for power (1000mw)
    return sum(model.x[g, t] for g in model.g) == model.dem[t]

model.pow = Constraint(model.t, rule=pow_rule)

def res_rule(model, t):
    # spinning reserve requirements (1000mw)
    return sum(model.maxpow[g] * model.n[g, t] for g in model.g) >= 1.15 * model.dem[t]

model.res = Constraint(model.t, rule=res_rule)

def st_rule(model, g, t):
    # start-up definition
    if t != '12pm-6am':
        return model.s[g, t] >= model.n[g, t] - model.n[g, pre_t[t]]
    else:
        return Constraint.Skip
model.st = Constraint(model.g, model.t, rule=st_rule)

# minimum generation level (1000mw)
def minu_rule(model, g, t):
    return model.x[g, t] >= model.minpow[g] * model.n[g, t]

model.minu = Constraint(model.g, model.t, rule=minu_rule)

# maximum generation level (1000mw)
def maxu_rule(model, g, t):
    return model.x[g, t] <= model.maxpow[g] * model.n[g, t]

model.maxu = Constraint(model.g, model.t, rule=maxu_rule)

# maximum number of generators in use
def maxn_rule(model, g, t):
    return model.n[g, t] <= model.number[g]
model.maxn = Constraint(model.g, model.t, rule=maxn_rule)