#adapted from bid.gms : Bid Evaluation (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_bid.html
import json
from pyomo.environ import *

model = ConcreteModel()

# Load JSON data
data = globals().get("data", {})
# with open('bid_inf_1_data.json') as f:
#     data = json.load(f)

# Extract vendors and segments from the bid_data
# Extract vendors and segments from the bid_data
vendors = list(data['bid_data'].keys())  # Extract all vendor names (keys of 'bid_data')
segments = sorted({int(segment) for vendor_data in data['bid_data'].values() for segment in vendor_data.keys()})  # Extract all unique segments

# vendors = ['a', 'b', 'c', 'd', 'e']
# segments = [1, 2, 3, 4, 5]

model.v = Set(initialize=vendors)  # vendors
model.s = Set(initialize=segments)  # segments

# Scalar
# model.req = Param(mutable=True, initialize=239600.48)  # requirements
model.req = Param(mutable=True, initialize=data['requirements']) # requirements


# Parameters
bid_init = {}
for vendor, segment_data in data['bid_data'].items():
    for segment, values in segment_data.items():
        bid_init[(vendor, segment, 'setup')] = values['setup']
        bid_init[(vendor, segment, 'price')] = values['price']
        bid_init[(vendor, segment, 'q-min')] = values['q-min']
        bid_init[(vendor, segment, 'q-max')] = values['q-max']
        

model.vs = Set(within=model.v * model.s, initialize=[(v, s) for v in vendors for s in segments if (v,s,'q-max') in bid_init.keys()])  # vendor bit possibilities
for (v,s) in model.vs:
    if (v,s+1) in model.vs:
        bid_init[(v, s + 1, 'setup')] = bid_init[(v, s, 'setup')] + bid_init[(v, s, 'q-max')] * (
                    bid_init[(v, s, 'price')] - bid_init[(v, s + 1, 'price')])

#bid data
model.setup = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'setup')] for vs in model.vs})
model.price = Param(model.vs,default=0, mutable=True, initialize={vs: bid_init[(*vs, 'price')] for vs in model.vs})
#divide qmin and qmax by 2
model.qmin = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-min')]/2 for vs in model.vs})
model.qmax = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-max')]/2 for vs in model.vs})

# Variables
model.c = Var(within=NonNegativeReals)  # total cost
model.pl = Var(model.vs, within=NonNegativeReals)  # purchase level
model.plb = Var(model.vs, within=Binary)  # purchase decision

# Constraints
def demand_rule(model):
    # demand constraint
    return model.req == sum(model.pl[vs] for vs in model.vs)
model.demand = Constraint(rule=demand_rule)

def costdef_rule(model):
    # cost definition
    return model.c == sum(model.price[vs]*model.pl[vs] + model.setup[vs]*model.plb[vs] for vs in model.vs)
model.costdef = Constraint(rule=costdef_rule)

def minpl_rule(model, v, s):
    # min purchase
    return model.pl[v, s] >= model.qmin[v, s]*model.plb[v, s]
model.minpl = Constraint(model.vs, rule=minpl_rule)

def maxpl_rule(model, v, s):
    # max purchase
    return model.pl[v, s] <= model.qmax[v, s]*model.plb[v, s]
model.maxpl = Constraint(model.vs, rule=maxpl_rule)

def oneonly_rule(model, v):
    # at most one deal
    return sum(model.plb[v, s] for s in model.s if (v,s) in model.vs) <= 1
model.oneonly = Constraint(model.v, rule=oneonly_rule)

# Objective
model.obj = Objective(expr=model.c, sense=minimize)
