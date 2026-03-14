#adapted from bid.gms : Bid Evaluation (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_bid.html
import json
from pyomo.environ import *

model = ConcreteModel()

# Load JSON data (injected via config; falls back to empty dict)
data = globals().get("data", {})
# with open('bid_data.json') as f:
#     data = json.load(f)

# Extract vendors and segments from the bid_data
vendors = list(data['bid_data'].keys())
# JSON keys are strings — convert to int to keep types consistent with model.s
segments = sorted({int(segment) for vendor_data in data['bid_data'].values() for segment in vendor_data.keys()})

model.v = Set(initialize=vendors, doc="vendors")
model.s = Set(initialize=segments, doc="segments")

model.req = Param(mutable=True, initialize=data['requirements'], doc="requirements")

# Build bid_init with int segment keys (matching model.s type)
bid_init = {}
for vendor, segment_data in data['bid_data'].items():
    for segment, values in segment_data.items():
        s = int(segment)  # JSON keys are strings; cast to int to match model.s
        bid_init[(vendor, s, 'setup')] = values['setup']
        bid_init[(vendor, s, 'price')] = values['price']
        bid_init[(vendor, s, 'q-min')] = values['q-min']
        bid_init[(vendor, s, 'q-max')] = values['q-max']

model.vs = Set(within=model.v * model.s, initialize=[(v, s) for v in vendors for s in segments if (v, s, 'q-max') in bid_init], doc="vendor-segment possibilities")
for (v, s) in model.vs:
    if (v, s + 1) in model.vs:
        bid_init[(v, s + 1, 'setup')] = bid_init[(v, s, 'setup')] + bid_init[(v, s, 'q-max')] * (
                    bid_init[(v, s, 'price')] - bid_init[(v, s + 1, 'price')])

# Bid data parameters
model.setup = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'setup')] for vs in model.vs}, doc="setup cost for selecting a vendor-segment deal")
model.price = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'price')] for vs in model.vs}, doc="unit purchase price for a vendor-segment deal")
model.qmin  = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-min')] for vs in model.vs}, doc="minimum purchase quantity if a deal is chosen")
model.qmax  = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-max')] for vs in model.vs}, doc="maximum purchase quantity for a vendor-segment deal")

# Variables
model.c   = Var(within=NonNegativeReals, doc="total cost")
model.pl  = Var(model.vs, within=NonNegativeReals, doc="purchase level")
model.plb = Var(model.vs, within=Binary, doc="purchase decision")

# Constraints
def demand_rule(model):
    return model.req == sum(model.pl[vs] for vs in model.vs)
model.demand = Constraint(rule=demand_rule, doc="total purchased quantity must meet requirement")

def costdef_rule(model):
    return model.c == sum(model.price[vs] * model.pl[vs] + model.setup[vs] * model.plb[vs] for vs in model.vs)
model.costdef = Constraint(rule=costdef_rule, doc="total cost definition")

def minpl_rule(model, v, s):
    return model.pl[v, s] >= model.qmin[v, s] * model.plb[v, s]
model.minpl = Constraint(model.vs, rule=minpl_rule, doc="minimum purchase level when a deal is selected")

def maxpl_rule(model, v, s):
    return model.pl[v, s] <= model.qmax[v, s] * model.plb[v, s]
model.maxpl = Constraint(model.vs, rule=maxpl_rule, doc="maximum purchase level for each deal")

def oneonly_rule(model, v):
    # at most one deal per vendor — skip if vendor has no valid segments
    terms = [model.plb[v, s] for s in model.s if (v, s) in model.vs]
    if not terms:
        return Constraint.Skip
    return sum(terms) <= 1
model.oneonly = Constraint(model.v, rule=oneonly_rule, doc="at most one accepted deal per vendor")

# Objective
model.obj = Objective(expr=model.c, sense=minimize)
