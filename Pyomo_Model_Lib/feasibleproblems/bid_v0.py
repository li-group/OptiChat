#adapted from bid.gms : Bid Evaluation (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_bid.html
from pyomo.environ import *

model = ConcreteModel()

# Sets
vendors = ['a', 'b', 'c', 'd', 'e']
segments = [1, 2, 3, 4, 5]
model.v = Set(initialize=vendors)  # vendors
model.s = Set(initialize=segments)  # segments

# Scalar
model.req = Param(mutable=True, initialize=239600.48)  # requirements

# Parameters
bid_init = {('a', 1, 'setup'): 3855.84, ('a', 1, 'price'): 61.150, ('a', 1, 'q-min'): 0, ('a', 1, 'q-max'): 33000,
            ('b', 1, 'setup'): 125804.84, ('b', 1, 'price'): 68.099, ('b', 1, 'q-min'): 22000, ('b', 1, 'q-max'): 70000,
            ('b', 2, 'setup'): 0, ('b', 2, 'price'): 66.049, ('b', 2, 'q-min'): 70000, ('b', 2, 'q-max'): 100000,
            ('b', 3, 'setup'): 0, ('b', 3, 'price'): 64.099, ('b', 3, 'q-min'): 100000, ('b', 3, 'q-max'): 150000,
            ('b', 4, 'setup'): 0, ('b', 4, 'price'): 62.119, ('b', 4, 'q-min'): 150000, ('b', 4, 'q-max'): 160000,
            ('c', 1, 'setup'): 13456.00, ('c', 1, 'price'): 62.190, ('c', 1, 'q-min'): 0, ('c', 1, 'q-max'): 165600,
            ('d', 1, 'setup'): 6583.98, ('d', 1, 'price'): 72.488, ('d', 1, 'q-min'): 0, ('d', 1, 'q-max'): 12000,
            ('e', 1, 'setup'): 0, ('e', 1, 'price'): 70.150, ('e', 1, 'q-min'): 0, ('e', 1, 'q-max'): 42000,
            ('e', 2, 'setup'): 0, ('e', 2, 'price'): 68.150, ('e', 2, 'q-min'): 42000, ('e', 2, 'q-max'): 77000}

model.vs = Set(within=model.v * model.s, initialize=[(v, s) for v in vendors for s in segments if (v,s,'q-max') in bid_init.keys()])  # vendor bit possibilities
for (v,s) in model.vs:
    if (v,s+1) in model.vs:
        bid_init[(v, s + 1, 'setup')] = bid_init[(v, s, 'setup')] + bid_init[(v, s, 'q-max')] * (
                    bid_init[(v, s, 'price')] - bid_init[(v, s + 1, 'price')])

#bid data
model.setup = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'setup')] for vs in model.vs})
model.price = Param(model.vs,default=0, mutable=True, initialize={vs: bid_init[(*vs, 'price')] for vs in model.vs})
model.qmin = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-min')] for vs in model.vs})
model.qmax = Param(model.vs, default=0, mutable=True, initialize={vs: bid_init[(*vs, 'q-max')] for vs in model.vs})

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