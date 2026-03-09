#adapted from marco.gms : Mini Oil Refining Model (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_marco.html

import json
from pyomo.environ import *

# Load external data
# data = json.load(open("macro_inf_1_data.json"))
data = globals().get("data", {})

# Extract sets from data
c = data["sets"]["c"]
cf = data["sets"]["cf"]
cr = data["sets"]["cr"]
ci = data["sets"]["ci"]
cd = data["sets"]["cd"]
p = data["sets"]["p"]
m = data["sets"]["m"]
q = data["sets"]["q"]

# Load bp_tuples and convert from list of lists to list of tuples
bp_tuples = [tuple(item) for item in data["sets"]["bp_tuples"]]

# Extract parameter data
params = data["parameters"]

# Load a_values and convert comma-separated keys to tuples
a_values = {tuple(k.split(',')): v for k, v in params["a_values"].items()}

# Load b_values and convert comma-separated keys to tuples
b_values = {tuple(k.split(',')): v for k, v in params["b_values"].items()}

# Load simple parameters
k_values = params["k_values"]
pd_values = params["pd_values"]
pr_values = params["pr_values"]
pf_values = params["pf_values"]
ur_values = params["ur_values"]
op_values = params["op_values"]

# Load qs_values and convert comma-separated keys to tuples
qs_values = {tuple(k.split(',')): v for k, v in params["qs_values"].items()}

# Load at_values and convert comma-separated keys to tuples
at_values = {tuple(k.split(',')): v for k, v in params["at_values"].items()}

# Load atc_values and convert comma-separated keys to tuples
atc_values = {tuple(k.split(',')): v for k, v in params["atc_values"].items()}

# Load demand parameter
demand_values = params["demand"]

# iterate over atc_values
for ci_ in ci:
    for cr_ in cr:
        for q_ in q:
            #if at_values has key (ci, q)
            if (ci_, q_) in at_values:
                atc_values[cr_,ci_,q_] = at_values[ci_, q_]

# Create a model
model = ConcreteModel()

# Define the sets
model.cr = Set(initialize=cr)
model.p = Set(initialize=p)
model.c = Set(initialize=c)
model.cf = Set(initialize=cf)
model.ci = Set(initialize=ci)
model.m = Set(initialize=m)
model.q = Set(initialize=q)
model.cd = Set(initialize=cd)
model.lim = Set(initialize=['lower', 'upper'])
model.bp = Set(within=model.cf*model.ci, initialize=bp_tuples)


# Define the parameters
model.a = Param(model.cr, model.c, model.p, default=0, mutable=True, initialize=a_values)
model.b = Param(model.m, model.p, default=0, mutable=True, initialize=b_values)
model.k = Param(model.m, default=0, mutable=True, initialize=k_values)
model.ur = Param(model.cr, default=0, mutable=True,initialize=ur_values)
model.qs = Param(model.lim, model.cf, model.q, mutable=True, default=0, initialize=qs_values)
model.atc = Param(model.cr, model.ci, model.q, default=0, mutable=True, initialize=atc_values)
model.pf = Param(model.cf, default=0, mutable=True, initialize=pf_values)
model.pr = Param(model.cr, default=0, mutable=True, initialize=pr_values)
model.pd = Param(model.cd, default=0, mutable=True, initialize=pd_values)
model.op = Param(model.p, default=0, mutable=True, initialize=op_values)
model.d = Param(model.cf, mutable=True, initialize=demand_values, doc='demand')

# Define the decision variables
model.z = Var(model.cr, model.p, domain=NonNegativeReals, doc="process level")
model.x = Var(model.cf, domain=NonNegativeReals, doc="final sales")
model.u = Var(model.cr, domain=NonNegativeReals, doc="purchase of crude oil")
model.ui = Var(model.cr, model.ci, domain=NonNegativeReals, doc="purchases of intermediate materials")
model.w = Var(model.cr, model.ci, model.cf, domain=NonNegativeReals, doc="blending process level")
model.phi = Var(domain=Reals, doc="total income")
model.phir = Var(domain=Reals, doc="revenue from final product sales")
model.phip = Var(domain=Reals, doc="input material cost")
model.phiw = Var(domain=Reals, doc="operating cost")


# Objective function
def objective_rule(model):
    return model.phir - model.phip - model.phiw
model.objective = Objective(rule=objective_rule, sense=maximize)

#material balances for crudes
def mbr_rule(model, cr):
    return sum(model.a[cr, 'crude', p] * model.z[cr, p] for p in model.p) + model.u[cr] >= 0
model.mbr = Constraint(model.cr, rule=mbr_rule)

# material balances for intermediates
def mb_rule(model, cr, ci):
    if ci in model.cd:
        return sum(model.a[cr, ci, p] * model.z[cr, p] for p in model.p) + model.ui[cr, ci] >= sum(model.w[cr, ci, cf] for cf in model.cf if (cf,ci) in model.bp)
    else:
        return sum(model.a[cr, ci, p] * model.z[cr, p] for p in model.p) >= sum(model.w[cr, ci, cf] for cf in model.cf if (cf,ci) in model.bp)
model.mb = Constraint(model.cr, model.ci, rule=mb_rule)

#capacity constraint
def cc_rule(model, m):
    return sum(model.b[m, p] * sum(model.z[cr, p] for cr in model.cr) for p in model.p) <= model.k[m]
model.cc = Constraint(model.m, rule=cc_rule)

#limits on crude oil purchases
def lcp_rule(model, cr):
    return model.u[cr] <= model.ur[cr]
model.lcp = Constraint(model.cr, rule=lcp_rule)

#blending balance
def bb_rule(model, cf):
    return model.x[cf] == sum(model.w[cr, ci, cf] for cr in model.cr for ci in model.ci if (cf,ci) in model.bp)
model.bb = Constraint(model.cf, rule=bb_rule)

#quality constraints lower bounds
def qlb_rule(model, cf, q):
    if ("lower", cf, q) in qs_values:
        return sum(model.atc[cr, ci, q] * model.w[cr, ci, cf] for cr in model.cr for ci in model.ci) >= model.qs["lower", cf, q] * model.x[cf]
    else:
        return Constraint.Skip
model.qlb = Constraint(model.cf, model.q, rule=qlb_rule)

#quality constraints upper bounds
def qub_rule(model, cf, q):
    if ("upper", cf, q) in qs_values:
        return sum(model.atc[cr, ci, q] * model.w[cr, ci, cf] for cr in model.cr for ci in model.ci) <= model.qs["upper", cf, q] * model.x[cf]
    else:
        return Constraint.Skip
model.qub = Constraint(model.cf, model.q, rule=qub_rule)

#revenue accounting
def arev_rule(model):
    return model.phir == sum(model.pf[cf] * model.x[cf] for cf in model.cf)
model.arev = Constraint(rule=arev_rule)

#material cost accounting
def amat_rule(model):
    return model.phip == sum(model.pr[cr] * model.u[cr] for cr in model.cr) + sum(model.pd[cd] * model.ui[cr, cd] for cd in model.cd for cr in model.cr)
model.amat = Constraint(rule=amat_rule)

#operating cost accounting
def aoper_rule(model):
    return model.phiw == sum(model.op[p] * sum(model.z[cr, p] for cr in model.cr) for p in model.p)
model.aoper = Constraint(rule=aoper_rule)

#satisfy demand
def demand_rule(model, cf):
    return model.x[cf] >= model.d[cf]
model.demand = Constraint(model.cf, rule=demand_rule)
