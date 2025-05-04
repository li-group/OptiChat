import pyomo.environ as pyo
import json

data = globals().get("data", {})
# with open('maintenance_data.json') as f:
#     data = json.load(f)

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.L = pyo.Set(initialize=data['sets']['Sections'])
model.I = pyo.Set(initialize=data['sets']['Event_Types'])
model.J = pyo.Set(initialize=data['sets']['Part_Numbers'])
model.T = pyo.Set(initialize=data['sets']['Months'])
model.K = pyo.Set(initialize=[tuple(k) for k in data['sets']['Section_Parts']], domain=model.L*model.J, 
                              doc="The Parts in Each Section")


# Define parameters
model.f = pyo.Param(model.K, model.I, initialize={
    eval(k): v for k, v in data['parameters']['Minimum_Maintenance_Periods'].items()
}, doc="Minimum Maintenance Periods for Each Event and Part", mutable=True)

model.c = pyo.Param(model.K, model.I, initialize={
    eval(k): v for k, v in data['parameters']['Event_Cost'].items()
}, doc="Event Cost (INR)", mutable=True)


model.w = pyo.Param(initialize=data['parameters']['Pullback_Window'], doc="Pullback Window (for event consolidation)", mutable=True)

model.oc = pyo.Param(initialize=data['parameters']['Shutdown_Cost'], doc="Shutdown Cost (INR)", mutable=True)

model.m = pyo.Param(initialize=data['parameters']['BigM'], doc="Big-M Value (maximum simultaneous events, lower case due to formatting rules)", mutable=True)

# Define variables
model.x = pyo.Var(model.K, model.I, model.T, domain=pyo.Binary, doc="Perform Event i for Part j in Section l at Time t")
model.o = pyo.Var(model.T, domain=pyo.Binary, doc="Perform Shutdown at Time t")
model.tc = pyo.Var(doc="Total Cost")

# Define constraints
def mtn_rule(model, l, j, i, t):
    if t + pyo.value(model.f[l, j, i]) > len(model.T):
         return pyo.Constraint.Feasible
    return sum(model.x[l, j, i, ft] for ft in model.T if ft >= t and ft <= t + pyo.value(model.f[l, j, i])) >= 1
model.mtn = pyo.Constraint(model.K, model.I, model.T, rule=mtn_rule, doc="Maintenance Scheduling Rule")

def shd_rule(model, t):
     return sum(model.x[l, j, i, t] for (l, j) in model.K for i in model.I) <= model.o[t]*model.m
model.shd = pyo.Constraint(model.T, rule=shd_rule, doc="Shutdown Rule")

def shw_rule(model, t):
     if t + pyo.value(model.w) > len(model.T):
          return pyo.Constraint.Feasible
     return sum(model.o[ft] for ft in model.T if ft >= t and ft <= t + pyo.value(model.w)) <= 1
model.shw = pyo.Constraint(model.T, rule=shw_rule, doc="Shutdown Window Rule")

def tcd_rule(model):
     return (sum(model.x[l, j, i, t]*model.c[l, j, i]
                 for (l, j) in model.K for i in model.I for t in model.T) 
            +model.oc*sum(model.o[t] for t in model.T)) == model.tc
model.tcd = pyo.Constraint(rule=tcd_rule, doc="Total Cost Definition")

# Define objective function
model.obj = pyo.Objective(expr=model.tc, sense=pyo.minimize)