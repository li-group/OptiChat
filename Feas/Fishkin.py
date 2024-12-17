import pyomo.environ as pyo

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.L = pyo.RangeSet(2, doc="Sections")
model.I = pyo.Set(initialize=["Repair", "Replace"], doc="Event Types")
model.J = pyo.RangeSet(10, doc="Part Numbers")
model.T = pyo.RangeSet(25, doc="Month Number (Time Index)")
model.K = pyo.Set(initialize=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
                              (2, 1), (2, 7), (2, 8), (2, 9), (2, 10)], domain=model.L*model.J, 
                              doc="The Parts in Each Section")

# Define parameters
model.f = pyo.Param(model.K, model.I, initialize={
    (1, 1, "Repair"): 3,
    (1, 1, "Replace"): 5,
    (1, 2, "Repair"): 3,
    (1, 2, "Replace"): 7,
    (1, 3, "Repair"): 3,
    (1, 3, "Replace"): 4,
    (1, 4, "Repair"): 4,
    (1, 4, "Replace"): 6,
    (1, 5, "Repair"): 4,
    (1, 5, "Replace"): 9,
    (1, 6, "Repair"): 5,
    (1, 6, "Replace"): 9,
    (2, 1, "Repair"): 4,
    (2, 1, "Replace"): 6,
    (2, 7, "Repair"): 5,
    (2, 7, "Replace"): 9,
    (2, 8, "Repair"): 4,
    (2, 8, "Replace"): 8,
    (2, 9, "Repair"): 3,
    (2, 9, "Replace"): 7,
    (2, 10, "Repair"): 4,
    (2, 10, "Replace"): 10,
}, doc="Minimum Maintenance Periods for Each Event and Part", mutable=True)

model.c = pyo.Param(model.K, model.I, initialize={
    (1, 1, "Repair"): 200,
    (1, 1, "Replace"): 300,
    (1, 2, "Repair"): 350,
    (1, 2, "Replace"): 250,
    (1, 3, "Repair"): 100,
    (1, 3, "Replace"): 300,
    (1, 4, "Repair"): 300,
    (1, 4, "Replace"): 500,
    (1, 5, "Repair"): 300,
    (1, 5, "Replace"): 100,
    (1, 6, "Repair"): 100,
    (1, 6, "Replace"): 100,
    (2, 1, "Repair"): 200,
    (2, 1, "Replace"): 300,
    (2, 7, "Repair"): 250,
    (2, 7, "Replace"): 150,
    (2, 8, "Repair"): 200,
    (2, 8, "Replace"): 300,
    (2, 9, "Repair"): 200,
    (2, 9, "Replace"): 180,
    (2, 10, "Repair"): 200,
    (2, 10, "Replace"): 190,
}, doc="Event Cost (INR)", mutable=True)

model.w = pyo.Param(initialize=1, doc="Pullback Window (for event consolidation)", mutable=True)

model.oc = pyo.Param(initialize=2000, doc="Shutdown Cost (INR)", mutable=True)

model.m = pyo.Param(initialize=22, doc="Big-M Value (maximum simultaneous events, lower case due to formatting rules)", mutable=True)

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