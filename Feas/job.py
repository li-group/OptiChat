import pyomo.environ as pyo

# Create the Pyomo model
model = pyo.ConcreteModel()


# Define sets
model.J = pyo.Set(initialize=[1,2,3,4,5,6,7], doc="Jobs")
model.P = pyo.Set(within=model.J*model.J, initialize=[(1,4), (2,4), (2,3), (3,5), (3,6), (4,7)], doc="Pairs of jobs where the first job is a predecessor of the second job")
# Define parameters and sets
model.mincost = pyo.Param(model.J, initialize = {1:1600, 2:2400, 3:2900, 4:1900, 5:3800, 6: 2900, 7: 1300}, doc = "Cost For Minimum Number of Days For Each Job", mutable = True)
model.maxcost = pyo.Param(model.J, initialize = {1:1000, 2:1800, 3:2000, 4:1300, 5:2000, 6: 2200, 7: 800}, doc = "Cost For Maximum Number of Days For Each Job", mutable = True)
model.mintime = pyo.Param(model.J, initialize = {1:6, 2:8, 3:16, 4:14, 5:4, 6: 12, 7: 2}, doc = "Minimum Time For Jobs", mutable = True)
model.maxtime = pyo.Param(model.J, initialize = {1:12, 2:16, 3:24, 4:20, 5:16, 6: 16, 7: 12}, doc = "Maximum Time For Jobs", mutable = True)
model.totaldays = pyo.Param(initialize = 40, doc = "Total Number Of Days For Work", mutable = True)
# Define variables
model.s = pyo.Var(model.J, within = pyo.PositiveIntegers, doc = "Start Times For Each Job")
model.t = pyo.Var(model.J, within = pyo.PositiveIntegers, doc = "Total Times For Each Job")


# Define constraints
#add min and max time constraints
def mintime_rule(model, j):
    return (model.t[j]) >= model.mintime[j]
model.mintimerule = pyo.Constraint(model.J, rule = mintime_rule, doc = "Enforce The Minimum Amount Of Time A Job Must Take To Complete")


def maxtime_rule(model, j):
    return (model.t[j]) <= model.maxtime[j]
model.maxtimerule = pyo.Constraint(model.J, rule = maxtime_rule, doc = "Enforce The Maximum Amount Of Time A Job Must Take To Complete")


#add total time constraints
def totaltime_rule(model, j):
    return (model.s[j] + model.t[j]) <= model.totaldays
model.totaltimerule = pyo.Constraint(model.J, rule = totaltime_rule, doc = "Total Amount Of Time To Complete All Jobs")


#add predecessor constraints
def predecessors_rule(model, i, j):
    return model.s[i] + model.t[i] <= model.s[j]
model.predecessorsrule = pyo.Constraint(model.P, rule=predecessors_rule, doc="Predecessor Constraints For Each Job")


# Define objective function
model.obj = pyo.Objective(expr=sum(
        (model.mincost[i] + ((model.t[i]-model.mintime[i])*((model.maxcost[i]-model.mincost[i])/(model.maxtime[i]-model.mintime[i])))) for i in model.J), sense=pyo.minimize)