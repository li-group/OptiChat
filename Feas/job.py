import pyomo.environ as pyo
import json
import ast
# Create the Pyomo model
model = pyo.ConcreteModel()

#import data from json
data = globals().get("data", {})
# with open('job_data.json') as f:
#     data = json.load(f)

# Define sets
model.J = pyo.Set(initialize=data['sets']['Jobs'], doc="Jobs")
# print("Contents of model.J:")
# for j in model.J:
#     print(j)
    
p_tuples = [ast.literal_eval(item) for item in data['sets']['job_pair']]
model.P = pyo.Set(within=model.J*model.J, initialize=p_tuples, doc="Pairs of jobs where the first job is a predecessor of the second job")
# Define parameters and sets
# mincost_data = {int(k): v for k, v in data["parameters"]["mincost"].items()}
# print(mincost_data)

model.mincost = pyo.Param(model.J, initialize = {int(k): v for k, v in data["parameters"]["mincost"].items()}, doc = "Cost For Minimum Number of Days For Each Job", mutable = True)
model.maxcost = pyo.Param(model.J, initialize = {int(k): v for k, v in data["parameters"]["maxcost"].items()}, doc = "Cost For Maximum Number of Days For Each Job", mutable = True)
model.mintime = pyo.Param(model.J, initialize = {int(k): v for k, v in data["parameters"]["mintime"].items()}, doc = "Minimum Time For Jobs", mutable = True)
model.maxtime = pyo.Param(model.J, initialize = {int(k): v for k, v in data["parameters"]["maxtime"].items()}, doc = "Maximum Time For Jobs", mutable = True)
model.totaldays = pyo.Param(initialize = data['parameters']['totaldays'], doc = "Total Number Of Days For Work", mutable = True)
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