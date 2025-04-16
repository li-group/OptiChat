from pyomo.environ import *
from pyomo.gdp import *
import json

# Load data from json
data = globals().get("data", {})
# with open("multipleMB_data.json", "r") as file:
#     data = json.load(file)


# create model
model = ConcreteModel()

# index set to simplify notation
model.J = Set(initialize=data["sets"]["JOBS"])
model.M = Set(initialize=data["sets"]["MACHINES"])
model.PAIRS = Set(initialize=model.J * model.J, dimen=2, filter=lambda m, j, k: j < k)

# decision variables
start_lower = data["bounds"]["start"]["lower"]
start_upper = data["bounds"]["start"]["upper"]
model.start = Var(model.J, bounds=(start_lower, start_upper))
model.makespan = Var(domain=NonNegativeReals)
model.early = Var(model.J, domain=NonNegativeReals)

# for binary assignment of jobs to machines
model.z = Var(model.J, model.M, domain=Binary)

# parameters
model.release = Param(model.J, mutable=True, initialize=data["parameters"]["release"])
model.duration = Param(model.J, mutable=True, initialize=data["parameters"]["duration"])
model.due = Param(model.J, mutable=True, initialize=data["parameters"]["due"])


# for modeling disjunctive constraints
model.y = Var(model.PAIRS, domain=Binary)
BigM = max(value(model.release[j]) for j in model.J) + sum(value(model.duration[j]) for j in model.J)

model.OBJ = Objective(expr=model.makespan, sense=minimize)

model.c1 = Constraint(model.J, rule=lambda m, j:
m.start[j] >= m.release[j])
model.c2 = Constraint(model.J, rule=lambda m, j:
m.start[j] + m.duration[j] + m.early[j] == m.due[j])
model.c3 = Constraint(model.J, rule=lambda m, j:
sum(m.z[j, mach] for mach in m.M) == 1)
model.c4 = Constraint(model.J, rule=lambda m, j:
m.start[j] + m.duration[j] <= m.makespan)
model.d1 = Constraint(model.M, model.PAIRS, rule=lambda m, mach, j, k:
m.start[j] + m.duration[j] <= m.start[k] + BigM * (m.y[j, k] + (1 - m.z[j, mach]) + (1 - m.z[k, mach])))
model.d2 = Constraint(model.M, model.PAIRS, rule=lambda m, mach, j, k:
m.start[k] + m.duration[k] <= m.start[j] + BigM * ((1 - m.y[j, k]) + (1 - m.z[j, mach]) + (1 - m.z[k, mach])))
