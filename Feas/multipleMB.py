from pyomo.environ import *
from pyomo.gdp import *

MACHINES = ['A','B']
JOBS = {
    'A': {'release': 2, 'duration': 5, 'due': 10},
    'B': {'release': 5, 'duration': 6, 'due': 21},
    'C': {'release': 4, 'duration': 8, 'due': 15},
    'D': {'release': 0, 'duration': 4, 'due': 10},
    'E': {'release': 0, 'duration': 2, 'due':  5},
    'F': {'release': 8, 'duration': 3, 'due': 15},
    'G': {'release': 9, 'duration': 2, 'due': 22},
}

# create model
model = ConcreteModel()

# index set to simplify notation
model.J = Set(initialize=list(JOBS.keys()))
model.M = Set(initialize=MACHINES)
model.PAIRS = Set(initialize=model.J * model.J, dimen=2, filter=lambda m, j, k: j < k)

# decision variables
model.start = Var(model.J, bounds=(0, 1000))
model.makespan = Var(domain=NonNegativeReals)
model.early = Var(model.J, domain=NonNegativeReals)

# for binary assignment of jobs to machines
model.z = Var(model.J, model.M, domain=Binary)

# parameters
model.release = Param(list(JOBS.keys()), mutable=True, initialize={key: value['release'] for key, value in JOBS.items()})  # release time
model.duration = Param(list(JOBS.keys()), mutable=True, initialize={key: value['duration'] for key, value in JOBS.items()})  # duration time
model.due = Param(list(JOBS.keys()), mutable=True, initialize={key: value['due'] for key, value in JOBS.items()})  # due time

# for modeling disjunctive constraints
model.y = Var(model.PAIRS, domain=Binary)
BigM = max([JOBS[j]['release'] for j in model.J]) + sum([JOBS[j]['duration'] for j in model.J])

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
