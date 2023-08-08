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
m = ConcreteModel()

# index set to simplify notation
m.J = Set(initialize=JOBS.keys())
m.M = Set(initialize=MACHINES)
m.PAIRS = Set(initialize=m.J * m.J, dimen=2, filter=lambda m, j, k: j < k)

# decision variables
m.start = Var(m.J, bounds=(0, 1000))
m.makespan = Var(domain=NonNegativeReals)
m.early = Var(m.J, domain=NonNegativeReals)

# for binary assignment of jobs to machines
m.z = Var(m.J, m.M, domain=Binary)

# parameters
m.release = Param(JOBS.keys(), mutable=True, initialize={key: value['release'] for key, value in JOBS.items()})  # release time
m.duration = Param(JOBS.keys(), mutable=True, initialize={key: value['duration'] for key, value in JOBS.items()})  # duration time
m.due = Param(JOBS.keys(), mutable=True, initialize={key: value['due'] for key, value in JOBS.items()})  # due time

# for modeling disjunctive constraints
m.y = Var(m.PAIRS, domain=Binary)
BigM = max([JOBS[j]['release'] for j in m.J]) + sum([JOBS[j]['duration'] for j in m.J])

m.OBJ = Objective(expr=m.makespan, sense=minimize)

m.c1 = Constraint(m.J, rule=lambda m, j:
m.start[j] >= m.release[j])
m.c2 = Constraint(m.J, rule=lambda m, j:
m.start[j] + m.duration[j] + m.early[j] == m.due[j])
m.c3 = Constraint(m.J, rule=lambda m, j:
sum(m.z[j, mach] for mach in m.M) == 1)
m.c4 = Constraint(m.J, rule=lambda m, j:
m.start[j] + m.duration[j] <= m.makespan)
m.d1 = Constraint(m.M, m.PAIRS, rule=lambda m, mach, j, k:
m.start[j] + m.duration[j] <= m.start[k] + BigM * (m.y[j, k] + (1 - m.z[j, mach]) + (1 - m.z[k, mach])))
m.d2 = Constraint(m.M, m.PAIRS, rule=lambda m, mach, j, k:
m.start[k] + m.duration[k] <= m.start[j] + BigM * ((1 - m.y[j, k]) + (1 - m.z[j, mach]) + (1 - m.z[k, mach])))

model = m

solver = SolverFactory('gurobi')
solver.solve(model, tee=True)