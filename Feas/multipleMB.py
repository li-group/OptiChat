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
model.J = Set(initialize=data["sets"]["JOBS"], doc="Set of jobs to be scheduled")
model.M = Set(initialize=data["sets"]["MACHINES"], doc="Set of available machines")
model.PAIRS = Set(initialize=model.J * model.J, dimen=2, filter=lambda m, j, k: j < k,
                  doc="Ordered pairs of distinct jobs (j, k) with j < k, used for disjunctive sequencing constraints")

# decision variables
start_lower = data["bounds"]["start"]["lower"]
start_upper = data["bounds"]["start"]["upper"]
model.start = Var(model.J, bounds=(start_lower, start_upper),
                  doc="Start time of each job (continuous, bounded by earliest and latest allowable start times)")
model.makespan = Var(domain=NonNegativeReals,
                     doc="Total schedule makespan: the latest job completion time, minimized as the objective")
model.early = Var(model.J, domain=NonNegativeReals,
                  doc="Earliness of each job: amount by which the job completes before its due date")

# for binary assignment of jobs to machines
model.z = Var(model.J, model.M, domain=Binary,
              doc="Binary assignment variable: 1 if job j is assigned to machine mach, 0 otherwise")

# parameters
model.release = Param(model.J, mutable=True, initialize=data["parameters"]["release"],
                      doc="Release time (earliest allowable start time) for each job")
model.duration = Param(model.J, mutable=True, initialize=data["parameters"]["duration"],
                       doc="Processing duration (time units required to complete) for each job")
model.due = Param(model.J, mutable=True, initialize=data["parameters"]["due"],
                  doc="Due date by which each job should ideally be completed")


# for modeling disjunctive constraints
model.y = Var(model.PAIRS, domain=Binary,
              doc="Binary sequencing variable: 1 if job j precedes job k on the same machine, 0 if k precedes j")
BigM = max(value(model.release[j]) for j in model.J) + sum(value(model.duration[j]) for j in model.J)

model.OBJ = Objective(expr=model.makespan, sense=minimize)

model.c1 = Constraint(model.J, rule=lambda m, j:
m.start[j] >= m.release[j],
doc="Release time constraint: each job must start no earlier than its release time")
model.c2 = Constraint(model.J, rule=lambda m, j:
m.start[j] + m.duration[j] + m.early[j] == m.due[j],
doc="Due date accounting constraint: job completion plus earliness equals the due date, tracking early finishing")
model.c3 = Constraint(model.J, rule=lambda m, j:
sum(m.z[j, mach] for mach in m.M) == 1,
doc="Machine assignment constraint: each job must be assigned to exactly one machine")
model.c4 = Constraint(model.J, rule=lambda m, j:
m.start[j] + m.duration[j] <= m.makespan,
doc="Makespan constraint: the makespan must be at least as large as each job's completion time")
model.d1 = Constraint(model.M, model.PAIRS, rule=lambda m, mach, j, k:
m.start[j] + m.duration[j] <= m.start[k] + BigM * (m.y[j, k] + (1 - m.z[j, mach]) + (1 - m.z[k, mach])),
doc="Disjunctive sequencing constraint (j before k): if both jobs are on the same machine and j precedes k, k must start after j finishes (BigM relaxation otherwise)")
model.d2 = Constraint(model.M, model.PAIRS, rule=lambda m, mach, j, k:
m.start[k] + m.duration[k] <= m.start[j] + BigM * ((1 - m.y[j, k]) + (1 - m.z[j, mach]) + (1 - m.z[k, mach])),
doc="Disjunctive sequencing constraint (k before j): if both jobs are on the same machine and k precedes j, j must start after k finishes (BigM relaxation otherwise)")
