# Ridesharing Problem
# This model is designed to find the assignment among vehicles and requests.
import json
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Load data from external JSON file
data = globals().get("data", {})
if not data:
    import os
    json_path = os.path.join(os.path.dirname(__file__), "ridesharing_data.json")
    with open(json_path, "r") as file:
        data = json.load(file)

# Create the Pyomo model
model = pyo.ConcreteModel()

#%% Define Sets
model.R = pyo.Set(initialize=data["sets"]["R"], doc="Requests")
model.V = pyo.Set(initialize=data["sets"]["V"], doc="Vehicles")
model.paths = pyo.Set(initialize=data["sets"]["paths"], doc="Paths")
model.L = pyo.Set(initialize=data["sets"]["L"], doc="Link")
model.h = pyo.Set(initialize=data["sets"]["h"], doc="horizon index set")
model.d = pyo.Set(initialize=data["sets"]["d"], doc="DNL horizon index set")

#%% Define parameters

# Vr: Convert string keys back to integers
vr_raw = data["parameters"]["Vr"]
vr_init = {int(k): v for k, v in vr_raw.items()}

model.Vr = pyo.Param(model.R, initialize=vr_init, doc="Potential vehicles for each request", mutable=True, within=pyo.Any)

# beta: Convert "v1,1,A" string keys back to tuple keys ('v1', 1, 'A')
beta_raw = data["parameters"]["beta"]
beta_init = {}
for key_str, value in beta_raw.items():
    parts = key_str.split(',')
    vehicle = parts[0]
    request = int(parts[1])
    path = parts[2]
    beta_init[(vehicle, request, path)] = value

model.beta = pyo.Param(model.V, model.R, model.paths, initialize=beta_init, doc="beta for each link", mutable=True)

# Cl: Link capacities
model.Cl = pyo.Param(model.L, initialize=data["parameters"]["Cl"], doc="capacity for each link", mutable=True)

# rho: Convert "l1,0" string keys back to tuple keys ('l1', 0)
rho_raw = data["parameters"]["rho"]
rho_init = {}
for key_str, value in rho_raw.items():
    parts = key_str.split(',')
    link = parts[0]
    h_val = int(parts[1])
    rho_init[(link, h_val)] = value

model.rho = pyo.Param(model.L, model.h, initialize=rho_init, doc="Rho values for each link and time frame", mutable=True)

# q: Convert "l1,0" string keys back to tuple keys ('l1', 0)
q_raw = data["parameters"]["q"]
q_init = {}
for key_str, value in q_raw.items():
    parts = key_str.split(',')
    link = parts[0]
    h_val = int(parts[1])
    q_init[(link, h_val)] = value

model.q = pyo.Param(model.L, model.h, initialize=q_init, doc="q values for each link and time frame", mutable=True)

# n: Convert "l1,0" string keys back to tuple keys ('l1', 0)
n_raw = data["parameters"]["n"]
n_init = {}
for key_str, value in n_raw.items():
    parts = key_str.split(',')
    link = parts[0]
    d_val = int(parts[1])
    n_init[(link, d_val)] = value

model.n = pyo.Param(model.L, model.d, initialize=n_init, doc="n values for each link and DNL time frame", mutable=True)

model.p = pyo.Param(initialize=data["parameters"]["p_val"], doc="Constant parameter p", mutable=True)
model.m = pyo.Param(initialize=data["parameters"]["m_val"], doc="Constant parameter m", mutable=True)

#%% Define variables
# Define the decision variable x[v, r, a] for each combination of r, v, and a
model.x = pyo.Var(model.V, model.R, model.paths, within=pyo.Binary, doc="Assignment variable")

model.alpha = pyo.Var(model.V, model.R, model.paths, model.L, model.h, within=pyo.Binary, doc="Link variable")


#%% Define constraints
# Constraint Eq.10 & 11
def eq_10_rule(model, r):
    return sum(model.x[v, r, a] for v in model.Vr[r].value for a in model.paths) <= 1

model.eq_10 = pyo.Constraint(model.R, rule=eq_10_rule, doc="Constraint Eq.10")

# Constraint Eq.12
def eq_12_rule(model):
    min_const = min(model.p.value, model.m.value)
    return sum(model.x[v, r, a] for r in model.R for v in model.Vr[r].value for a in model.paths) == min_const

model.eq_12 = pyo.Constraint(rule=eq_12_rule, doc="Constraint Eq.12")


#%% Define objective function

def objective_rule(model):
    J1 = sum(model.x[v, r, a] * model.beta[v, r, a]
               for r in model.R
               for v in model.Vr[r].value
               for a in model.paths)
    A = {}
    for l in model.L:
        for h in model.h:
            A[l, h] = sum(model.x[v, r, a] * model.alpha[v, r, a, l, h]
                        for r in model.R
                        for v in model.Vr[r].value
                        for a in model.paths)
    J2 = sum((1 - model.rho[l, h]) * A[l, h] for l in model.L for h in model.h)
    return J1 + J2 #change to sum instead of min(J1, J2)

# Set the objective
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
