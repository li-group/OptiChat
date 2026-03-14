#adapted from thai.gms : Thai Navy Problem (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_thai.html
from pyomo.environ import *
import json

data = globals().get("data", {})
# with open("thai_data.json", "r") as file:
#     data = json.load(file)

# Sets
ports = data["sets"]["ports"]
voyages = data["sets"]["voyages"]
ship_classes = data["sets"]["ship_classes"]
ship_capability = [tuple(x) for x in data["sets"]["ship_capability"]]

# Parameters
number_of_men = data["parameters"]["number_of_men"]
ship_capacity = data["parameters"]["ship_capacity"]
number_of_ships = data["parameters"]["number_of_ships"]


# Weights
w1=data["parameters"]["weights"]["w1"]
w2 = data["parameters"]["weights"]["w2"]
w3 = data["parameters"]["weights"]["w3"]

# Voyage-port assignments and distances
dist = data["parameters"]["dist"]
assignment = set(tuple(x) for x in data["sets"]["assignment"])

def voyage_capability_filter(model, j, k):
    val = True
    for i in ports:
        if (j,i) in assignment and (i,k) not in ship_capability:
            val = False
            break
    #print(j,k,val)
    return val 

# Model
model = ConcreteModel()

# Sets
model.i = Set(initialize=ports, doc="ports where men must be evacuated")
model.j = Set(initialize=voyages, doc="candidate voyages")
model.k = Set(initialize=ship_classes, doc="ship classes")
model.a = Set(within=model.j*model.i, initialize=assignment, doc="voyage-port assignments")
model.sc = Set(within=model.i*model.k, initialize=ship_capability, doc="ship classes capable of serving each port")
model.vc = Set(initialize=model.j*model.k, filter=voyage_capability_filter, doc="feasible voyage and ship-class combinations")


# Parameters
model.d = Param(model.i, mutable=True, initialize=number_of_men, doc="number of men to evacuate from each port")
model.shipcap = Param(model.k, mutable=True, initialize=ship_capacity, doc="capacity of each ship class")
model.n = Param(model.k, mutable=True, initialize=number_of_ships, doc="number of available ships in each class")
model.dist = Param(model.j, mutable=True, initialize=dist, doc="distance associated with each voyage")

# Variables
model.z = Var(model.j, model.k, domain=NonNegativeIntegers, doc="number of times voyage jk is used")
model.y = Var(model.j, model.k, model.i, domain=NonNegativeReals, doc="number of men transported from port i via voyage jk")

# Constraints
def demand_rule(model, i): #pick up all the men at port i
    return sum(model.y[j,k,i] for j, k in model.vc if (j,i) in model.a) >= model.d[i]
model.demand = Constraint(model.i, rule=demand_rule, doc="evacuation demand satisfaction at each port")

def voycap_rule(model, j, k):#observe variable capacity of voyage jk
    if (j, k) in model.vc:
        return sum(model.y[j,k,i] for i in model.i if (j,i) in model.a) <= model.shipcap[k]*model.z[j,k]
    else:
        return Constraint.Skip
model.voycap = Constraint(model.j, model.k, rule=voycap_rule, doc="voyage capacity limit by ship class")

def shiplim_rule(model, k): #observe limit of class k
    return sum(model.z[j,k] for j in model.j if (j, k) in model.vc) <= model.n[k]
model.shiplim = Constraint(model.k, rule=shiplim_rule, doc="availability limit on ships by class")

# Objective
model.obj = Objective(expr=(
    w1*sum(model.z[j,k] for j, k in model.vc) +
    w2*sum(model.dist[j]*model.z[j,k] for j, k in model.vc) +
    w3*sum(model.dist[j]*model.y[j,k,i] for j, k, i in model.y)
), sense=minimize)
