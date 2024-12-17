#adapted from thai.gms : Thai Navy Problem (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_thai.html
from pyomo.environ import *

# Sets
ports = ['chumphon', 'surat', 'nakon', 'songkhla']
voyages = ['v-{:02}'.format(i+1) for i in range(15)]
ship_classes = ['small', 'medium', 'large']
ship_capability = [
    ('chumphon', 'small'), ('chumphon', 'medium'), ('chumphon', 'large'),
    ('surat', 'medium'), ('surat', 'large'),
    ('nakon', 'medium'), ('nakon', 'large'),
    ('songkhla', 'large')
]

# Parameters
number_of_men = {
    'chumphon': 475,
    'surat': 659,
    'nakon': 672,
    'songkhla': 1123
}
ship_capacity = {
    'small': 100,
    'medium': 200,
    'large': 600
}
number_of_ships = {
    'small': 2,
    'medium': 3,
    'large': 4
}

# Weights
w1 = 1.00
w2 = 0.01
w3 = 0.0001

# Voyage-port assignments and distances
dist = {
    'v-01': 370, 
    'v-02': 460, 
    'v-03': 600, 
    'v-04': 750, 
    'v-05': 515, 
    'v-06': 640, 
    'v-07': 810, 
    'v-08': 665, 
    'v-09': 665, 
    'v-10': 800, 
    'v-11': 720, 
    'v-12': 860, 
    'v-13': 840, 
    'v-14': 865, 
    'v-15': 920
}

assignment={('v-01','chumphon'),('v-02','surat'),('v-03','nakon'),('v-04','songkhla'),('v-05','chumphon'),('v-05','surat'),('v-06','chumphon'),('v-06','nakon'),('v-07','chumphon'),('v-07','songkhla'),('v-08','surat'),('v-08','nakon'),('v-09','surat'),('v-09','songkhla'),('v-10','nakon'),('v-10','songkhla'),('v-11','chumphon'),('v-11','surat'),('v-11','nakon'),('v-12','chumphon'),('v-12','surat'),('v-12','songkhla'),('v-13','chumphon'),('v-13','nakon'),('v-13','songkhla'),('v-14','surat'),('v-14','nakon'),('v-14','songkhla'),('v-15','chumphon'),('v-15','surat'),('v-15','nakon'),('v-15','songkhla')}

def voyage_capability_filter(model, j, k):
    val = True
    for i in ports:
        if (j,i) in assignment and (i,k) not in ship_capability:
            val = False
            break
    print(j,k,val)
    return val 

# Model
model = ConcreteModel()

# Sets
model.i = Set(initialize=ports)
model.j = Set(initialize=voyages)
model.k = Set(initialize=ship_classes)
model.a = Set(within=model.j*model.i, initialize=assignment)
model.sc = Set(within=model.i*model.k, initialize=ship_capability)
model.vc = Set(initialize=model.j*model.k, filter=voyage_capability_filter)


# Parameters
model.d = Param(model.i, mutable=True, initialize=number_of_men)
model.shipcap = Param(model.k, mutable=True, initialize=ship_capacity)
model.n = Param(model.k, mutable=True, initialize=number_of_ships)
model.dist = Param(model.j, mutable=True, initialize=dist)

# Variables
model.z = Var(model.j, model.k, domain=NonNegativeIntegers) #number of times voyage jk is used
model.y = Var(model.j, model.k, model.i, domain=NonNegativeReals) # number of men transported from port i via voyage jk

# Constraints
def demand_rule(model, i): #pick up all the men at port i
    return sum(model.y[j,k,i] for j, k in model.vc if (j,i) in model.a) >= model.d[i]
model.demand = Constraint(model.i, rule=demand_rule)

def voycap_rule(model, j, k):#observe variable capacity of voyage jk
    if (j, k) in model.vc:
        return sum(model.y[j,k,i] for i in model.i if (j,i) in model.a) <= model.shipcap[k]*model.z[j,k]
    else:
        return Constraint.Skip
model.voycap = Constraint(model.j, model.k, rule=voycap_rule)

def shiplim_rule(model, k): #observe limit of class k
    return sum(model.z[j,k] for j in model.j if (j, k) in model.vc) <= model.n[k]
model.shiplim = Constraint(model.k, rule=shiplim_rule)

# Objective
model.obj = Objective(expr=(
    w1*sum(model.z[j,k] for j, k in model.vc) +
    w2*sum(model.dist[j]*model.z[j,k] for j, k in model.vc) +
    w3*sum(model.dist[j]*model.y[j,k,i] for j, k, i in model.y)
), sense=minimize)
