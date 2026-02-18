#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:45:01 2023

@author: geconsta
"""
import json
from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Load demand distribution data from JSON file
data = globals().get("data", {})
# with open("aircraft_data.json", "r") as file:
#     data = json.load(file)


# Sets   
# Extract Aircraft Types
I = list(data["cost_per_aircraft"].keys())  # Extracts 'a', 'b', 'c', 'd'

# Extract Routes
J = list(data["demand_distribution"].keys())  # Extracts 'route-1', 'route-2', etc.

# Extract Demand States (assuming all routes share the same demand states)
H = sorted([int(h) for h in data["demand_distribution"][J[0]].keys()])  # Extracts 1, 2, 3, 4, 5


model.i = Set(initialize=I, doc='Aircraft types and unassigned passengers')
model.j = Set(initialize=J, doc='Assigned and unassigned routes')
model.h = Set(initialize=H, doc='Demand states')
model.hp = Set(initialize=H, doc='Demand states')


# Demand distribution on route j
dd_init = {(route, int(state)): demand for route, states in data["demand_distribution"].items() for state, demand in states.items()}

# probability of demand state h on route j
lambda_init = {(route, int(state)): prob for route, states in data["lambda_init"].items() for state, prob in states.items()}

# costs per aircraft (1000s)
c_init = {(aircraft, route): cost for aircraft, routes in data["cost_per_aircraft"].items() for route, cost in routes.items()}

# passenger capacity of aircraft i on route j
p_init = {(aircraft, route): capacity for aircraft, routes in data["passenger_capacity"].items() for route, capacity in routes.items()}
aa_init = data["aircraft_availability"]  # Aircraft availability
k_init = data["k_init"] 


# Parameters


model.dd = Param(model.j, model.h, initialize=dd_init, mutable = True, doc= 'demand distribution on route j')
model.lambda_ = Param(model.j, model.h, initialize=lambda_init, mutable = True, doc= 'probability of demand state h on route j')
model.c = Param(model.i, model.j, initialize=c_init, mutable = True, doc='costs per aircraft (1000s)')
model.p = Param(model.i, model.j, initialize=p_init, mutable = True, doc='passenger capacity of aircraft i on route j')
model.aa = Param(model.i, initialize=aa_init, mutable = True, doc='aircraft availability')
#revenue lost (1000 per 100 bumped)
model.k = Param(model.j, initialize=k_init, mutable = True, doc='revenue lost (1000 per 100 bumped')

def ed_init(model, j):
    return sum(lambda_init[j,h]*dd_init[j,h] for h in model.h)
model.ed = Param(model.j, initialize=ed_init, mutable = True, doc='expected demand')

# def gamma_init(model, j, h):
#     return sum(lambda_init[j,hp] for hp in model.h if hp>=h)
# model.gamma = Param(model.j, model.h, initialize=gamma_init, mutable = True)

def deltb_init(model, j, h):
    if dd_init[j,h] > 0:
        if h >= 2:
            return dd_init[j,h] - dd_init[j,h-1]
        else:
            return dd_init[j,h]
    else:
        return 0
model.deltb = Param(model.j, model.h, initialize=deltb_init, mutable = True, doc='incremental passenger load in demand states')

# Variables
# number of aircraft type i assigned to route j
model.x = Var(model.i, model.j, domain=NonNegativeReals, doc='number of aircraft type i assigned to route j')
# passengers actually carried
model.y = Var(model.j, model.h, domain=NonNegativeReals, doc='passengers actually carried')
# passengers bumped
model.b = Var(model.j, model.h, domain=NonNegativeReals, doc='passengers bumped')
# operating cost
model.oc = Var(domain=NonNegativeReals, doc='operating cost')
# bumping cost
model.bc = Var(domain=NonNegativeReals, doc='bumping cost')
# total expected costs
model.phi = Var()

# Objective
model.obj = Objective(expr=model.oc + model.bc, sense=minimize, doc='objective function')

# Constraints

# model.db = Constraint(model.j, rule=lambda model, j: sum(model.p[i,j]*model.x[i,j] for i in model.i) >= sum(model.y[j,h] for h in model.h))
# model.bcd1 = Constraint(rule=lambda model: model.bc == sum(model.k[j]*(model.ed[j] - sum(model.y[j,h]*model.gamma[j,h] for h in model.h)) for j in model.j))

# aircraft balance'
model.ab = Constraint(model.i, rule=lambda model, i: sum(model.x[i,j] for j in model.j) <= model.aa[i], doc='aircraft balance')
# definition of boarded passengers
model.yd = Constraint(model.j, model.h, rule=lambda model, j, h: model.y[j, h] <= sum(model.p[i, j]*model.x[i, j] for i in model.i), doc='definition of boarded passengers')
# definition of bumped passengers
model.bd = Constraint(model.j, model.h, rule=lambda model, j, h: model.b[j, h] == model.dd[j, h] - model.y[j, h], doc='definition of bumped passengers')
# operating cost definition
model.ocd = Constraint(rule=lambda model: model.oc == sum(model.c[i, j]*model.x[i, j] for i in model.i for j in model.j), doc='operating cost definition')
# bumping cost definition: version 2
model.bcd2 = Constraint(rule=lambda model: model.bc == sum(model.k[j]*model.lambda_[j, h]*model.b[j, h] for j in model.j for h in model.h), doc='bumping cost definition: version 2')


model.yup = Constraint(model.j, model.h, rule=lambda model, j,h: model.y[j,h] <= model.deltb[j,h])