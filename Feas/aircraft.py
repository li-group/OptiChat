#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:45:01 2023

@author: geconsta
"""

from pyomo.environ import *

# Create a model
model = ConcreteModel()

# Sets
#'aircraft types and unassigned passengers'
I = ['a', 'b', 'c', 'd']
#'assigned and unassigned routes'
J = ['route-1', 'route-2', 'route-3', 'route-4', 'route-5']
#'demand states'
H = [1, 2, 3, 4, 5]

model.i = Set(initialize=I, doc='Aircraft types and unassigned passengers')
model.j = Set(initialize=J, doc='Assigned and unassigned routes')
model.h = Set(initialize=H, doc='Demand states')
model.hp = Set(initialize=H, doc='Demand states')

# Demand distribution on route j
dd_init = {
    ('route-1', 1): 200, ('route-1', 2): 220, ('route-1', 3): 250, ('route-1', 4): 270, ('route-1', 5): 300,
    ('route-2', 1): 50,  ('route-2', 2): 150, ('route-2', 3): 0,   ('route-2', 4): 0,   ('route-2', 5): 0,
    ('route-3', 1): 140, ('route-3', 2): 160, ('route-3', 3): 180, ('route-3', 4): 200, ('route-3', 5): 220,
    ('route-4', 1): 10,  ('route-4', 2): 50,  ('route-4', 3): 80,  ('route-4', 4): 100, ('route-4', 5): 340,
    ('route-5', 1): 580, ('route-5', 2): 600, ('route-5', 3): 620, ('route-5', 4): 0,   ('route-5', 5): 0}
# probability of demand state h on route j
lambda_init = {
    ('route-1', 1): 0.2, ('route-1', 2): 0.05,('route-1', 3): 0.35,('route-1', 4): 0.2, ('route-1', 5): 0.2,
    ('route-2', 1): 0.3, ('route-2', 2): 0.7, ('route-2', 3): 0,   ('route-2', 4): 0,   ('route-2', 5): 0,
    ('route-3', 1): 0.1, ('route-3', 2): 0.2, ('route-3', 3): 0.4, ('route-3', 4): 0.2, ('route-3', 5): 0.1,
    ('route-4', 1): 0.2, ('route-4', 2): 0.2, ('route-4', 3): 0.3, ('route-4', 4): 0.2, ('route-4', 5): 0.1,
    ('route-5', 1): 0.1, ('route-5', 2): 0.8, ('route-5', 3): 0.1, ('route-5', 4): 0,   ('route-5', 5): 0}
# costs per aircraft (1000s)
c_init = {
    ('a','route-1'): 18, ('a','route-2'): 21, ('a','route-3'): 18, ('a','route-4'): 16, ('a','route-5'): 10,
    ('b','route-1'): 0,  ('b','route-2'): 15, ('b','route-3'): 16, ('b','route-4'): 14, ('b','route-5'): 9,
    ('c','route-1'): 0,  ('c','route-2'): 10, ('c','route-3'): 0,  ('c','route-4'): 9,  ('c','route-5'): 6,
    ('d','route-1'): 17, ('d','route-2'): 16, ('d','route-3'): 17, ('d','route-4'): 15, ('d','route-5'): 10}
# passenger capacity of aircraft i on route j
p_init = {
    ('a','route-1'): 16, ('a','route-2'): 15, ('a','route-3'): 28, ('a','route-4'): 23, ('a','route-5'): 81,
    ('b','route-1'): 0,  ('b','route-2'): 10, ('b','route-3'): 14, ('b','route-4'): 15, ('b','route-5'): 57,
    ('c','route-1'): 0,  ('c','route-2'): 5,  ('c','route-3'): 0,  ('c','route-4'): 7,  ('c','route-5'): 29,
    ('d','route-1'): 9,  ('d','route-2'): 11, ('d','route-3'): 22, ('d','route-4'): 17, ('d','route-5'): 55}

# Parameters


model.dd = Param(model.j, model.h, initialize=dd_init, mutable = True, doc= 'demand distribution on route j')
model.lambda_ = Param(model.j, model.h, initialize=lambda_init, mutable = True, doc= 'probability of demand state h on route j')
model.c = Param(model.i, model.j, initialize=c_init, mutable = True, doc='costs per aircraft (1000s)')
model.p = Param(model.i, model.j, initialize=p_init, mutable = True, doc='passenger capacity of aircraft i on route j')
model.aa = Param(model.i, initialize={
    'a': 10, 'b': 19, 'c': 25, 'd': 15}, mutable = True, doc='aircraft availability')
#revenue lost (1000 per 100 bumped)
model.k = Param(model.j, initialize={
    'route-1': 13, 'route-2': 13, 'route-3': 7,'route-4': 7,'route-5': 1}, mutable = True, doc='revenue lost (1000 per 100 bumped')

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