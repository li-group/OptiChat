#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:29:40 2023

@author: geconsta
"""

import json
from pyomo.environ import *

model = ConcreteModel()

#load data from json file
data = globals().get("data", {})
# with open("diet_inf_1_data.json", "r") as file:
#     data = json.load(file)

# Access nutrient requirements
nutrient_requirements = data['nutrient_requirements']


# Access nutritive values (food items and their nutrients)
nutritive_values = data['nutritive_values']


# Sets
model.n = Set(initialize=list(data['nutrient_requirements'].keys()), doc='nutrients')
model.f = Set(initialize=list(data['nutritive_values'].keys()), doc='foods')


# Importing required values dynamically from the JSON file
model.b = Param(model.n, initialize=nutrient_requirements, mutable=True, doc='required daily allowances of nutrients')


# to initialize model.a with the nutritive values
initialized_data = {
    (food, nutrient): value
    for food, nutrients in nutritive_values.items()
    for nutrient, value in nutrients.items()
}

model.a = Param(model.f, model.n, initialize=initialized_data, mutable=True, doc='nutritive value of foods (per dollar spent)')


# Variables
model.x = Var(model.f, within=NonNegativeReals, bounds=(0, None), initialize=0, doc='dollars of food f to be purchased daily (dollars)')
model.cost = Var(within=NonNegativeReals, bounds=(0, None), initialize=0, doc='total food bill (dollars)')

# Constraints
def nutrient_balance_rule(model, n):
    return sum(model.a[f, n] * model.x[f] for f in model.f) >= model.b[n]
model.nb = Constraint(model.n, rule=nutrient_balance_rule, doc='nutrient balance (units)')

def cost_balance_rule(model):
    return model.cost == sum(model.x[f] for f in model.f)
model.cb = Constraint(rule=cost_balance_rule, doc='cost balance   (dollars)')

def max_cal_rule(model):
    return sum(model.a[f, 'calorie'] * model.x[f] for f in model.f) <= 1.2*model.b['calorie']
model.max_cal = Constraint(rule=max_cal_rule,doc='maximum calories')

# Objective
model.obj = Objective(expr=model.cost, sense=minimize, doc='objective function')
