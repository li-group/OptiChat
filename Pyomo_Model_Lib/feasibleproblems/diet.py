#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:29:40 2023

@author: geconsta
"""

from pyomo.environ import *

model = ConcreteModel()

# Sets
model.n = Set(initialize=['calorie', 'protein', 'calcium', 'iron', 'vitamin-a', 'vitamin-b1', 'vitamin-b2', 'niacin', 'vitamin-c'], doc='nutrients')
model.f = Set(initialize=['wheat', 'cornmeal', 'cannedmilk', 'margarine', 'cheese', 'peanut-b', 'lard', 'liver', 'porkroast', 'salmon', 'greenbeans', 'cabbage', 'onions', 'potatoes', 'spinach', 'sweet-pot', 'peaches', 'prunes', 'limabeans', 'navybeans'], doc='foods')

# Parameters
model.b = Param(model.n, initialize={'calorie': 3, 'protein': 70, 'calcium': 0.8, 'iron': 12, 'vitamin-a': 5, 'vitamin-b1': 1.8, 'vitamin-b2': 2.7, 'niacin': 18, 'vitamin-c': 75}, mutable = True, doc='required daily allowances of nutrients')

model.a = Param(model.f, model.n, initialize={
    ('wheat', 'calorie'): 44.7, ('wheat', 'protein'): 1411, ('wheat', 'calcium'): 2.0, ('wheat', 'iron'): 365, ('wheat', 'vitamin-a'): 0, ('wheat', 'vitamin-b1'): 55.4, ('wheat', 'vitamin-b2'): 33.3, ('wheat', 'niacin'): 441, ('wheat', 'vitamin-c'): 0,
    ('cornmeal', 'calorie'): 36, ('cornmeal', 'protein'): 897, ('cornmeal', 'calcium'): 1.7, ('cornmeal', 'iron'): 99, ('cornmeal', 'vitamin-a'): 30.9, ('cornmeal', 'vitamin-b1'): 17.4, ('cornmeal', 'vitamin-b2'): 7.9, ('cornmeal', 'niacin'): 106, ('cornmeal', 'vitamin-c'): 0,
    ('cannedmilk', 'calorie'): 8.4, ('cannedmilk', 'protein'): 422, ('cannedmilk', 'calcium'): 15.1, ('cannedmilk', 'iron'): 9, ('cannedmilk', 'vitamin-a'): 26, ('cannedmilk', 'vitamin-b1'): 3, ('cannedmilk', 'vitamin-b2'): 23.5, ('cannedmilk', 'niacin'): 11, ('cannedmilk', 'vitamin-c'): 60,
    ('margarine', 'calorie'): 20.6, ('margarine', 'protein'): 17, ('margarine', 'calcium'): 0.6, ('margarine', 'iron'): 6, ('margarine', 'vitamin-a'): 55.8, ('margarine', 'vitamin-b1'): 0.2, ('margarine', 'vitamin-b2'): 0, ('margarine', 'niacin'): 0, ('margarine', 'vitamin-c'): 0,
    ('cheese', 'calorie'): 7.4, ('cheese', 'protein'): 448, ('cheese', 'calcium'): 16.4, ('cheese', 'iron'): 19, ('cheese', 'vitamin-a'): 28.1, ('cheese', 'vitamin-b1'): 0.8, ('cheese', 'vitamin-b2'): 10.3, ('cheese', 'niacin'): 4, ('cheese', 'vitamin-c'): 0,
    ('peanut-b', 'calorie'): 15.7, ('peanut-b', 'protein'): 661, ('peanut-b', 'calcium'): 1, ('peanut-b', 'iron'): 48, ('peanut-b', 'vitamin-a'): 0, ('peanut-b', 'vitamin-b1'): 9.6, ('peanut-b', 'vitamin-b2'): 8.1, ('peanut-b', 'niacin'): 471, ('peanut-b', 'vitamin-c'): 0,
    ('lard', 'calorie'): 41.7, ('lard', 'protein'): 0, ('lard', 'calcium'): 0, ('lard', 'iron'): 0, ('lard', 'vitamin-a'): 0, ('lard', 'vitamin-b1'): 0, ('lard', 'vitamin-b2'): 0.5, ('lard', 'niacin'): 5, ('lard', 'vitamin-c'): 0,
    ('liver', 'calorie'): 2.2, ('liver', 'protein'): 333, ('liver', 'calcium'): 0.2, ('liver', 'iron'): 139, ('liver', 'vitamin-a'): 169.2, ('liver', 'vitamin-b1'): 6.4, ('liver', 'vitamin-b2'): 50.8, ('liver', 'niacin'): 316, ('liver', 'vitamin-c'): 525,
    ('porkroast', 'calorie'): 4.4, ('porkroast', 'protein'): 249, ('porkroast', 'calcium'): 0.3, ('porkroast', 'iron'): 37, ('porkroast', 'vitamin-a'): 0, ('porkroast', 'vitamin-b1'): 18.2, ('porkroast', 'vitamin-b2'): 3.6, ('porkroast', 'niacin'): 79, ('porkroast', 'vitamin-c'): 0,
    ('salmon', 'calorie'): 5.8, ('salmon', 'protein'): 705, ('salmon', 'calcium'): 6.8, ('salmon', 'iron'): 45, ('salmon', 'vitamin-a'): 3.5, ('salmon', 'vitamin-b1'): 1, ('salmon', 'vitamin-b2'): 4.9, ('salmon', 'niacin'): 209, ('salmon', 'vitamin-c'): 0,
    ('greenbeans', 'calorie'): 2.4, ('greenbeans', 'protein'): 138, ('greenbeans', 'calcium'): 3.7, ('greenbeans', 'iron'): 80, ('greenbeans', 'vitamin-a'): 69, ('greenbeans', 'vitamin-b1'): 4.3, ('greenbeans', 'vitamin-b2'): 5.8, ('greenbeans', 'niacin'): 37, ('greenbeans', 'vitamin-c'): 862,
    ('cabbage', 'calorie'): 2.6, ('cabbage', 'protein'): 125, ('cabbage', 'calcium'): 4, ('cabbage', 'iron'): 36, ('cabbage', 'vitamin-a'): 7.2, ('cabbage', 'vitamin-b1'): 9, ('cabbage', 'vitamin-b2'): 4.5, ('cabbage', 'niacin'): 26, ('cabbage', 'vitamin-c'): 5369,
    ('onions', 'calorie'): 5.8, ('onions', 'protein'): 166, ('onions', 'calcium'): 3.8, ('onions', 'iron'): 59, ('onions', 'vitamin-a'): 16.6, ('onions', 'vitamin-b1'): 4.7, ('onions', 'vitamin-b2'): 5.9, ('onions', 'niacin'): 21, ('onions', 'vitamin-c'): 1184,
    ('potatoes', 'calorie'): 14.3, ('potatoes', 'protein'): 336, ('potatoes', 'calcium'): 1.8, ('potatoes', 'iron'): 118, ('potatoes', 'vitamin-a'): 6.7, ('potatoes', 'vitamin-b1'): 29.4, ('potatoes', 'vitamin-b2'): 7.1, ('potatoes', 'niacin'): 198, ('potatoes', 'vitamin-c'): 2522,
    ('spinach', 'calorie'): 1.1, ('spinach', 'protein'): 106, ('spinach', 'calcium'): 0, ('spinach', 'iron'): 138, ('spinach', 'vitamin-a'): 918.4, ('spinach', 'vitamin-b1'): 5.7, ('spinach', 'vitamin-b2'): 13.8, ('spinach', 'niacin'): 33, ('spinach', 'vitamin-c'): 2755,
    ('sweet-pot', 'calorie'): 9.6, ('sweet-pot', 'protein'): 138, ('sweet-pot', 'calcium'): 2.7, ('sweet-pot', 'iron'): 54, ('sweet-pot', 'vitamin-a'): 290.7, ('sweet-pot', 'vitamin-b1'): 8.4, ('sweet-pot', 'vitamin-b2'): 5.4, ('sweet-pot', 'niacin'): 83, ('sweet-pot', 'vitamin-c'): 1912,
    ('peaches', 'calorie'): 8.5, ('peaches', 'protein'): 87, ('peaches', 'calcium'): 1.7, ('peaches', 'iron'): 173, ('peaches', 'vitamin-a'): 86.8, ('peaches', 'vitamin-b1'): 1.2, ('peaches', 'vitamin-b2'): 4.3, ('peaches', 'niacin'): 55, ('peaches', 'vitamin-c'): 57,
    ('prunes', 'calorie'): 12.8, ('prunes', 'protein'): 99, ('prunes', 'calcium'): 2.5, ('prunes', 'iron'): 154, ('prunes', 'vitamin-a'): 85.7, ('prunes', 'vitamin-b1'): 3.9, ('prunes', 'vitamin-b2'): 4.3, ('prunes', 'niacin'): 65, ('prunes', 'vitamin-c'): 257,
    ('limabeans', 'calorie'): 17.4, ('limabeans', 'protein'): 1055, ('limabeans', 'calcium'): 3.7, ('limabeans', 'iron'): 459, ('limabeans', 'vitamin-a'): 5.1, ('limabeans', 'vitamin-b1'): 26.9, ('limabeans', 'vitamin-b2'): 38.2, ('limabeans', 'niacin'): 93, ('limabeans', 'vitamin-c'): 0,
    ('navybeans', 'calorie'): 26.9, ('navybeans', 'protein'): 1691, ('navybeans', 'calcium'): 11.4, ('navybeans', 'iron'): 792, ('navybeans', 'vitamin-a'): 0, ('navybeans', 'vitamin-b1'): 38.4, ('navybeans', 'vitamin-b2'): 24.6, ('navybeans', 'niacin'): 217, ('navybeans', 'vitamin-c'): 0
}, mutable = True, doc='nutritive value of foods (per dollar spent)')

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

# Objective
model.obj = Objective(expr=model.cost, sense=minimize, doc='objective function')