#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:11:42 2023

@author: geconsta
"""

from pyomo.environ import *

model = ConcreteModel()

# Sets
model.q = Set(initialize=['summer', 'fall', 'winter', 'spring'], doc='quarters')
model.s = Set(initialize=['first', 'second'], doc='shifts')
model.l = Set(initialize=[1, 2, 3, 4], doc='production levels')
model.prp =  Set(initialize=['labor', 'motor'], doc='production relationship parameters')
model.scp =  Set(initialize=['fixed', 'labor'], doc='shift cost parameters')

# Parameters
model.d = Param(model.q, initialize={'spring': 24000}, default =0, mutable = True, doc='demand        (motors per season)')
model.lc = Param(model.q, initialize={'summer': 15000}, default =0, mutable = True, doc='leasing cost (dollars per season)')
model.ei = Param(model.q, initialize={'summer': 84}, default =0, mutable = True, doc= 'initial employment')
model.mc = Param(initialize=100, mutable = True, doc='material cost  (dollars per motor)')
model.sr = Param(initialize=2, mutable = True, doc='space rental   (dollars per motor)')
model.hc = Param(initialize=900, mutable = True, doc='hiring cost (dollars per employee)')
model.fc = Param(initialize=150, mutable = True, doc='firing cost (dollars per employee)')

model.delt = Param(model.q, initialize={q: 1/1.03**(qq - 1) for qq, q in enumerate(model.q, start=1)}, mutable = True, doc='discount factor')
model.invmax = Param(initialize=sum(model.d[q] for q in model.q), mutable = True, doc='upper bound on inventory  (motors)')

model.pr = Param(model.prp, model.l, initialize={
    ('labor', 1): 20,   ('labor', 2): 40,   ('labor', 3): 50,   ('labor', 4): 60,
    ('motor', 1): 1000, ('motor', 2): 3000, ('motor', 3): 4500, ('motor', 4): 5800
}, default =0, doc='production relationship')

model.sc = Param(model.scp, model.s, initialize={
    ('fixed', 'first'): 10000, ('fixed', 'second'): 16000,
    ('labor', 'first'): 3500,  ('labor', 'second'): 4100
}, default =0, mutable = True, doc='shift cost (dollars per shift)')

# Variables
model.cost = Var()
model.dpc = Var(model.q, doc='direct production cost      (1000 $ per season)')
model.isc = Var(model.q, doc='inventory storage cost      (1000 $ per season)')
model.wfc = Var(model.q, doc='workforce fluctuation cost  (1000 $ per season)')
model.src = Var(model.q, within=NonNegativeReals, doc='space rental cost           (1000 $ per season)')
model.p = Var(model.q, within=NonNegativeReals, doc='production                  (motors per season)')

model.ss = Var(model.l, model.q, model.s, within=NonNegativeReals, doc='production segments                 (sos2 type)')
model.ssb = Var(model.l, model.q, model.s, within=Binary, doc='0-1 needed for ss sos2 formulation')
model.inv = Var(model.q, within=NonNegativeReals, doc='inventory                   (motors per season)')
model.lease = Var(within=Binary, doc='lease-rent option')
model.e = Var(model.q, doc='total employment                    (employees)')
model.se = Var(model.q, model.s, doc='shift employment          (employees per shift)')
model.shift = Var(model.q, model.s, within=Binary, doc='shift use indicator                    (binary)')
model.h = Var(model.q, within=NonNegativeReals, doc='hirings in quarter                  (employees)')
model.f = Var(model.q, within=NonNegativeReals, doc='firings in quarter                  (employees)')

# Constraints

def ddpc_rule(model, q):
    return model.dpc[q] == (model.mc*model.p[q] + sum(model.sc['fixed', s]*model.shift[q, s] + model.sc['labor', s]*model.se[q, s] for s in model.s))/1000
model.ddpc = Constraint(model.q, rule=ddpc_rule, doc='direct production cost definition     (1000 $)')

def sbp_rule(model, q):
    return model.p[q] == sum(model.pr['motor', l]*model.ss[l, q, s] for l in model.l for s in model.s)
model.sbp = Constraint(model.q, rule=sbp_rule, doc='sos product balance                   (motors)')

def sbse_rule(model, q, s):
    return model.se[q, s] == sum(model.pr['labor', l]*model.ss[l, q, s] for l in model.l)
model.sbse = Constraint(model.q, model.s, rule=sbse_rule, doc='sos shift employment balance       (employees)')

def scc_rule(model, q, s):
    return sum(model.ss[l, q, s] for l in model.l) == model.shift[q, s]
model.scc = Constraint(model.q, model.s, rule=scc_rule, doc='sos shift link')

def invb_rule(model, q):
    key_list = list(model.q)
    q_idx = next((idx for idx, key in enumerate(key_list) if key == q)) 
    if q_idx > 0:
        return model.inv[q] == model.inv[model.q.prev(q)] + model.p[q] - model.d[q]
    else:
        return model.inv[q] == model.p[q] - model.d[q]
model.invb = Constraint(model.q, rule=invb_rule, doc='inventory balance                     (motors)')

def disc_rule(model, q):
    return model.isc[q] == (model.lc[q]*model.lease + model.src[q])/1000
model.disc = Constraint(model.q, rule=disc_rule, doc='inventory storage cost definition     (1000 $)')

def dsrc_rule(model, q):
    return model.src[q] >= model.sr*(model.inv[q] - model.invmax*model.lease)
model.dsrc = Constraint(model.q, rule=dsrc_rule, doc='definition: space rental')

def dwfc_rule(model, q):
    return model.wfc[q] == (model.hc*model.h[q] + model.fc*model.f[q])/1000
model.dwfc = Constraint(model.q, rule=dwfc_rule, doc= 'workforce fluctuation cost definition (1000 $)')

def ed_rule(model, q):
    return model.e[q] == sum(model.se[q, s] for s in model.s)
model.ed = Constraint(model.q, rule=ed_rule, doc='total employment definition        (employees)')

def eb1_rule(model, q):
    key_list = list(model.q)
    q_idx = next((idx for idx, key in enumerate(key_list) if key == q))
    if q_idx > 0:
        return model.e[q] == model.e[model.q.prev(q)] + model.h[q] - model.f[q] + model.ei[q]
    else:
        return model.e[q] == model.h[q] - model.f[q] + model.ei[q]
model.eb1 = Constraint(model.q, rule=eb1_rule, doc='employment balance type 1          (employees)')

def messb_rule(model, q, s):
    return sum(model.ssb[l, q, s] for l in model.l) == 1
model.messb = Constraint(model.q, model.s, rule=messb_rule, doc='mutual exclusivity for ssb')

def lssb_rule(model, l, q, s):
    if l > 2:
        return model.ss[l-1, q, s] + model.ss[l, q, s] <= model.ssb[l-2, q, s] + model.ssb[l-1, q, s] + model.ssb[l, q, s]
    elif l == 2:
        return model.ss[l-1, q, s] + model.ss[l, q, s] <= model.ssb[l-1, q, s] + model.ssb[l, q, s]
    elif l == 1:
        return model.ss[l, q, s] <= model.ssb[l, q, s]
model.lssb = Constraint(model.l, model.q, model.s, rule=lssb_rule, doc='ss - ssb linkage')

model.pup = Constraint(rule=lambda model: model.p['spring'] <= 0.8*len(model.s)*max(model.pr['motor',l] for l in model.l))

# Objective
model.obj = Objective(expr=sum(model.delt[q]*(model.dpc[q] + model.isc[q] + model.wfc[q]) for q in model.q), sense=minimize)