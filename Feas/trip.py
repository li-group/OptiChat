import pyomo.environ as pyo
import numpy as np

## Given problem parameters
n = 4
h = 5
edges = [(1,2),(1,3),(2,1),(2,4),(3,1),(3,4),(4,2),(4,3)]

## Fix the random seed
np.random.seed(0)

## Big M constant
M = 5000

## Create the Pyomo model
model = pyo.ConcreteModel()

## Define sets
# traffic network G = (N,A)
model.N = pyo.RangeSet(n, doc="nodes")
model.A = pyo.Set(initialize=edges, doc="edges")

# time-expanded version of G
model.S = pyo.RangeSet(0,h,1, doc="periods")
model.cal_N = pyo.Set(initialize=model.A*model.S, doc="time-expanded nodes")
def cal_A_init(model):
    return [(x,tau,y,tau+s) for x in model.N for y in model.N if (x,y) in model.A \
                            for tau in pyo.RangeSet(0,h-1,1) for s in pyo.RangeSet(1,h,1) if (tau+s<=h)]
model.cal_A = pyo.Set(initialize=cal_A_init, doc="time-expanded edges")

## Define variables
model.delta = pyo.Var(model.cal_A, within=pyo.Boolean, initialize = 0, doc="link to choose")
model.f_d = pyo.Var(model.cal_A*model.N, within=pyo.NonNegativeReals, initialize = 0, 
                    doc="traffic volume of each link with destination d")

## Define parametes
model.beta = pyo.Param(model.N, model.N, initialize={
    (1,1): 0.000, (1,2): 2.000, (1,3): 1.768, (1,4): 3.576,
    (2,1): 2.000, (2,2): 0.000, (2,3): 3.768, (2,4): 2.348,
    (3,1): 1.768, (3,2): 3.768, (3,3): 0.000, (3,4): 1.808,
    (4,1): 3.576, (4,2): 2.348, (4,3): 1.808, (4,4): 0.000
}, doc="trip completion penalties", mutable = True)

model.nu = pyo.Param(model.N, model.N, pyo.RangeSet(0,h-1,1), initialize={
    (1,1,0): 0,  (1,2,0): 39, (1,3,0): 17, (1,4,0): 10,
    (2,1,0): 40, (2,2,0): 0,  (2,3,0): 7,  (2,4,0): 38,
    (3,1,0): 14, (3,2,0): 5,  (3,3,0): 0,  (3,4,0): 14,
    (4,1,0): 10, (4,2,0): 36, (4,3,0): 14, (4,4,0): 0,
    (1,1,1): 0,  (1,2,1): 45, (1,3,1): 17, (1,4,1): 11,
    (2,1,1): 44, (2,2,1): 0,  (2,3,1): 5,  (2,4,1): 31,
    (3,1,1): 21, (3,2,1): 5,  (3,3,1): 0,  (3,4,1): 15,
    (4,1,1): 12, (4,2,1): 44, (4,3,1): 15, (4,4,1): 0,
    (1,1,2): 0,  (1,2,2): 41, (1,3,2): 16, (1,4,2): 11,
    (2,1,2): 38, (2,2,2): 0,  (2,3,2): 7,  (2,4,2): 29,
    (3,1,2): 17, (3,2,2): 6,  (3,3,2): 0,  (3,4,2): 13,
    (4,1,2): 13, (4,2,2): 36, (4,3,2): 13, (4,4,2): 0,
    (1,1,3): 0,  (1,2,3): 37, (1,3,3): 14, (1,4,3): 12,
    (2,1,3): 38, (2,2,3): 0,  (2,3,3): 7,  (2,4,3): 42,
    (3,1,3): 15, (3,2,3): 6,  (3,3,3): 0,  (3,4,3): 13,
    (4,1,3): 9,  (4,2,3): 35, (4,3,3): 12, (4,4,3): 0,
    (1,1,4): 0,  (1,2,4): 44, (1,3,4): 17, (1,4,4): 12,
    (2,1,4): 42, (2,2,4): 0,  (2,3,4): 5,  (2,4,4): 41,
    (3,1,4): 17, (3,2,4): 6,  (3,3,4): 0,  (3,4,4): 16,
    (4,1,4): 12, (4,2,4): 30, (4,3,4): 13, (4,4,4): 0,
}, doc="travel demands", mutable=True)

model.Txy = pyo.Param(model.N,model.N, initialize={
    (1,2): 2.000, (1,3): 1.768, (2,4): 2.348, (3,4):1.808,
    (2,1): 2.000, (3,1): 1.768, (4,2): 2.348, (4,3):1.808
}, doc="freeflow travel time", mutable = True)

model.Cxy = pyo.Param(model.N,model.N, initialize={
    (1,2): 25.00, (1,3): 12.50, (2,4): 33.75, (3,4):12.50,
    (2,1): 25.00, (3,1): 12.50, (4,2): 33.75, (4,3):12.50
}, doc="Practical capacity", mutable = True)

def c_init_rule(model, x, tau_x, y, tau_y):
    if tau_y == h:
        return M
    else:
        s = tau_y - tau_x
        value = s*model.Cxy[x,y].value*((s/model.Txy[x,y].value - 1)/0.15)**0.25
        if isinstance(value, complex):
            return 0.0
        else:
            return value
model.c = pyo.Param(model.cal_A, initialize = c_init_rule, doc="link capacity", mutable = True)

## Define constraints
# Assumption 1: no dispersion of platoons within links
def no_dispersion_rule_1(model, x, tau_x, y, tau_y):
    return sum(model.f_d[x, tau_x, y, tau_y, d] for d in model.N) <= M*model.delta[x, tau_x, y, tau_y]
model.no_dispersion_const_1 = pyo.Constraint(model.cal_A, rule=no_dispersion_rule_1, 
                                             doc="no dispersion constraint - 1")

def no_dispersion_rule_2(model, x, y, tau_x):
    if tau_x < h and (x,y) in model.A:
        return sum(model.delta[x, tau_x, y, tau_x+s] for s in pyo.RangeSet(1,h-tau_x,1)) <= 1
    else:
        return pyo.Constraint.Skip
model.no_dispersion_const_2 = pyo.Constraint(model.N, model.N, model.S, rule=no_dispersion_rule_2, 
                                             doc="no dispersion constraint - 2")

# Assumption 2: Link consistency
def link_consistency_rule(model, x, y, tau_x, tau_w):
    if tau_x < tau_w and tau_w<h and (x,y) in model.A:
        expr_left = tau_x + sum(s*model.delta[x, tau_x, y, tau_x+s] for s in pyo.RangeSet(1,h-tau_x,1))
        expr_right = tau_w + sum(s*model.delta[x, tau_w, y, tau_w+s] for s in pyo.RangeSet(1,h-tau_w,1)) \
                           + M*(1 - sum(model.delta[x, tau_w, y, tau_w+s] for s in pyo.RangeSet(1,h-tau_w,1)))
        return expr_left <= expr_right
    else:
        return pyo.Constraint.Skip
model.link_consistency_const = pyo.Constraint(model.N, model.N, model.S, model.S, rule=link_consistency_rule, 
                                             doc="link consistency constraint")

# Assumption 4: Flow conservation except at trip completion
def flow_conservation_rule(model, x, d, tau_x):
    if tau_x < h and x != d:
        expr_left_1 = sum(sum(model.f_d[x, tau_x, y, tau_x+s, d] for y in model.N if (x,y) in model.A) for s in pyo.RangeSet(1,h-tau_x,1))
        if tau_x == 0:
            expr_left_2 = 0
        else:                 
            expr_left_2 = sum(sum(model.f_d[y, tau_x-s, x, tau_x, d] for y in model.N if (y,x) in model.A) for s in pyo.RangeSet(1,tau_x,1))
        expr_left = expr_left_1 - expr_left_2
        expr_right = np.random.normal(model.nu[x,d,tau_x].value, np.sqrt(0.1)*model.nu[x,d,tau_x].value)
        return expr_left == expr_right
    else:
        return pyo.Constraint.Skip
model.flow_conservation_const = pyo.Constraint(model.N, model.N, model.S, rule=flow_conservation_rule, 
                                             doc="flow conservation constraint")

# Assumption 5: Capacitated congestion modeling
def capacitated_congestion_rule(model, x, y, tau_x):
    if tau_x < h and (x,y) in model.A:
        expr_left = sum(sum(sum(model.f_d[x, tau_x-s1, y, tau_x+s2, d] for d in model.N) for s2 in pyo.RangeSet(1,h-tau_x,1)) for s1 in pyo.RangeSet(0, tau_x, 1))
        expr_right_1 = sum(model.delta[x, tau_x, y, tau_x+s]*model.c[x, tau_x, y, tau_x+s] for s in pyo.RangeSet(1,h-tau_x,1))
        expr_right_2 = M*(1 - sum(model.delta[x, tau_x, y, tau_x+s] for s in pyo.RangeSet(1,h-tau_x,1)))
        expr_right = expr_right_1 + expr_right_2
        return expr_left <= expr_right
    else:
        return pyo.Constraint.Skip
model.capacity_const = pyo.Constraint(model.N, model.N, model.S, rule=capacitated_congestion_rule, 
                                             doc="capacity constraint")

# Define objective function
def obj_func(model):
    obj_1 = sum(sum(sum(sum(sum(model.f_d[x, tau_x-s1, y, tau_x+s2, d] for d in model.N) for s2 in pyo.RangeSet(1,h-tau_x,1)) for s1 in pyo.RangeSet(0,tau_x,1)) for x in model.N for y in model.N if (x,y) in model.A) for tau_x in pyo.RangeSet(0,h-1,1))
    obj_2 = sum(sum(sum(model.beta[y,d]*model.f_d[x, h-s, y, h, d] for s in pyo.RangeSet(1,h)) for x in model.N for y in model.N if (x,y) in model.A) for d in model.N)
    obj = obj_1 + obj_2
    return obj
model.obj = pyo.Objective(rule=obj_func, sense=pyo.minimize)