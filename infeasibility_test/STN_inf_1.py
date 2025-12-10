from pyomo.environ import *
import numpy as np

# planning horizon
H = 10

Kondili = {
    # time grid
    'TIME': range(0, H + 1),

    # states
    'STATES': {
        'Feed_A': {'capacity': 500, 'initial': 500, 'price': 0},
        'Feed_B': {'capacity': 500, 'initial': 500, 'price': 0},
        'Feed_C': {'capacity': 500, 'initial': 500, 'price': 0},
        'Hot_A': {'capacity': 100, 'initial': 0, 'price': -100},
        'Int_AB': {'capacity': 200, 'initial': 0, 'price': -100},
        'Int_BC': {'capacity': 150, 'initial': 0, 'price': -100},
        'Impure_E': {'capacity': 100, 'initial': 0, 'price': -100},
        'Product_1': {'capacity': 500, 'initial': 0, 'price': 10},
        'Product_2': {'capacity': 500, 'initial': 0, 'price': 10},
    },

    # state-to-task arcs indexed by (state, task)
    'ST_ARCS': {
        ('Feed_A', 'Heating'): {'rho': 1.0},
        ('Feed_B', 'Reaction_1'): {'rho': 0.5},
        ('Feed_C', 'Reaction_1'): {'rho': 0.5},
        ('Feed_C', 'Reaction_3'): {'rho': 0.2},
        ('Hot_A', 'Reaction_2'): {'rho': 0.4},
        ('Int_AB', 'Reaction_3'): {'rho': 0.8},
        ('Int_BC', 'Reaction_2'): {'rho': 0.6},
        ('Impure_E', 'Separation'): {'rho': 1.0},
    },

    # task-to-state arcs indexed by (task, state)
    'TS_ARCS': {
        ('Heating', 'Hot_A'): {'dur': 1, 'rho': 1.0},
        ('Reaction_2', 'Product_1'): {'dur': 2, 'rho': 0.4},
        ('Reaction_2', 'Int_AB'): {'dur': 2, 'rho': 0.6},
        ('Reaction_1', 'Int_BC'): {'dur': 2, 'rho': 1.0},
        ('Reaction_3', 'Impure_E'): {'dur': 1, 'rho': 1.0},
        ('Separation', 'Int_AB'): {'dur': 2, 'rho': 0.1},
        ('Separation', 'Product_2'): {'dur': 1, 'rho': 0.9},
    },

    # unit data indexed by (unit, task)
    'UNIT_TASKS': {
        ('Heater', 'Heating'): {'Bmin': 0, 'Bmax': 100, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_1', 'Reaction_1'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_1', 'Reaction_2'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_1', 'Reaction_3'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_2', 'Reaction_1'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_2', 'Reaction_2'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Reactor_2', 'Reaction_3'): {'Bmin': 0, 'Bmax': 80, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
        ('Still', 'Separation'): {'Bmin': 0, 'Bmax': 200, 'Cost': 1, 'vCost': 0, 'Tclean': 0},
    },
}

STN = Kondili

STATES = STN['STATES']
ST_ARCS = STN['ST_ARCS']
TS_ARCS = STN['TS_ARCS']
UNIT_TASKS = STN['UNIT_TASKS']
TIME = STN['TIME']
H = max(TIME)

# set of tasks
TASKS = set([i for (j, i) in UNIT_TASKS])

# S[i] input set of states which feed task i
S = {i: set() for i in TASKS}
for (s, i) in ST_ARCS:
    S[i].add(s)

# S_[i] output set of states fed by task i
S_ = {i: set() for i in TASKS}
for (i, s) in TS_ARCS:
    S_[i].add(s)

# parameter rho[(i,s)] input fraction of task i from state s
rho = {(i, s): ST_ARCS[(s, i)]['rho'] for (s, i) in ST_ARCS}

# parameter rho_[(i,s)] output fraction of task i to state s
rho_ = {(i, s): TS_ARCS[(i, s)]['rho'] for (i, s) in TS_ARCS}

# parameter P[(i,s)] time for task i output to state s
P = {(i, s): TS_ARCS[(i, s)]['dur'] for (i, s) in TS_ARCS}

# parameter p[i] completion time for task i
p = {i: max([P[(i, s)] for s in S_[i]]) for i in TASKS}

# K[i] set of units capable of task i
K = {i: set() for i in TASKS}
for (j, i) in UNIT_TASKS:
    K[i].add(j)

# T[s] set of tasks receiving material from state s
T = {s: set() for s in STATES}
for (s, i) in ST_ARCS:
    T[s].add(i)

# set of tasks producing material for state s
T_ = {s: set() for s in STATES}
for (i, s) in TS_ARCS:
    T_[s].add(i)

# parameter C[s] storage capacity for state s
C = {s: STATES[s]['capacity'] for s in STATES}

UNITS = set([j for (j, i) in UNIT_TASKS])

# I[j] set of tasks performed with unit j
I = {j: set() for j in UNITS}
for (j, i) in UNIT_TASKS:
    I[j].add(i)

# parameter Bmax[(i,j)] maximum capacity of unit j for task i
Bmax = {(i, j): UNIT_TASKS[(j, i)]['Bmax'] / 4 for (j, i) in UNIT_TASKS}

# parameter Bmin[(i,j)] minimum capacity of unit j for task i
Bmin = {(i, j): UNIT_TASKS[(j, i)]['Bmin'] / 4 for (j, i) in UNIT_TASKS}

# parameter Pi[(s,t)] external entrance/exit
Pi = {(s, t): -10
if s in ['Product_1', 'Product_2'] and t > H / 2 else
10 if s in ['Feed_A', 'Feed_B', 'Feed_C'] and t > H / 2 else
0
      for s in STATES for t in TIME
      }

TIME = np.array(TIME)

model = ConcreteModel()

# W[i,j,t] 1 if task i starts in unit j at time t
model.W = Var(TASKS, UNITS, TIME, domain=Boolean)

# B[i,j,t,] size of batch assigned to task i in unit j at time t
model.B = Var(TASKS, UNITS, TIME, domain=NonNegativeReals)

# S[s,t] inventory of state s at time t
model.S = Var(list(STATES.keys()), TIME, domain=NonNegativeReals)

# Q[j,t] inventory of unit j at time t
model.Q = Var(UNITS, TIME, domain=NonNegativeReals)

# store pamameters
model.rho = Param(list(rho.keys()), mutable=True, initialize=rho)
model.rho_ = Param(list(rho_.keys()), mutable=True, initialize=rho_)
model.C = Param(list(C.keys()), mutable=True, initialize=C)
model.Bmax = Param(list(Bmax.keys()), mutable=True, initialize=Bmax)
model.Bmin = Param(list(Bmin.keys()), mutable=True, initialize=Bmin)
model.Pi = Param(list(Pi.keys()), mutable=True, initialize=Pi)

# Objective function

# project value
model.Value = Var(domain=NonNegativeReals)
model.valuec = Constraint(expr=model.Value == sum([STATES[s]['price'] * model.S[s, H] for s in STATES]))

# project cost
model.Cost = Var(domain=NonNegativeReals)
model.costc = Constraint(expr=model.Cost == sum([UNIT_TASKS[(j, i)]['Cost'] * model.W[i, j, t] +
                                                 UNIT_TASKS[(j, i)]['vCost'] * model.B[i, j, t] for i in TASKS for j in
                                                 K[i] for t in TIME]))

model.obj = Objective(expr=model.Value - model.Cost, sense=maximize)

# Constraints
model.cons = ConstraintList()

# units assignment
for j in UNITS:
    for t in TIME:
        lhs = 0
        for i in I[j]:
            for tprime in TIME:
                if tprime >= (t - p[i] + 1 - UNIT_TASKS[(j, i)]['Tclean']) and tprime <= t:
                    lhs += model.W[i, j, tprime]
        model.cons.add(lhs <= 1)

# state capacity limits
model.sc = Constraint(list(STATES.keys()), TIME, rule=lambda model, s, t: model.S[s, t] <= model.C[s])

# state mass balances
for s in STATES.keys():
    rhs = STATES[s]['initial']
    for t in TIME:
        rhs += model.Pi[(s, t)]
        for i in T_[s]:
            for j in K[i]:
                if t >= P[(i, s)]:
                    rhs += model.rho_[(i, s)] * model.B[i, j, max(TIME[TIME <= t - P[(i, s)]])]
        for i in T[s]:
            rhs -= model.rho[(i, s)] * sum([model.B[i, j, t] for j in K[i]])
        model.cons.add(model.S[s, t] == rhs)
        rhs = model.S[s, t]

    # unit capacity limits
for t in TIME:
    for j in UNITS:
        for i in I[j]:
            model.cons.add(model.W[i, j, t] * model.Bmin[i, j] <= model.B[i, j, t])
            model.cons.add(model.B[i, j, t] <= model.W[i, j, t] * model.Bmax[i, j])

