# Medium-Term Planning of Single-Stage Continuous Multiproduct Plants 2008

# This model considers the optimal medium-term planning of a single-stage plant. 
# The plant manufactures several types of products in one processing machine over a planning horizon.
# The total available processing time is divided into multiple weeks.

# Source: https://pubs.acs.org/doi/10.1021/ie800646q
import pyomo.environ as pyo

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.C = pyo.Set(initialize=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10'], doc='Customers')
model.I = pyo.Set(initialize=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], doc='Products')
model.J = pyo.Set(initialize=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], doc='Products')
model.W = pyo.Set(initialize=[1, 2, 3, 4, 5, 6, 7, 8], doc='Weeks')

# Define parameters
prices_data = {
    'A': 10, 'B': 12, 'C': 13, 'D': 12, 'E': 15,
    'F': 10, 'G': 8, 'H': 14, 'I': 7, 'J': 15
}

changeover_times_data = {
    ('A', 'B'): 45, ('A', 'C'): 45, ('A', 'D'): 45, ('A', 'E'): 60, ('A', 'F'): 80, ('A', 'G'): 30, ('A', 'H'): 25, ('A', 'I'): 70, ('A', 'J'): 55,
    ('B', 'A'): 55, ('B', 'C'): 55, ('B', 'D'): 40, ('B', 'E'): 60, ('B', 'F'): 80, ('B', 'G'): 80, ('B', 'H'): 30, ('B', 'I'): 30, ('B', 'J'): 55,
    ('C', 'A'): 60, ('C', 'B'): 100,('C', 'D'): 100, ('C', 'E'): 75, ('C', 'F'): 60, ('C', 'G'): 80, ('C', 'H'): 80, ('C', 'I'): 75, ('C', 'J'): 75,
    ('D', 'A'): 60, ('D', 'B'): 100,('D', 'C'): 30, ('D', 'E'): 45, ('D', 'F'): 45, ('D', 'G'): 45, ('D', 'H'): 60, ('D', 'I'): 80, ('D', 'J'): 100,
    ('E', 'A'): 60, ('E', 'B'): 60,('E', 'C'): 55, ('E', 'D'): 30, ('E', 'F'): 35, ('E', 'G'): 30, ('E', 'H'): 35, ('E', 'I'): 60, ('E', 'J'): 90,
    ('F', 'A'): 75, ('F', 'B'): 75, ('F', 'C'): 60, ('F', 'D'): 100, ('F', 'E'): 75, ('F', 'G'): 100, ('F', 'H'): 75, ('F', 'I'): 100, ('F', 'J'): 60,
    ('G', 'A'): 80, ('G', 'B'): 100, ('G', 'C'): 30, ('G', 'D'): 60, ('G', 'E'): 100, ('G', 'F'): 85, ('G', 'H'): 60, ('G', 'I'): 100, ('G', 'J'): 65,
    ('H', 'A'): 60, ('H', 'B'): 60, ('H', 'C'): 60, ('H', 'D'): 60, ('H', 'E'): 60, ('H', 'F'): 60, ('H', 'G'): 60, ('H', 'I'): 60, ('H', 'J'): 60,
    ('I', 'A'): 80, ('I', 'B'): 80, ('I', 'C'): 30, ('I', 'D'): 30, ('I', 'E'): 60, ('I', 'F'): 70, ('I', 'G'): 55, ('I', 'H'): 85, ('I', 'J'): 100,
    ('J', 'A'): 100, ('J', 'B'): 100, ('J', 'C'): 60, ('J', 'D'): 80, ('J', 'E'): 80, ('J', 'F'): 30, ('J', 'G'): 45, ('J', 'H'): 100, ('J', 'I'): 100, 
}

demands_data = {
    ('C1', 'A', 1): 5, ('C1', 'A', 5): 5,
    ('C5', 'A', 1): 5, ('C5', 'A', 5): 5,
    ('C1', 'C', 1): 2, ('C1', 'C', 2): 2, ('C1', 'C', 3): 2, ('C1', 'C', 4): 2, ('C1', 'C', 5): 3, ('C1', 'C', 6): 3, ('C1', 'C', 7): 3, ('C1', 'C', 8): 3,
    ('C5', 'C', 1): 2, ('C5', 'C', 2): 2, ('C5', 'C', 3): 2, ('C5', 'C', 4): 2, ('C5', 'C', 5): 3, ('C5', 'C', 6): 3, ('C5', 'C', 7): 3, ('C5', 'C', 8): 3,
    ('C2', 'D', 1): 3, ('C2', 'D', 3): 3, ('C2', 'D', 5): 3, ('C2', 'D', 7): 3,
    ('C6', 'D', 1): 3, ('C6', 'D', 3): 3, ('C6', 'D', 5): 3, ('C6', 'D', 7): 3,
    ('C2', 'E', 1): 5, ('C2', 'E', 3): 5, ('C2', 'E', 5): 5, ('C2', 'E', 7): 5,
    ('C6', 'E', 1): 5, ('C6', 'E', 3): 5, ('C6', 'E', 5): 5, ('C6', 'E', 7): 5,
    ('C2', 'H', 2): 12, ('C2', 'H', 6): 12, ('C2', 'H', 8): 12,
    ('C6', 'H', 2): 12, ('C6', 'H', 6): 12, ('C6', 'H', 8): 12,    
    ('C3', 'B', 1): 4, ('C3', 'B', 5): 4,
    ('C7', 'B', 1): 4, ('C7', 'B', 5): 4,
    ('C9', 'B', 1): 4, ('C9', 'B', 5): 4,
    ('C3', 'G', 3): 5, ('C7', 'G', 3): 5, ('C9', 'G', 3): 5, 
    ('C3', 'J', 2): 6, ('C3', 'J', 4): 6, ('C3', 'J', 6): 6, ('C3', 'J', 8): 6,
    ('C7', 'J', 2): 6, ('C7', 'J', 4): 6, ('C7', 'J', 6): 6, ('C7', 'J', 8): 6,
    ('C9', 'J', 2): 6, ('C9', 'J', 4): 6, ('C9', 'J', 6): 6, ('C9', 'J', 8): 6,
    ('C4', 'A', 1): 7, ('C4', 'A', 5): 7, ('C8', 'A', 1): 7, ('C8', 'A', 5): 7, ('C10', 'A', 1): 7, ('C10', 'A', 5): 7,     
    ('C4', 'B', 2): 5, ('C4', 'B', 4): 5, ('C4', 'B', 7): 5, ('C8', 'B', 2): 5, ('C8', 'B', 4): 5, ('C8', 'B', 7): 5, ('C10', 'B', 2): 5, ('C10', 'B', 4): 5, ('C10', 'B', 7): 5, 
    ('C4', 'C', 1): 5, ('C4', 'C', 4): 5, ('C4', 'C', 7): 5, ('C8', 'C', 1): 5, ('C8', 'C', 4): 5, ('C8', 'C', 7): 5, ('C10', 'C', 1): 5, ('C10', 'C', 4): 5, ('C10', 'C', 7): 5, 
    ('C4', 'D', 1): 10, ('C4', 'D', 6): 10, ('C8', 'D', 1): 10, ('C8', 'D', 6): 10, ('C10', 'D', 1): 10, ('C10', 'D', 6): 10, 
    ('C4', 'E', 1): 11, ('C4', 'E', 3): 11, ('C4', 'E', 5): 11, ('C4', 'E', 7): 11, ('C8', 'E', 1): 11, ('C8', 'E', 3): 11, ('C8', 'E', 5): 11, ('C8', 'E', 7): 11, ('C10', 'E', 1): 11, ('C10', 'E', 3): 11, ('C10', 'E', 5): 11, ('C10', 'E', 7): 11, 
    ('C4', 'F', 1): 8, ('C4', 'F', 4): 8, ('C4', 'F', 7): 8, ('C8', 'F', 1): 8, ('C8', 'F', 4): 8, ('C8', 'F', 7): 8, ('C10', 'F', 1): 8, ('C10', 'F', 4): 8, ('C10', 'F', 7): 8, 
    ('C4', 'G', 1): 4, ('C4', 'G', 3): 4, ('C4', 'G', 5): 4, ('C4', 'G', 7): 4, ('C8', 'G', 1): 4, ('C8', 'G', 3): 4, ('C8', 'G', 5): 4, ('C8', 'G', 7): 4, ('C10', 'G', 1): 4, ('C10', 'G', 3): 4, ('C10', 'G', 5): 4, ('C10', 'G', 7): 4, 
    ('C4', 'H', 1): 1, ('C4', 'H', 2): 1, ('C4', 'H', 3): 1, ('C4', 'H', 4): 3, ('C4', 'H', 5): 3, ('C4', 'H', 6): 3, ('C4', 'H', 7): 1, ('C4', 'H', 8): 1,
    ('C8', 'H', 1): 1, ('C8', 'H', 2): 1, ('C8', 'H', 3): 1, ('C8', 'H', 4): 3, ('C8', 'H', 5): 3, ('C8', 'H', 6): 3, ('C8', 'H', 7): 1, ('C8', 'H', 8): 1,
    ('C10', 'H', 1): 1, ('C10', 'H', 2): 1, ('C10', 'H', 3): 1, ('C10', 'H', 4): 3, ('C10', 'H', 5): 3, ('C10', 'H', 6): 3, ('C10', 'H', 7): 1, ('C10', 'H', 8): 1,
    ('C4', 'I', 1): 5, ('C4', 'I', 2): 5, ('C4', 'I', 3): 5, ('C4', 'I', 4): 5, ('C4', 'I', 5): 5, ('C4', 'I', 6): 5, ('C4', 'I', 7): 5, ('C4', 'I', 8): 5,
    ('C8', 'I', 1): 5, ('C8', 'I', 2): 5, ('C8', 'I', 3): 5, ('C8', 'I', 4): 5, ('C8', 'I', 5): 5, ('C8', 'I', 6): 5, ('C8', 'I', 7): 5, ('C8', 'I', 8): 5,
    ('C10', 'I', 1): 5, ('C10', 'I', 2): 5, ('C10', 'I', 3): 5, ('C10', 'I', 4): 5, ('C10', 'I', 5): 5, ('C10', 'I', 6): 5, ('C10', 'I', 7): 5, ('C10', 'I', 8): 5,
    ('C4', 'J', 2): 3, ('C4', 'J', 4): 3, ('C4', 'J', 5): 3, ('C4', 'J', 7): 3, ('C8', 'J', 2): 3, ('C8', 'J', 4): 3, ('C8', 'J', 5): 3, ('C8', 'J', 7): 3, ('C10', 'J', 2): 3, ('C10', 'J', 4): 3, ('C10', 'J', 5): 3, ('C10', 'J', 7): 3, 
    }

demands_data = {key: value * 2 for key, value in demands_data.items()}

model.ps = pyo.Param(model.I, model.C, initialize=lambda model, i, c: prices_data[i] if c != 'C10' else prices_data[i] * 1.5, doc='Unit selling price of product to customer', mutable = True)
model.cb = pyo.Param(model.I, model.C, initialize=lambda model, i, c: (prices_data[i] * 0.2) if c != 'C10' else (prices_data[i] * 0.20 * 1.5), doc='Unit backlog penalty cost of product to customer', mutable = True)
model.ci = pyo.Param(model.I, initialize=lambda model, i: prices_data[i] * 0.05, doc='Unit inventory cost of product', mutable = True)
model.tau = pyo.Param(model.I, model.J, initialize=lambda model, i, j: changeover_times_data.get((i, j), 0) /60, default = 0, doc='Changeover time from product i to j in hours', mutable=True)
model.cc = pyo.Param(model.I, model.J, initialize=lambda model, i, j: model.tau[i, j] * 10, doc='Changeover cost from product i to j', mutable=True)
model.d = pyo.Param(model.C, model.I, model.W, initialize=demands_data, default=0, doc='Demand of customer c for product i in week w', mutable=True)
model.ri = pyo.Param(model.I, initialize=110, doc='Processing rate of product i (ton/week)', mutable=True)
model.t_l = pyo.Param(initialize=5, doc='Lower bound for processing time in a week (hours)', mutable = True)
model.t_u = pyo.Param(initialize=168, doc='Upper bound for processing time in a week (hours)', mutable = True)
model.bigm = pyo.Param(initialize=1e4, doc="big M value", mutable=True)

# Define variables
model.e = pyo.Var(model.I, model.W, within=pyo.Binary, doc="1 if product i is processed during week w; 0 otherwise")
model.f = pyo.Var(model.I, model.W, within=pyo.Binary, doc="1 if product i is the first one in week w; 0 otherwise")
model.l = pyo.Var(model.I, model.W, within=pyo.Binary, doc="1 if product i is the last one in week w; 0 otherwise")
model.z = pyo.Var(model.I, model.I, model.W, within=pyo.Binary, doc="1 if product i immediately precedes product j during week w; 0 otherwise")
model.zf = pyo.Var(model.I, model.J, model.W, within=pyo.NonNegativeReals, bounds=(0,1), doc="Changeover between weeks w-1 and w from product i to j")
model.o = pyo.Var(model.I, model.W, within=pyo.NonNegativeReals, doc="Order index of product i during week w")
model.p = pyo.Var(model.I, model.W, within=pyo.NonNegativeReals, doc="Amount of product i produced during week w")
model.s = pyo.Var(model.C, model.I, model.W, within=pyo.NonNegativeReals, doc="Sales volume of product i to customer c during week w")
model.t = pyo.Var(model.I, model.W, within=pyo.NonNegativeReals, doc="Processing time of product i during week w")
model.v = pyo.Var(model.I, model.W, within=pyo.NonNegativeReals, doc="Inventory volume of product i at the end of week w")
model.delta = pyo.Var(model.C, model.I, model.W, within=pyo.NonNegativeReals, doc="Backlog of product i to customer c at the end of week w")

# Define Objective function
def objective_rule(model):
    # Revenue from sales
    revenue = sum(model.ps[i, c] * model.s[c, i, w] for c in model.C for i in model.I for w in model.W)
    # Changeover costs between products within the same week
    changeover_costs = sum(model.cc[i, j] * model.z[i, j, w] for i in model.I for j in model.I if i != j for w in model.W)
    # Changeover costs between weeks
    changeover_weekly_costs = sum(model.cc[i, j] * model.zf[i, j, w] for i in model.I for j in model.I if i != j for w in model.W if w > 1)
    # Backlog penalty costs
    backlog_costs = sum(model.cb[i, c] * model.delta[c, i, w] for c in model.C for i in model.I for w in model.W)
    # Inventory costs
    inventory_costs = sum(model.ci[i] * model.v[i, w] for i in model.I for w in model.W)
    
    return revenue - (changeover_costs + changeover_weekly_costs + backlog_costs + inventory_costs)

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

# Define constraints
# Constraint for exactly one first product each week
def first_product_rule(model, w):
    return sum(model.f[i, w] for i in model.I) == 1
model.fpc = pyo.Constraint(model.W, rule=first_product_rule, doc='Exactly one first product each week')

# Constraint for exactly one last product each week
def last_product_rule(model, w):
    return sum(model.l[i, w] for i in model.I) == 1
model.lpc = pyo.Constraint(model.W, rule=last_product_rule, doc='Exactly one last product each week')

# Remaining assignmnet constraints
def first_prod_in_week(model, i, w):
    return model.f[i, w] <= model.e[i, w]
model.fpw = pyo.Constraint(model.I, model.W, rule=first_prod_in_week, doc='First product can only be assigned if processed in week')

def last_prod_in_week(model, i, w):
    return model.l[i, w] <= model.e[i, w]
model.lpw = pyo.Constraint(model.I, model.W, rule=last_prod_in_week, doc='Last product can only be assigned if processed in week')

# Changeover constraints
def sequence_1(model, j, w):
    return sum(model.z[i, j, w] for i in model.I if i != j) == model.e[j, w] - model.f[j, w]
model.spc = pyo.Constraint(model.I, model.W, rule=sequence_1, doc="If not first, product must be preceded by another")

def sequence_2(model, i, w):
    return sum(model.z[i, j, w] for j in model.I if i != j) == model.e[i, w] - model.l[i, w]
model.sfc = pyo.Constraint(model.I, model.W, rule=sequence_2, doc="If not last, product must be followed by another")

def sequence_3(model, j, w):
    if w == 1:
        return pyo.Constraint.Skip  # Skip the first week for this constraint
    return sum(model.zf[i, j, w] for i in model.I) == model.f[j, w]
model.change_l = pyo.Constraint(model.I, model.W , rule=sequence_3, doc="Link changeover to first product of the week")

def sequence_4(model, i, w):
    if w == 1:
        return pyo.Constraint.Skip  # Skip the first week for this constraint
    return sum(model.zf[i, j, w] for j in model.I) == model.l[i, w-1]
model.change_n = pyo.Constraint(model.I, model.W , rule=sequence_4, doc="Link last product of week to changeover at start of next week")

# Subtour elimination constraints
def subtour_rule(model, i, j, w):
    if i == j:
        return pyo.Constraint.Skip
    return model.o[j, w] - (model.o[i, w] + 1) >= -model.bigm * (1 - model.z[i, j, w])

model.subrule = pyo.Constraint(model.I, model.I, model.W, rule=subtour_rule, doc="Sub-tour elimination constraint on order indices")

def index_rule(model, i, w):
    return model.o[i, w] <= model.bigm * model.e[i, w]

model.ind = pyo.Constraint(model.I, model.W, rule=index_rule, doc="Set order index to zero for not processed products")

def index2a_rule(model, i, w):
    return model.o[i, w] >= model.f[i, w]

def index2b_rule(model, i, w):
    return model.o[i, w] <= sum(model.e[j, w] for j in model.I)

model.ind2a = pyo.Constraint(model.I, model.W, rule=index2a_rule, doc="Order index at least as large as first-product indicator")
model.ind2b = pyo.Constraint(model.I, model.W, rule=index2b_rule, doc="Order index does not exceed total products processed")

def time_lb(model, i, w):
    return model.t[i, w] >= model.t_l * model.e[i, w]
model.plb = pyo.Constraint(model.I, model.W, rule=time_lb, doc="Lower bound on processing time")

def time_ub(model, i, w):
    return model.t[i, w] <= model.t_u * model.e[i, w]
model.pub = pyo.Constraint(model.I, model.W, rule=time_ub, doc="Upper bound on processing time")

def total_time(model, w):
    if w == 1:  # Skip the first week
        return pyo.Constraint.Skip
    processing_time = sum(model.t[i, w] for i in model.I)
    changeover_time = sum((model.z[i, j, w] + model.zf[i, j, w]) * model.tau[i, j] for i in model.I for j in model.I if i != j)
    return processing_time + changeover_time <= model.t_u
model.tt = pyo.Constraint(model.W, rule=total_time, doc="Total processing and changeover time for all weeks except the first")

def time_first(model, w):
    if w != 1:  # Apply only to the first week
        return pyo.Constraint.Skip
    processing_time = sum(model.t[i, w] for i in model.I)
    changeover_time = sum(model.z[i, j, w] * model.tau[i, j] for i in model.I for j in model.I if i != j)
    return processing_time + changeover_time <= model.t_u
model.ttf = pyo.Constraint(model.W, rule=time_first, doc="Total processing and changeover time for the first week")

def production_amount(model, i, w):
    return model.p[i, w] == model.ri[i] * model.t[i, w]
model.pa = pyo.Constraint(model.I, model.W, rule=production_amount, doc="Product amount produced per week")

def backlog_rule(model, c, i, w):
    previous_backlog = model.delta[c, i, w-1] if w > 1 else 0
    return model.delta[c, i, w] == previous_backlog + model.d[c, i, w] - model.s[c, i, w]
model.back = pyo.Constraint(model.C, model.I, model.W, rule=backlog_rule, doc="Backlog of product to customer per week")

def inventory_rule(model, i, w):
    previous_inventory = model.v[i, w-1] if w > 1 else 0
    return model.v[i, w] == previous_inventory + model.p[i, w] - sum(model.s[c, i, w] for c in model.C)
model.ic = pyo.Constraint(model.I, model.W, rule=inventory_rule, doc="Inventory of product per week")