# Medium-Term Planning of Single-Stage Continuous Multiproduct Plants 2008

# This model considers the optimal medium-term planning of a single-stage plant.
# The plant manufactures several types of products in one processing machine over a planning horizon.
# The total available processing time is divided into multiple weeks.

# Source: https://pubs.acs.org/doi/10.1021/ie800646q
import json
import pyomo.environ as pyo

data = globals().get("data", {})
# with open("pp_data.json", "r") as file:
#     data = json.load(file)

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.C = pyo.Set(initialize=data["sets"]["C"], doc='Customers')
model.I = pyo.Set(initialize=data["sets"]["I"], doc='Products')
model.J = pyo.Set(initialize=data["sets"]["I"], doc='Products')
model.W = pyo.Set(initialize=data["sets"]["W"], doc='Weeks')

# Extract parameters from data
params = data["parameters"]
prices_data = params["prices_data"]

# Convert changeover_times_data from string keys to tuples
changeover_times_data = {tuple(k.split(",")): v for k, v in params["changeover_times_data"].items()}

# Convert demands_data from string keys to tuples and apply doubling
demands_data = {}
for key_str, value in params["demands_data"].items():
    parts = key_str.split(",")
    # Convert week (third element) to int
    key_tuple = (parts[0], parts[1], int(parts[2]))
    # Apply doubling
    demands_data[key_tuple] = value * 2

# Extract scalar parameters
ri = params["ri"]
t_l = params["t_l"]
t_u = params["t_u"]
bigm = params["bigm"]

# Define parameters
model.ps = pyo.Param(model.I, model.C, initialize=lambda model, i, c: prices_data[i] if c != 'C10' else prices_data[i] * 1.5, doc='Unit selling price of product to customer', mutable = True)
model.cb = pyo.Param(model.I, model.C, initialize=lambda model, i, c: (prices_data[i] * 0.2) if c != 'C10' else (prices_data[i] * 0.20 * 1.5), doc='Unit backlog penalty cost of product to customer', mutable = True)
model.ci = pyo.Param(model.I, initialize=lambda model, i: prices_data[i] * 0.05, doc='Unit inventory cost of product', mutable = True)
model.tau = pyo.Param(model.I, model.J, initialize=lambda model, i, j: changeover_times_data.get((i, j), 0) /60, default = 0, doc='Changeover time from product i to j in hours', mutable=True)
model.cc = pyo.Param(model.I, model.J, initialize=lambda model, i, j: model.tau[i, j] * 10, doc='Changeover cost from product i to j', mutable=True)
model.d = pyo.Param(model.C, model.I, model.W, initialize=demands_data, default=0, doc='Demand of customer c for product i in week w', mutable=True)
model.ri = pyo.Param(model.I, initialize=ri, doc='Processing rate of product i (ton/week)', mutable=True)
model.t_l = pyo.Param(initialize=t_l, doc='Lower bound for processing time in a week (hours)', mutable = True)
model.t_u = pyo.Param(initialize=t_u, doc='Upper bound for processing time in a week (hours)', mutable = True)
model.bigm = pyo.Param(initialize=bigm, doc="big M value", mutable=True)

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
