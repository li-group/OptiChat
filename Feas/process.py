import numpy as np
import pyomo.environ as pyo
import pyomo.opt as po
import json

# Load data from json
data = globals().get("data", {})
# with open("process_data.json", "r") as file:
#     data = json.load(file)

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.cond_hot_index = pyo.Set(initialize = data["sets"]["Condesnser_hot_streams"], doc = "Condesnser hot streams")
model.reb_cold_index = pyo.Set(initialize = data["sets"]["Reboiler_cold_streams"], doc = "Reboiler cold streams")
model.C_hu_index = pyo.Set(initialize = data["sets"]["Number_of_hot_utilities"], doc = "Number of hot utilities")
model.F_index = pyo.Set(initialize = data["sets"]["superstructure_columns"], doc = "Columns in the superstructure")
model.F_subset = pyo.Set(initialize=data["sets"]["ABC_stream_division"], doc = "number of division of ABC stream")
model.F_index_first_six = pyo.Set(initialize=data["sets"]["A/BC_type_column"], doc = "A/BC types column")
model.F_index_between_six = pyo.Set(initialize=data["sets"]["AB/C_type_column"], doc = "AB/C types column")
model.F_index_next_between_six = pyo.Set(initialize=data["sets"]["A/B_type_column"], doc = "A/B type columns")
model.F_index_last_six = pyo.Set(initialize=data["sets"]["B/C_type_column"], doc = "B/C types of column")

# print()
# for f in model.reb_cold_index:
#     print(f)

# Define parameters
cond_hot_dict = {int(k): v for k, v in data["parameters"]["cond_hot_temp"].items()}
reb_cold_dict = {int(k): v for k, v in data["parameters"]["reb_cold_temp"].items()}
C_hu_dict = {int(k): v / 1e6 for k, v in data["parameters"]["C_hu_cost"].items()}


params = data["parameters"]
model.FC = pyo.Param(initialize=params["Fixed_Cost"], doc = "Fixed cost", mutable = True)
model.F_tot = pyo.Param(initialize=params["Total_Flow_Rate"], doc = "total flow rate", mutable = True) 
model.K = pyo.Param(initialize=params["K"], doc = "Slope for heat duty vs flow rates", mutable = True) 
model.V = pyo.Param(initialize=params["V"], doc = "Slope of column cost of flowrates", mutable = True)
model.alpha = pyo.Param(initialize=params["alpha"], doc = "payout time for capital cost", mutable = True)
model.beta = pyo.Param(initialize=params["beta"], doc = "Income tax correction parameter", mutable = True)
model.C_cu = pyo.Param(initialize=params["C_cu"]/1e6, doc = "Cost of cold water", mutable = True)
model.hex_cost = pyo.Param(initialize = params["hex_cost"], doc = 'cost associated with heat exchanger')
model.C_hu = pyo.Param(model.C_hu_index, initialize=C_hu_dict, doc = "Cost of hot utilities i.e. LP, MP and HP steam", mutable = True)
model.Stream_Matrix = pyo.Param(model.reb_cold_index, model.cond_hot_index, initialize=dict(
    ((i, j), 1 if (i in reb_cold_dict) and (j in cond_hot_dict) and (cond_hot_dict[j] >= reb_cold_dict[i]) else 0) for i in model.reb_cold_index for j in model.cond_hot_index
), doc = "Possible connections between hot and cold", mutable = True)


# Define variables
model.F = pyo.Var(model.F_index, domain=pyo.NonNegativeReals, doc = "FLowrates in columns")
model.q = pyo.Var(model.reb_cold_index, model.cond_hot_index, domain=pyo.NonNegativeReals, doc = "heat transferred from hot uitlity or condenser to cold utility or reboiler", initialize=0)
model.y = pyo.Var(model.F_index, domain=pyo.Binary, doc = "Column selection")
model.Q1 = pyo.Var(model.F_index, domain=pyo.NonNegativeReals, doc = "Heta duties of the column")
model.QH = pyo.Var(model.cond_hot_index, domain=pyo.NonNegativeReals, doc = "heat associated with hot streams")
model.QC = pyo.Var(model.reb_cold_index, domain=pyo.NonNegativeReals, doc = "heat associated with cold streams")

# Define objective function
def objective_rule(model):
    dist_cost = sum(model.FC * model.y[j] + model.V * model.F[j] for j in model.F_index)
    hex_cost = sum(
        (model.Stream_Matrix[i, j] * 300 * model.q[i, j] / abs(reb_cold_dict[i] - cond_hot_dict[j])) if reb_cold_dict[i] - cond_hot_dict[j] != 0 else (model.Stream_Matrix[i, j] * model.hex_cost * model.q[i, j] / 10) for i in model.reb_cold_index for j in model.cond_hot_index
    )
    cold_utility_cost = sum(model.C_cu * model.Stream_Matrix[25, j] * model.q[25, j] for j in model.cond_hot_index)
    hot_utility_cost = sum(model.C_hu[j - 24] * model.q[i, j] for i in model.reb_cold_index for j in model.cond_hot_index if j > 24)
    return (dist_cost + hex_cost) / model.alpha + model.beta * (cold_utility_cost + hot_utility_cost)

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Define constraints
def mass_balance_rule(model):
    return sum(model.F[j] for j in model.F_subset) == model.F_tot

model.mass_balance_constraint = pyo.Constraint(rule=mass_balance_rule, doc = 'mass balance constraint')

def intermediate_BC_balance_rule(model):
    return sum(model.F[j] for j in model.F_index_first_six) * 0.6 == sum(model.F[j] for j in model.F_index_last_six)

model.intermediate_BC_balance_constraint = pyo.Constraint(rule=intermediate_BC_balance_rule, doc = 'intermediate BC mass balance')

def intermediate_AB_balance_rule(model):
    return sum(model.F[j] for j in model.F_index_between_six) * 0.9 == sum(model.F[j] for j in model.F_index_next_between_six)

model.intermediate_AB_balance_constraint = pyo.Constraint(rule=intermediate_AB_balance_rule, doc = 'intermediate AB mass balance')

def q_constraint_rule(model, j):
    return model.Q1[j] == model.K * model.F[j]

def qh_constraint_rule(model, j):
    return model.QH[j] == model.Q1[j]

def f_constraint_rule(model, j):
    return model.F[j] <= model.F_tot * model.y[j]

def qc_constraint_rule(model, j):
    return model.QC[j] == model.Q1[j]

model.q_constraint = pyo.Constraint(model.F_index, rule=q_constraint_rule, doc = 'heat duty calculation constraint')
model.qh_constraint = pyo.Constraint(model.F_index, rule=qh_constraint_rule, doc = 'equating hot stream duties with column duty constraint')
model.f_constraint = pyo.Constraint(model.F_index, rule=f_constraint_rule, doc = 'flowrates being 0 is column not selected constraint')
model.qc_constraint = pyo.Constraint(model.F_index, rule = qc_constraint_rule, doc = 'equating cold stream duties with column duty constraint')

def hot_stream_constraint_rule(model, j):
    return sum(model.q[i, j] * model.Stream_Matrix[i, j] for i in model.reb_cold_index) == model.QH[j]

model.hot_stream_constraints = pyo.Constraint(model.cond_hot_index, rule=hot_stream_constraint_rule, doc = 'hot stream duties matching constraint')

def cold_stream_constraint_rule(model, i):
    return sum(model.q[i, j] * model.Stream_Matrix[i, j] for j in model.cond_hot_index) == model.QC[i]

model.cold_stream_constraints = pyo.Constraint(model.reb_cold_index, rule=cold_stream_constraint_rule, doc = 'cold stream duties matching constraint')
