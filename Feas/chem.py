# Morari, M, and Grossmann, I E, Eds, Chemical Engineering Optimization Models with GAMS. Computer Aids for Chemical Engineering Corporation, 1991.
# Raman, R, and Grossmann, I E, Relation between MINLP Modeling and Logical Inverence for Chemical Process Synthesis. Computers and Chemical Engineering 15, 2 (1991), 73-84.

# Given a set of possible chemical reactions (rxn 01-22), verify a chemical of interest can be synthesized from a set of available chemicals/raw materials and catalysts.

# Source: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_reaction.html

import json
import pyomo.environ as pyo

data = globals().get("data", {})
# with open("chem_data.json", "r") as file:
#     data = json.load(file)

# Create the Pyomo model
model = pyo.ConcreteModel(doc='Model for verifying the synthesis of acetone from available chemicals')

# Define sets
# R : reaction ID, C : chemical ID
C_range = data["sets"]["C_range"]
model.R = pyo.Set(initialize=data["sets"]["R"], doc='Reactions')
model.C = pyo.RangeSet(C_range[0], C_range[1], doc='Chemicals')
model.RP = pyo.Set(initialize=[tuple(rp) for rp in data["sets"]["RP"]], doc='Valid reaction and product pairs')

''' model.C list:
        y01  'ch3co2c2h5',        y02  'naoc2h5'
        y03  'c2h5oh',            y04  'ch3coch2co2c2h5'
        y05  'h3o-hydronium ion', y06  'ch3coch3'
        y07  'co2',               y08  'ch3cn'
        y09  'ch3mgi',            y10  'c2h5oc2h5'
        y11  'ch3c(nmgi)ch3',     y12  'h2o'
        y13  'hcl',               y14  'ch3cho'
        y15  'ch3ch(oh)ch3',      y16  'cro3'
        y17  'h2so4',             y18  'ch2=c(ch3)2'
        y19  'o3',                y20  'hco2h'
        y21  'ch3i',              y22  'mg'
        y23  'ch3co2ch3',         y24  'hoc(ch3)3'
        y25  'ch4',               y26  'i2'
        y27  'hi',                y28  'o2'
        y29  'cr2o3',             y30  'ch3cl'
        y31  'nacn',              y32  'nacl'
        y33  'cl2',               y34  'ch3cooh'

'''

# Define parameters (reaction connections/mapping)
# rm is a binary parameter that indicates the reaction mapping of reactants to products for each reaction
# The mapping is defined as a dictionary with keys as a tuple of (reaction, product, reactant) and values as binary values (1 = necessary reactant for the reaction)

# Convert string keys from JSON to tuple keys
rm_raw = data["parameters"]["rm"]
rm_values = {(k.split(',')[0], int(k.split(',')[1]), int(k.split(',')[2])): v for k, v in rm_raw.items()}

model.rm = pyo.Param(model.R, model.C, model.C, initialize=rm_values, domain=pyo.Binary, doc='Reaction mappings from product to reactant', mutable=True)

# Define variables
# y : binary variable indicating whether a chemical is present
model.y = pyo.Var(model.C, domain=pyo.Binary, doc='presence of chemicals')

# Define constraints
def reactant_constraints(model, r, prod):
    if any((r, prod, react) in model.rm for react in model.C):
        reactant_expression = sum(model.rm[r, prod, react] * (1 - model.y[react]) for react in model.C if (r, prod, react) in model.rm)
        return reactant_expression >= (1 - model.y[prod])
    else:
        return pyo.Constraint.Skip
model.reaction_cons = pyo.Constraint(model.R, model.C, rule=reactant_constraints, doc='required reactants must be present to synthesize a product')

avail_c = data["parameters"]["avail_c"]
unavail_c = data["parameters"]["unavail_c"]

def material_constraints(model, c):
# chemical/material availability constraints
    if c in avail_c:
        return model.y[c] == 1
    elif c in unavail_c:
        return model.y[c] == 0
    else:
        return pyo.Constraint.Skip
model.material_cons = pyo.Constraint(model.C, rule=material_constraints, doc='fixes availability of raw materials and unavailable chemicals')

# Define objective function
model.obj = pyo.Objective(expr=model.y[6], sense=pyo.minimize, doc='verify acetone production feasibility')
