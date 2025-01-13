# Morari, M, and Grossmann, I E, Eds, Chemical Engineering Optimization Models with GAMS. Computer Aids for Chemical Engineering Corporation, 1991.
# Raman, R, and Grossmann, I E, Relation between MINLP Modeling and Logical Inverence for Chemical Process Synthesis. Computers and Chemical Engineering 15, 2 (1991), 73-84.

# Given a set of possible chemical reactions (rxn 01-22), verify a chemical of interest can be synthesized from a set of available chemicals/raw materials and catalysts.

# Source: https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_reaction.html 

import pyomo.environ as pyo

# Create the Pyomo model
model = pyo.ConcreteModel(doc='Model for verifying the synthesis of acetone from available chemicals')

# Define sets 
# R : reaction ID, C : chemical ID
model.R = pyo.Set(initialize=['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', 
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', 
                              '21', '22'], doc='Reactions')
model.C = pyo.RangeSet(1, 34, doc='Chemicals')
model.RP = pyo.Set(initialize=[('01', 4), ('02', 6), ('03', 7), ('04', 3), ('05', 11),
                                ('06', 6), ('07', 15), ('08', 6), ('09', 6), ('10', 20),
                                ('11', 9), ('12', 24), ('13', 18), ('14', 21), ('15', 27),
                                ('16', 14), ('17', 32), ('18', 8), ('19', 30), ('20', 13),
                                ('21', 1), ('22', 34)
                               ], doc='Valid reaction and product pairs')

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

model.rm = pyo.Param(model.R, model.C, model.C, initialize={
    ('01', 4, 1): 1, ('01', 4, 2): 1, ('01', 4, 3): 1,
    ('02', 6, 4): 1, ('02', 6, 5): 1,
    ('03', 7, 4): 1, ('03', 7, 5): 1,
    ('04', 3, 4): 1, ('04', 3, 5): 1,
    ('05', 11, 8): 1, ('05', 11, 9): 1, ('05', 11, 10): 1,
    ('06', 6, 11): 1, ('06', 6, 12): 1, ('06', 6, 13): 1,
    ('07', 15, 14): 1, ('07', 15, 9): 1, ('07', 15, 10): 1, ('07', 15, 5): 1,
    ('08', 6, 15): 1, ('08', 6, 16): 1, ('08', 6, 17): 1,
    ('09', 6, 18): 1, ('09', 6, 19): 1, ('09', 6, 12): 1,
    ('10', 20, 18): 1, ('10', 20, 19): 1, ('10', 20, 12): 1,
    ('11', 9, 21): 1, ('11', 9, 22): 1,
    ('12', 24, 9): 1, ('12', 24, 23): 1,
    ('13', 18, 24): 1, ('13', 18, 17): 1,
    ('14', 21, 25): 1, ('14', 21, 26): 1,
    ('15', 27, 25): 1, ('15', 27, 26): 1,
    ('16', 14, 3): 1, ('16', 14, 28): 1, ('16', 14, 29): 1,
    ('17', 32, 30): 1, ('17', 32, 31): 1, ('17', 32, 12): 1,
    ('18', 8, 30): 1, ('18', 8, 31): 1, ('18', 8, 12): 1,
    ('19', 30, 25): 1, ('19', 30, 33): 1,
    ('20', 13, 25): 1, ('20', 13, 33): 1,
    ('21', 1, 34): 1, ('21', 1, 3): 1,
    ('22', 34, 14): 1, ('22', 34, 28): 1
    }, domain=pyo.Binary, doc='Reaction mappings from product to reactant', mutable=True)

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
model.reaction_cons = pyo.Constraint(model.R, model.C, rule=reactant_constraints)

avail_c = [2, 3, 5, 10, 12, 13, 17, 22, 25, 26, 28, 31, 33]
unavail_c = [16, 19]

def material_constraints(model, c):
# chemical/material availability constraints 
    if c in avail_c: 
        return model.y[c] == 1
    elif c in unavail_c:
        return model.y[c] == 0
    else:
        return pyo.Constraint.Skip
model.material_cons = pyo.Constraint(model.C, rule=material_constraints)

# Define objective function
model.obj = pyo.Objective(expr=model.y[6], sense=pyo.minimize, doc='verify acetone production feasibility')
