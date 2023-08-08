from pyomo.environ import *

model = ConcreteModel()

# Sets
model.p = Set(initialize=['one', 'two', 'three'], doc='production facilities')
model.d = Set(initialize=['east', 'south', 'west', 'north'], doc='distribution centers')
model.c = Set(initialize=[1, 2, 3, 4, 5], doc='customer zones')
model.m = Set(initialize=['january', 'february', 'march', 'april'], doc='month')
model.pf = Set(initialize=['min-prod', 'max-prod', 'over-prod', 'prod-cost', 'over-cost'], doc='production facility parameters')
model.dcp = Set(initialize=['max-invent', 'hold-cost'], doc='distribution center parameters')
model.czp = Set(initialize=['min-demand', 'max-demand', 'revenue'], doc='customer zone parameters')

# Parameters
model.pfd = Param(model.p, model.pf, initialize ={
    ('one', 'max-prod'):  5000,  ('one', 'over-prod'): 1000, ('one', 'prod-cost'): 35, ('one', 'over-cost'): 45,
    ('two', 'min-prod'):  1200,  ('two', 'max-prod'):  3000, ('two', 'over-prod'): 500,  ('two', 'prod-cost'): 40, ('two', 'over-cost'): 43,
    ('three', 'min-prod'):  700, ('three', 'max-prod'):  1500, ('three', 'prod-cost'): 38}, default = 0, mutable = True, doc='production facility data')
model.fdec = Param(model.p, model.d, initialize ={
   ('one', 'east'):  10, ('one', 'south'): 12,
   ('two', 'south'): 8, ('two', 'west'):  4, ('two', 'north'):  5,
   ('three', 'west'):  6, ('three', 'north'):  8}, default = 0, doc='first distribution echelon cost ($ per unit)')
model.sdec = Param(model.d, model.c, initialize ={
   ('east',1):  15, ('east',2): 19,
   ('south',2): 20, ('south',3): 22, ('south',4): 18,
   ('west',2):  16, ('west',4): 18, ('west',5): 19,
   ('north',4): 15, ('north',5): 21
   }, default = 0, doc='second distribution echelon cost ($ per unit)')    
model.dcd = Param(model.d, model.dcp, initialize ={
   ('east','max-invent'):  3000, ('east','hold-cost'):  2,
   ('south','max-invent'): 2500, ('south','hold-cost'): 2, 
   ('west','max-invent'):  4000, ('west','hold-cost'):  1, 
   ('north','max-invent'): 2500, ('north','hold-cost'): 3
   }, default = 0, mutable = True, doc='distribution center data') 
model.czd = Param(model.c, model.czp, initialize ={
   (1,'min-demand'): 2000, (1,'max-demand'):  2500, (1,'revenue'):  70,
   (2,'min-demand'): 0,    (2,'max-demand'):  2500, (2,'revenue'):  68,
   (3,'min-demand'): 2000, (3,'max-demand'):  3000, (3,'revenue'):  65,
   (4,'min-demand'): 1500, (4,'max-demand'):  2000, (4,'revenue'):  72,
   (5,'min-demand'): 1500, (5,'max-demand'):  3000, (5,'revenue'):  71
   }, default = 0, mutable = True, doc='customer zone data') 
model.pc = Param(model.p, model.m, initialize={
    ('one','january'): 35, ('one','february'): 36, ('one','march'): 37, ('one','april'): 38, 
    ('two','january'): 40, ('two','february'): 41, ('two','march'): 42, ('two','april'): 43,
    ('three','january'): 38, ('three','february'): 39, ('three','march'): 40, ('three','april'): 41,
    }, mutable = True, doc='production cost normal shift') 
model.pco = Param(model.p, model.m, initialize={    
    ('one','january'): 45, ('one','february'): 46, ('one','march'): 47, ('one','april'): 49, 
    ('two','january'): 43, ('two','february'): 44, ('two','march'): 45, ('two','april'): 47,
    ('three','january'): 0, ('three','february'): 1, ('three','march'): 2, ('three','april'): 4,
    }, mutable = True, doc='production cost overtime') 

model.revfac = Param(model.m, initialize={
    'january': 1, 'february': 1, 'march': 1.1, 'april': 1.1}, mutable = True, doc='revenue factor')

# Variables
model.x = Var(model.p, model.d, model.m, within=NonNegativeReals, doc='shipments from production to distribution')
model.y = Var(model.d, model.c, model.m, within=NonNegativeReals, doc='shipments from distribution centers to markets')
model.pn = Var(model.p, model.m, within=NonNegativeReals, doc='production')
model.po = Var(model.p, model.m, within=NonNegativeReals, doc='production: overtime')
model.s = Var(model.d, model.m, within=NonNegativeReals, doc='storage level')
model.dm = Var(model.c, doc='demand level')
model.h = Var(model.d, model.m, within=NonNegativeReals, doc='handling')
model.profit = Var()
model.revenue = Var()
model.transport = Var()
model.production = Var()
model.holding = Var()

# Constraints
def ib_rule(model, d, m):
    key_list = list(model.m)
    m_idx = next((idx for idx, key in enumerate(key_list) if key == m))
    if m_idx == 0:
        return model.h[d,m] == sum(model.x[p,d,m] for p in model.p if model.fdec[p,d] != 0)
    else:
        return model.h[d,m] == model.s[d,model.m.prev(m)] + sum(model.x[p,d,m] for p in model.p if model.fdec[p,d] != 0)
model.ib = Constraint(model.d, model.m, rule=ib_rule, doc='inventory balance')

def pb_rule(model, p, m):
    return model.pn[p,m] + model.po[p,m] == sum(model.x[p,d,m] for d in model.d if model.fdec[p,d] != 0)
model.pb = Constraint(model.p, model.m, rule=pb_rule, doc= 'production balance')

def hb_rule(model, d, m):
    return model.s[d,m] == model.h[d,m] - sum(model.y[d,c,m] for c in model.c if model.sdec[d,c] != 0)
model.hb = Constraint(model.d, model.m, rule=hb_rule, doc= 'handling balance')

def db_rule(model, c, m):
    return sum(model.y[d,c,m] for d in model.d if model.sdec[d,c] != 0) == model.dm[c]
model.db = Constraint(model.c, model.m, rule=db_rule, doc='demand balance')

def ar_rule(model):
    return model.revenue == sum(model.revfac[m]*model.czd[c,"revenue"]*model.y[d,c,m] for d in model.d for c in model.c for m in model.m if model.sdec[d,c] != 0)
model.ar = Constraint(rule=ar_rule, doc= 'revenue balance')

def at_rule(model):
    return model.transport == sum(sum(model.fdec[p,d]*model.x[p,d,m] for p in model.p) + sum(model.sdec[d,c]*model.y[d,c,m] for c in model.c) for d in model.d for m in model.m)
model.at = Constraint(rule=at_rule, doc= 'transport balance')

def ap_rule(model):
    return model.production == sum(model.pc[p,m]*model.pn[p,m] + model.pco[p,m]*model.po[p,m] for p in model.p for m in model.m)
model.ap = Constraint(rule=ap_rule, doc= 'production cost balance')

def ah_rule(model):
    return model.holding == sum(model.dcd[d,"hold-cost"]*model.s[d,m] for d in model.d for m in model.m)
model.ah = Constraint(rule=ah_rule, doc='inventory holding cost definition')

def apr_rule(model):
    return model.profit == model.revenue - model.transport - model.production - model.holding + 10
model.apr = Constraint(rule=apr_rule, doc= 'profit definition')

def slo_rule(model,d):
    return model.s[d,'april'] >= 200
model.slo = Constraint(model.d,rule=slo_rule)

def hup_rule(model,d,m):
    return model.h[d,m] <= model.dcd[d,'max-invent']
model.hup = Constraint(model.d,model.m,rule=hup_rule)

def pnlo_rule(model,p,m):
    return model.pn[p,m] >= model.pfd[p,'min-prod']
model.pnlo = Constraint(model.p,model.m,rule=pnlo_rule)

def pnup_rule(model,p,m):
    return model.pn[p,m] <= model.pfd[p,'max-prod']
model.pnup = Constraint(model.p,model.m,rule=pnup_rule)

def poup_rule(model,p,m):
    return model.po[p,m] <= model.pfd[p,'over-prod']
model.poup = Constraint(model.p,model.m,rule=poup_rule)

def dmlo_rule(model,c):
    return model.dm[c] >= model.czd[c,'min-demand']
model.dmlo = Constraint(model.c,rule=dmlo_rule)

def dmup_rule(model,c):
    return model.dm[c] <= model.czd[c,'max-demand']
model.dmup = Constraint(model.c,rule=dmup_rule)

# Objective
model.obj = Objective(expr=model.profit, sense=maximize)