import json
from pyomo.environ import *

data = globals().get("data", {})
# with open("pdi_data.json", "r") as file:
#     data = json.load(file)

model = ConcreteModel()

# Sets
model.p = Set(initialize=data["sets"]["p"], doc='production facilities')
model.d = Set(initialize=data["sets"]["d"], doc='distribution centers')
model.c = Set(initialize=data["sets"]["c"], doc='customer zones')
model.m = Set(initialize=data["sets"]["m"], doc='month')
model.pf = Set(initialize=data["sets"]["pf"], doc='production facility parameters')
model.dcp = Set(initialize=data["sets"]["dcp"], doc='distribution center parameters')
model.czp = Set(initialize=data["sets"]["czp"], doc='customer zone parameters')

# Helper function to parse comma-separated tuple keys from JSON
def _parse_str_str(param_dict):
    return {(k.split(',')[0], k.split(',')[1]): v for k, v in param_dict.items()}

def _parse_str_int(param_dict):
    return {(k.split(',')[0], int(k.split(',')[1])): v for k, v in param_dict.items()}

def _parse_int_str(param_dict):
    return {(int(k.split(',')[0]), k.split(',')[1]): v for k, v in param_dict.items()}

params = data["parameters"]

# Parameters
model.pfd = Param(model.p, model.pf, initialize=_parse_str_str(params["pfd"]), default=0, mutable=True, doc='production facility data')
model.fdec = Param(model.p, model.d, initialize=_parse_str_str(params["fdec"]), default=0, doc='first distribution echelon cost ($ per unit)')
model.sdec = Param(model.d, model.c, initialize=_parse_str_int(params["sdec"]), default=0, doc='second distribution echelon cost ($ per unit)')
model.dcd = Param(model.d, model.dcp, initialize=_parse_str_str(params["dcd"]), default=0, mutable=True, doc='distribution center data')
model.czd = Param(model.c, model.czp, initialize=_parse_int_str(params["czd"]), default=0, mutable=True, doc='customer zone data')
model.pc = Param(model.p, model.m, initialize=_parse_str_str(params["pc"]), mutable=True, doc='production cost normal shift')
model.pco = Param(model.p, model.m, initialize=_parse_str_str(params["pco"]), mutable=True, doc='production cost overtime')
model.revfac = Param(model.m, initialize=params["revfac"], mutable=True, doc='revenue factor')

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
model.pb = Constraint(model.p, model.m, rule=pb_rule, doc='production balance')

def hb_rule(model, d, m):
    return model.s[d,m] == model.h[d,m] - sum(model.y[d,c,m] for c in model.c if model.sdec[d,c] != 0)
model.hb = Constraint(model.d, model.m, rule=hb_rule, doc='handling balance')

def db_rule(model, c, m):
    return sum(model.y[d,c,m] for d in model.d if model.sdec[d,c] != 0) == model.dm[c]
model.db = Constraint(model.c, model.m, rule=db_rule, doc='demand balance')

def ar_rule(model):
    return model.revenue == sum(model.revfac[m]*model.czd[c,"revenue"]*model.y[d,c,m] for d in model.d for c in model.c for m in model.m if model.sdec[d,c] != 0)
model.ar = Constraint(rule=ar_rule, doc='revenue balance')

def at_rule(model):
    return model.transport == sum(sum(model.fdec[p,d]*model.x[p,d,m] for p in model.p) + sum(model.sdec[d,c]*model.y[d,c,m] for c in model.c) for d in model.d for m in model.m)
model.at = Constraint(rule=at_rule, doc='transport balance')

def ap_rule(model):
    return model.production == sum(model.pc[p,m]*model.pn[p,m] + model.pco[p,m]*model.po[p,m] for p in model.p for m in model.m)
model.ap = Constraint(rule=ap_rule, doc='production cost balance')

def ah_rule(model):
    return model.holding == sum(model.dcd[d,"hold-cost"]*model.s[d,m] for d in model.d for m in model.m)
model.ah = Constraint(rule=ah_rule, doc='inventory holding cost definition')

def apr_rule(model):
    return model.profit == model.revenue - model.transport - model.production - model.holding + 10
model.apr = Constraint(rule=apr_rule, doc='profit definition')

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
