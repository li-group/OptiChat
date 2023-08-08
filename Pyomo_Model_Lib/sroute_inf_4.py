from pyomo.environ import *
import pyomo.environ as pe

# Set
I = ['boston', 'chicago', 'dallas', 'kansas-cty', 'losangeles', 'memphis', 'portland', 'salt-lake', 'wash-dc']
model = pe.ConcreteModel()
model.i = pe.Set(initialize=I, doc='cities')
model.r = pe.Set(model.i, model.i, doc='routes')

# Alias
model.ip = pe.Set(initialize = I)
model.ipp = pe.Set(initialize = I)

# Parameter
uarc_init = {
    ('boston', 'chicago'): 58, 
    ('boston', 'wash-dc'): 25,
    ('chicago', 'kansas-cty'): 29, 
    ('chicago', 'memphis'): 32,
    ('chicago', 'portland'): 130, 
    ('chicago', 'salt-lake'): 85,
    ('dallas', 'kansas-cty'): 29, 
    ('dallas', 'losangeles'): 85,
    ('dallas', 'memphis'): 28, 
    ('dallas', 'salt-lake'): 75,
    ('kansas-cty', 'memphis'): 27, 
    ('kansas-cty', 'salt-lake'): 66,
    ('kansas-cty', 'wash-dc'): 62, 
    ('losangeles', 'portland'): 58,
    ('losangeles', 'salt-lake'): 43, 
    ('memphis', 'wash-dc'): 53,
    ('portland', 'salt-lake'): 48
}
model.uarc = pe.Param(model.i, model.ip, initialize=uarc_init, mutable = True, doc='undirected arcs')

model.routes_chicago = pe.Param(initialize=10, mutable = True, doc='routes from chicago')

def darc_init(model, i, ip):
    if (i,ip) in uarc_init.keys():
        return uarc_init[i, ip]
    elif (ip,i) in uarc_init.keys():
        return uarc_init[ip, i]
    else:
        return 0
model.darc = pe.Param(model.i, model.ip, initialize=darc_init, default = 0, doc='directed arcs')

# Variable
model.x = pe.Var(model.i, model.ip, model.ipp, within=NonNegativeReals, doc='arcs taken')
model.cost = pe.Var()

# Equation
def nb_rule(model, i, ip):
    if i != ip:
        return sum(model.x[i, ipp, ip] for ipp in model.i if model.darc[ipp, ip] != 0) >= sum(model.x[i, ip, ipp] for ipp in model.i if model.darc[ip, ipp] != 0) + 1
    else:
        return Constraint.Skip
model.nb = pe.Constraint(model.i, model.ip, rule=nb_rule, doc='node balance')

def cd_rule(model):
    return model.cost == sum(model.darc[ip, ipp] * model.x[i, ip, ipp] for i in model.i for ip in model.i for ipp in model.i)
model.cd = pe.Constraint(rule=cd_rule, doc='cost definition')

def chicago_rule(model):
    return sum(model.x['chicago', ip, ipp] for ip in model.i for ipp in model.i) <= model.routes_chicago
model.minx = pe.Constraint(rule=chicago_rule,doc ='max arcs taken from Chicago limit')

# Objective
model.obj = pe.Objective(expr=model.cost, sense=minimize)