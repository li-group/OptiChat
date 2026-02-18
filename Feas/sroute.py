from pyomo.environ import *
import pyomo.environ as pe
import json

data = globals().get("data", {})
# with open("sroute_data.json", "r") as file:
#     data = json.load(file)

# Set
model = pe.ConcreteModel()
model.i = pe.Set(initialize=data["sets"]["cities"], doc='cities')
model.r = pe.Set(model.i, model.i, doc='routes')

# Alias
model.ip = pe.Set(initialize =  data["sets"]["cities"])
model.ipp = pe.Set(initialize = data["sets"]["cities"])

# Parameter

uarc_init = {
    tuple(k.split(",")): v for k, v in data["parameters"]["uarc"].items()
}

model.uarc = pe.Param(model.i, model.ip, initialize=uarc_init, mutable = True, doc='undirected arcs')


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

# Objective
model.obj = pe.Objective(expr=model.cost, sense=minimize)