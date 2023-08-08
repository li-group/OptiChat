#adapted from marco.gms : Mini Oil Refining Model (GAMS Model Library)
#https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_marco.html

from pyomo.environ import *
# Set
c = ['crude', 'butane', 'mid-c', 'w-tex', 'sr-gas', 'sr-naphtha', 'sr-dist', 'sr-gas-oil', 'sr-res', 'rf-gas', 'fuel-gas', 'cc-gas', 'cc-gas-oil', 'hydro-res', 'premium', 'regular', 'distillate', 'fuel-oil']#'all commodities'
cf = ['premium', 'regular', 'distillate', 'fuel-oil', 'fuel-gas']#final product
cr = ['mid-c', 'w-tex']#crude oils
ci = ['butane', 'sr-gas', 'sr-naphtha', 'sr-dist', 'sr-gas-oil', 'sr-res', 'rf-gas', 'fuel-gas', 'cc-gas', 'cc-gas-oil', 'hydro-res']#intermediates
cd = ['butane']#domestic products
p = ['a-dist', 'n-reform', 'cc-dist', 'cc-gas-oil', 'hydro']#processes
m = ['a-still', 'reformer', 'c-crack', 'hydro']#productive units
q = ['octane', 'vapor-pr', 'density', 'sulfur']   #'quality attributes'
# tuples for blending possibility
bp_tuples = [('premium', 'butane'), ('premium', 'sr-gas'), ('premium', 'rf-gas'), ('premium', 'cc-gas'), ('premium', 'sr-naphtha'),
             ('regular', 'butane'), ('regular', 'sr-gas'), ('regular', 'rf-gas'), ('regular', 'cc-gas'), ('regular', 'sr-naphtha'),
             ('distillate', 'sr-dist'), ('distillate', 'sr-naphtha'), ('distillate', 'sr-gas-oil'), ('distillate', 'cc-gas-oil'),
             ('fuel-oil', 'sr-gas-oil'), ('fuel-oil', 'sr-res'), ('fuel-oil', 'cc-gas-oil'), ('fuel-oil', 'hydro-res'),
             ('fuel-gas', 'fuel-gas')]
# Table a
#input output coefficients
a_values = {}
a_values[('mid-c', 'crude', 'a-dist')] = -1.0
a_values[('mid-c', 'sr-gas', 'a-dist')] = .236
a_values[('mid-c', 'sr-naphtha', 'a-dist')] = .223
a_values[('mid-c', 'sr-naphtha', 'n-reform')] = -1.0
a_values[('mid-c', 'sr-dist', 'a-dist')] = .087
a_values[('mid-c', 'sr-dist', 'cc-dist')] = -1.0
a_values[('mid-c', 'sr-gas-oil', 'a-dist')] = .111
a_values[('mid-c', 'sr-gas-oil', 'cc-gas-oil')] = -1.0
a_values[('mid-c', 'sr-res', 'a-dist')] = .315
a_values[('mid-c', 'rf-gas', 'n-reform')] = .807
a_values[('mid-c', 'fuel-gas', 'a-dist')] = .029
a_values[('mid-c', 'fuel-gas', 'n-reform')] = .129
a_values[('mid-c', 'fuel-gas', 'cc-dist')] = .30
a_values[('mid-c', 'fuel-gas', 'cc-gas-oil')] = .31
a_values[('mid-c', 'cc-gas', 'cc-dist')] = .59
a_values[('mid-c', 'cc-gas', 'cc-gas-oil')] = .59
a_values[('mid-c', 'cc-gas-oil', 'cc-dist')] = .21
a_values[('mid-c', 'cc-gas-oil', 'cc-gas-oil')] = .22
a_values[('w-tex', 'crude', 'a-dist')] = -1.0
a_values[('w-tex', 'sr-gas', 'a-dist')] = .180
a_values[('w-tex', 'sr-naphtha', 'a-dist')] = .196
a_values[('w-tex', 'sr-naphtha', 'n-reform')] = -1.0
a_values[('w-tex', 'sr-dist', 'a-dist')] = .073
a_values[('w-tex', 'sr-dist', 'cc-dist')] = -1.0
a_values[('w-tex', 'sr-gas-oil', 'a-dist')] = .091
a_values[('w-tex', 'sr-gas-oil', 'cc-gas-oil')] = -1.0
a_values[('w-tex', 'sr-res', 'hydro')] = -1.0
a_values[('w-tex', 'rf-gas', 'n-reform')] = .836
a_values[('w-tex', 'fuel-gas', 'a-dist')] = .017
a_values[('w-tex', 'fuel-gas', 'n-reform')] = .099
a_values[('w-tex', 'fuel-gas', 'cc-dist')] = .36
a_values[('w-tex', 'fuel-gas', 'cc-gas-oil')] = .38
a_values[('w-tex', 'cc-gas', 'cc-dist')] = .58
a_values[('w-tex', 'cc-gas', 'cc-gas-oil')] = .60
a_values[('w-tex', 'cc-gas-oil', 'cc-dist')] = .15
a_values[('w-tex', 'cc-gas-oil', 'cc-gas-oil')] = .15
a_values[('w-tex', 'hydro-res', 'hydro')] = .97


# Table b capacity utilization
b_values = {('a-still', 'a-dist'): 1.0,
            ('reformer', 'n-reform'): 1.0,
            ('c-crack', 'cc-gas-oil'): 1.0,
            ('c-crack', 'cc-gas-oil'): 1.0}

# Parameter k initial capacity 
k_values = {'a-still': 100,
            'reformer': 20,
            'c-crack': 30}

# Parameter pd: prices of domestic products ($ pb)
pd_values = {'butane': 6.75}

# Parameter pr: prices of crude oils
pr_values = {'mid-c': 7.50,
             'w-tex': 6.50}

# Parameter pf: prices of final products
pf_values = {'premium': 10.5,
             'regular': 9.1,
             'distillate': 7.7,
             'fuel-gas': 1.5,
             'fuel-oil': 6.65}

# Parameter ur: upper bnd on crude oil  (1000 bpd)
ur_values = {'mid-c': 200, 'w-tex': 200}

# Parameter op:  operating cost              ($ pb)
op_values = {'a-dist': 0.1,
             'n-reform': 0.15,
             'cc-dist': 0.8,
             'cc-gas-oil': 0.08,
             'hydro': 0.1}

# Table qs: 'product quality specifications'
qs_values = {('lower', 'premium', 'octane'): 90,
             ('lower', 'regular', 'octane'): 86,
             ('upper', 'premium', 'vapor-pr'): 12.7,
             ('upper', 'regular', 'vapor-pr'): 12.7,
             ('upper', 'distillate', 'density'): 306,
             ('upper', 'distillate', 'sulfur'): 0.5,
             ('upper', 'fuel-oil', 'density'): 352,
             ('upper', 'fuel-oil', 'sulfur'): 3.5}

# Table at: 'attributes for blending'
at_values = {('sr-gas', 'octane'): 78.5,
             ('sr-gas', 'vapor-pr'): 18.4,
             ('sr-naphtha', 'octane'): 65.0,
             ('sr-naphtha', 'vapor-pr'): 6.54,
             ('rf-gas', 'octane'): 104.0,
             ('rf-gas', 'vapor-pr'): 2.57,
             ('cc-gas', 'octane'): 93.7,
             ('cc-gas', 'vapor-pr'): 6.9,
             ('butane', 'octane'): 91.8,
             ('butane', 'vapor-pr'): 199.2}

# Table atc: 'attributes for blending by crude'
atc_values = {('mid-c', 'sr-naphtha', 'density'): 272.0,
              ('mid-c', 'sr-naphtha', 'sulfur'): 0.283,
              ('mid-c', 'sr-dist', 'density'): 292.0,
              ('mid-c', 'sr-dist', 'sulfur'): 0.526,
              ('mid-c', 'sr-gas-oil', 'density'): 295.0,
              ('mid-c', 'sr-gas-oil', 'sulfur'): 0.980,
              ('mid-c', 'cc-gas-oil', 'density'): 294.4,
              ('mid-c', 'cc-gas-oil', 'sulfur'): 0.353,
              ('mid-c', 'sr-res', 'density'): 343.0,
              ('mid-c', 'sr-res', 'sulfur'): 4.7,
              ('w-tex', 'sr-naphtha', 'density'): 272.0,
              ('w-tex', 'sr-naphtha', 'sulfur'): 1.48,
              ('w-tex', 'sr-dist', 'density'): 297.6,
              ('w-tex', 'sr-dist', 'sulfur'): 2.83,
              ('w-tex', 'sr-gas-oil', 'density'): 303.3,
              ('w-tex', 'sr-gas-oil', 'sulfur'): 5.05,
              ('w-tex', 'sr-res', 'density'): 365.0,
              ('w-tex', 'sr-res', 'sulfur'): 11.00,
              ('w-tex', 'cc-gas-oil', 'density'): 299.1,
              ('w-tex', 'cc-gas-oil', 'sulfur'): 1.31,
              ('w-tex', 'hydro-res', 'density'): 365.0,
              ('w-tex', 'hydro-res', 'sulfur'): 6.00}

# iterate over atc_values
for ci_ in ci:
    for cr_ in cr:
        for q_ in q:
            #if at_values has key (ci, q)
            if (ci_, q_) in at_values:
                atc_values[cr_,ci_,q_] = at_values[ci_, q_]

# Create a model
model = ConcreteModel()

# Define the sets
model.cr = Set(initialize=cr)
model.p = Set(initialize=p)
model.c = Set(initialize=c)
model.cf = Set(initialize=cf)
model.ci = Set(initialize=ci)
model.m = Set(initialize=m)
model.q = Set(initialize=q)
model.cd = Set(initialize=cd)
model.lim = Set(initialize=['lower', 'upper'])
model.bp = Set(within=model.cf*model.ci, initialize=bp_tuples)


# Define the parameters
model.a = Param(model.cr, model.c, model.p, default=0, mutable=True, initialize=a_values)
model.b = Param(model.m, model.p, default=0, mutable=True, initialize=b_values)
model.k = Param(model.m, default=0, mutable=True, initialize=k_values)
model.ur = Param(model.cr, default=0, mutable=True,initialize=ur_values)
model.qs = Param(model.lim, model.cf, model.q, default=0, initialize=qs_values)
model.atc = Param(model.cr, model.ci, model.q, default=0, mutable=True, initialize=atc_values)
model.pf = Param(model.cf, default=0, mutable=True, initialize=pf_values)
model.pr = Param(model.cr, default=0, mutable=True, initialize=pr_values)
model.pd = Param(model.cd, default=0, mutable=True, initialize=pd_values)
model.op = Param(model.p, default=0, mutable=True, initialize=op_values)

# Define the decision variables
model.z = Var(model.cr, model.p, domain=NonNegativeReals) #process level'
model.x = Var(model.cf, domain=NonNegativeReals)#final sales
model.u = Var(model.cr, domain=NonNegativeReals)#purchase of crude oil
model.ui = Var(model.cr, model.ci, domain=NonNegativeReals)#purchases of intermediate materials
model.w = Var(model.cr, model.ci, model.cf, domain=NonNegativeReals)#blending process level
model.phi = Var(domain=Reals)#total income
model.phir = Var(domain=Reals)#revenue from final product sales
model.phip = Var(domain=Reals)#input material cost
model.phiw = Var(domain=Reals)#operating cost


# Objective function
def objective_rule(model):
    return model.phir - model.phip - model.phiw
model.objective = Objective(rule=objective_rule, sense=maximize)

#material balances for crudes
def mbr_rule(model, cr):
    return sum(model.a[cr, 'crude', p] * model.z[cr, p] for p in model.p) + model.u[cr] >= 0
model.mbr = Constraint(model.cr, rule=mbr_rule)

# material balances for intermediates
def mb_rule(model, cr, ci):
    if ci in model.cd:
        return sum(model.a[cr, ci, p] * model.z[cr, p] for p in model.p) + model.ui[cr, ci] >= sum(model.w[cr, ci, cf] for cf in model.cf if (cf,ci) in model.bp)
    else:
        return sum(model.a[cr, ci, p] * model.z[cr, p] for p in model.p) >= sum(model.w[cr, ci, cf] for cf in model.cf if (cf,ci) in model.bp)
model.mb = Constraint(model.cr, model.ci, rule=mb_rule)

#capacity constraint
def cc_rule(model, m):
    return sum(model.b[m, p] * sum(model.z[cr, p] for cr in model.cr) for p in model.p) <= model.k[m]
model.cc = Constraint(model.m, rule=cc_rule)

#limits on crude oil purchases
def lcp_rule(model, cr):
    return model.u[cr] <= model.ur[cr]
model.lcp = Constraint(model.cr, rule=lcp_rule)

#blending balance
def bb_rule(model, cf):
    return model.x[cf] == sum(model.w[cr, ci, cf] for cr in model.cr for ci in model.ci if (cf,ci) in model.bp)
model.bb = Constraint(model.cf, rule=bb_rule)

#quality constraints lower bounds
def qlb_rule(model, cf, q):
    if model.qs["lower", cf, q] != 0:
        return sum(model.atc[cr, ci, q] * model.w[cr, ci, cf] for cr in model.cr for ci in model.ci) >= model.qs["lower", cf, q] * model.x[cf]
    else:
        return Constraint.Skip
model.qlb = Constraint(model.cf, model.q, rule=qlb_rule)

#quality constraints upper bounds
def qub_rule(model, cf, q):
    if model.qs["upper", cf, q] != 0:
        return sum(model.atc[cr, ci, q] * model.w[cr, ci, cf] for cr in model.cr for ci in model.ci) <= model.qs["upper", cf, q] * model.x[cf]
    else:
        return Constraint.Skip
model.qub = Constraint(model.cf, model.q, rule=qub_rule)

#revenue accounting
def arev_rule(model):
    return model.phir == sum(model.pf[cf] * model.x[cf] for cf in model.cf)
model.arev = Constraint(rule=arev_rule)

#material cost accounting
def amat_rule(model):
    return model.phip == sum(model.pr[cr] * model.u[cr] for cr in model.cr) + sum(model.pd[cd] * model.ui[cr, cd] for cd in model.cd for cr in model.cr)
model.amat = Constraint(rule=amat_rule)

#operating cost accounting
def aoper_rule(model):
    return model.phiw == sum(model.op[p] * sum(model.z[cr, p] for cr in model.cr) for p in model.p)
model.aoper = Constraint(rule=aoper_rule)