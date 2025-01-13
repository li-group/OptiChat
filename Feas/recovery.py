import numpy as np
import pyomo.environ as pyo

np.random.seed(15)

len_K = 10
len_I = 5
len_J = 3
len_M = 5
len_L = 10
len_N = 2

# Initialize Parameters Randomly as per the Paper
d = np.random.uniform(350,550,len_L)
r = np.random.uniform(450,650,len_K)
s = 0.2
cc = np.random.uniform(1500,2000,len_I)
ce = np.random.uniform(1500,2000,len_M)
cr = np.random.uniform(2000,3000,len_J)
cd = np.random.uniform(800,1000,len_N)

# Initialize Costs Randomly as per the Paper
f = np.random.uniform(210000,2400000,len_I)
g = np.random.uniform(4500000,4900000,len_J)
h = np.random.uniform(160000,200000,len_M)
c = np.random.uniform(45,55,(len_K,len_I))
a = np.random.uniform(45,55,(len_I,len_J))
b = np.random.uniform(45,55,(len_J,len_M))
e = np.random.uniform(45,55,(len_M,len_L))
v = np.random.uniform(45,55,(len_I,len_N))
pi = np.random.uniform(4500,6000,len_L)


# Create Model 
model = pyo.ConcreteModel()

# Decision Variables
model.X = pyo.Var(range(len_K), range(len_I), domain=pyo.NonNegativeReals, doc="Quantity of returned products shipped from customer zone k to collection/inspection center i")
model.U = pyo.Var(range(len_I), range(len_J), domain=pyo.NonNegativeReals, doc="Quantity of recoverable products shipped from collection/inspection center i to recovery center j")
model.P = pyo.Var(range(len_J), range(len_M), domain=pyo.NonNegativeReals, doc="Quantity of recovered products shipped from recovery center j to redistribution center m")
model.Q = pyo.Var(range(len_M), range(len_L), domain=pyo.NonNegativeReals, doc="Quantity of recovered products shipped from redistribution center m to customer zone l")
model.T = pyo.Var(range(len_I), range(len_N), domain=pyo.NonNegativeReals, doc="Quantity of scrapped products shipped from collection/inspection center i to disposal center n")          
model.delta = pyo.Var(range(len_L), domain=pyo.NonNegativeReals, doc="Quantity of non-satisfied demand of customer l")
model.Y = pyo.Var(range(len_I), domain=pyo.Binary, doc="1 if a collection=inspection center is opened at location i")                 
model.Z = pyo.Var(range(len_J), domain=pyo.Binary, doc="1 if a recovery center is opened at location j")               
model.W = pyo.Var(range(len_M), domain=pyo.Binary, doc="1 if a redistribution center is opened at location m")

# Model Parameters
model.d = pyo.Param(range(len_L), initialize={i: val for i, val in enumerate(d)}, mutable=True, doc="Demand of customer l for recovered products")
model.r = pyo.Param(range(len_K), initialize={i: val for i, val in enumerate(r)}, mutable=True, doc="Returns of used products from customer k")
model.s = pyo.Param(initialize=s, mutable=True, doc="Average disposal fraction")
model.cc = pyo.Param(range(len_I), initialize={i: val for i, val in enumerate(cc)}, mutable=True, doc="Capacity of handling returned products at collection/inspection i")
model.ce = pyo.Param(range(len_M), initialize={i: val for i, val in enumerate(ce)}, mutable=True, doc="Capacity of handling recovered products at redistribution center m")
model.cr = pyo.Param(range(len_J), initialize={i: val for i, val in enumerate(cr)}, mutable=True, doc="Capacity of handling recoverable products at recovery center j")
model.cd = pyo.Param(range(len_N), initialize={i: val for i, val in enumerate(cd)}, mutable=True, doc="Capacity of handling scrapped products at disposal center n")

model.f = pyo.Param(range(len_I), initialize={i: val for i, val in enumerate(f)}, mutable=True, doc="Fixed cost of opening collection/inspection center i")
model.g = pyo.Param(range(len_J), initialize={i: val for i, val in enumerate(g)}, mutable=True, doc="Fixed cost of opening recovery center j")
model.h = pyo.Param(range(len_M), initialize={i: val for i, val in enumerate(h)}, mutable=True, doc="Fixed cost of opening redistribution center m")
model.c = pyo.Param(range(len_K),range(len_I), initialize={(i, j): c[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])}, mutable=True, doc="Shipping cost per unit of returned products from customer zone k to collection/inspection center i")
model.a = pyo.Param(range(len_I,),range(len_J), initialize={(i, j): a[i, j] for i in range(a.shape[0]) for j in range(a.shape[1])}, mutable=True, doc="Shipping cost per unit of recoverable products from collection/inspection center i to recovery center j")
model.b = pyo.Param(range(len_J),range(len_M), initialize={(i, j): b[i, j] for i in range(b.shape[0]) for j in range(b.shape[1])}, mutable=True, doc="Shipping cost per unit of recovered products from recovery center j to redistribution center m")
model.e = pyo.Param(range(len_M),range(len_L), initialize={(i, j): e[i, j] for i in range(e.shape[0]) for j in range(e.shape[1])}, mutable=True, doc="Shipping cost per unit of recovered products from redistribution center m to customer zone l")
model.v = pyo.Param(range(len_I),range(len_N), initialize={(i, j): v[i, j] for i in range(v.shape[0]) for j in range(v.shape[1])}, mutable=True, doc="Shipping cost per unit of scrapped products from collection/inspection center i to disposal center n")
model.pi = pyo.Param(range(len_L), initialize={i: val for i, val in enumerate(pi)}, mutable=True, doc="Penalty cost per unit of non-satisfied demand of customer l")

# Helper Expressions in order to Understand the Optimized Result
model.construction_cost = pyo.Expression(expr=(
    sum(model.f[i] * model.Y[i] for i in range(len_I))
    + sum(model.g[i] * model.Z[i] for i in range(len_J))
    + sum(model.h[i] * model.W[i] for i in range(len_M))
))
model.shipping_cost = pyo.Expression(expr=(
    sum(model.c[i,j] * model.X[i,j] for i in range(len_K) for j in range(len_I))
    + sum(model.a[i,j] * model.U[i,j] for i in range(len_I) for j in range(len_J))
    + sum(model.b[i,j] * model.P[i,j] for i in range(len_J) for j in range(len_M))
    + sum(model.e[i,j] * model.Q[i,j] for i in range(len_M) for j in range(len_L))
    + sum(model.v[i,j] * model.T[i,j] for i in range(len_I) for j in range(len_N))
))
model.penalty_cost = pyo.Expression(expr=(
    sum(model.pi[i] * model.delta[i] for i in range(len_L))
))

# Objective Function
def objective_fun(model):
    my_sum = (
        sum(model.f[i] * model.Y[i] for i in range(len_I))
        + sum(model.g[i] * model.Z[i] for i in range(len_J))
        + sum(model.h[i] * model.W[i] for i in range(len_M))
        + sum(model.c[i,j] * model.X[i,j] for i in range(len_K) for j in range(len_I))
        + sum(model.a[i,j] * model.U[i,j] for i in range(len_I) for j in range(len_J))
        + sum(model.b[i,j] * model.P[i,j] for i in range(len_J) for j in range(len_M))
        + sum(model.e[i,j] * model.Q[i,j] for i in range(len_M) for j in range(len_L))
        + sum(model.v[i,j] * model.T[i,j] for i in range(len_I) for j in range(len_N))
        + sum(model.pi[i] * model.delta[i] for i in range(len_L))
    )
    return my_sum
model.obj = pyo.Objective(rule=objective_fun, sense=pyo.minimize, doc="Objective Function")



def constraint_1(model, l):
    return sum(model.Q[m,l] for m in range(len_M)) + model.delta[l] - model.d[l] >= 0
model.constraint_1 = pyo.Constraint(range(len_L), rule=constraint_1, doc="This constraint states that for each second market customer, deliveries plus unmet demand must be greater than or equal to the demand.  The reason for this constraint (and for the existence of unmet demand as a variable at all, instead of calculating it directly) is that it allows us to exceed demand at a particular customer while keeping unmet demand nonnegative.")

def constraint_2(model, k):
    return sum(model.X[k,i] for i in range(len_I)) - model.r[k] == 0
model.constraint_2 = pyo.Constraint(range(len_K), rule=constraint_2, doc="Flow of Returned Products from primary customers equals their necessary surrender of products")

def constraint_3(model, i):
    return sum(model.U[i,j] for j in range(len_J)) - (1-model.s) * sum(model.X[k,i] for k in range(len_K)) == 0
model.constraint_3 = pyo.Constraint(range(len_I), rule=constraint_3, doc="Mass balance around the collection/inspection centers for products NOT going to disposal.")

def constraint_4(model, i):
    return sum(model.T[i,n] for n in range(len_N)) - model.s * sum(model.X[k,i] for k in range(len_K)) == 0
model.constraint_4 = pyo.Constraint(range(len_I), rule=constraint_4, doc="Mass balance around the collection/inspection centers for products going to disposal.")

def constraint_5(model, m):
    return sum(model.P[j,m] for j in range(len_J)) - sum(model.Q[m,l] for l in range(len_L)) == 0
model.constraint_5 = pyo.Constraint(range(len_M), rule=constraint_5, doc="Mass balance around the Redistribution centers.")

def constraint_6(model, j):
    return sum(model.P[j,m] for m in range(len_M)) - sum(model.U[i,j] for i in range(len_I)) <= 0
model.constraint_6 = pyo.Constraint(range(len_J), rule=constraint_6, doc="Flow to Recovery centers is greater than or equal to the flow from Recovery centers.")

def constraint_7(model, i):
    return sum(model.X[k,i] for k in range(len_K)) - model.Y[i] * model.cc[i] <= 0
model.constraint_7 = pyo.Constraint(range(len_I), rule=constraint_7, doc="Capacity for Collection/inspection centers is greater than or equal to throughput.")

def constraint_8(model, j):
    return sum(model.U[i,j] for i in range(len_I)) - model.Z[j] * model.cr[j] <= 0
model.constraint_8 = pyo.Constraint(range(len_J), rule=constraint_8, doc="Capacity for Recovery Centers is greater than or equal to throughput.")

def constraint_9(model, m):
    return sum(model.P[j,m] for j in range(len_J)) - model.W[m] * model.ce[m] <= 0
model.constraint_9 = pyo.Constraint(range(len_M), rule=constraint_9, doc="Capacity for Redistribution Centers is greater than or equal to throughput.")

def constraint_10(model, n):
    return sum(model.T[i,n] for i in range(len_I)) - model.cd[n] <= 0
model.constraint_10 = pyo.Constraint(range(len_N), rule=constraint_10, doc="Capacity for Disposal Centers is greater than or equal to the flow thereto.")