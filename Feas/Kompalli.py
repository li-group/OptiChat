import pyomo.environ as pyo

#create model
model = pyo.ConcreteModel(name="Multi-Echelon Supply Chain Optimization Model")

# Data
num_periods = 30 # Number of periods
num_stages = 4 # Number of stages

unit_price = [2, 1.5, 1.0, 0.75] # unit sales price at stages [0, 1, 2, 3]
holding_cost = [0.15, 0.10, 0.05, 0] # holding cost at stages [0, 1, 2, 3]
supply_capacity = [100, 90, 80] # production capacity at stages [1, 2, 3]
lead_time = [3, 5, 10] # lead times at stages [0, 1, 2]
demand_cost = [0.10, 0.075, 0.05, 0.025] # unit backlog cost at stages [0, 1, 2, 3]
unit_cost = [1.5, 1.0, 0.75, 0.5] # unit replenishment cost at stages [0, 1, 2, 3]

discount = 0.97 #  discount factor
 
D = [16, 10, 11, 16, 18, 18, 20, 25, 29, 14, 23, 16, 19, 27, 17, 21, 7, 21, 20, 20, 19, 25, 27, 21, 18, 29, 24, 17, 22, 16] # poisson distribution of demand for each period

init_inv = [100, 100, 200] # initial inventory levels at stages [0, 1, 2]

#define sets
model.N = pyo.RangeSet(0,num_periods-1, doc= "Set of time periods excluding the last period") 
model.N1 = pyo.RangeSet(0,num_periods, doc= "Set of time periods")
model.M = pyo.RangeSet(0,num_stages-1, doc= "Set of all stages")
model.M0 = pyo.RangeSet(0,num_stages-2, doc= "Set of all stages exculding the last stage which has no inventory")
    
#define parameters
model.up = pyo.Param(model.M, initialize = {i:unit_price[i] for i in model.M}, mutable=True, doc= "sales price for each stage in dollars") 
model.uc = pyo.Param(model.M, initialize = {i:unit_cost[i] for i in model.M}, mutable=True, doc= "replenishment cost for each stage in dollars")
model.dc = pyo.Param(model.M, initialize = {i:demand_cost[i] for i in model.M}, mutable=True, doc= "cost for unfulfilled demand at each stage in dollars")
model.hc = pyo.Param(model.M, initialize = {i:holding_cost[i] for i in model.M}, mutable=True, doc= "inventory holding cost at each stage in dollars")
model.sc = pyo.Param(model.M0, initialize = {i:supply_capacity[i] for i in model.M0}, mutable=True, doc= "production capacity at each stage")
model.lt = pyo.Param(model.M0, initialize = {i:lead_time[i] for i in model.M0}, doc= "lead times in between stages")
model.dis = pyo.Param(initialize = discount, mutable=True, doc= "time-valued discount ")
model.np = pyo.Param(initialize=num_periods, mutable=True, doc= "number of periods")
model.dmd = pyo.Param(model.N, initialize = {i:D[i] for i in model.N}, mutable=True, doc= "retailer demand at each period")
    
#define variables
model.i = pyo.Var(model.N1,model.M0,domain=pyo.NonNegativeReals, doc= "on hand inventory at each time period and stage excluding the last stage") 
model.t = pyo.Var(model.N1,model.M0,domain=pyo.NonNegativeReals, doc= "pipeline inventory in between each time period and stage excluding the last stage") 
model.r = pyo.Var(model.N,model.M0,domain=pyo.NonNegativeReals, doc= "reorder quantities for each time peiod and stage excluding the last stage")
model.s = pyo.Var(model.N,model.M,domain=pyo.NonNegativeReals, doc= "sales for each time period and stage") 
model.b = pyo.Var(model.N,model.M,domain=pyo.NonNegativeReals, doc= "backlogs for each time period and stage") 
model.p = pyo.Var(model.N,domain=pyo.Reals, doc= "profit at each time period and stage") 

#initialize
for m in model.M0:
    model.i[0,m].fix(init_inv[m])
    model.t[0,m].fix(0)
    
#define constraints
model.inv_bal = pyo.ConstraintList(doc="Balances on hand inventory at stages 0, 1, 2 over all time periods")
model.sales1 = pyo.ConstraintList(doc="Limits initial stage sales to available inventory and arrivals")
model.sales3 = pyo.ConstraintList(doc="Sets sales at the initial time period for all stages and time periods based on demand and previous backlogs")
model.sales5 = pyo.ConstraintList(doc="Ensures sales match reorders from the previous stage for later stages over all time periods")
model.reorder6 = pyo.ConstraintList(doc="Caps reorder quantities at stages 0, 1, 2 to not exceed supply capacity")
model.reorder8 = pyo.ConstraintList(doc="Limits reorders based on available inventory at the next stage for stages 0, 1, 2")
model.pip_bal = pyo.ConstraintList(doc="Tracks inventory in transit between stages for all time periods, adjusting for deliveries and reorders")
model.unfulfilled = pyo.ConstraintList(doc="Tracks demand not met immediately at each stage for all time periods, leading to backlogs")
model.profit = pyo.ConstraintList(doc= "Profit obtained at each period")
    
for n in model.N:
    model.profit.add(
    model.p[n] == model.dis**n * (
        sum(model.up[m] * model.s[n, m] for m in model.M)
        - (sum(model.uc[m] * model.r[n, m] for m in model.M0) 
           + model.uc[model.M.at(len(model.M))] * model.s[n, model.M.at(len(model.M))])
        - sum(model.dc[m] * model.b[n, m] for m in model.M)
        - sum(model.hc[m] * model.i[n + 1, m] for m in model.M0))) 
            
    for m in model.M0:
        if n - model.lt[m] >= 0:
            model.inv_bal.add(model.i[n+1,m] == model.i[n,m] + model.r[n - model.lt[m],m] - model.s[n,m])
        else:
            model.inv_bal.add(model.i[n+1,m] == model.i[n,m] - model.s[n,m])
        if n - model.lt[m] >= 0:
            model.pip_bal.add(model.t[n+1,m] == model.t[n,m] - model.r[n - model.lt[m],m] + model.r[n,m])
        else:
            model.pip_bal.add(model.t[n+1,m] == model.t[n,m] + model.r[n,m])
        model.reorder6.add(model.r[n,m] <= model.sc[m])
        if (m < model.M.at(len(model.M)) & (num_stages > 2)): 
            model.reorder8.add(model.r[n,m] <= model.i[n,m+1])
                
    for m in model.M:            
        if m == 0:
            if n - model.lt[m] >= 0:
                model.sales1.add(model.s[n,m] <= model.i[n,m] + model.r[n - model.lt[m],m])
            else:
                model.sales1.add(model.s[n,m] <= model.i[n,m])
                
            if n-1>=0:
                model.sales3.add(model.s[n,m] <= model.dmd[n] + model.b[n-1,m])
            else:
                model.sales3.add(model.s[n,m] <= model.dmd[n])
        else:
            model.sales5.add(model.s[n,m] == model.r[n,m-1])
                    
        if m == 0:
                if n-1>=0:
                    model.unfulfilled.add(model.b[n,m] == model.dmd[n] + model.b[n-1,m] - model.s[n,m])
                else:
                    model.unfulfilled.add(model.b[n,m] == model.dmd[n] - model.s[n,m])
        else:
                if n-1>=0:
                    model.unfulfilled.add(model.b[n,m] == model.r[n,m-1] + model.b[n-1,m] - model.s[n,m])
                else:
                    model.unfulfilled.add(model.b[n,m] == model.r[n,m-1] - model.s[n,m])

#objective function
model.obj = pyo.Objective(
    expr = sum(model.p[n] for n in model.N),
    sense = pyo.maximize, doc = "maximize the profit over all periods")