import numpy as np
import pyomo.environ as pyo
# import cloudpickle as pickle


time_periods = 5
seed = None
data = {
'retailers_idx' :[0, 1], # Index of retailers
'distributors_idx' :[2, 3, 4], # Index of distributors
'producers_idx': [5, 6], # Index of producers
'raw_distributors_idx': [7, 8], # Index of raw distributors
'unit_price' :{0:5, 1:4}, # unit sales price at stages [0, 1]
'holding_cost' : {0:0.5, 1:0.4, 2:0.2, 3:0.3, 4:0.2, 5:0.3, 6:0.3}, # holding cost at stages [0, 1, 2, 3]
'supply_capacity' : {5: 40, 6: 40, 7: 50, 8: 40}, # production capacity at stages [1, 2, 3]
'reordering_routes': [(2,0), (2,1), (3,0), (3,1), (4,1), (5,2), (5,3), (5,4), (6,2), (6,4), (7,5), (7,6), (8,5), (8,6)], # reordering routes
'lead_time' : {(2,0): 1, (2,1):1, (3,0):1, (3,1):1, (4,1):1, (5,2):1, (5,3):1, (5,4):1, (6,2):1, (6,4):1, (7,5):1, (7,6):1, (8,5):1, (8,6):1}, # lead times at stages [0, 1, 2]
'demand_cost' : {0:1.5, 1:1}, # unit backlog cost at stages [0, 1, 2, 3]
'unit_cost' : {0:0.5, 1:0.3, 2:0.8, 3:0.7, 4:0.8, 5:0.1, 6:0.7, 7:0.5, 8:0.3, 9:0.4, 10:0.1, 11:0.7, 12:0.1, 13:0.2},
'discount' : 0.8, #  discount factor
'init_inv' :{0:20, 1:15, 2:30, 3:40, 4:30, 5:20, 6:30}, # Initial inventory
'demand_mean': {0:20, 1:10}, # Demand mean
'discount': 0.92, # Time value for money (Discount factor)
'uncertainty': 0.1,
'num_main_nodes':7, # Total number of main nodes (Retailer + Distributor + Producer)
'jin':{0: [2, 3], 1: [2, 3, 4], 2: [5, 6], 3: [5], 4: [5, 6], 5: [7, 8], 6: [7, 8]},
'jout':{2: [0, 1], 3: [0, 1], 4: [1], 5: [2, 3, 4], 6: [2, 4], 7: [5, 6], 8:[5, 6]},
'reorder_mapping' : {(2,0): 0, (2,1):1, (3,0):2, (3,1):3, (4,1):4, (5,2):5, (5,3):6, (5,4):7, (6,2):8, (6,4):9, (7,5):10, (7,6):11, (8,5):12, (8,6):13}, # lead times at stages [0, 1, 2]
'reverse_reorder_mapping': {
    0: (2, 0),
    1: (2, 1),
    2: (3, 0),
    3: (3, 1),
    4: (4, 1),
    5: (5, 2),
    6: (5, 3),
    7: (5, 4),
    8: (6, 2),
    9: (6, 4),
    10: (7, 5),
    11: (7, 6),
    12: (8, 5),
    13: (8, 6)
}}


if seed is None:
    rng = np.random.default_rng()
else:
    rng = np.random.default_rng(seed=42)

model = pyo.ConcreteModel()

# ------------------- sets -------------------------------------------------
model.T0 = pyo.RangeSet(0, time_periods)
model.T  = pyo.RangeSet(1, time_periods)
model.Tp = pyo.RangeSet(1, time_periods + 1)

model.R  = pyo.Set(initialize=data["retailers_idx"])
model.D  = pyo.Set(initialize=data["distributors_idx"])
model.P  = pyo.Set(initialize=data["producers_idx"])
model.S  = pyo.Set(initialize=data["raw_distributors_idx"])

model.MAIN   = model.R | model.D | model.P
model.MAIN2  = model.P | model.S
model.ROUTES = pyo.Set(initialize=data["reordering_routes"])

model.JIN  = pyo.Set(data["jin"].keys(), within=pyo.Any, initialize=data["jin"])
model.JOUT = pyo.Set(data["jout"].keys(), within=pyo.Any, initialize=data["jout"])

# ------------------- static parameters ------------------------------------
model.unit_price  = pyo.Param(model.MAIN, initialize=data["unit_price"])
model.h_cost      = pyo.Param(model.MAIN, initialize=data["holding_cost"])
model.cap         = pyo.Param(model.MAIN2, initialize=data["supply_capacity"])
model.lt          = pyo.Param(model.ROUTES, initialize=data["lead_time"])
model.b_cost      = pyo.Param(model.R, initialize=data["demand_cost"])

# remap unit_cost keys from (j,k)→idx if needed
updated_unit_cost_dict = {data['reverse_reorder_mapping'].get(old_key, old_key): value for old_key, value in data['unit_cost'].items()}
model.u_cost = pyo.Param(model.ROUTES, initialize=updated_unit_cost_dict)

model.init_inv = pyo.Param(model.MAIN, mutable = True, initialize=data["init_inv"])

# ------------------- *mutable* demand parameters --------------------------
demand_dict = {
    (t, r): float(rng.integers(low=0, high=10))    # random demand 8…19
    for t in model.T                             # (1 … time_periods)
    for r in model.R
}
model.demand = pyo.Param(model.T, model.R,
                        mutable=True,
                        initialize=demand_dict)

# ------------------- decision variables -----------------------------------
model.I = pyo.Var(model.Tp, model.MAIN,             domain=pyo.NonNegativeReals)
model.Tinv = pyo.Var(model.Tp, model.ROUTES,        domain=pyo.NonNegativeReals)
model.Rqty = pyo.Var(model.T,  model.ROUTES,        domain=pyo.NonNegativeReals)
model.Sales= pyo.Var(model.T,  model.R,             domain=pyo.NonNegativeReals)
model.Back = pyo.Var(model.T,  model.R,             domain=pyo.NonNegativeReals)

# ------------------- constraints ------------------------------------------
def inv_balance(m, t, j):
    if j in m.P | m.D:
        if t == 0:
            return m.I[t+1, j] == m.init_inv[j]
        return (
            m.I[t+1, j] ==
            m.I[t, j]
            + sum(m.Rqty[t - m.lt[k, j], k, j]
                    for k in m.JIN[j] if t - m.lt[k, j] >= 1)
            - sum(m.Rqty[t, j, k] for k in m.JOUT[j])
        )
    else:
        if t == 0:
            return m.I[t+1, j] == m.init_inv[j]
        return (
            m.I[t+1, j] ==
            m.I[t, j]
            + sum(m.Rqty[t - m.lt[k, j], k, j]
                    for k in m.JIN[j] if t - m.lt[k, j] >= 1)
            - m.Sales[t, j]
        )
model.inv_bal = pyo.Constraint(model.T0, model.MAIN, rule=inv_balance)

def pipe_balance(m, t, j, k):
    if t == 0:
        return m.Tinv[t+1, j, k] == 0
    if t - m.lt[j, k] >= 1:
        return (
            m.Tinv[t+1, j, k] ==
            m.Tinv[t, j, k] - m.Rqty[t - m.lt[j, k], j, k] + m.Rqty[t, j, k]
        )
    return (
        m.Tinv[t+1, j, k] ==
        m.Tinv[t, j, k] + m.Rqty[t, j, k]
    )
model.pipe_bal = pyo.Constraint(model.T0, model.ROUTES, rule=pipe_balance)

def reorder_cap(m, t, j):
    if j in m.MAIN2:
        return sum(m.Rqty[t, j, k] for k in m.JOUT[j]) <= m.cap[j]
    return pyo.Constraint.Skip
model.re_cap = pyo.Constraint(model.T, model.MAIN2, rule=reorder_cap)

def reorder_inv(m, t, j):
    if j in m.D | m.P:
        return sum(m.Rqty[t, j, k] for k in m.JOUT[j]) <= m.I[t, j]
    return pyo.Constraint.Skip
model.re_inv = pyo.Constraint(model.T, model.D | model.P, rule=reorder_inv)

def sales_demand(m, t, r):
    rhs = m.demand[t, r] + (m.Back[t-1, r] if t > 1 else 0)
    return m.Sales[t, r] <= rhs
model.sales1 = pyo.Constraint(model.T, model.R, rule=sales_demand)

def sales_stock(m, t, r):
    avail = m.I[t, r] + sum(m.Rqty[t - m.lt[k, r], k, r]
                            for k in m.JIN[r] if t - m.lt[k, r] >= 1)
    return m.Sales[t, r] <= avail
model.sales2 = pyo.Constraint(model.T, model.R, rule=sales_stock)

def backlog_def(m, t, r):
    if t == 1:
        return m.Back[t, r] == m.demand[t, r] - m.Sales[t, r]
    return m.Back[t, r] == m.demand[t, r] + m.Back[t-1, r] - m.Sales[t, r]
model.backlog = pyo.Constraint(model.T, model.R, rule=backlog_def)

# # end-horizon inventory target
model.final_inv = pyo.Constraint(
    model.MAIN, rule=lambda m, j: m.I[time_periods + 1, j] == m.init_inv[j]
)

# ------------------- objective -------------------------------------------
revenue = sum(model.unit_price[r] * model.Sales[t, r]   for t in model.T for r in model.R)
cost_re = sum(model.u_cost[route] * model.Rqty[t, route] for t in model.T for route in model.ROUTES)
cost_bk = sum(model.b_cost[r] * model.Back[t, r]        for t in model.T for r in model.R)
cost_h  = sum(model.h_cost[j] * model.I[t+1, j]         for t in model.T for j in model.MAIN)

model.obj = pyo.Objective(expr = revenue - cost_re - cost_bk - cost_h,
                        sense = pyo.maximize)
    


# with open("supply_chain_model.pkl", "wb") as f:
#     pickle.dump(model, f)




