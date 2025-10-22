import pyomo.environ as pyo
import numpy as np

## Data

data = {
    'S': ['DC1', 'DC2', 'DC3'],  # Distribution Centers
    'R': ['Store_A', 'Store_B', 'Store_C', 'Store_D'],  # Retail Stores
    'T': [0, 1, 2, 3, 4, 5, 6],  # Week 0 through Week 6 (7 weeks)
    
    'c': {  # Transportation cost per unit (based on distance)
        # DC1 is closest to Store_A and Store_B
        ('DC1', 'Store_A'): 5,  ('DC1', 'Store_B'): 7,  ('DC1', 'Store_C'): 15, ('DC1', 'Store_D'): 18,
        # DC2 is in the middle
        ('DC2', 'Store_A'): 12, ('DC2', 'Store_B'): 10, ('DC2', 'Store_C'): 8,  ('DC2', 'Store_D'): 11,
        # DC3 is closest to Store_C and Store_D
        ('DC3', 'Store_A'): 20, ('DC3', 'Store_B'): 16, ('DC3', 'Store_C'): 6,  ('DC3', 'Store_D'): 5,
    },
    
    # Initial inventories - DCs have good stock, stores are at different levels
    's0_i': {'DC1': 450, 'DC2': 380, 'DC3': 420},  # Total: 1250 units available
    's0_j': {'Store_A': 35, 'Store_B': 22, 'Store_C': 18, 'Store_D': 28},  # Stores starting at different levels
    
    # Need calculation: RUTL - Net_Inventory (only when NI < RUTL)
    # Stores have different capacity needs based on size and sales volume
    'need': {
        # Store_A: Large store, high capacity needs
        ('Store_A', 0): 45, ('Store_A', 1): 38, ('Store_A', 2): 42, ('Store_A', 3): 35, 
        ('Store_A', 4): 40, ('Store_A', 5): 48, ('Store_A', 6): 45,
        
        # Store_B: Medium store, moderate needs
        ('Store_B', 0): 38, ('Store_B', 1): 35, ('Store_B', 2): 32, ('Store_B', 3): 40, 
        ('Store_B', 4): 38, ('Store_B', 5): 42, ('Store_B', 6): 36,
        
        # Store_C: Small store, lower needs but growing
        ('Store_C', 0): 27, ('Store_C', 1): 30, ('Store_C', 2): 28, ('Store_C', 3): 32, 
        ('Store_C', 4): 35, ('Store_C', 5): 38, ('Store_C', 6): 40,
        
        # Store_D: Medium-large store, consistent needs
        ('Store_D', 0): 32, ('Store_D', 1): 35, ('Store_D', 2): 38, ('Store_D', 3): 36, 
        ('Store_D', 4): 40, ('Store_D', 5): 42, ('Store_D', 6): 38,
    },
    
    # Aim: Normalized need based on priority and fairness
    # Total need at t=0: 45+38+27+32 = 142
    # Available capacity ~1250, but we allocate conservatively
    # Store_A gets 90% of need (high priority, flagship store)
    # Store_B gets 85% of need (standard priority)
    # Store_C gets 75% of need (newer, smaller store)
    # Store_D gets 80% of need (standard priority)
    'aim': {
        # Store_A: 90% of need (flagship store priority)
        ('Store_A', 0): 40, ('Store_A', 1): 34, ('Store_A', 2): 38, ('Store_A', 3): 32, 
        ('Store_A', 4): 36, ('Store_A', 5): 43, ('Store_A', 6): 40,
        
        # Store_B: 85% of need
        ('Store_B', 0): 32, ('Store_B', 1): 30, ('Store_B', 2): 27, ('Store_B', 3): 34, 
        ('Store_B', 4): 32, ('Store_B', 5): 36, ('Store_B', 6): 31,
        
        # Store_C: 75% of need (lower priority, building up gradually)
        ('Store_C', 0): 20, ('Store_C', 1): 23, ('Store_C', 2): 21, ('Store_C', 3): 24, 
        ('Store_C', 4): 26, ('Store_C', 5): 29, ('Store_C', 6): 30,
        
        # Store_D: 80% of need
        ('Store_D', 0): 26, ('Store_D', 1): 28, ('Store_D', 2): 30, ('Store_D', 3): 29, 
        ('Store_D', 4): 32, ('Store_D', 5): 34, ('Store_D', 6): 30,
    },
    
    'M': 10000,  # Big M value
    
    # Reorder points: Based on lead time demand + safety stock
    # RP = (avg weekly demand * max lead time) + safety stock
    'RP': {
        ('Store_A', 0): 25, ('Store_A', 1): 25, ('Store_A', 2): 25, ('Store_A', 3): 25, 
        ('Store_A', 4): 25, ('Store_A', 5): 25, ('Store_A', 6): 25,
        
        ('Store_B', 0): 20, ('Store_B', 1): 20, ('Store_B', 2): 20, ('Store_B', 3): 20, 
        ('Store_B', 4): 20, ('Store_B', 5): 20, ('Store_B', 6): 20,
        
        ('Store_C', 0): 15, ('Store_C', 1): 15, ('Store_C', 2): 15, ('Store_C', 3): 15, 
        ('Store_C', 4): 15, ('Store_C', 5): 15, ('Store_C', 6): 15,
        
        ('Store_D', 0): 22, ('Store_D', 1): 22, ('Store_D', 2): 22, ('Store_D', 3): 22, 
        ('Store_D', 4): 22, ('Store_D', 5): 22, ('Store_D', 6): 22,
    },
    
    # Reorder up to levels: Maximum inventory capacity
    'RUTL': {
        ('Store_A', 0): 80, ('Store_A', 1): 80, ('Store_A', 2): 80, ('Store_A', 3): 80, 
        ('Store_A', 4): 80, ('Store_A', 5): 80, ('Store_A', 6): 80,
        
        ('Store_B', 0): 60, ('Store_B', 1): 60, ('Store_B', 2): 60, ('Store_B', 3): 60, 
        ('Store_B', 4): 60, ('Store_B', 5): 60, ('Store_B', 6): 60,
        
        ('Store_C', 0): 45, ('Store_C', 1): 45, ('Store_C', 2): 45, ('Store_C', 3): 45, 
        ('Store_C', 4): 45, ('Store_C', 5): 45, ('Store_C', 6): 45,
        
        ('Store_D', 0): 60, ('Store_D', 1): 60, ('Store_D', 2): 60, ('Store_D', 3): 60, 
        ('Store_D', 4): 60, ('Store_D', 5): 60, ('Store_D', 6): 60,
    },
    
    # Demand: Actual customer demand (slightly seasonal pattern)
    # Week 5-6 show increased demand (weekend/promotion effect)
    'd': {
        # Store_A: High volume store
        ('Store_A', 0): 18, ('Store_A', 1): 20, ('Store_A', 2): 19, ('Store_A', 3): 21, 
        ('Store_A', 4): 22, ('Store_A', 5): 28, ('Store_A', 6): 26,
        
        # Store_B: Medium volume
        ('Store_B', 0): 15, ('Store_B', 1): 16, ('Store_B', 2): 14, ('Store_B', 3): 17, 
        ('Store_B', 4): 18, ('Store_B', 5): 22, ('Store_B', 6): 20,
        
        # Store_C: Lower volume, growing
        ('Store_C', 0): 10, ('Store_C', 1): 11, ('Store_C', 2): 12, ('Store_C', 3): 13, 
        ('Store_C', 4): 14, ('Store_C', 5): 16, ('Store_C', 6): 15,
        
        # Store_D: Medium-high volume
        ('Store_D', 0): 16, ('Store_D', 1): 17, ('Store_D', 2): 18, ('Store_D', 3): 17, 
        ('Store_D', 4): 19, ('Store_D', 5): 24, ('Store_D', 6): 21,
    },
    
    # Lead times: Varies by distance
    # DC1 to nearby stores: 1 week, far stores: 2 weeks
    # DC2: mostly 1 week (centrally located)
    # DC3 to nearby stores: 1 week, far stores: 2 weeks
    'LT': {
        # DC1 lead times
        ('DC1', 'Store_A', 0): 1, ('DC1', 'Store_A', 1): 1, ('DC1', 'Store_A', 2): 1, 
        ('DC1', 'Store_A', 3): 1, ('DC1', 'Store_A', 4): 1, ('DC1', 'Store_A', 5): 1, ('DC1', 'Store_A', 6): 1,
        
        ('DC1', 'Store_B', 0): 1, ('DC1', 'Store_B', 1): 1, ('DC1', 'Store_B', 2): 1, 
        ('DC1', 'Store_B', 3): 1, ('DC1', 'Store_B', 4): 1, ('DC1', 'Store_B', 5): 1, ('DC1', 'Store_B', 6): 1,
        
        ('DC1', 'Store_C', 0): 2, ('DC1', 'Store_C', 1): 2, ('DC1', 'Store_C', 2): 2, 
        ('DC1', 'Store_C', 3): 2, ('DC1', 'Store_C', 4): 2, ('DC1', 'Store_C', 5): 2, ('DC1', 'Store_C', 6): 2,
        
        ('DC1', 'Store_D', 0): 2, ('DC1', 'Store_D', 1): 2, ('DC1', 'Store_D', 2): 2, 
        ('DC1', 'Store_D', 3): 2, ('DC1', 'Store_D', 4): 2, ('DC1', 'Store_D', 5): 2, ('DC1', 'Store_D', 6): 2,
        
        # DC2 lead times (centrally located - all 1 week)
        ('DC2', 'Store_A', 0): 1, ('DC2', 'Store_A', 1): 1, ('DC2', 'Store_A', 2): 1, 
        ('DC2', 'Store_A', 3): 1, ('DC2', 'Store_A', 4): 1, ('DC2', 'Store_A', 5): 1, ('DC2', 'Store_A', 6): 1,
        
        ('DC2', 'Store_B', 0): 1, ('DC2', 'Store_B', 1): 1, ('DC2', 'Store_B', 2): 1, 
        ('DC2', 'Store_B', 3): 1, ('DC2', 'Store_B', 4): 1, ('DC2', 'Store_B', 5): 1, ('DC2', 'Store_B', 6): 1,
        
        ('DC2', 'Store_C', 0): 1, ('DC2', 'Store_C', 1): 1, ('DC2', 'Store_C', 2): 1, 
        ('DC2', 'Store_C', 3): 1, ('DC2', 'Store_C', 4): 1, ('DC2', 'Store_C', 5): 1, ('DC2', 'Store_C', 6): 1,
        
        ('DC2', 'Store_D', 0): 1, ('DC2', 'Store_D', 1): 1, ('DC2', 'Store_D', 2): 1, 
        ('DC2', 'Store_D', 3): 1, ('DC2', 'Store_D', 4): 1, ('DC2', 'Store_D', 5): 1, ('DC2', 'Store_D', 6): 1,
        
        # DC3 lead times
        ('DC3', 'Store_A', 0): 2, ('DC3', 'Store_A', 1): 2, ('DC3', 'Store_A', 2): 2, 
        ('DC3', 'Store_A', 3): 2, ('DC3', 'Store_A', 4): 2, ('DC3', 'Store_A', 5): 2, ('DC3', 'Store_A', 6): 2,
        
        ('DC3', 'Store_B', 0): 2, ('DC3', 'Store_B', 1): 2, ('DC3', 'Store_B', 2): 2, 
        ('DC3', 'Store_B', 3): 2, ('DC3', 'Store_B', 4): 2, ('DC3', 'Store_B', 5): 2, ('DC3', 'Store_B', 6): 2,
        
        ('DC3', 'Store_C', 0): 1, ('DC3', 'Store_C', 1): 1, ('DC3', 'Store_C', 2): 1, 
        ('DC3', 'Store_C', 3): 1, ('DC3', 'Store_C', 4): 1, ('DC3', 'Store_C', 5): 1, ('DC3', 'Store_C', 6): 1,
        
        ('DC3', 'Store_D', 0): 1, ('DC3', 'Store_D', 1): 1, ('DC3', 'Store_D', 2): 1, 
        ('DC3', 'Store_D', 3): 1, ('DC3', 'Store_D', 4): 1, ('DC3', 'Store_D', 5): 1, ('DC3', 'Store_D', 6): 1,
    },
    
    # Initial net inventory (on-hand + in-transit)
    'NI_init': {
        'Store_A': 35,  # Same as s0_j if no in-transit
        'Store_B': 22,
        'Store_C': 18,
        'Store_D': 28,
    },
    
    'incoming_q': {},  # No external incoming quantities in this scenario,

    'inventory_limit': {
    'DC1': 150,     # Capacity limit
    'DC2': 350,
    'DC3': 380,
    'Store_A': 70,
    'Store_B': 50,
    'Store_C': 40,
    'Store_D': 55,
    }
}


# Building model

"""
    Create the Fair Allocation Model as specified in the PDF.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing all required data:
        - S: set of senders
        - R: set of recipients  
        - T: set of time periods {t0, ..., tk}
        - c: cost matrix c[i,j]
        - s0_i: initial inventory for senders
        - s0_j: initial inventory for recipients
        - need: need[j,t]
        - aim: aim[j,t]
        - M: big M value
        - NI_init: initial net inventory NI[j,t0]
        - RP: reorder point RP[j,t]
        - RUTL: reorder up to level RUTL[j,t]
        - d: demand/sales d[j,t]
        - LT: lead time LT[i,j,t]
        - incoming_q: incoming quantities
"""
    
model = pyo.ConcreteModel()

# Sets
model.S = pyo.Set(initialize=data['S'], doc = "Set of all Senders/Suppliers")  # Set of senders
model.R = pyo.Set(initialize=data['R'], doc = "Set of all Recipients")  # Set of recipients
model.T = pyo.Set(initialize=data['T'], doc = "Set of all time periods")  # Time periods
model.T_plus = pyo.Set(initialize=data['T'][1:], doc = "Set of all time periods except 0")  # T excluding t0

# Parameters
model.c = pyo.Param(model.S, model.R, initialize=data['c'], doc = "Cost of item to go from i to j", mutable = True)  # Cost matrix
model.s0_i = pyo.Param(model.S, initialize=data['s0_i'], doc = "Initial inventory of senders", mutable = True)  # Initial sender inventory
model.s0_j = pyo.Param(model.R, initialize=data['s0_j'], doc = "Initial inventory of receipients", mutable = True)  # Initial recipient inventory
model.need = pyo.Param(model.R, model.T, initialize=data['need'], doc = "Need at location j at time t", mutable = True)  # Need at location j, time t
model.aim = pyo.Param(model.R, model.T, initialize=data['aim'], doc = " Aimed quantity at receiver j at time t", mutable = True)  # Aimed quantity
model.M = pyo.Param(initialize=data['M'], mutable = True)  # Big M
model.RP = pyo.Param(model.R, model.T, initialize=data['RP'], doc = "Reoder Point", mutable = True)  # Reorder point
model.RUTL = pyo.Param(model.R, model.T, initialize=data['RUTL'], doc = "Reorder up to level", mutable = True)  # Reorder up to level
model.d = pyo.Param(model.R, model.T, initialize=data['d'], doc = "Demand at receiver j at time t", mutable = True)  # Demand/sales
model.LT = pyo.Param(model.S, model.R, model.T, initialize=data['LT'],doc = "Lead time from supplier i to receiver j at time t", mutable = True)  # Lead time
model.need_min_param = pyo.Param(initialize = 10, doc = "limit for need quantites", mutable = True)

# Decision Variables
model.U_plus = pyo.Var(within=pyo.NonNegativeReals)  # U+
model.U_minus = pyo.Var(within=pyo.NonNegativeReals)  # U-
model.q = pyo.Var(model.S, model.R, model.T, within=pyo.NonNegativeReals, doc = "Quantity ordered from location i to j at time t")  # Quantity from i to j at t
model.x = pyo.Var(model.S, model.R, model.T, within=pyo.Binary, doc = "Binary: 1 if shipping from i to j at t")  # Binary: 1 if shipping from i to j at t
model.s = pyo.Var(pyo.Set(initialize=model.S | model.R), model.T, within=pyo.NonNegativeReals, doc = "On-hand inventory at location i at time t")  # Inventory at location i/j, time t
model.NI = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc = "Net inventory at receiver j at time t")  # Net inventory at j, time t
model.sls = pyo.Var(model.R, model.T, within=pyo.NonNegativeReals, doc = "Expected Sales at j at time t")  # Expected sales at j, time t

# Objective Function
def objective_rule(model):
    return (model.U_plus + model.U_minus + 
            sum(model.c[i,j] * model.x[i,j,t] for i in model.S for j in model.R for t in model.T) -
            2 * sum(model.q[i,j,t] for i in model.S for j in model.R for t in model.T))
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Constraints

def need_bound_constraint(model, r, t):
    return model.need[r, t] <= model.need_min_param

model.need_min = pyo.Constraint(model.R, model.T, rule = need_bound_constraint, doc = "Need bounded my minimum need")

# Sender capacity constraint
def sender_capacity_rule(model, i, t):
    return sum(model.q[i,j,t] for j in model.R) <= model.s[i,t]
model.sender_capacity = pyo.Constraint(model.S, model.T, rule=sender_capacity_rule, doc = "Sender capacity constraint: sum over j,t of q[i,j,t] <= s[i,t] for all i in S, t")

# Recipient need constraint
def recipient_need_rule(model, j, t):
    return sum(model.q[i,j,t] for i in model.S) <= model.need[j,t]
model.recipient_need = pyo.Constraint(model.R, model.T, rule=recipient_need_rule, doc = "Recipient need constraint: sum over i of q[i,j,t] <= need[j,t] for all j in R, t")

# Linking constraint
def linking_rule(model, i, j, t):
    return model.q[i,j,t] <= model.need[j,t] * model.x[i,j,t]
model.linking = pyo.Constraint(model.S, model.R, model.T, rule=linking_rule, doc = "Linking constraint: q[i,j,t] <= need[j,t] * x[i,j,t] for all i,j,t")

# U_minus constraint
def u_minus_rule(model, j, t):
    return model.U_minus >= model.aim[j,t] - sum(model.q[i,j,t] for i in model.S)
model.u_minus_constraint = pyo.Constraint(model.R, model.T, rule=u_minus_rule, doc = " U_minus constraint: U_minus >= aim[j,t] - sum over i of q[i,j,t] for all j in R, t")

# U_plus constraint
def u_plus_rule(model, j, t):
    return model.U_plus >= sum(model.q[i,j,t] for i in model.S) - model.aim[j,t]
model.u_plus_constraint = pyo.Constraint(model.R, model.T, rule=u_plus_rule, doc = "U_plus constraint: U_plus >= sum over i of q[i,j,t] - aim[j,t] for all j in R, t")

# Initial sender inventory
def init_sender_inventory_rule(model, i):
    return model.s[i,0] == model.s0_i[i]
model.init_sender_inventory = pyo.Constraint(model.S, rule=init_sender_inventory_rule, doc = "Initial sender inventory: s[i,t0] = s0_i for all i in S")

# Initial recipient inventory
def init_recipient_inventory_rule(model, j):
    return model.s[j,0] == model.s0_j[j]
model.init_recipient_inventory = pyo.Constraint(model.R, rule=init_recipient_inventory_rule, doc = "Initial recipient inventory: s[j,t0] = s0_j for all j in R")

# Sender inventory update
def sender_inventory_update_rule(model, i, t):
    if t < data["T"][-1]:
        return model.s[i,t + 1] == model.s[i,t] - sum(model.q[i,j,t] for j in model.R)
    else:
        return pyo.Constraint.Skip
model.sender_inventory_update = pyo.Constraint(model.S, model.T, rule=sender_inventory_update_rule, doc = "Sender inventory update: s[i,t+1] = s[i,t] - sum over j of q[i,j,t] for all i in S, t")

# Big M constraint
def big_m_rule(model, j, t):
    return model.M * sum(model.x[i,j,t] for i in model.S) >= model.NI[j,t] - model.RP[j,t]
model.big_m_constraint = pyo.Constraint(model.R, model.T, rule=big_m_rule, doc = "Big M constraint: (M * sum over i of x[i,j,t]) >= NI[j,t] - RP[j,t] for all j in R, t")

# RUTL constraint
def rutl_rule(model, j, t):
    return model.s[j,t] + sum(model.q[i,j,t] for i in model.S) <= model.RUTL[j,t]
model.rutl_constraint = pyo.Constraint(model.R, model.T, rule=rutl_rule, doc = "RUTL constraint: s[j,t] + sum over i of q[i,j,t] <= RUTL[j,t] for all j in R, t")

# Sales constraint
def sales_rule(model, j, t):
    return model.sls[j,t] <= model.s[j,t]
model.sales_upper1 = pyo.Constraint(model.R, model.T, rule=sales_rule, doc = "Sales constraint: sls[j,t] = s[j,t] for all j in R, t")

def sales_rule2(model, j, t):
    return model.sls[j,t] <= model.d[j,t]
model.sales_upper2 = pyo.Constraint(model.R, model.T, rule=sales_rule2, doc = "Sales constraint: sls[j,t] <= d[j,t] for all j in R, t")

def recipient_inventory_update_rule(model, j, t):
    if t == model.T[-1]:
        return pyo.Constraint.Skip
    
    # Calculate incoming quantities considering lead times
    incoming = sum(
        model.q[i, j, t - pyo.value(model.LT[i, j, t])]
        for i in model.S
        if t - pyo.value(model.LT[i, j, t]) in model.T
    )
    
    return model.s[j, t + 1] == model.s[j, t] - model.sls[j, t] + incoming

model.recipient_inventory_update = pyo.Constraint(model.R, model.T, rule=recipient_inventory_update_rule, doc = "s[j,t+1] = s[j,t] - sls[j,t] + sum over i of q[i,j,t-LT[i,j,t]]")

# Net inventory update: Complex constraint
def net_inventory_update_rule(model, j, t):
    if t == model.T[-1]:
        return pyo.Constraint.Skip
    
    # Start with current inventory
    ni_value = model.s[j, t]
    
    # Add in-transit quantities: Sum over i, sum over u in (t-LT[i,j,t], t)
    for i in model.S:
        lt = pyo.value(model.LT[i, j, t])
        # Sum over u in (t-LT[i,j,t], t] - note: open interval on left, closed on right
        for u in range(t - lt + 1, t + 1):
            if u in model.T:
                ni_value += model.q[i, j, u]
    
    return model.NI[j, t + 1] == ni_value

model.net_inventory_update = pyo.Constraint(model.R, model.T, rule=net_inventory_update_rule, doc = "NI[j,t+1] = s[j,t] + sum over i sum over u in (t-LT[i,j,t], t) of q[i,j,u]")

def inventory_capacity_rule(model, i, t):
    return model.s[i, t] <= data['inventory_limit'][i]

model.inventory_capacity = pyo.Constraint(
    pyo.Set(initialize=model.S | model.R), 
    model.T, 
    rule=inventory_capacity_rule, doc = "Inventory Capacity Constraint"
)
