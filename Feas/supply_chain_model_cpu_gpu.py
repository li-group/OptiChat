from pyomo.environ import *
import pandas as pd
from pathlib import Path

try:
    base_path = Path(__file__).resolve().parent.parent / 'data' / 'cpu_gpu_supply_chain'
except NameError:
    base_path = Path.cwd() / 'data' / 'cpu_gpu_supply_chain'

demands_df = pd.read_csv(base_path / 'demands.csv')
processing_df = pd.read_csv(base_path / 'processing.csv')
transportation_df = pd.read_csv(base_path / 'transportation.csv')

model = ConcreteModel()

# Sets
model.suppliers = Set(initialize=processing_df[processing_df['node_type'] == 'supplier']['node_id'].unique(), doc='Supplier facilities')
model.processors = Set(initialize=processing_df[processing_df['node_type'] == 'processor']['node_id'].unique(), doc='Processor facilities')
model.stores = Set(initialize=demands_df['node_id'].unique(), doc='Retail stores')
model.products = Set(initialize=processing_df['product_type'].unique(), doc='Product types')

# Parameters
# Supplier capacity and cost
def supplier_capacity_init(model, node_id, product_type):
    row = processing_df[(processing_df['node_id'] == node_id) & 
                       (processing_df['product_type'] == product_type)]
    if not row.empty:
        return row['capacity'].values[0]
    return 0
model.supplier_capacity = Param(model.suppliers, model.products, 
                               initialize=supplier_capacity_init, doc='Supplier capacity')

def supplier_cost_init(model, node_id, product_type):
    row = processing_df[(processing_df['node_id'] == node_id) & 
                       (processing_df['product_type'] == product_type)]
    if not row.empty:
        return row['cost_per_unit'].values[0]
    return 0
model.supplier_cost = Param(model.suppliers, model.products, 
                           initialize=supplier_cost_init, doc='Supplier production cost per unit')

# Processor capacity and cost
def processor_capacity_init(model, node_id, product_type):
    row = processing_df[(processing_df['node_id'] == node_id) & 
                       (processing_df['product_type'] == product_type)]
    if not row.empty:
        return row['capacity'].values[0]
    return 0
model.processor_capacity = Param(model.processors, model.products, 
                               initialize=processor_capacity_init, doc='Processor capacity')

def processor_cost_init(model, node_id, product_type):
    row = processing_df[(processing_df['node_id'] == node_id) & 
                       (processing_df['product_type'] == product_type)]
    if not row.empty:
        return row['cost_per_unit'].values[0]
    return 0
model.processor_cost = Param(model.processors, model.products, 
                           initialize=processor_cost_init, doc='Processor cost per unit')

# Transportation cost and capacity
def transport_cost_init(model, origin, destination, product_type):
    row = transportation_df[(transportation_df['origin'] == origin) & 
                           (transportation_df['destination'] == destination) & 
                           (transportation_df['product_type'] == product_type)]
    if not row.empty:
        return row['cost_per_unit'].values[0]
    return float('inf')  # No route exists
model.transport_cost = Param(model.suppliers | model.processors, 
                            model.processors | model.stores, 
                            model.products, 
                            initialize=transport_cost_init, 
                            doc='Transportation cost per unit')

def transport_capacity_init(model, origin, destination, product_type):
    row = transportation_df[(transportation_df['origin'] == origin) & 
                           (transportation_df['destination'] == destination) & 
                           (transportation_df['product_type'] == product_type)]
    if not row.empty:
        return row['capacity'].values[0]
    return 0
model.transport_capacity = Param(model.suppliers | model.processors, 
                                model.processors | model.stores, 
                                model.products, 
                                initialize=transport_capacity_init, 
                                doc='Transportation capacity')

# Demand and revenue
def demand_init(model, node_id, product_type):
    row = demands_df[(demands_df['node_id'] == node_id) & 
                    (demands_df['product_type'] == product_type)]
    if not row.empty:
        return row['demand'].values[0]
    return 0
model.demand = Param(model.stores, model.products, 
                    initialize=demand_init, doc='Store demand')

def revenue_init(model, node_id, product_type):
    row = demands_df[(demands_df['node_id'] == node_id) & 
                    (demands_df['product_type'] == product_type)]
    if not row.empty:
        return row['revenue_per_unit'].values[0]
    return 0
model.revenue = Param(model.stores, model.products, 
                     initialize=revenue_init, doc='Revenue per unit at store')

# Variables
model.supply = Var(model.suppliers, model.products, within=NonNegativeReals, doc='Production at supplier')
model.process = Var(model.processors, model.products, within=NonNegativeReals, doc='Processing at processor')

model.ship_supplier_processor = Var(
    model.suppliers, model.processors, model.products, 
    within=NonNegativeReals, 
    doc='Shipment from supplier to processor'
)

model.ship_processor_store = Var(
    model.processors, model.stores, model.products, 
    within=NonNegativeReals, 
    doc='Shipment from processor to store'
)

model.total_revenue = Var(doc='Total revenue')
model.total_production_cost = Var(doc='Total production cost')
model.total_processing_cost = Var(doc='Total processing cost')
model.total_transport_cost = Var(doc='Total transport cost')
model.profit = Var(doc='Total profit')

# Constraints
def supplier_capacity_rule(model, s, p):
    return model.supply[s, p] <= model.supplier_capacity[s, p]
model.supplier_capacity_constr = Constraint(
    model.suppliers, model.products, rule=supplier_capacity_rule,
    doc='Supplier cannot exceed capacity'
)

def processor_capacity_rule(model, a, p):
    return model.process[a, p] <= model.processor_capacity[a, p]
model.processor_capacity_constr = Constraint(
    model.processors, model.products, rule=processor_capacity_rule,
    doc='Processor cannot exceed capacity'
)

def supplier_balance_rule(model, s, p):
    return model.supply[s, p] == sum(model.ship_supplier_processor[s, a, p] for a in model.processors)
model.supplier_balance_constr = Constraint(
    model.suppliers, model.products, rule=supplier_balance_rule,
    doc='Supplier output equals shipments to processors'
)

def processor_balance_rule(model, a, p):
    return model.process[a, p] == sum(model.ship_supplier_processor[s, a, p] for s in model.suppliers)
model.processor_balance_constr = Constraint(
    model.processors, model.products, rule=processor_balance_rule,
    doc='Processor input equals shipments from suppliers'
)

def processor_output_rule(model, a, p):
    return sum(model.ship_processor_store[a, st, p] for st in model.stores) <= model.process[a, p]
model.processor_output_constr = Constraint(
    model.processors, model.products, rule=processor_output_rule,
    doc='Processor output cannot exceed processed amount'
)

def demand_satisfaction_rule(model, st, p):
    return sum(model.ship_processor_store[a, st, p] for a in model.processors) <= model.demand[st, p]
model.demand_satisfaction_constr = Constraint(
    model.stores, model.products, rule=demand_satisfaction_rule,
    doc='Cannot exceed store demand'
)

def transport_capacity_supplier_processor_rule(model, s, a, p):
    return model.ship_supplier_processor[s, a, p] <= model.transport_capacity[s, a, p]
model.transport_capacity_sp_constr = Constraint(
    model.suppliers, model.processors, model.products, 
    rule=transport_capacity_supplier_processor_rule,
    doc='Supplier-processor transport capacity'
)

def transport_capacity_processor_store_rule(model, a, st, p):
    return model.ship_processor_store[a, st, p] <= model.transport_capacity[a, st, p]
model.transport_capacity_ps_constr = Constraint(
    model.processors, model.stores, model.products, 
    rule=transport_capacity_processor_store_rule,
    doc='Processor-store transport capacity'
)

# Cost and revenue calculations
def total_revenue_rule(model):
    return model.total_revenue == sum(
        model.revenue[st, p] * model.ship_processor_store[a, st, p] 
        for a in model.processors 
        for st in model.stores 
        for p in model.products
    )
model.total_revenue_constr = Constraint(rule=total_revenue_rule)

def total_production_cost_rule(model):
    return model.total_production_cost == sum(
        model.supplier_cost[s, p] * model.supply[s, p] 
        for s in model.suppliers 
        for p in model.products
    )
model.total_production_cost_constr = Constraint(rule=total_production_cost_rule)

def total_processing_cost_rule(model):
    return model.total_processing_cost == sum(
        model.processor_cost[a, p] * model.process[a, p] 
        for a in model.processors 
        for p in model.products
    )
model.total_processing_cost_constr = Constraint(rule=total_processing_cost_rule)

def total_transport_cost_rule(model):
    # Cost from suppliers to processors
    sp_cost = sum(
        model.transport_cost[s, a, p] * model.ship_supplier_processor[s, a, p] 
        for s in model.suppliers 
        for a in model.processors 
        for p in model.products
    )
    
    # Cost from processors to stores
    ps_cost = sum(
        model.transport_cost[a, st, p] * model.ship_processor_store[a, st, p] 
        for a in model.processors 
        for st in model.stores 
        for p in model.products
    )
    
    return model.total_transport_cost == sp_cost + ps_cost
model.total_transport_cost_constr = Constraint(rule=total_transport_cost_rule)

def profit_rule(model):
    return model.profit == (
        model.total_revenue - 
        model.total_production_cost - 
        model.total_processing_cost - 
        model.total_transport_cost
    )
model.profit_constr = Constraint(rule=profit_rule)

# Objective
model.obj = Objective(expr=model.profit, sense=maximize)