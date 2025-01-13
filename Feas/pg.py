import pyomo.environ as pyo
import time

# Create the Pyomo model
model = pyo.ConcreteModel()

# Define sets
model.T = pyo.Set(initialize=[t+1 for t in range(24)], doc="Hour of the day")

# Define parameters
power_lim_wind = [
    130, 150, 140, 160, 100, 120,
    150, 180, 170, 160, 120, 130,
    150, 176, 185, 120, 130, 140,
    170, 190, 120, 170, 130, 150
]
model.power_lim_wind = pyo.Param(
    model.T, initialize={t: power_lim_wind[t-1] for t in model.T}, mutable=True,
    doc="power limit of wind generation at each time"
)
power_lim_pv = [
    0, 0, 0, 0, 0, 5,
    10, 30, 60, 100, 130, 140,
    150, 140, 130, 100, 60, 30,
    10, 5, 0, 0, 0, 0
]
model.power_lim_pv = pyo.Param(
    model.T, initialize={t: power_lim_pv[t-1] for t in model.T}, mutable=True,
    doc="power limit of pv generation at each time"
)
model.power_lim_fuel = pyo.Param(
    model.T, initialize={t: 80 for t in model.T}, mutable=True,
    doc="power limit of fuel cell generation at each time"
)
model.power_lim_charge = pyo.Param(
    model.T, initialize={t: 200 for t in model.T}, mutable=True,
    doc="power limit of storage charging at each time"
)
model.power_lim_discharge = pyo.Param(
    model.T, initialize={t: 50 for t in model.T}, mutable=True,
    doc="power limit of storage discharging at each time"
)
load_demand = [
    160, 140, 150, 120, 110, 100,
    170, 180, 200, 220, 230, 240,
    240, 230, 220, 210, 210, 220,
    230, 240, 250, 200, 190, 180
]
model.load_demand = pyo.Param(
    model.T, initialize={t: load_demand[t-1] for t in model.T}, mutable=True,
    doc="power load demand at each time"
)
model.initial_battery = pyo.Param(
    initialize=100, mutable=True, doc="initial power in the battery")
model.power_lim_battery = pyo.Param(
    model.T, initialize={t: 200 for t in model.T}, mutable=True,
    doc="power limit of battery storage at each time"
)
model.cost_wind = pyo.Param(
    model.T, initialize={t: 0.4 for t in model.T}, mutable=True,
    doc="unit cost for wind power at each time"
)
model.cost_pv = pyo.Param(
    model.T, initialize={t: 0.4 for t in model.T}, mutable=True,
    doc="unit cost for pv power at each time"
)
model.cost_fuel = pyo.Param(
    model.T, initialize={t: 0.9 for t in model.T}, mutable=True,
    doc="unit cost for fuel power at each time"
)
model.cost_charge = pyo.Param(
    model.T, initialize={t: 0.4 for t in model.T}, mutable=True,
    doc="unit cost for charge power at each time"
)
model.cost_discharge = pyo.Param(
    model.T, initialize={t: 0.6 for t in model.T}, mutable=True,
    doc="unit cost for discharge power at each time"
)
model.cost_undelivered = pyo.Param(
    model.T, initialize={t: 1.5 for t in model.T}, mutable=True,
    doc="unit cost for undelivered power at each time"
)
model.cost_excess = pyo.Param(
    model.T, initialize={t: 0 for t in model.T}, mutable=True,
    doc="unit cost for excess power at each time"
)

# Define variables
model.wind = pyo.Var(model.T, within=pyo.NonNegativeReals,
                     doc="wind generation")
model.pv = pyo.Var(model.T, within=pyo.NonNegativeReals, doc="pv generation")
model.fuel = pyo.Var(model.T, within=pyo.NonNegativeReals,
                     doc="fuel generation")
model.charge = pyo.Var(
    model.T, within=pyo.NonNegativeReals, doc="power charged")
model.discharge = pyo.Var(
    model.T, within=pyo.NonNegativeReals, doc="power discharged")
model.undelivered = pyo.Var(
    model.T, within=pyo.NonNegativeReals, doc="undelivered power")
model.excess = pyo.Var(
    model.T, within=pyo.NonNegativeReals, doc="excess generation")
model.x = pyo.Var(model.T, within=pyo.Binary,
                  doc="whether battery should be discharged")
model.y = pyo.Var(model.T, within=pyo.Binary,
                  doc="whether battery should be charged")
model.battery = pyo.Var(
    model.T, within=pyo.NonNegativeReals, doc="power in battery")

# Define constraints
model.wind_gen_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.wind[t] <= m.power_lim_wind[t],
    doc="wind generation cannot exceed upper limit"
)
model.pv_gen_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.pv[t] <= m.power_lim_pv[t],
    doc="pv generation cannot exceed upper limit"
)
model.fuel_gen_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.fuel[t] <= m.power_lim_fuel[t],
    doc="fuel generation cannot exceed upper limit"
)
model.battery_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.battery[t] <= m.power_lim_battery[t],
    doc="battery storage cannot exceed upper limit"
)
model.discharge_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.discharge[t] <= m.power_lim_discharge[t] * m.x[t],
    doc="battery discharging cannot exceed discharging limit"
)
model.charge_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.charge[t] <= m.power_lim_charge[t] * m.y[t],
    doc="battery charging cannot exceed charging limit"
)
model.no_simultaneous_charge_discharge_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.x[t] + m.y[t] <= 1,
    doc="battery cannot charge and discharge at the same time"
)
model.discharge_limit_by_battery_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.discharge[t] <= (
        m.initial_battery if t == 1 else m.battery[t-1]),
    doc="battery cannot discharge more than storage at previous time"
)
model.discharge_limit_by_battery_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.charge[t] + (
        m.initial_battery if t == 1 else m.battery[t-1]) <= m.power_lim_battery[t],
    doc="battery cannot charge to exceed the capacity"
)
model.state_of_battery_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.battery[t] == (
        m.initial_battery if t == 1 else m.battery[t-1]) - m.discharge[t] + m.charge[t],
    doc="how the battery storage should be upated with charge and discharge"
)
model.power_balance_constraint = pyo.Constraint(
    model.T, rule=lambda m, t: m.wind[t] + m.pv[t] + m.fuel[t] +
    m.discharge[t] + m.undelivered[t] == m.load_demand[t] + m.charge[t] + m.excess[t],
    doc="generation must be equal to demand"
)

# Define auxillary variable and constraint
model.tc = pyo.Var(doc="total cost")
model.total_cost_definition = pyo.Constraint(
    rule=lambda m: m.tc == sum(
        m.wind[t] * m.cost_wind[t] + m.pv[t] * m.cost_pv[t] +
        m.fuel[t] * m.cost_fuel[t] + m.charge[t] * m.cost_charge[t] +
        m.discharge[t] * m.cost_discharge[t] + m.undelivered[t] * m.cost_undelivered[t] +
        m.excess[t] * m.cost_excess[t] for t in model.T),
    doc="total cost of the network, which is the objective value"
)

# Define objective
model.obj = pyo.Objective(expr=model.tc, sense=pyo.minimize, doc="we want to minimize the total cost")

