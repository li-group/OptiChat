\* Source Pyomo model name=K_system_for_IIS *\

min 
DummyObjective:
+0.0 ONE_VAR_CONSTANT

s.t.

c_u_K_budget_:
+0.5 Scenario_SCEN008_y
+0.5 Scenario_SCEN009_y
+0.5 Scenario_SCEN010_y
<= 40.0

c_l_Scenario_SCEN008_DUMMY_:
+0 ONE_VAR_CONSTANT
>= -55.0

c_l_Scenario_SCEN008_DEMAND_:
+1 Scenario_SCEN008_y
>= 25.0

c_l_Scenario_SCEN009_DUMMY_:
+0 ONE_VAR_CONSTANT
>= -55.0

c_l_Scenario_SCEN009_DEMAND_:
+1 Scenario_SCEN009_y
>= 35.0

c_l_Scenario_SCEN010_DUMMY_:
+0 ONE_VAR_CONSTANT
>= -55.0

c_l_Scenario_SCEN010_DEMAND_:
+1 Scenario_SCEN010_y
>= 45.0

bounds
   1 <= ONE_VAR_CONSTANT <= 1
   0.0 <= Scenario_SCEN008_y <= 1.7976931348623157e+308
   0.0 <= Scenario_SCEN009_y <= 1.7976931348623157e+308
   0.0 <= Scenario_SCEN010_y <= 1.7976931348623157e+308
end
