\ Base/reference scenario LP for dummy_demand_tssp
Minimize
 obj: 1 x + 5 y
Subject To
 dummy: x >= 0
 demand_coverage: x + y >= 10
Bounds
 0 <= x <= 100
 0 <= y
End
