# Norm_2_one_sided_sos1 scalability summary

- Norm: **L2**
- Formulation: **one_sided**
- Complementarity: native Gurobi SOS1, with `PreSOS1BigM=0` by default.
- Violation margin: `Delta(xi) <= -0`.
- Solver objective is squared Euclidean distance; reported radius is its square root.

## Per-instance results

| Instance | Family | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Radius | Radius bound | MIP gap | True gap |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.004762 | 0.1625 | 0.1625 | 0 | 0 |
| lands_instance_small_1_1_2 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.005971 | 0.6219 | 0.6219 | 0 | 0 |
| lands_instance_small_1_1_3 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.009089 | 1.24 | 1.24 | 0 | -1.421e-14 |
| lands_instance_small_1_1_4 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.01046 | 1.646 | 1.646 | 0 | 0 |
| lands_instance_small_1_1_5 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.009511 | 0.9901 | 0.9901 | 0 | 8.527e-14 |
| lands_instance_small_2_1_1 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.03038 | 0.1178 | 0.1178 | 0 | 0 |
| lands_instance_small_2_1_2 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.0382 | 0.6542 | 0.6542 | 0 | 5.684e-14 |
| lands_instance_small_2_1_3 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.05527 | 3.55 | 3.55 | 0 | -2.842e-14 |
| lands_instance_small_2_1_4 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.05417 | 0.4108 | 0.4108 | 0 | 5.684e-14 |
| lands_instance_small_2_1_5 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.04463 | 1.487 | 1.487 | 0 | 1.137e-13 |
| lands_instance_small_3_1_1 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.3429 | 2.481 | 2.481 | 0 | 2.842e-14 |
| lands_instance_small_3_1_2 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.3376 | 0.4523 | 0.4523 | 0 | -1.705e-13 |
| lands_instance_small_3_1_3 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.5009 | 0.8608 | 0.8608 | 0 | 8.527e-14 |
| lands_instance_small_3_1_4 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.0605 | 0.3606 | 0.3606 | 0 | 2.842e-14 |
| lands_instance_small_3_1_5 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.07949 | 0.03355 | 0.03355 | 0 | -5.684e-14 |
| lands_instance_med_1_1_1 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 6.046 | 0.1309 | 0.1309 | 0 | -1.137e-13 |
| lands_instance_med_1_1_2 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 13.4 | 0.5134 | 0.5134 | 0 | 5.684e-13 |
| lands_instance_med_1_1_3 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 17.93 | 0.7316 | 0.7316 | 0 | -5.684e-14 |
| lands_instance_med_1_1_4 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 20.43 | 0.4923 | 0.4923 | 0 | -9.095e-13 |
| lands_instance_med_1_1_5 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 14.3 | 0.06708 | 0.06708 | 0 | 0 |
| lands_instance_med_2_1_1 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 1.155 | 0 | 1 | -2.842e-13 |
| lands_instance_med_2_1_2 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.9839 | 9.261e-08 | 1 | 5.684e-14 |
| lands_instance_med_2_1_3 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.6703 | 1.145e-07 | 1 | -1.762e-12 |
| lands_instance_med_2_1_4 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.7423 | 0 | 1 | 2.274e-13 |
| lands_instance_med_2_1_5 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 1.864 | 1.366e-07 | 1 | 7.901e-12 |
| lands_instance_med_3_1_1 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 1.717 | 1.154e-07 | 1 | 1.137e-13 |
| lands_instance_med_3_1_2 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.8021 | 1.115e-07 | 1 | -1.137e-13 |
| lands_instance_med_3_1_3 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 1.287 | 0 | 1 | -1.137e-13 |
| lands_instance_med_3_1_4 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_5 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.9936 | 7.3e-08 | 1 | 5.684e-14 |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median radius | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 525 | 175 | 5 | 5 | 0 | 14.3 | 2.634e+05 | 0.4923 |  |
| med | 10 | 20 | 300 | 35 | 35 | 1005 | 335 | 5 | 0 | 5 | 300 | 2.913e+06 | 0.9839 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 1950 | 650 | 5 | 0 | 5 | 300 | 1.8e+06 | 1.14 | 1 |
| small | 10 | 4 | 12 | 7 | 7 | 57 | 19 | 5 | 5 | 0 | 0.009089 | 103 | 0.9901 |  |
| small | 10 | 6 | 30 | 11 | 11 | 123 | 41 | 5 | 5 | 0 | 0.04463 | 777 | 0.6542 |  |
| small | 10 | 8 | 48 | 14 | 14 | 186 | 62 | 5 | 5 | 0 | 0.3376 | 7084 | 0.4523 |  |
