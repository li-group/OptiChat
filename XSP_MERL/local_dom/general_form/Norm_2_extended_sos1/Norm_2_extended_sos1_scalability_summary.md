# Norm_2_extended_sos1 scalability summary

- Norm: **L2**
- Formulation: **extended**
- Complementarity: native Gurobi SOS1, with `PreSOS1BigM=0` by default.
- Violation margin: `Delta(xi) <= -0`.
- Solver objective is squared Euclidean distance; reported radius is its square root.

## Per-instance results

| Instance | Family | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Radius | Radius bound | MIP gap | True gap |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.03093 | 0.1625 | 0.1625 | 0 | 2.842e-14 |
| lands_instance_small_1_1_2 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.00683 | 0.6219 | 0.6219 | 0 | 0 |
| lands_instance_small_1_1_3 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.014 | 1.24 | 1.24 | 0 | -1.421e-14 |
| lands_instance_small_1_1_4 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.02492 | 1.646 | 1.646 | 0 | 0 |
| lands_instance_small_1_1_5 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.01133 | 0.9901 | 0.9901 | 0 | 2.842e-14 |
| lands_instance_small_2_1_1 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.21 | 0.1178 | 0.1178 | 0 | 0 |
| lands_instance_small_2_1_2 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.4759 | 0.6542 | 0.6542 | 0 | 2.842e-14 |
| lands_instance_small_2_1_3 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.6272 | 3.55 | 3.55 | 0 | 0 |
| lands_instance_small_2_1_4 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.7906 | 0.4108 | 0.4108 | 0 | 5.684e-14 |
| lands_instance_small_2_1_5 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.3087 | 1.487 | 1.487 | 0 | 1.137e-13 |
| lands_instance_small_3_1_1 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 28.15 | 2.481 | 2.48 | 9.872e-05 | 5.684e-14 |
| lands_instance_small_3_1_2 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 6.323 | 0.4523 | 0.4523 | 0 | -5.684e-14 |
| lands_instance_small_3_1_3 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 4.033 | 0.8608 | 0.8608 | 0 | 5.684e-14 |
| lands_instance_small_3_1_4 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 1.077 | 0.3606 | 0.3606 | 0 | -5.258e-13 |
| lands_instance_small_3_1_5 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 2.56 | 0.03355 | 0.03355 | 0 | -5.684e-14 |
| lands_instance_med_1_1_1 | med | 10 | 15 | 150 | 25 | 25 | 350 | TIME_LIMIT | 300 | 0.1309 | 1.192e-07 | 1 | -5.684e-14 |
| lands_instance_med_1_1_2 | med | 10 | 15 | 150 | 25 | 25 | 350 | OPTIMAL | 253.6 | 0.5134 | 0.5134 | 7.965e-05 | 1.145e-11 |
| lands_instance_med_1_1_3 | med | 10 | 15 | 150 | 25 | 25 | 350 | TIME_LIMIT | 300 | 0.7316 | 0 | 1 | -5.684e-14 |
| lands_instance_med_1_1_4 | med | 10 | 15 | 150 | 25 | 25 | 350 | TIME_LIMIT | 300 | 0.4923 | 0 | 1 | -8.527e-14 |
| lands_instance_med_1_1_5 | med | 10 | 15 | 150 | 25 | 25 | 350 | TIME_LIMIT | 300 | 0.06708 | 0 | 1 | 0 |
| lands_instance_med_2_1_1 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_2 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 |  | 9.261e-08 |  |  |
| lands_instance_med_2_1_3 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 | 0.6703 | 1.145e-07 | 1 | -2.842e-13 |
| lands_instance_med_2_1_4 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_5 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 | 1.864 | 1.366e-07 | 1 | -4.434e-12 |
| lands_instance_med_3_1_1 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 1.154e-07 |  |  |
| lands_instance_med_3_1_2 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 1.115e-07 |  |  |
| lands_instance_med_3_1_3 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_4 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_5 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 7.3e-08 |  |  |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median radius | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 725 | 350 | 5 | 1 | 4 | 300 | 3.623e+06 | 0.4923 | 1 |
| med | 10 | 20 | 300 | 35 | 35 | 1375 | 670 | 5 | 0 | 5 | 300 | 1.764e+06 | 1.267 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 2650 | 1300 | 5 | 0 | 5 | 300 | 9.296e+05 |  |  |
| small | 10 | 4 | 12 | 7 | 7 | 83 | 38 | 5 | 5 | 0 | 0.014 | 203 | 0.9901 |  |
| small | 10 | 6 | 30 | 11 | 11 | 175 | 82 | 5 | 5 | 0 | 0.4759 | 1.197e+04 | 0.6542 |  |
| small | 10 | 8 | 48 | 14 | 14 | 262 | 124 | 5 | 5 | 0 | 4.033 | 7.818e+04 | 0.4523 |  |
