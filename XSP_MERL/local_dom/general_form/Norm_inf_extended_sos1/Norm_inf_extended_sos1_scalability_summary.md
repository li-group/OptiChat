# Norm_inf_extended_sos1 scalability summary

- Norm: **L-infinity**
- Formulation: **extended**
- Complementarity: native Gurobi SOS1, with `PreSOS1BigM=0` by default.
- Violation margin: `Delta(xi) <= -0`.
- Solver objective is the L-infinity radius `rho`.

## Per-instance results

| Instance | Family | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Radius | Radius bound | MIP gap | True gap |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.008188 | 0.157 | 0.157 | 0 | 0 |
| lands_instance_small_1_1_2 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.007529 | 0.6219 | 0.6219 | 0 | 0 |
| lands_instance_small_1_1_3 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.01471 | 1.24 | 1.24 | 0 | -1.421e-14 |
| lands_instance_small_1_1_4 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.01243 | 1.233 | 1.233 | 0 | 1.705e-13 |
| lands_instance_small_1_1_5 | small | 10 | 4 | 12 | 7 | 7 | 38 | OPTIMAL | 0.01217 | 0.9557 | 0.9557 | 0 | 1.137e-13 |
| lands_instance_small_2_1_1 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.05001 | 0.1178 | 0.1178 | 0 | 0 |
| lands_instance_small_2_1_2 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.04173 | 0.4621 | 0.4621 | 0 | -1.421e-14 |
| lands_instance_small_2_1_3 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.09754 | 2.313 | 2.313 | 0 | -1.705e-13 |
| lands_instance_small_2_1_4 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.2698 | 0.2668 | 0.2668 | 0 | 5.684e-14 |
| lands_instance_small_2_1_5 | small | 10 | 6 | 30 | 11 | 11 | 82 | OPTIMAL | 0.06907 | 0.8142 | 0.8142 | 0 | -5.684e-14 |
| lands_instance_small_3_1_1 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 0.1807 | 1.555 | 1.555 | 0 | -8.527e-14 |
| lands_instance_small_3_1_2 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 2.915 | 0.3054 | 0.3054 | 0 | 0 |
| lands_instance_small_3_1_3 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 0.547 | 0.4078 | 0.4078 | 0 | 5.684e-14 |
| lands_instance_small_3_1_4 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 0.09623 | 0.3606 | 0.3606 | 0 | -2.842e-14 |
| lands_instance_small_3_1_5 | small | 10 | 8 | 48 | 14 | 14 | 124 | OPTIMAL | 0.1318 | 0.02391 | 0.02391 | 0 | 5.684e-14 |
| lands_instance_med_1_1_1 | med | 10 | 15 | 150 | 25 | 25 | 350 | OPTIMAL | 215.2 | 0.06193 | 0.06193 | 0 | -8.527e-14 |
| lands_instance_med_1_1_2 | med | 10 | 15 | 150 | 25 | 25 | 350 | OPTIMAL | 40.35 | 0.3508 | 0.3508 | 0 | 0 |
| lands_instance_med_1_1_3 | med | 10 | 15 | 150 | 25 | 25 | 350 | OPTIMAL | 142.9 | 0.3482 | 0.3482 | 0 | 0 |
| lands_instance_med_1_1_4 | med | 10 | 15 | 150 | 25 | 25 | 350 | OPTIMAL | 21.06 | 0.2153 | 0.2153 | 0 | 0 |
| lands_instance_med_1_1_5 | med | 10 | 15 | 150 | 25 | 25 | 350 | TIME_LIMIT | 300 | 0.03714 | 0.00878 | 0.7636 | -1.137e-13 |
| lands_instance_med_2_1_1 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_2 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_3 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 | 0.3063 | 0 | 1 | 5.684e-14 |
| lands_instance_med_2_1_4 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 | 0.2629 | 0 | 1 | 1.137e-13 |
| lands_instance_med_2_1_5 | med | 10 | 20 | 300 | 35 | 35 | 670 | TIME_LIMIT | 300 | 0.9753 | 0 | 1 | -1.137e-13 |
| lands_instance_med_3_1_1 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_2 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_3 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_4 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_5 | med | 10 | 30 | 600 | 50 | 50 | 1300 | TIME_LIMIT | 300 |  | 0 |  |  |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median radius | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 726 | 350 | 5 | 4 | 1 | 142.9 | 3.438e+06 | 0.2153 | 0.7636 |
| med | 10 | 20 | 300 | 35 | 35 | 1376 | 670 | 5 | 0 | 5 | 300 | 3.366e+06 | 0.3063 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 2651 | 1300 | 5 | 0 | 5 | 300 | 9.674e+05 |  |  |
| small | 10 | 4 | 12 | 7 | 7 | 84 | 38 | 5 | 5 | 0 | 0.01217 | 79 | 0.9557 |  |
| small | 10 | 6 | 30 | 11 | 11 | 176 | 82 | 5 | 5 | 0 | 0.06907 | 1683 | 0.4621 |  |
| small | 10 | 8 | 48 | 14 | 14 | 263 | 124 | 5 | 5 | 0 | 0.1807 | 5899 | 0.3606 |  |
