# Norm_inf_one_sided_sos1 scalability summary

- Norm: **L-infinity**
- Formulation: **one_sided**
- Complementarity: native Gurobi SOS1, with `PreSOS1BigM=0` by default.
- Violation margin: `Delta(xi) <= -0`.
- Solver objective is the L-infinity radius `rho`.

## Per-instance results

| Instance | Family | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Radius | Radius bound | MIP gap | True gap |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.006488 | 0.157 | 0.157 | 0 | 1.421e-14 |
| lands_instance_small_1_1_2 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.004583 | 0.6219 | 0.6219 | 0 | 5.684e-14 |
| lands_instance_small_1_1_3 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.0062 | 1.24 | 1.24 | 0 | 2.842e-14 |
| lands_instance_small_1_1_4 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.009106 | 1.233 | 1.233 | 0 | -5.684e-14 |
| lands_instance_small_1_1_5 | small | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.0076 | 0.9557 | 0.9557 | 0 | 1.137e-13 |
| lands_instance_small_2_1_1 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.02263 | 0.1178 | 0.1178 | 0 | -2.842e-14 |
| lands_instance_small_2_1_2 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.0221 | 0.4621 | 0.4621 | 0 | -2.842e-14 |
| lands_instance_small_2_1_3 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.06526 | 2.313 | 2.313 | 0 | -2.842e-13 |
| lands_instance_small_2_1_4 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.03955 | 0.2668 | 0.2668 | 0 | 5.684e-14 |
| lands_instance_small_2_1_5 | small | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.03572 | 0.8142 | 0.8142 | 0 | -5.684e-14 |
| lands_instance_small_3_1_1 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.1276 | 1.555 | 1.555 | 0 | 0 |
| lands_instance_small_3_1_2 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.131 | 0.3054 | 0.3054 | 0 | 5.684e-14 |
| lands_instance_small_3_1_3 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.1413 | 0.4078 | 0.4078 | 0 | 1.137e-13 |
| lands_instance_small_3_1_4 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.03326 | 0.3606 | 0.3606 | 0 | 0 |
| lands_instance_small_3_1_5 | small | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.04293 | 0.02391 | 0.02391 | 0 | 1.137e-13 |
| lands_instance_med_1_1_1 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 1.892 | 0.06193 | 0.06193 | 0 | -2.842e-14 |
| lands_instance_med_1_1_2 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 3.779 | 0.3508 | 0.3508 | 0 | 0 |
| lands_instance_med_1_1_3 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 4.829 | 0.3482 | 0.3482 | 0 | 5.684e-14 |
| lands_instance_med_1_1_4 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 3.259 | 0.2153 | 0.2153 | 0 | -1.137e-13 |
| lands_instance_med_1_1_5 | med | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 0.6573 | 0.03714 | 0.03714 | 0 | -5.684e-14 |
| lands_instance_med_2_1_1 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.372 | 0 | 1 | -2.274e-13 |
| lands_instance_med_2_1_2 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.4123 | 0.2014 | 0.5116 | 5.684e-14 |
| lands_instance_med_2_1_3 | med | 10 | 20 | 300 | 35 | 35 | 335 | OPTIMAL | 191 | 0.3063 | 0.3063 | 0 | 5.684e-14 |
| lands_instance_med_2_1_4 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.2629 | 0 | 1 | 1.137e-13 |
| lands_instance_med_2_1_5 | med | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 0.9753 | 0 | 1 | -2.274e-13 |
| lands_instance_med_3_1_1 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.6155 | 0 | 1 | 1.705e-13 |
| lands_instance_med_3_1_2 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.294 | 0 | 1 | -1.137e-13 |
| lands_instance_med_3_1_3 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.7152 | 0 | 1 | -1.137e-13 |
| lands_instance_med_3_1_4 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.6111 | 0 | 1 | 1.137e-13 |
| lands_instance_med_3_1_5 | med | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 0.6949 | 0 | 1 | 1.137e-13 |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median radius | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 526 | 175 | 5 | 5 | 0 | 3.259 | 1.014e+05 | 0.2153 |  |
| med | 10 | 20 | 300 | 35 | 35 | 1006 | 335 | 5 | 1 | 4 | 300 | 5.761e+06 | 0.372 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 1951 | 650 | 5 | 0 | 5 | 300 | 3.695e+06 | 0.6155 | 1 |
| small | 10 | 4 | 12 | 7 | 7 | 58 | 19 | 5 | 5 | 0 | 0.006488 | 77 | 0.9557 |  |
| small | 10 | 6 | 30 | 11 | 11 | 124 | 41 | 5 | 5 | 0 | 0.03572 | 573 | 0.4621 |  |
| small | 10 | 8 | 48 | 14 | 14 | 187 | 62 | 5 | 5 | 0 | 0.1276 | 3604 | 0.3606 |  |
