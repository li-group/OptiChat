# Extended SOS1 local-dominance scalability report

This report uses KKT+SOS1 optimality on both the EV and x* recourse LPs. Native SOS1 is retained with `PreSOS1BigM=0`.

## Per-instance results

| Instance | S | nx | ny | r | m | SOS1 | eps | Status | Runtime (s) | Gamma/incumbent | Best bound | MIP gap | True gap |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | 10 | 4 | 12 | 7 | 7 | 38 | 0 | OPTIMAL | 0.01232 | 0.1625 | 0.1625 | 0 | 1.421e-14 |
| lands_instance_small_1_1_2 | 10 | 4 | 12 | 7 | 7 | 38 | 0 | OPTIMAL | 0.005507 | 0.6219 | 0.6219 | 0 | 1.137e-13 |
| lands_instance_small_1_1_3 | 10 | 4 | 12 | 7 | 7 | 38 | 0 | OPTIMAL | 0.008155 | 1.24 | 1.24 | 0 | -1.421e-14 |
| lands_instance_small_1_1_4 | 10 | 4 | 12 | 7 | 7 | 38 | 0 | OPTIMAL | 0.0297 | 1.646 | 1.646 | 0 | -1.705e-13 |
| lands_instance_small_1_1_5 | 10 | 4 | 12 | 7 | 7 | 38 | 0 | OPTIMAL | 0.01339 | 0.9901 | 0.9901 | 0 | 8.527e-14 |
| lands_instance_small_2_1_1 | 10 | 6 | 30 | 11 | 11 | 82 | 0 | OPTIMAL | 0.04232 | 0.1178 | 0.1178 | 0 | 2.842e-14 |
| lands_instance_small_2_1_2 | 10 | 6 | 30 | 11 | 11 | 82 | 0 | OPTIMAL | 0.03515 | 0.6542 | 0.6542 | 0 | 0 |
| lands_instance_small_2_1_3 | 10 | 6 | 30 | 11 | 11 | 82 | 0 | OPTIMAL | 0.0875 | 4.083 | 4.083 | 0 | -6.395e-14 |
| lands_instance_small_2_1_4 | 10 | 6 | 30 | 11 | 11 | 82 | 0 | OPTIMAL | 0.06259 | 0.6741 | 0.6741 | 0 | 2.842e-14 |
| lands_instance_small_2_1_5 | 10 | 6 | 30 | 11 | 11 | 82 | 0 | OPTIMAL | 0.08868 | 2.834 | 2.834 | 0 | 5.684e-14 |
| lands_instance_small_3_1_1 | 10 | 8 | 48 | 14 | 14 | 124 | 0 | OPTIMAL | 0.5272 | 3.409 | 3.409 | 0 | 0 |
| lands_instance_small_3_1_2 | 10 | 8 | 48 | 14 | 14 | 124 | 0 | OPTIMAL | 0.3438 | 0.6814 | 0.6814 | 0 | 0 |
| lands_instance_small_3_1_3 | 10 | 8 | 48 | 14 | 14 | 124 | 0 | OPTIMAL | 1.25 | 1.371 | 1.371 | 0 | 0 |
| lands_instance_small_3_1_4 | 10 | 8 | 48 | 14 | 14 | 124 | 0 | OPTIMAL | 0.6366 | 0.3606 | 0.3606 | 0 | 2.842e-14 |
| lands_instance_small_3_1_5 | 10 | 8 | 48 | 14 | 14 | 124 | 0 | OPTIMAL | 0.1557 | 0.03596 | 0.03596 | 0 | -5.684e-14 |
| lands_instance_med_1_1_1 | 10 | 15 | 150 | 25 | 25 | 350 | 0 | OPTIMAL | 228.5 | 0.222 | 0.222 | 0 | 0 |
| lands_instance_med_1_1_2 | 10 | 15 | 150 | 25 | 25 | 350 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_1_1_3 | 10 | 15 | 150 | 25 | 25 | 350 | 0 | TIME_LIMIT | 300 | 1.469 | 0.2331 | 0.8413 | 0 |
| lands_instance_med_1_1_4 | 10 | 15 | 150 | 25 | 25 | 350 | 0 | OPTIMAL | 61.03 | 0.9107 | 0.9106 | 6.026e-05 | 5.4e-13 |
| lands_instance_med_1_1_5 | 10 | 15 | 150 | 25 | 25 | 350 | 0 | OPTIMAL | 60.47 | 0.1141 | 0.1141 | 0 | 5.684e-13 |
| lands_instance_med_2_1_1 | 10 | 20 | 300 | 35 | 35 | 670 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_2 | 10 | 20 | 300 | 35 | 35 | 670 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_2_1_3 | 10 | 20 | 300 | 35 | 35 | 670 | 0 | TIME_LIMIT | 300 | 1.216 | 0 | 1 | 0 |
| lands_instance_med_2_1_4 | 10 | 20 | 300 | 35 | 35 | 670 | 0 | TIME_LIMIT | 300 | 2.008 | 0 | 1 | 2.274e-13 |
| lands_instance_med_2_1_5 | 10 | 20 | 300 | 35 | 35 | 670 | 0 | TIME_LIMIT | 300 | 4.797 | 0 | 1 | 2.331e-12 |
| lands_instance_med_3_1_1 | 10 | 30 | 600 | 50 | 50 | 1300 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_2 | 10 | 30 | 600 | 50 | 50 | 1300 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_3 | 10 | 30 | 600 | 50 | 50 | 1300 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_4 | 10 | 30 | 600 | 50 | 50 | 1300 | 0 | TIME_LIMIT | 300 |  | 0 |  |  |
| lands_instance_med_3_1_5 | 10 | 30 | 600 | 50 | 50 | 1300 | 0 | TIME_LIMIT | 300 | 2.656 | 0 | 1 | 5.23e-12 |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | eps | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 750 | 350 | 0 | 5 | 3 | 2 | 228.5 | 3.715e+06 | 0.8413 |
| med | 10 | 20 | 300 | 35 | 35 | 1410 | 670 | 0 | 5 | 0 | 5 | 300 | 2.773e+06 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 2700 | 1300 | 0 | 5 | 0 | 5 | 300 | 1.053e+06 | 1 |
| small | 10 | 4 | 12 | 7 | 7 | 90 | 38 | 0 | 5 | 5 | 0 | 0.01232 | 115 |  |
| small | 10 | 6 | 30 | 11 | 11 | 186 | 82 | 0 | 5 | 5 | 0 | 0.06259 | 1761 |  |
| small | 10 | 8 | 48 | 14 | 14 | 276 | 124 | 0 | 5 | 5 | 0 | 0.5272 | 1.978e+04 |  |
