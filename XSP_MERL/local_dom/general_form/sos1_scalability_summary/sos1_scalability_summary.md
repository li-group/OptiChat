# SOS1 local-dominance scalability report

## Per-instance results

| Instance | S | nx | ny | r | m | SOS1 | Status | Runtime (s) | Gamma/incumbent | Best bound | MIP gap |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.01726 | 0.1625 | 0.1625 | 0 |
| lands_instance_small_1_1_2 | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.01214 | 0.6219 | 0.6219 | 0 |
| lands_instance_small_1_1_3 | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.005906 | 1.24 | 1.24 | 0 |
| lands_instance_small_1_1_4 | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.01197 | 1.646 | 1.646 | 0 |
| lands_instance_small_1_1_5 | 10 | 4 | 12 | 7 | 7 | 19 | OPTIMAL | 0.005807 | 0.9901 | 0.9901 | 0 |
| lands_instance_small_2_1_1 | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.02122 | 0.1178 | 0.1178 | 0 |
| lands_instance_small_2_1_2 | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.02114 | 0.6542 | 0.6542 | 0 |
| lands_instance_small_2_1_3 | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.06341 | 4.083 | 4.083 | 0 |
| lands_instance_small_2_1_4 | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.03272 | 0.6741 | 0.6741 | 0 |
| lands_instance_small_2_1_5 | 10 | 6 | 30 | 11 | 11 | 41 | OPTIMAL | 0.05108 | 2.834 | 2.834 | 0 |
| lands_instance_small_3_1_1 | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.1697 | 3.409 | 3.409 | 0 |
| lands_instance_small_3_1_2 | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.1147 | 0.6814 | 0.6814 | 0 |
| lands_instance_small_3_1_3 | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.203 | 1.371 | 1.371 | 0 |
| lands_instance_small_3_1_4 | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.0633 | 0.3606 | 0.3606 | 0 |
| lands_instance_small_3_1_5 | 10 | 8 | 48 | 14 | 14 | 62 | OPTIMAL | 0.04265 | 0.03596 | 0.03596 | 0 |
| lands_instance_med_1_1_1 | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 14.68 | 0.222 | 0.222 | 0 |
| lands_instance_med_1_1_2 | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 2.805 | 0.8465 | 0.8465 | 0 |
| lands_instance_med_1_1_3 | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 16.13 | 1.469 | 1.469 | 0 |
| lands_instance_med_1_1_4 | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 4.953 | 0.9107 | 0.9107 | 0 |
| lands_instance_med_1_1_5 | 10 | 15 | 150 | 25 | 25 | 175 | OPTIMAL | 2.658 | 0.1141 | 0.1141 | 0 |
| lands_instance_med_2_1_1 | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 3.967 | 0 | 1 |
| lands_instance_med_2_1_2 | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 1.695 | 0 | 1 |
| lands_instance_med_2_1_3 | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 1.216 | 0 | 1 |
| lands_instance_med_2_1_4 | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 2.08 | 0 | 1 |
| lands_instance_med_2_1_5 | 10 | 20 | 300 | 35 | 35 | 335 | TIME_LIMIT | 300 | 4.797 | 0 | 1 |
| lands_instance_med_3_1_1 | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 5.438 | 0 | 1 |
| lands_instance_med_3_1_2 | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 6.841 | 0 | 1 |
| lands_instance_med_3_1_3 | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 5.786 | 0 | 1 |
| lands_instance_med_3_1_4 | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 3.904 | 0 | 1 |
| lands_instance_med_3_1_5 | 10 | 30 | 600 | 50 | 50 | 650 | TIME_LIMIT | 300 | 8.467 | 0 | 1 |

## Grouped scalability summary

| Family | S | nx | ny | r | m | Vars | SOS1 | N | Optimal | Time limit | Median runtime (s) | Median nodes | Median nonoptimal gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| med | 10 | 15 | 150 | 25 | 25 | 550 | 175 | 5 | 5 | 0 | 4.953 | 1.721e+05 |  |
| med | 10 | 20 | 300 | 35 | 35 | 1040 | 335 | 5 | 0 | 5 | 300 | 6.202e+06 | 1 |
| med | 10 | 30 | 600 | 50 | 50 | 2000 | 650 | 5 | 0 | 5 | 300 | 2.871e+06 | 1 |
| small | 10 | 4 | 12 | 7 | 7 | 64 | 19 | 5 | 5 | 0 | 0.01197 | 63 |  |
| small | 10 | 6 | 30 | 11 | 11 | 134 | 41 | 5 | 5 | 0 | 0.03272 | 1160 |  |
| small | 10 | 8 | 48 | 14 | 14 | 200 | 62 | 5 | 5 | 0 | 0.1147 | 2955 |  |
