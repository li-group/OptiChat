# Hadamard Big-M local-dominance scalability report

## Per-instance results

| Instance | S | nx | ny | r | m | binaries | log10 Mp | log10 Mmu | LP relax | Status | MIP runtime (s) | Gamma/incumbent | Best bound | MIP gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| lands_instance_small_1_1_1 | 10 | 4 | 12 | 7 | 7 | 19 | 3.85 | 4.934 | 0 | OPTIMAL | 0.01325 | 0.1625 | 0.1625 | 0 |
| lands_instance_small_1_1_2 | 10 | 4 | 12 | 7 | 7 | 19 | 4.31 | 4.96 | 0 | OPTIMAL | 0.009099 | 0.6219 | 0.6219 | 0 |
| lands_instance_small_1_1_3 | 10 | 4 | 12 | 7 | 7 | 19 | 4.231 | 4.999 | 0 | OPTIMAL | 0.005811 | 1.24 | 1.24 | 0 |
| lands_instance_small_1_1_4 | 10 | 4 | 12 | 7 | 7 | 19 | 4.318 | 4.959 | 0 | OPTIMAL | 0.01074 | 1.646 | 1.646 | 0 |
| lands_instance_small_1_1_5 | 10 | 4 | 12 | 7 | 7 | 19 | 4.316 | 4.953 | 0 | OPTIMAL | 0.006837 | 0.9901 | 0.9901 | 0 |
| lands_instance_small_2_1_1 | 10 | 6 | 30 | 11 | 11 | 41 | 6.811 | 7.751 | 0 | OPTIMAL | 0.03388 | 3.128 | 3.128 | 0 |
| lands_instance_small_2_1_2 | 10 | 6 | 30 | 11 | 11 | 41 | 6.847 | 7.756 | 0 | OPTIMAL | 0.01947 | 0.6542 | 0.6542 | 0 |
| lands_instance_small_2_1_3 | 10 | 6 | 30 | 11 | 11 | 41 | 6.973 | 7.747 | 0 | OPTIMAL | 0.04901 | 4.083 | 4.083 | 0 |
| lands_instance_small_2_1_4 | 10 | 6 | 30 | 11 | 11 | 41 | 6.948 | 7.765 | 0 | OPTIMAL | 0.03064 | 0.6741 | 0.6741 | 0 |
| lands_instance_small_2_1_5 | 10 | 6 | 30 | 11 | 11 | 41 | 7.105 | 7.767 | 0 | OPTIMAL | 0.112 | 2.834 | 2.834 | 0 |
| lands_instance_small_3_1_1 | 10 | 8 | 48 | 14 | 14 | 62 | 9.271 | 10.06 | 0 | INFEASIBLE | 4.201e-04 |  | inf |  |
| lands_instance_small_3_1_2 | 10 | 8 | 48 | 14 | 14 | 62 | 9.136 | 10.06 | 0 | INFEASIBLE | 5.350e-04 |  | inf |  |
| lands_instance_small_3_1_3 | 10 | 8 | 48 | 14 | 14 | 62 | 9.28 | 10.06 | 0 | INFEASIBLE | 4.292e-04 |  | inf |  |
| lands_instance_small_3_1_4 | 10 | 8 | 48 | 14 | 14 | 62 | 9.137 | 10.06 | 0 | INFEASIBLE | 4.981e-04 |  | inf |  |
| lands_instance_small_3_1_5 | 10 | 8 | 48 | 14 | 14 | 62 | 9.018 | 10.06 | 0 | INFEASIBLE | 5.112e-04 |  | inf |  |
| lands_instance_med_1_1_1 | 10 | 15 | 150 | 25 | 25 | 175 | 18.73 | 19.51 | 0 | OPTIMAL | 0.001491 | 0 | 0 | 0 |
| lands_instance_med_1_1_2 | 10 | 15 | 150 | 25 | 25 | 175 | 18.37 | 19.52 | 0 | OPTIMAL | 0.001483 | 0 | 0 | 0 |
| lands_instance_med_1_1_3 | 10 | 15 | 150 | 25 | 25 | 175 | 18.81 | 19.51 | 0 | OPTIMAL | 0.001531 | 0 | -2.842e-14 | 0 |
| lands_instance_med_1_1_4 | 10 | 15 | 150 | 25 | 25 | 175 | 18.64 | 19.52 | 0 | OPTIMAL | 0.001647 | 0 | -1.421e-14 | 0 |
| lands_instance_med_1_1_5 | 10 | 15 | 150 | 25 | 25 | 175 | 18.38 | 19.51 | 0 | OPTIMAL | 0.001775 | 0 | 0 | 0 |
| lands_instance_med_2_1_1 | 10 | 20 | 300 | 35 | 35 | 335 | 28.18 | 29.06 | 0 | OPTIMAL | 0.002289 | 0 | -7.105e-15 | 0 |
| lands_instance_med_2_1_2 | 10 | 20 | 300 | 35 | 35 | 335 | 28.12 | 29.06 | 0 | OPTIMAL | 0.002249 | 0 | -2.487e-14 | 0 |
| lands_instance_med_2_1_3 | 10 | 20 | 300 | 35 | 35 | 335 | 28.32 | 29.06 | 0 | OPTIMAL | 0.00232 | 0 | -5.329e-14 | 0 |
| lands_instance_med_2_1_4 | 10 | 20 | 300 | 35 | 35 | 335 | 28.28 | 29.06 | 0 | OPTIMAL | 0.002401 | 0 | 0 | 0 |
| lands_instance_med_2_1_5 | 10 | 20 | 300 | 35 | 35 | 335 | 28.17 | 29.06 | 0 | OPTIMAL | 0.002262 | 0 | -2.132e-14 | 0 |
| lands_instance_med_3_1_1 | 10 | 30 | 600 | 50 | 50 | 650 | 43.89 | 44.52 | 0 | OPTIMAL | 0.005888 | 0 | -7.816e-14 | 0 |
| lands_instance_med_3_1_2 | 10 | 30 | 600 | 50 | 50 | 650 | 43.88 | 44.52 | 0 | OPTIMAL | 0.006215 | 0 | -7.105e-14 | 0 |
| lands_instance_med_3_1_3 | 10 | 30 | 600 | 50 | 50 | 650 | 43.62 | 44.52 | 0 | OPTIMAL | 0.005917 | 0 | -2.842e-14 | 0 |
| lands_instance_med_3_1_4 | 10 | 30 | 600 | 50 | 50 | 650 | 43.64 | 44.51 | 0 | OPTIMAL | 0.006246 | 0 | 0 | 0 |
| lands_instance_med_3_1_5 | 10 | 30 | 600 | 50 | 50 | 650 | 43.7 | 44.52 | 0 | OPTIMAL | 0.005909 | 0 | -2.132e-14 | 0 |
| lands_instance_hard_1_1_1 | 10 | 50 | 1500 | 80 | 80 | 1580 | 77.22 | 78.16 | 0 | OPTIMAL | 0.02082 | 0 | -6.395e-14 | 0 |
| lands_instance_hard_1_1_2 | 10 | 50 | 1500 | 80 | 80 | 1580 | 77.32 | 78.16 | 0 | OPTIMAL | 0.01565 | 0 | -4.263e-14 | 0 |
| lands_instance_hard_1_1_3 | 10 | 50 | 1500 | 80 | 80 | 1580 | 77.54 | 78.16 | 0 | OPTIMAL | 0.01516 | 0 | -9.948e-14 | 0 |
| lands_instance_hard_1_1_4 | 10 | 50 | 1500 | 80 | 80 | 1580 | 77.51 | 78.16 | 0 | OPTIMAL | 0.01547 | 0 | -1.421e-14 | 0 |
| lands_instance_hard_1_1_5 | 10 | 50 | 1500 | 80 | 80 | 1580 | 77.31 | 78.16 | 0 | OPTIMAL | 0.01498 | 0 | -3.553e-14 | 0 |
| lands_instance_hard_2_1_1 | 10 | 75 | 3750 | 125 | 125 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_2_1_2 | 10 | 75 | 3750 | 125 | 125 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_2_1_3 | 10 | 75 | 3750 | 125 | 125 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_2_1_4 | 10 | 75 | 3750 | 125 | 125 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_2_1_5 | 10 | 75 | 3750 | 125 | 125 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_3_1_1 | 10 | 100 | 7500 | 175 | 175 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_3_1_2 | 10 | 100 | 7500 | 175 | 175 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_3_1_3 | 10 | 100 | 7500 | 175 | 175 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_3_1_4 | 10 | 100 | 7500 | 175 | 175 |  |  |  |  | ERROR |  |  |  |  |
| lands_instance_hard_3_1_5 | 10 | 100 | 7500 | 175 | 175 |  |  |  |  | ERROR |  |  |  |  |

## Grouped scalability summary

| Family | S | nx | ny | r | m | N | Optimal | Time limit | Median MIP runtime (s) | Median nodes | Median nonoptimal gap | Median LP relax | Median log10 Mp | Median log10 Mmu |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard | 10 | 100 | 7500 | 175 | 175 | 5 | 0 | 0 |  |  |  |  |  |  |
| hard | 10 | 50 | 1500 | 80 | 80 | 5 | 5 | 0 | 0.01547 | 1 |  | 0 | 77.32 | 78.16 |
| hard | 10 | 75 | 3750 | 125 | 125 | 5 | 0 | 0 |  |  |  |  |  |  |
| med | 10 | 15 | 150 | 25 | 25 | 5 | 5 | 0 | 0.001531 | 1 |  | 0 | 18.64 | 19.51 |
| med | 10 | 20 | 300 | 35 | 35 | 5 | 5 | 0 | 0.002289 | 1 |  | 0 | 28.18 | 29.06 |
| med | 10 | 30 | 600 | 50 | 50 | 5 | 5 | 0 | 0.005917 | 1 |  | 0 | 43.7 | 44.52 |
| small | 10 | 4 | 12 | 7 | 7 | 5 | 5 | 0 | 0.009099 | 19 |  | 0 | 4.31 | 4.959 |
| small | 10 | 6 | 30 | 11 | 11 | 5 | 5 | 0 | 0.03388 | 228 |  | 0 | 6.948 | 7.756 |
| small | 10 | 8 | 48 | 14 | 14 | 5 | 0 | 0 | 4.981e-04 | 0 |  | 0 | 9.137 | 10.06 |
