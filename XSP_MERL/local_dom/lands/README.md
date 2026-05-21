# LandS Pyomo deterministic equivalent

This folder contains a data-driven Pyomo implementation of the two-stage LandS stochastic LP.

Files:

- `lands_data.json`: instance data, scenario probabilities, expected reference solution, and EV/EEV/VSS reference values.
- `lands_model.py`: model class that builds and solves the deterministic equivalent, solves the expected-value problem, evaluates EEV, computes VSS, and saves diagnostics.
- `lands_diagnostics.json`: generated diagnostics output with first-stage and second-stage vectors, feasibility slacks, RP/EV/EEV/VSS metrics, and scenario costs.
- `test_lands.py`: regression tests checking the reported stochastic solution and EV/EEV/VSS values.
- `requirements.txt`: Python dependencies.

Install and run:

```bash
pip install -r requirements.txt
python lands_model.py
python test_lands.py
```

The main script writes:

```text
lands_diagnostics.json
```

## Stochastic-program solution

Expected first-stage solution:

```text
x[1] = 8/3
x[2] = 4
x[3] = 10/3
x[4] = 2
RP objective = 381.85333333333335
```

## Expected-value diagnostics

Here the only random parameter is demand of mode 1:

```text
E[d_1] = 0.3(3) + 0.4(5) + 0.3(7) = 5
E[d_2] = 3
E[d_3] = 2
```

Expected-value first-stage solution from the deterministic EV problem:

```text
x_EV[1] = 0.8333333333333333
x_EV[2] = 3
x_EV[3] = 4.166666666666666
x_EV[4] = 4
EV objective = 378.6666666666667
```

Evaluating this fixed `x_EV` in the original stochastic model gives:

```text
EEV = 383.9866666666667
VSS = EEV - RP = 2.1333333333333258
```

For this minimization problem, positive VSS means the stochastic model improves over the policy obtained from the expected-value approximation.
