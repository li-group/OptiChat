# mptsp.py
from pathlib import Path
import re
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix, vstack

base = Path.cwd() / 'MPTSPs_D0_50'

def load_instance(base: Path):
    lines = base.joinpath('prob.txt').read_text().splitlines()
    N = K = None
    cbar_entries = []
    section = None
    coords = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('DIMENSION'):
            N = int(line.split()[-1])
        elif line.startswith('N_PATH'):
            K = int(line.split()[-1])
        elif line == 'NODE_COORD_SECTION':
            section = 'nodes'
            continue
        elif line == 'EDGE_WEIGHT_SECTION':
            section = 'edges'
            continue
        elif section == 'nodes':
            parts = line.split()
            if len(parts) >= 3:
                coords.append((int(parts[0]) - 1, float(parts[1]), float(parts[2])))
        elif section == 'edges':
            parts = line.split()
            if len(parts) >= 3:
                cbar_entries.append((int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])))
    cbar = np.zeros((N, N), dtype=float)
    for i, j, w in cbar_entries:
        cbar[i, j] = w
    scen_files = sorted(base.glob('Scenario*.dat'), key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
    C = np.empty((len(scen_files), N, N, K), dtype=float)
    for s, p in enumerate(scen_files):
        vals = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line == 'C_ijk':
                continue
            vals.append(float(line))
        if len(vals) != N * N * K:
            raise ValueError(f'{p.name}: expected {N*N*K} costs, found {len(vals)}')
        C[s] = np.array(vals, dtype=float).reshape(N, N, K)
    Q = C.min(axis=3).mean(axis=0)
    np.fill_diagonal(Q, 1e9)  # objective penalty; diagonal is also fixed to zero below
    return N, K, cbar, C, Q, coords, scen_files

def subtours_from_edges(edges, N):
    succ = {i: j for i, j in edges}
    unseen = set(range(N))
    tours = []
    while unseen:
        start = next(iter(unseen))
        tour = []
        cur = start
        while cur in unseen:
            unseen.remove(cur)
            tour.append(cur)
            cur = succ[cur]
        tours.append(tour)
    return tours

def solve_tsp_with_cuts(Q):
    N = Q.shape[0]
    nvar = N * N
    c = Q.reshape(-1)
    rows = []
    lb = []
    ub = []
    # assignment constraints
    A = lil_matrix((2 * N, nvar), dtype=float)
    b = np.ones(2 * N)
    for i in range(N):
        for j in range(N):
            A[i, i * N + j] = 1.0
    for j in range(N):
        for i in range(N):
            A[N + j, i * N + j] = 1.0
    A = A.tocsr()
    lower = b.copy()
    upper = b.copy()
    # bounds and integrality
    lo = np.zeros(nvar)
    hi = np.ones(nvar)
    for i in range(N):
        hi[i * N + i] = 0.0
    bounds = Bounds(lo, hi)
    integrality = np.ones(nvar, dtype=int)
    cuts = []
    for it in range(1, 101):
        if cuts:
            Ac = lil_matrix((len(cuts), nvar), dtype=float)
            for r, T in enumerate(cuts):
                T = set(T)
                for i in T:
                    for j in range(N):
                        if j not in T:
                            Ac[r, i * N + j] = 1.0
            A_all = vstack([A, Ac.tocsr()], format='csr')
            lb_all = np.concatenate([lower, np.ones(len(cuts))])
            ub_all = np.concatenate([upper, np.full(len(cuts), np.inf)])
        else:
            A_all = A
            lb_all = lower
            ub_all = upper
        res = milp(c=c, integrality=integrality, bounds=bounds,
                   constraints=LinearConstraint(A_all, lb_all, ub_all),
                   options={'time_limit': 120, 'mip_rel_gap': 0.0, 'disp': False})
        if not res.success:
            raise RuntimeError(f'MILP failed at iteration {it}: {res.message}')
        y = res.x.reshape(N, N)
        edges = [(i, int(np.argmax(y[i]))) for i in range(N)]
        tours = subtours_from_edges(edges, N)
        tours.sort(key=len)
        print(f'iter={it:02d} obj={res.fun:.4f} subtours={len(tours)} sizes={[len(t) for t in tours]} cuts={len(cuts)}')
        if len(tours) == 1:
            return res.fun, edges, tours[0], y, it, len(cuts)
        # add one cut for every proper subtour found
        for T in tours:
            if len(T) < N:
                cuts.append(T)
    raise RuntimeError('Reached iteration limit')

def make_cost_matrices(C):
    """
    C shape: (S, N, N, K)

    Q_rp[i,j] = average over scenarios of the best path in each scenario.
                 This is the reduced recourse-problem cost.

    Q_ev[i,j] = best path after replacing each path cost by its average value.
                 This is the expected-value problem cost.
    """
    S, N, _, K = C.shape

    # RP reduced stochastic cost:
    # For each scenario, choose cheapest path k, then average over scenarios.
    Q_rp = C.min(axis=3).mean(axis=0)

    # EV deterministic cost:
    # First average each path over scenarios, then choose cheapest expected path.
    C_mean = C.mean(axis=0)          # shape: (N, N, K)
    Q_ev = C_mean.min(axis=2)        # shape: (N, N)

    # Disallow self loops.
    np.fill_diagonal(Q_rp, 1e9)
    np.fill_diagonal(Q_ev, 1e9)

    return Q_rp, Q_ev


def edge_cost(edges, Q):
    """Cost of a first-stage tour under arc-cost matrix Q."""
    return sum(Q[i, j] for i, j in edges)


def recover_ev_path_choices(edges, C):
    """
    For the EV deterministic problem, recover which path k is chosen
    using the expected path costs.

    Returns:
        ev_path_choice[(i,j)] = k_star
    """
    C_mean = C.mean(axis=0)  # shape: (N, N, K)
    ev_path_choice = {}

    for i, j in edges:
        k_star = int(np.argmin(C_mean[i, j, :]))
        ev_path_choice[(i, j)] = k_star

    return ev_path_choice


def recover_second_stage_choices(edges, C):
    """
    Given a fixed first-stage tour, recover the true second-stage recourse
    decisions for every scenario.

    Returns:
        x_choice[(s,i,j)] = k_star

    This means that in scenario s, for selected edge i -> j,
    path k_star is chosen.
    """
    S, N, _, K = C.shape
    x_choice = {}

    for s in range(S):
        for i, j in edges:
            k_star = int(np.argmin(C[s, i, j, :]))
            x_choice[(s, i, j)] = k_star

    return x_choice

import csv
import json


RP_REFERENCE_OBJ = 23544.19
EV_REFERENCE_OBJ = 63607.37


def read_solution_file(path: Path, N: int):
    """
    Reads RP.txt, EV.txt, EEV.txt, etc.

    Format:
        first line: objective value
        following lines: i j selected edge

    Returns:
        file_obj: float
        edges: list[(i,j)]
        x: N x N binary matrix
    """
    lines = path.read_text().splitlines()

    file_obj = float(lines[0].strip())
    edges = []

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        i, j = map(int, line.split())
        edges.append((i, j))

    if len(edges) != N:
        raise ValueError(f"{path.name}: expected {N} selected edges, found {len(edges)}")

    x = edges_to_matrix(edges, N)
    validate_tour_edges(edges, N, name=path.name)

    return file_obj, edges, x


def edges_to_matrix(edges, N):
    """
    Convert selected edges into binary first-stage decision matrix x[i,j].
    """
    x = np.zeros((N, N), dtype=int)

    for i, j in edges:
        x[i, j] = 1

    return x


def matrix_to_edges(x):
    """
    Convert binary decision matrix x[i,j] into edge list.
    """
    edges = []

    N = x.shape[0]
    for i in range(N):
        for j in range(N):
            if int(round(x[i, j])) == 1:
                edges.append((i, j))

    return edges


def validate_tour_edges(edges, N, name="solution"):
    """
    Basic TSP tour validation:
      - exactly N edges
      - each node has exactly one outgoing edge
      - each node has exactly one incoming edge
      - no self loops
    """
    if len(edges) != N:
        raise ValueError(f"{name}: expected {N} edges, found {len(edges)}")

    out_count = np.zeros(N, dtype=int)
    in_count = np.zeros(N, dtype=int)

    for i, j in edges:
        if i == j:
            raise ValueError(f"{name}: self-loop found at node {i}")
        out_count[i] += 1
        in_count[j] += 1

    bad_out = np.where(out_count != 1)[0]
    bad_in = np.where(in_count != 1)[0]

    if len(bad_out) > 0:
        raise ValueError(f"{name}: nodes with bad outgoing degree: {bad_out.tolist()}")

    if len(bad_in) > 0:
        raise ValueError(f"{name}: nodes with bad incoming degree: {bad_in.tolist()}")


def find_and_store_x_decisions(base, N, out_dir):
    """
    a. Find and store x^star and x^EV decisions.

    Here x means the first-stage tour decision:
        x[i,j] = 1 if edge i -> j is selected.

    x^star is read from RP.txt.
    x^EV is read from EV.txt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rp_file_obj, xstar_edges, xstar = read_solution_file(base / "RP.txt", N)
    ev_file_obj, xev_edges, xev = read_solution_file(base / "EV.txt", N)

    store_edges_csv(out_dir / "x_star_edges.csv", xstar_edges)
    store_edges_csv(out_dir / "x_ev_edges.csv", xev_edges)

    np.savetxt(out_dir / "x_star_matrix.csv", xstar, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "x_ev_matrix.csv", xev, fmt="%d", delimiter=",")

    return {
        "rp_file_obj": rp_file_obj,
        "ev_file_obj": ev_file_obj,
        "xstar_edges": xstar_edges,
        "xev_edges": xev_edges,
        "xstar": xstar,
        "xev": xev,
    }


def store_edges_csv(path, edges):
    """
    Store selected first-stage edges.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "value"])

        for i, j in edges:
            writer.writerow([i, j, 1])


def first_stage_cost(edges, cbar):
    """
    b. Compute c^T x for a given first-stage tour.
    """
    return float(sum(cbar[i, j] for i, j in edges))


def compute_and_store_first_stage_costs(out_dir, xstar_edges, xev_edges, cbar):
    """
    b. Find and store first-stage costs for x^star and x^EV.
    """
    cTx_star = first_stage_cost(xstar_edges, cbar)
    cTx_ev = first_stage_cost(xev_edges, cbar)

    path = out_dir / "first_stage_costs.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solution", "first_stage_cost_cTx"])
        writer.writerow(["x_star", cTx_star])
        writer.writerow(["x_EV", cTx_ev])

    return cTx_star, cTx_ev


def scenario_recourse_raw(edges, cbar, C):
    """
    Compute raw scenario recourse values.

    Raw Q_s(x) is:

        Q_s(x) = sum_{(i,j) selected}
                 [ min_k C[s,i,j,k] - cbar[i,j] ]

    This is the second-stage deviation term for scenario s.

    Shape:
        returns array of length S
    """
    S = C.shape[0]
    Q_raw = np.zeros(S, dtype=float)

    for s in range(S):
        total = 0.0

        for i, j in edges:
            best_path_cost = float(np.min(C[s, i, j, :]))
            total += best_path_cost - cbar[i, j]

        Q_raw[s] = total

    return Q_raw


def scenario_total_realized_cost(edges, C):
    """
    Scenario realized cost after recourse:

        realized_s(x) = sum_{(i,j) selected} min_k C[s,i,j,k]

    This equals c^T x + raw Q_s(x).
    """
    S = C.shape[0]
    realized = np.zeros(S, dtype=float)

    for s in range(S):
        total = 0.0

        for i, j in edges:
            total += float(np.min(C[s, i, j, :]))

        realized[s] = total

    return realized


def compute_and_store_Qs(out_dir, xstar_edges, xev_edges, cbar, C, probabilities=None):
    """
    c. Find and store Q_s(x) for x^star and x^EV.

    Important convention:

    - Q_raw_s is the unweighted scenario recourse deviation.
    - Q_s is the probability-weighted contribution used in the objective.

    Therefore:

        objective = c^T x + sum_s Q_s(x)

    where:

        Q_s(x) = p_s * Q_raw_s(x)
    """
    S = C.shape[0]

    if probabilities is None:
        probabilities = np.full(S, 1.0 / S, dtype=float)
    else:
        probabilities = np.asarray(probabilities, dtype=float)

    if len(probabilities) != S:
        raise ValueError(f"Expected {S} probabilities, got {len(probabilities)}")

    Qstar_raw = scenario_recourse_raw(xstar_edges, cbar, C)
    Qev_raw = scenario_recourse_raw(xev_edges, cbar, C)

    Qstar = probabilities * Qstar_raw
    Qev = probabilities * Qev_raw

    realized_star = scenario_total_realized_cost(xstar_edges, C)
    realized_ev = scenario_total_realized_cost(xev_edges, C)

    path = out_dir / "scenario_Q_values.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "probability",
            "Q_raw_x_star",
            "Q_weighted_x_star",
            "realized_cost_x_star",
            "Q_raw_x_EV",
            "Q_weighted_x_EV",
            "realized_cost_x_EV",
        ])

        for s in range(S):
            writer.writerow([
                s + 1,
                probabilities[s],
                Qstar_raw[s],
                Qstar[s],
                realized_star[s],
                Qev_raw[s],
                Qev[s],
                realized_ev[s],
            ])

    return {
        "probabilities": probabilities,
        "Qstar_raw": Qstar_raw,
        "Qev_raw": Qev_raw,
        "Qstar": Qstar,
        "Qev": Qev,
        "realized_star": realized_star,
        "realized_ev": realized_ev,
    }


def compute_and_store_delta_first_stage(out_dir, cTx_star, cTx_ev):
    """
    d. Compute and store:

        delta_first_stage = c^T x^EV - c^T x^star
    """
    delta_first_stage = float(cTx_ev - cTx_star)

    path = out_dir / "delta_first_stage.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        writer.writerow(["delta_first_stage", delta_first_stage])

    return delta_first_stage


def compute_and_store_delta_s_and_labels(out_dir, Q_data):
    """
    e and g. Compute per-scenario delta_s and label each scenario.

    Delta definition:

        Delta_s = Q_s(x^EV) - Q_s(x^star)

    Scenario labels:

        good if Delta_s <= 0
        bad  if Delta_s > 0

    Since Q_weighted_s = p_s * Q_raw_s, the sign is the same for raw
    and weighted deltas when p_s > 0.
    """
    Qstar_raw = Q_data["Qstar_raw"]
    Qev_raw = Q_data["Qev_raw"]
    Qstar = Q_data["Qstar"]
    Qev = Q_data["Qev"]
    probabilities = Q_data["probabilities"]

    delta_raw = Qev_raw - Qstar_raw
    delta_weighted = Qev - Qstar

    labels = np.where(delta_raw <= 0.0, "good", "bad")

    good_set = [int(s + 1) for s, label in enumerate(labels) if label == "good"]
    bad_set = [int(s + 1) for s, label in enumerate(labels) if label == "bad"]

    path = out_dir / "scenario_deltas_and_labels.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario",
            "probability",
            "Q_raw_x_star",
            "Q_raw_x_EV",
            "delta_raw",
            "Q_weighted_x_star",
            "Q_weighted_x_EV",
            "delta_weighted",
            "label",
        ])

        for s in range(len(labels)):
            writer.writerow([
                s + 1,
                probabilities[s],
                Qstar_raw[s],
                Qev_raw[s],
                delta_raw[s],
                Qstar[s],
                Qev[s],
                delta_weighted[s],
                labels[s],
            ])

    return {
        "delta_raw": delta_raw,
        "delta_weighted": delta_weighted,
        "labels": labels,
        "good_set": good_set,
        "bad_set": bad_set,
    }


def ev_deterministic_objective(edges, C):
    """
    EV objective for a fixed EV tour.

    This is:

        sum_{(i,j) selected} min_k E_s[C[s,i,j,k]]

    This should match 63607.37 for the EV solution.
    """
    C_mean = C.mean(axis=0)  # shape: N x N x K

    return float(sum(np.min(C_mean[i, j, :]) for i, j in edges))


def check_reference_objectives(
    cTx_star,
    cTx_ev,
    Q_data,
    xev_edges,
    C,
    rp_reference=RP_REFERENCE_OBJ,
    ev_reference=EV_REFERENCE_OBJ,
    tol=1e-2,
):
    """
    f. Check the hard-coded verified objectives.

    For x^star:

        c^T x^star + sum_s Q_s(x^star) == 23544.19

    For x^EV:

        EV deterministic objective == 63607.37

    Also computes EEV:

        c^T x^EV + sum_s Q_s(x^EV)
    """
    rp_obj = float(cTx_star + np.sum(Q_data["Qstar"]))
    eev_obj = float(cTx_ev + np.sum(Q_data["Qev"]))
    ev_obj = ev_deterministic_objective(xev_edges, C)

    print("\nObjective checks")
    print("----------------")
    print(f"RP check:  c^T x_star + sum_s Q_s(x_star) = {rp_obj:.6f}")
    print(f"Expected RP reference                    = {rp_reference:.6f}")
    print(f"EV check:  deterministic EV objective    = {ev_obj:.6f}")
    print(f"Expected EV reference                    = {ev_reference:.6f}")
    print(f"EEV:       c^T x_EV + sum_s Q_s(x_EV)    = {eev_obj:.6f}")

    if not np.isclose(rp_obj, rp_reference, atol=tol):
        raise AssertionError(
            f"RP objective mismatch: got {rp_obj}, expected {rp_reference}"
        )

    if not np.isclose(ev_obj, ev_reference, atol=tol):
        raise AssertionError(
            f"EV objective mismatch: got {ev_obj}, expected {ev_reference}"
        )

    return {
        "rp_obj": rp_obj,
        "ev_obj": ev_obj,
        "eev_obj": eev_obj,
    }


def write_summary_json(
    out_dir,
    cTx_star,
    cTx_ev,
    delta_first_stage,
    objective_data,
    delta_data,
    min_bad_data,
):
    """
    Store final summary.
    """
    summary = {
        "first_stage_cost_x_star": float(cTx_star),
        "first_stage_cost_x_EV": float(cTx_ev),
        "delta_first_stage": float(delta_first_stage),

        "RP_objective_x_star": float(objective_data["rp_obj"]),
        "EV_deterministic_objective_x_EV": float(objective_data["ev_obj"]),
        "EEV_true_expected_cost_x_EV": float(objective_data["eev_obj"]),
        "VSS": float(objective_data["eev_obj"] - objective_data["rp_obj"]),

        "number_good_scenarios": len(delta_data["good_set"]),
        "number_bad_scenarios": len(delta_data["bad_set"]),
        "good_scenarios_1_based": delta_data["good_set"],
        "bad_scenarios_1_based": delta_data["bad_set"],

        "C_comp": float(min_bad_data["C_comp"]),
        "bad_scenario_weighted_sum": float(min_bad_data["bad_total"]),
        "VSS_decomposed": float(min_bad_data["VSS_decomposed"]),
        "minimum_bad_set_size": int(min_bad_data["minimum_bad_set_size"]),
        "minimum_bad_set_1_based": min_bad_data["selected_bad_scenarios_1_based"],
        "minimum_bad_set_weighted_sum": float(min_bad_data["selected_bad_weighted_sum"]),
        "C_comp_plus_minimum_bad_set": float(min_bad_data["C_comp_plus_selected_sum"]),
    }

    path = out_dir / "summary.json"

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_all_inspections(base: Path):
    """
    Runs a-g.
    """
    N, K, cbar, C, Q, coords, scen_files = load_instance(base)

    print(f"Loaded N={N}, K={K}, scenarios={len(scen_files)}")

    out_dir = base / "inspection_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # a. x^star and x^EV decisions
    decisions = find_and_store_x_decisions(base, N, out_dir)

    xstar_edges = decisions["xstar_edges"]
    xev_edges = decisions["xev_edges"]

    # b. first-stage costs
    cTx_star, cTx_ev = compute_and_store_first_stage_costs(
        out_dir,
        xstar_edges,
        xev_edges,
        cbar,
    )

    # c. Q_s(x) values
    Q_data = compute_and_store_Qs(
        out_dir,
        xstar_edges,
        xev_edges,
        cbar,
        C,
    )

    # d. delta first stage
    delta_first_stage = compute_and_store_delta_first_stage(
        out_dir,
        cTx_star,
        cTx_ev,
    )

    # e and g. per-scenario deltas and labels
    delta_data = compute_and_store_delta_s_and_labels(out_dir, Q_data)

    # h. minimum bad scenario set
    min_bad_data = compute_and_store_minimum_bad_scenario_set(
        out_dir,
        delta_first_stage,
        Q_data,
        delta_data,
    )

    # i. Build infeasible K-system and save IIS
    iis_data = build_and_save_iis_for_min_bad_set(
        base=base,
        out_dir=out_dir,
        xev_edges=xev_edges,
        C=C,
        cbar=cbar,
        Q_data=Q_data,
        min_bad_data=min_bad_data,
    )

    # f. hard-coded reference objective checks
    objective_data = check_reference_objectives(
        cTx_star,
        cTx_ev,
        Q_data,
        xev_edges,
        C,
    )

    # summary
    summary = write_summary_json(
        out_dir,
        cTx_star,
        cTx_ev,
        delta_first_stage,
        objective_data,
        delta_data,
        min_bad_data,
    )

    print("\nSummary")
    print("-------")
    print(f"c^T x_star: {cTx_star:.6f}")
    print(f"c^T x_EV:   {cTx_ev:.6f}")
    print(f"delta_first_stage: {delta_first_stage:.6f}")
    print(f"RP objective:  {objective_data['rp_obj']:.6f}")
    print(f"EV objective:  {objective_data['ev_obj']:.6f}")
    print(f"EEV objective: {objective_data['eev_obj']:.6f}")
    print(f"VSS:           {objective_data['eev_obj'] - objective_data['rp_obj']:.6f}")
    print(f"Good scenarios: {len(delta_data['good_set'])}")
    print(f"Bad scenarios:  {len(delta_data['bad_set'])}")
    print(f"C_comp: {min_bad_data['C_comp']:.6f}")
    print(f"Bad weighted total: {min_bad_data['bad_total']:.6f}")
    print(f"VSS decomposed: {min_bad_data['VSS_decomposed']:.6f}")
    print(f"Minimum bad set size: {min_bad_data['minimum_bad_set_size']}")
    print(f"Minimum bad set, 1-based: {min_bad_data['selected_bad_scenarios_1_based']}")
    print(f"C_comp + selected bad sum: {min_bad_data['C_comp_plus_selected_sum']:.6f}")

    print(f"\nWrote outputs to: {out_dir}")

    return {
        "decisions": decisions,
        "first_stage": {
            "cTx_star": cTx_star,
            "cTx_ev": cTx_ev,
            "delta_first_stage": delta_first_stage,
        },
        "Q_data": Q_data,
        "delta_data": delta_data,
        "min_bad_data": min_bad_data,
        "iis_data": iis_data,
        "objective_data": objective_data,
        "summary": summary,
    }

def compute_and_store_minimum_bad_scenario_set(
    out_dir,
    delta_first_stage,
    Q_data,
    delta_data,
    strict_tol=1e-10,
):
    """
    Find and store the minimum bad scenario set K.

    Definitions:

        Delta_0 = c^T x_EV - c^T x_star

        Delta_s = Q_raw_s(x_EV) - Q_raw_s(x_star)

        weighted_delta_s = p_s * Delta_s

        G = {s : Delta_s <= 0}
        B = {s : Delta_s > 0}

        C_comp = Delta_0 + sum_{s in G} p_s Delta_s

    We seek the smallest K subset of B such that:

        sum_{s in K} p_s Delta_s + C_comp > 0

    Since every bad scenario has positive weighted regret, the smallest-cardinality
    set is obtained by sorting bad scenarios by p_s Delta_s descending and
    taking the smallest prefix satisfying the condition.

    Scenario ids stored in output files are 1-based.
    """

    probabilities = Q_data["probabilities"]

    # delta_raw_s = Delta_s
    delta_raw = delta_data["delta_raw"]

    # weighted_delta_s = p_s Delta_s
    weighted_delta = probabilities * delta_raw

    labels = delta_data["labels"]

    good_indices = [s for s, label in enumerate(labels) if label == "good"]
    bad_indices = [s for s, label in enumerate(labels) if label == "bad"]

    # Compensation from first-stage difference and good scenarios
    C_comp = float(delta_first_stage + np.sum(weighted_delta[good_indices]))

    # Full VSS decomposition
    bad_total = float(np.sum(weighted_delta[bad_indices]))
    VSS_decomposed = float(C_comp + bad_total)

    # If compensation is already positive, the empty set satisfies the constraint.
    selected_bad = []
    selected_sum = 0.0

    if C_comp > strict_tol:
        selected_bad = []
        selected_sum = 0.0
    else:
        # Sort bad scenarios by p_s Delta_s descending.
        bad_sorted = sorted(
            bad_indices,
            key=lambda s: weighted_delta[s],
            reverse=True,
        )

        for s in bad_sorted:
            selected_bad.append(s)
            selected_sum += float(weighted_delta[s])

            if selected_sum + C_comp > strict_tol:
                break

        if selected_sum + C_comp <= strict_tol:
            raise RuntimeError(
                "Could not find a bad scenario subset satisfying the condition. "
                f"C_comp={C_comp}, bad_total={bad_total}, "
                f"C_comp + bad_total={C_comp + bad_total}"
            )

    selected_bad_set_1_based = [int(s + 1) for s in selected_bad]

    # Store ranked bad scenarios
    ranked_path = out_dir / "bad_scenarios_ranked.csv"

    bad_sorted_all = sorted(
        bad_indices,
        key=lambda s: weighted_delta[s],
        reverse=True,
    )

    cumulative = 0.0

    with open(ranked_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "scenario",
            "probability",
            "Delta_s_raw",
            "p_s_Delta_s",
            "cumulative_bad_contribution",
            "C_comp_plus_cumulative",
            "selected_in_min_bad_set",
        ])

        for rank, s in enumerate(bad_sorted_all, start=1):
            cumulative += float(weighted_delta[s])

            writer.writerow([
                rank,
                s + 1,
                probabilities[s],
                delta_raw[s],
                weighted_delta[s],
                cumulative,
                C_comp + cumulative,
                int(s in selected_bad),
            ])

    # Store minimum bad set
    min_set_path = out_dir / "minimum_bad_scenario_set.csv"

    cumulative = 0.0

    with open(min_set_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "selection_order",
            "scenario",
            "probability",
            "Delta_s_raw",
            "p_s_Delta_s",
            "cumulative_selected_contribution",
            "C_comp_plus_cumulative",
        ])

        for order, s in enumerate(selected_bad, start=1):
            cumulative += float(weighted_delta[s])

            writer.writerow([
                order,
                s + 1,
                probabilities[s],
                delta_raw[s],
                weighted_delta[s],
                cumulative,
                C_comp + cumulative,
            ])

    # Store decomposition summary
    summary_path = out_dir / "minimum_bad_scenario_summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        writer.writerow(["Delta_0_first_stage", delta_first_stage])
        writer.writerow(["good_scenario_weighted_sum", float(np.sum(weighted_delta[good_indices]))])
        writer.writerow(["C_comp", C_comp])
        writer.writerow(["bad_scenario_weighted_sum", bad_total])
        writer.writerow(["VSS_decomposed", VSS_decomposed])
        writer.writerow(["minimum_bad_set_size", len(selected_bad_set_1_based)])
        writer.writerow(["minimum_bad_set_1_based", selected_bad_set_1_based])
        writer.writerow(["selected_bad_weighted_sum", selected_sum])
        writer.writerow(["C_comp_plus_selected_sum", C_comp + selected_sum])

    return {
        "C_comp": C_comp,
        "bad_total": bad_total,
        "VSS_decomposed": VSS_decomposed,
        "selected_bad_indices_0_based": selected_bad,
        "selected_bad_scenarios_1_based": selected_bad_set_1_based,
        "minimum_bad_set_size": len(selected_bad_set_1_based),
        "selected_bad_weighted_sum": float(selected_sum),
        "C_comp_plus_selected_sum": float(C_comp + selected_sum),
    }


def build_and_save_iis_for_min_bad_set(
    base,
    out_dir,
    xev_edges,
    C,
    cbar,
    Q_data,
    min_bad_data,
    model_name="K_system_min_bad_IIS",
    feasibility_tol=1e-9,
):
    """
    Build the infeasible K-system and save its IIS.

    Mathematical system:

        find {y_s}_{s in K}

        s.t.
            for each s in K and each EV edge (i,j):
                sum_k y[s,i,j,k] = 1

            budget:
                sum_{s in K} p_s q_s^T y_s
                <= sum_{s in K} p_s Q_s(x_star) - C_comp

    Here:

        q_s[i,j,k] = C[s,i,j,k] - cbar[i,j]

    and:

        Q_s(x_star) = sum_{(i,j) in x_star} min_k q_s[i,j,k]

    The model is infeasible because K was selected so that:

        sum_{s in K} p_s Delta_s + C_comp > 0

    where:

        Delta_s = Q_s(x_EV) - Q_s(x_star)

    Outputs:
        K_system_full.lp
        K_system_full.mps
        K_system_iis.ilp
        K_system_iis_constraints.csv
        K_system_iis_variables.csv
        K_system_iis_summary.json
    """

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "This IIS extraction function requires gurobipy. "
            "Install/use Gurobi's Python package in the same environment."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)

    probabilities = Q_data["probabilities"]
    Qstar_raw = Q_data["Qstar_raw"]
    Qev_raw = Q_data["Qev_raw"]

    K_indices = list(min_bad_data["selected_bad_indices_0_based"])
    C_comp = float(min_bad_data["C_comp"])

    S, N, _, num_paths = C.shape

    if len(K_indices) == 0:
        raise ValueError(
            "Minimum bad scenario set K is empty. "
            "The current IIS builder expects at least one selected bad scenario."
        )

    # ------------------------------------------------------------
    # Budget RHS
    # ------------------------------------------------------------
    # RHS = sum_{s in K} p_s Q_s(x_star) - C_comp
    budget_rhs = float(
        sum(probabilities[s] * Qstar_raw[s] for s in K_indices) - C_comp
    )

    # Minimum possible LHS under x_EV recourse:
    # min_y sum_{s in K} p_s q_s^T y_s
    # = sum_{s in K} p_s Q_s(x_EV)
    min_possible_lhs = float(
        sum(probabilities[s] * Qev_raw[s] for s in K_indices)
    )

    infeasibility_margin = float(min_possible_lhs - budget_rhs)

    print("\nBuilding K-system IIS model")
    print("---------------------------")
    print(f"Selected K scenarios, 1-based: {[s + 1 for s in K_indices]}")
    print(f"C_comp: {C_comp:.10f}")
    print(f"Budget RHS: {budget_rhs:.10f}")
    print(f"Minimum possible budget LHS: {min_possible_lhs:.10f}")
    print(f"Infeasibility margin LHS_min - RHS: {infeasibility_margin:.10f}")

    if infeasibility_margin <= feasibility_tol:
        print(
            "WARNING: infeasibility margin is not clearly positive. "
            "Gurobi may not report the model as infeasible."
        )

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------
    m = gp.Model(model_name)
    m.Params.OutputFlag = 1
    m.Params.FeasibilityTol = feasibility_tol
    m.Params.IntFeasTol = feasibility_tol

    y = {}

    for s in K_indices:
        for i, j in xev_edges:
            for k in range(num_paths):
                y[s, i, j, k] = m.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_s{s+1}_i{i}_j{j}_k{k}",
                )

    m.update()

    # ------------------------------------------------------------
    # Recourse feasibility constraints for fixed x_EV
    # ------------------------------------------------------------
    # Since x_EV fixes the selected tour edges, each selected edge
    # must choose exactly one path in each scenario.
    recourse_constrs = {}

    for s in K_indices:
        for i, j in xev_edges:
            cname = f"rec_s{s+1}_edge_{i}_{j}_choose_one"

            recourse_constrs[s, i, j] = m.addConstr(
                gp.quicksum(y[s, i, j, k] for k in range(num_paths)) == 1.0,
                name=cname,
            )

    # ------------------------------------------------------------
    # Budget constraint
    # ------------------------------------------------------------
    budget_lhs = gp.quicksum(
        probabilities[s] * (C[s, i, j, k] - cbar[i, j]) * y[s, i, j, k]
        for s in K_indices
        for i, j in xev_edges
        for k in range(num_paths)
    )

    budget_constr = m.addConstr(
        budget_lhs <= budget_rhs,
        name="budget_K",
    )

    m.setObjective(0.0, GRB.MINIMIZE)
    m.update()

    # Save full model before solving
    full_lp_path = out_dir / "K_system_full.lp"
    full_mps_path = out_dir / "K_system_full.mps"

    m.write(str(full_lp_path))
    m.write(str(full_mps_path))

    print(f"\nWrote full K-system model to:")
    print(full_lp_path)
    print(full_mps_path)

    # ------------------------------------------------------------
    # Optimize feasibility model
    # ------------------------------------------------------------
    m.optimize()

    status = m.Status

    if status != GRB.INFEASIBLE:
        raise RuntimeError(
            f"K-system was expected to be infeasible, but Gurobi status is {status}. "
            "Check K selection, Q_s convention, and budget definition."
        )

    print("\nModel is infeasible. Computing IIS...")

    # ------------------------------------------------------------
    # Compute and save IIS
    # ------------------------------------------------------------
    m.computeIIS()

    iis_ilp_path = out_dir / "K_system_iis.ilp"
    m.write(str(iis_ilp_path))

    print(f"Wrote IIS subsystem to:")
    print(iis_ilp_path)

    # ------------------------------------------------------------
    # Save IIS constraints table
    # ------------------------------------------------------------
    iis_constraints_path = out_dir / "K_system_iis_constraints.csv"

    with open(iis_constraints_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "constraint_name",
            "in_IIS",
            "sense",
            "rhs",
        ])

        for constr in m.getConstrs():
            writer.writerow([
                constr.ConstrName,
                int(constr.IISConstr),
                constr.Sense,
                constr.RHS,
            ])

    # ------------------------------------------------------------
    # Save IIS variable bounds table
    # ------------------------------------------------------------
    iis_variables_path = out_dir / "K_system_iis_variables.csv"

    with open(iis_variables_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variable_name",
            "lower_bound",
            "upper_bound",
            "IIS_lower_bound",
            "IIS_upper_bound",
            "vtype",
        ])

        for var in m.getVars():
            writer.writerow([
                var.VarName,
                var.LB,
                var.UB,
                int(var.IISLB),
                int(var.IISUB),
                var.VType,
            ])

    # ------------------------------------------------------------
    # Extract compact IIS summary
    # ------------------------------------------------------------
    iis_constraint_names = [
        constr.ConstrName
        for constr in m.getConstrs()
        if constr.IISConstr
    ]

    iis_recourse_constraints = [
        name for name in iis_constraint_names
        if name.startswith("rec_s")
    ]

    iis_budget_included = "budget_K" in iis_constraint_names

    # Parse scenario ids from IIS recourse constraints
    iis_scenarios = set()
    iis_edges = []

    for name in iis_recourse_constraints:
        # rec_s12_edge_4_9_choose_one
        parts = name.split("_")
        s_part = parts[1]      # s12
        i_part = parts[3]
        j_part = parts[4]

        s_id = int(s_part[1:])
        i_id = int(i_part)
        j_id = int(j_part)

        iis_scenarios.add(s_id)
        iis_edges.append((s_id, i_id, j_id))

    iis_summary = {
        "K_scenarios_1_based": [int(s + 1) for s in K_indices],
        "K_size": len(K_indices),
        "C_comp": C_comp,
        "budget_rhs": budget_rhs,
        "minimum_possible_lhs": min_possible_lhs,
        "infeasibility_margin": infeasibility_margin,
        "iis_budget_constraint_included": bool(iis_budget_included),
        "number_constraints_in_IIS": len(iis_constraint_names),
        "number_recourse_constraints_in_IIS": len(iis_recourse_constraints),
        "IIS_scenarios_1_based": sorted(int(s) for s in iis_scenarios),
        "IIS_recourse_edges": [
            {
                "scenario": int(s_id),
                "i": int(i_id),
                "j": int(j_id),
            }
            for s_id, i_id, j_id in iis_edges
        ],
    }

    iis_summary_path = out_dir / "K_system_iis_summary.json"

    with open(iis_summary_path, "w") as f:
        json.dump(iis_summary, f, indent=2)

    print(f"Wrote IIS constraint table to:")
    print(iis_constraints_path)

    print(f"Wrote IIS variable table to:")
    print(iis_variables_path)

    print(f"Wrote IIS summary to:")
    print(iis_summary_path)

    print("\nIIS summary")
    print("-----------")
    print(f"Budget constraint in IIS: {iis_budget_included}")
    print(f"Constraints in IIS: {len(iis_constraint_names)}")
    print(f"Recourse constraints in IIS: {len(iis_recourse_constraints)}")
    print(f"IIS scenarios, 1-based: {sorted(iis_scenarios)}")

    return {
        "model": m,
        "K_indices": K_indices,
        "budget_rhs": budget_rhs,
        "minimum_possible_lhs": min_possible_lhs,
        "infeasibility_margin": infeasibility_margin,
        "iis_constraint_names": iis_constraint_names,
        "iis_summary": iis_summary,
        "paths": {
            "full_lp": full_lp_path,
            "full_mps": full_mps_path,
            "iis_ilp": iis_ilp_path,
            "iis_constraints_csv": iis_constraints_path,
            "iis_variables_csv": iis_variables_path,
            "iis_summary_json": iis_summary_path,
        },
    }



if __name__ == "__main__":
    results = run_all_inspections(base)








# if __name__ == '__main__':
#     N, K, cbar, C, Q_old, coords, scen_files = load_instance(base)

#     print(f'Loaded N={N}, K={K}, scenarios={len(scen_files)}')

#     Q_rp, Q_ev = make_cost_matrices(C)

#     # ------------------------------------------------------------
#     # Solve RP reduced problem
#     # This should reproduce the stochastic optimum around 23544.2
#     # ------------------------------------------------------------
#     print('\nSolving reduced RP problem...')
#     rp_obj, rp_edges, rp_tour, rp_y, rp_iters, rp_ncuts = solve_tsp_with_cuts(Q_rp)

#     print('\nRP optimal objective:', rp_obj)
#     print('RP tour order 0-based:', rp_tour)

#     # ------------------------------------------------------------
#     # Solve EV problem
#     # ------------------------------------------------------------
#     print('\nSolving EV problem...')
#     ev_obj, ev_edges, ev_tour, ev_y, ev_iters, ev_ncuts = solve_tsp_with_cuts(Q_ev)

#     print('\nEV objective:', ev_obj)
#     print('EV tour order 0-based:', ev_tour)

#     print('\nEV selected edges 0-based:')
#     for i, j in ev_edges:
#         print(i, j)

#     # ------------------------------------------------------------
#     # Evaluate EV solution under the true stochastic recourse model
#     # This is usually called EEV.
#     # ------------------------------------------------------------
#     eev_cost = edge_cost(ev_edges, Q_rp)

#     print('\nEEV / true expected cost of EV solution:', eev_cost)

#     # ------------------------------------------------------------
#     # Optional: compute VSS
#     # ------------------------------------------------------------
#     vss = eev_cost - rp_obj

#     print('\nVSS = EEV - RP:', vss)

#     # ------------------------------------------------------------
#     # Optional: recover path choices
#     # ------------------------------------------------------------
#     ev_path_choice = recover_ev_path_choices(ev_edges, C)
#     second_stage_choice = recover_second_stage_choices(ev_edges, C)

#     print('\nEV deterministic path choices for selected edges:')
#     for i, j in ev_edges:
#         print(f'edge {i} -> {j}: path {ev_path_choice[(i, j)]}')

#     print('\nExample true second-stage choices for scenario 0:')
#     for i, j in ev_edges:
#         print(f'scenario 0, edge {i} -> {j}: path {second_stage_choice[(0, i, j)]}')

# if __name__ == '__main__':
#     N, K, cbar, C, Q, coords, scen_files = load_instance(base)
#     print(f'Loaded N={N}, K={K}, scenarios={len(scen_files)}')
#     obj, edges, tour, y, iters, ncuts = solve_tsp_with_cuts(Q)
#     print('\nOptimal reduced RP objective:', obj)
#     print('Tour order 0-based:', tour)
#     print('Selected edges 0-based:')
#     for e in edges:
#         print(e[0], e[1])
