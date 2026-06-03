"""Benchmark instance generator for the LandS energy planning TSSP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from general_model import validate_instance, evaluate_recourse


CLASS_OFFSETS = {"small": 0, "med": 100000, "hard": 200000}
SCENARIO_COUNT_BY_REFERENCE = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}

DEFAULT_GENERATION_CONFIG = {
    "first_stage_cost_range": (5.0, 20.0),
    "second_stage_cost_range": (3.0, 60.0),
    "plant_cost_factor_range": (0.75, 1.25),
    "mode_cost_factor_range": (0.75, 1.25),
    "noise_factor_range": (0.85, 1.15),
    "base_demand_range": (1.0, 6.0),
    "demand_multiplier_range_by_class": {
        "small": (0.75, 1.25),
        "med": (0.65, 1.35),
        "hard": (0.55, 1.45),
    },
    # The following structured defaults are deliberately used to make the
    # first-stage decision meaningful. Cheap-capital plants are only moderate
    # generalists, while higher-capital-cost plants are operational specialists.
    # This creates a genuine investment-versus-recourse tradeoff, so x_EV and
    # x_star need not have the same c^T x value.
    "structured_tradeoff": True,
    "generalist_fraction": 0.25,
    "specialist_low_cost_range": (2.0, 8.0),
    "generalist_second_stage_cost_range": (24.0, 42.0),
    "nonspecialist_second_stage_cost_range": (65.0, 110.0),
    "structured_demand_regimes": True,
    "dominant_mode_multiplier_range_by_class": {
        "small": (1.55, 2.35),
        "med": (1.65, 2.55),
        "hard": (1.75, 2.75),
    },
    "nondominant_mode_multiplier_range_by_class": {
        "small": (0.35, 0.75),
        "med": (0.30, 0.70),
        "hard": (0.25, 0.65),
    },
    "budget_safety_factor": 2.75,
    "minimum_capacity_safety_factor": 1.05,
    "probability_method": "uniform",
    "generator_version": "1.1",
}


def get_size_config() -> Dict[str, Dict[int, Dict[str, int]]]:
    """Return default benchmark size templates.

    The n_scenarios values are defaults for the template. The optional scenario
    reference Y in generate_benchmark_suite can override them using
    SCENARIO_COUNT_BY_REFERENCE.
    """
    return {
        "small": {
            1: {"n_plants": 4, "n_demand_modes": 3, "n_scenarios": 3},
            2: {"n_plants": 6, "n_demand_modes": 5, "n_scenarios": 5},
            3: {"n_plants": 8, "n_demand_modes": 6, "n_scenarios": 10},
        },
        "med": {
            1: {"n_plants": 15, "n_demand_modes": 10, "n_scenarios": 15},
            2: {"n_plants": 20, "n_demand_modes": 15, "n_scenarios": 20},
            3: {"n_plants": 30, "n_demand_modes": 20, "n_scenarios": 30},
        },
        "hard": {
            1: {"n_plants": 50, "n_demand_modes": 30, "n_scenarios": 40},
            2: {"n_plants": 75, "n_demand_modes": 50, "n_scenarios": 45},
            3: {"n_plants": 100, "n_demand_modes": 75, "n_scenarios": 50},
        },
    }


def _matrix_payload(mat: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(mat, dtype=float)
    return {"rows": int(arr.shape[0]), "cols": int(arr.shape[1]), "data": arr.reshape(-1).tolist()}


def generate_first_stage_data(n_plants: int, rng: np.random.Generator, config: Mapping[str, Any]) -> np.ndarray:
    c_min, c_max = config.get("first_stage_cost_range", DEFAULT_GENERATION_CONFIG["first_stage_cost_range"])
    return rng.uniform(float(c_min), float(c_max), size=n_plants).round(6)


def generate_second_stage_costs(
    n_plants: int,
    n_demand_modes: int,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Generate plant-mode operating costs in plant-major order.

    By default, generated instances use a structured investment/operation
    tradeoff. The plants with the lowest capital costs are made moderate
    generalists, while more expensive plants become very cheap specialists for
    selected demand modes and expensive for all other modes. This breaks the
    near-degeneracy where EV and SP both buy the same cheapest-capacity mix.
    Set ``structured_tradeoff=False`` in the config to recover the older fully
    random behavior.
    """
    if bool(config.get("structured_tradeoff", True)):
        c_arr = np.asarray(config.get("_first_stage_costs", []), dtype=float)
        if c_arr.shape[0] != n_plants:
            raise ValueError(
                "structured_tradeoff=True requires _first_stage_costs in the generation config."
            )

        order = np.argsort(c_arr)
        expensive_order = order[::-1]
        n_generalists = max(1, int(np.ceil(float(config.get("generalist_fraction", 0.25)) * n_plants)))
        generalists = order[:n_generalists]

        ns_lo, ns_hi = config.get(
            "nonspecialist_second_stage_cost_range",
            DEFAULT_GENERATION_CONFIG["nonspecialist_second_stage_cost_range"],
        )
        gen_lo, gen_hi = config.get(
            "generalist_second_stage_cost_range",
            DEFAULT_GENERATION_CONFIG["generalist_second_stage_cost_range"],
        )
        low_lo, low_hi = config.get(
            "specialist_low_cost_range",
            DEFAULT_GENERATION_CONFIG["specialist_low_cost_range"],
        )

        costs = rng.uniform(float(ns_lo), float(ns_hi), size=(n_plants, n_demand_modes))
        costs[generalists, :] = rng.uniform(
            float(gen_lo), float(gen_hi), size=(len(generalists), n_demand_modes)
        )

        # Assign each demand mode a specialist, biased toward the capital-costly
        # plants. If there are more modes than plants, specialists are reused.
        for j in range(n_demand_modes):
            specialist = int(expensive_order[j % n_plants])
            costs[specialist, j] = rng.uniform(float(low_lo), float(low_hi))

        return costs.round(6).reshape(-1)

    f_min, f_max = config.get("second_stage_cost_range", DEFAULT_GENERATION_CONFIG["second_stage_cost_range"])
    base = rng.uniform(float(f_min), float(f_max), size=(n_plants, n_demand_modes))
    alpha_lo, alpha_hi = config.get("plant_cost_factor_range", DEFAULT_GENERATION_CONFIG["plant_cost_factor_range"])
    beta_lo, beta_hi = config.get("mode_cost_factor_range", DEFAULT_GENERATION_CONFIG["mode_cost_factor_range"])
    eps_lo, eps_hi = config.get("noise_factor_range", DEFAULT_GENERATION_CONFIG["noise_factor_range"])
    alpha = rng.uniform(float(alpha_lo), float(alpha_hi), size=(n_plants, 1))
    beta = rng.uniform(float(beta_lo), float(beta_hi), size=(1, n_demand_modes))
    eps = rng.uniform(float(eps_lo), float(eps_hi), size=(n_plants, n_demand_modes))
    costs = base * alpha * beta * eps
    return costs.round(6).reshape(-1)


def generate_scenario_probabilities(
    n_scenarios: int,
    rng: np.random.Generator,
    method: str = "uniform",
) -> np.ndarray:
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")
    if method == "uniform":
        return np.full(n_scenarios, 1.0 / n_scenarios, dtype=float)
    if method == "random":
        raw = rng.uniform(0.05, 1.0, size=n_scenarios)
        return raw / raw.sum()
    raise ValueError("Probability generation method must be 'uniform' or 'random'.")


def generate_scenario_demands(
    n_demand_modes: int,
    n_scenarios: int,
    rng: np.random.Generator,
    config: Mapping[str, Any],
) -> np.ndarray:
    """Generate scenario demands with optional mode-dominant regimes.

    In the structured regime mode, every scenario has one demand mode that is
    unusually high and the remaining modes are relatively low. Across scenarios,
    the dominant mode rotates. The mean scenario therefore looks balanced, while
    the stochastic model sees tail-like mode-specific realizations. This is what
    encourages a different first-stage capacity portfolio in SP than in EV.
    """
    d_lo, d_hi = config.get("base_demand_range", DEFAULT_GENERATION_CONFIG["base_demand_range"])
    size_class = config.get("size_class", "small")
    base = rng.uniform(float(d_lo), float(d_hi), size=(n_demand_modes, 1))

    if bool(config.get("structured_demand_regimes", True)):
        dom_map = config.get(
            "dominant_mode_multiplier_range_by_class",
            DEFAULT_GENERATION_CONFIG["dominant_mode_multiplier_range_by_class"],
        )
        nondom_map = config.get(
            "nondominant_mode_multiplier_range_by_class",
            DEFAULT_GENERATION_CONFIG["nondominant_mode_multiplier_range_by_class"],
        )
        dom_lo, dom_hi = dom_map.get(size_class, (1.6, 2.4))
        low_lo, low_hi = nondom_map.get(size_class, (0.3, 0.7))

        demands = np.zeros((n_demand_modes, n_scenarios), dtype=float)
        dominant_modes = np.arange(n_scenarios) % n_demand_modes
        rng.shuffle(dominant_modes)
        for s, dom_j in enumerate(dominant_modes):
            multipliers = rng.uniform(float(low_lo), float(low_hi), size=n_demand_modes)
            multipliers[int(dom_j)] = rng.uniform(float(dom_lo), float(dom_hi))
            # A mild scenario-wide shock prevents all scenarios from having the
            # exact same total demand after the mode mix changes.
            system_shock = rng.uniform(0.9, 1.1)
            demands[:, s] = base[:, 0] * multipliers * system_shock
        return demands.round(6)

    mult_map = config.get(
        "demand_multiplier_range_by_class",
        DEFAULT_GENERATION_CONFIG["demand_multiplier_range_by_class"],
    )
    m_lo, m_hi = mult_map.get(size_class, (0.7, 1.3))
    multipliers = rng.uniform(float(m_lo), float(m_hi), size=(n_demand_modes, n_scenarios))
    return (base * multipliers).round(6)


def build_lands_matrices(
    c: Sequence[float],
    demands: np.ndarray,
    minimum_capacity: float,
    budget: float,
) -> Dict[str, Any]:
    c_arr = np.asarray(c, dtype=float)
    demands_arr = np.asarray(demands, dtype=float)
    if demands_arr.ndim != 2:
        raise ValueError("demands must have shape (n_demand_modes, n_scenarios).")
    n_plants = c_arr.size
    n_demand_modes, n_scenarios = demands_arr.shape
    n_y = n_plants * n_demand_modes
    n_rows = n_plants + n_demand_modes

    A = np.vstack([np.ones(n_plants), -c_arr])
    b = np.asarray([minimum_capacity, -budget], dtype=float)

    T = np.zeros((n_rows, n_plants), dtype=float)
    W = np.zeros((n_rows, n_y), dtype=float)
    H = np.zeros((n_rows, n_rows), dtype=float)

    for i in range(n_plants):
        T[i, i] = -1.0
        for j in range(n_demand_modes):
            W[i, i * n_demand_modes + j] = 1.0

    for j in range(n_demand_modes):
        row = n_plants + j
        for i in range(n_plants):
            W[row, i * n_demand_modes + j] = -1.0
        H[row, n_plants + j] = -1.0

    xis = {}
    for s in range(n_scenarios):
        xi = np.zeros(n_rows, dtype=float)
        xi[n_plants:] = demands_arr[:, s]
        xis[f"xi_{s + 1}"] = xi.tolist()

    return {
        "A": _matrix_payload(A),
        "b": b.tolist(),
        "T": _matrix_payload(T),
        "W": _matrix_payload(W),
        "H": _matrix_payload(H),
        **xis,
    }


def _make_seed(
    base_seed: int,
    size_class: str,
    size_template_id: int,
    scenario_ref_id: int,
    replicate_id: int,
) -> int:
    if size_class not in CLASS_OFFSETS:
        raise ValueError(f"Unknown size_class {size_class!r}.")
    return int(base_seed + CLASS_OFFSETS[size_class] + 1000 * size_template_id + scenario_ref_id + replicate_id)


def generate_lands_instance(
    size_class: str,
    size_template_id: int,
    replicate_id: int,
    base_seed: int = 12345,
    n_scenarios: int = 3,
    scenario_ref_id: int = 1,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate one reproducible LandS instance.

    The folder/name convention is lands_instance_{class}_X_Y_Z, where X is the
    size template, Y is the scenario-count reference, and Z is the replicate.
    """
    size_config = get_size_config()
    if size_class not in size_config:
        raise ValueError(f"Unknown size_class {size_class!r}; choose from {sorted(size_config)}.")
    if size_template_id not in size_config[size_class]:
        raise ValueError(f"Unknown size_template_id {size_template_id!r} for class {size_class!r}.")

    gen_config = dict(DEFAULT_GENERATION_CONFIG)
    if config:
        gen_config.update(dict(config))
    gen_config["size_class"] = size_class

    dims = size_config[size_class][size_template_id]
    n_plants = int(dims["n_plants"])
    n_demand_modes = int(dims["n_demand_modes"])
    if n_scenarios is None:
        n_scenarios = int(dims["n_scenarios"])
    if n_scenarios > 50:
        raise ValueError("n_scenarios must not exceed 50.")

    seed = _make_seed(base_seed, size_class, size_template_id, scenario_ref_id, replicate_id)
    rng = np.random.default_rng(seed)

    c = generate_first_stage_data(n_plants, rng, gen_config)
    gen_config["_first_stage_costs"] = c
    d = generate_second_stage_costs(n_plants, n_demand_modes, rng, gen_config)
    p_s = generate_scenario_probabilities(
        n_scenarios,
        rng,
        method=gen_config.get("probability_method", "uniform"),
    )
    demands = generate_scenario_demands(n_demand_modes, n_scenarios, rng, gen_config)

    cap_factor = float(gen_config.get("minimum_capacity_safety_factor", 1.05))
    budget_factor = float(gen_config.get("budget_safety_factor", 1.35))
    minimum_capacity = float(cap_factor * np.max(np.sum(demands, axis=0)))
    budget = float(budget_factor * minimum_capacity * np.min(c))

    instance_name = f"lands_instance_{size_class}_{size_template_id}_{scenario_ref_id}_{replicate_id}"
    matrices = build_lands_matrices(c, demands, minimum_capacity, budget)

    metadata = {
        "instance_name": instance_name,
        "problem_family": "LandS",
        "size_class": size_class,
        "size_template_id": int(size_template_id),
        "scenario_reference_id": int(scenario_ref_id),
        "replicate_id": int(replicate_id),
        "seed": int(seed),
        "n_plants": int(n_plants),
        "n_demand_modes": int(n_demand_modes),
        "n_scenarios": int(n_scenarios),
        "num_first_stage_variables": int(n_plants),
        "num_second_stage_variables_per_scenario": int(n_plants * n_demand_modes),
        "num_second_stage_constraints_per_scenario": int(n_plants + n_demand_modes),
        "minimum_capacity": float(minimum_capacity),
        "budget": float(budget),
        "cost_generation_ranges": {
            "first_stage_cost_range": list(gen_config["first_stage_cost_range"]),
            "second_stage_cost_range": list(gen_config["second_stage_cost_range"]),
            "structured_tradeoff": bool(gen_config.get("structured_tradeoff", True)),
            "specialist_low_cost_range": list(gen_config["specialist_low_cost_range"]),
            "generalist_second_stage_cost_range": list(
                gen_config["generalist_second_stage_cost_range"]
            ),
            "nonspecialist_second_stage_cost_range": list(
                gen_config["nonspecialist_second_stage_cost_range"]
            ),
        },
        "demand_generation_ranges": {
            "base_demand_range": list(gen_config["base_demand_range"]),
            "demand_multiplier_range": list(
                gen_config["demand_multiplier_range_by_class"].get(size_class, (0.7, 1.3))
            ),
            "structured_demand_regimes": bool(gen_config.get("structured_demand_regimes", True)),
            "dominant_mode_multiplier_range": list(
                gen_config["dominant_mode_multiplier_range_by_class"].get(size_class, (1.6, 2.4))
            ),
            "nondominant_mode_multiplier_range": list(
                gen_config["nondominant_mode_multiplier_range_by_class"].get(size_class, (0.3, 0.7))
            ),
        },
        "generator_version": str(gen_config.get("generator_version", "1.0")),
    }

    data = {
        "metadata": metadata,
        **matrices,
        "c": c.tolist(),
        "d": np.asarray(d, dtype=float).tolist(),
        "p_s": p_s.tolist(),
        "constraint_form": "T x + W y_s <= H xi_s",
    }
    # Backward-compatible scalar probability fields.
    for s, p in enumerate(p_s, start=1):
        data[f"p_{s}"] = [float(p)]

    validate_generated_instance(data)
    return data


def validate_generated_instance(data: Mapping[str, Any]) -> bool:
    """Run consistency and practical feasibility checks for a generated instance."""
    validate_instance(data)
    meta = data.get("metadata", {})
    n_plants = int(meta.get("n_plants", len(data["c"])))
    n_demand_modes = int(meta.get("n_demand_modes", len(data["H"]["data"])))
    m = float(meta.get("minimum_capacity", data["b"][0]))
    budget = float(meta.get("budget", -data["b"][1]))
    c = np.asarray(data["c"], dtype=float)

    if c.size != n_plants:
        raise ValueError("Metadata n_plants does not match len(c).")
    if budget + 1e-8 < m * np.min(c):
        raise ValueError("Budget is too small for the cheap-plant feasibility certificate.")

    cheapest = int(np.argmin(c))
    x_cert = np.zeros(n_plants, dtype=float)
    x_cert[cheapest] = m
    if float(c @ x_cert) > budget + 1e-8:
        raise ValueError("Generated first-stage certificate violates budget.")

    scenario_ids = sorted([k for k in data if k.startswith("xi_")], key=lambda key: int(key.split("_")[1]))
    for sid in scenario_ids:
        try:
            rec = evaluate_recourse(
                data,
                x=x_cert,
                xi=data[sid],
                backend="scipy",
                solver="highs",
            )
        except Exception as e:
            print(f"SciPy failed, falling back to Pyomo/Gurobi: {e}")

            rec = evaluate_recourse(
                data,
                x=x_cert,
                xi=data[sid],
                backend="pyomo",
                solver="gurobi",
            )
        if not rec["success"]:
            raise ValueError(f"Generated recourse problem is infeasible for {sid}.")
    return True


def write_instance(data: Mapping[str, Any], root_dir: str | Path, instance_name: str) -> Path:
    root = Path(root_dir)
    inst_dir = root / instance_name
    input_dir = inst_dir / "input_data"
    output_dir = inst_dir / "output_data"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / f"{instance_name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    return path


def generate_benchmark_suite(
    root_dir: str | Path,
    size_classes: Sequence[str] = ("small", "med", "hard"),
    num_replicates: int = 5,
    base_seed: int = 12345,
    scenario_reference_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Generate a suite of benchmark instances under root_dir."""
    size_config = get_size_config()
    written = []
    scenario_reference_ids = tuple(scenario_reference_ids or (1,))

    for size_class in size_classes:
        if size_class not in size_config:
            raise ValueError(f"Unknown size_class {size_class!r}.")
        for template_id, dims in size_config[size_class].items():
            for scenario_ref_id in scenario_reference_ids:
                n_scenarios = SCENARIO_COUNT_BY_REFERENCE.get(
                    int(scenario_ref_id), int(dims["n_scenarios"])
                )
                for replicate_id in range(1, num_replicates + 1):
                    data = generate_lands_instance(
                        size_class=size_class,
                        size_template_id=int(template_id),
                        scenario_ref_id=int(scenario_ref_id),
                        replicate_id=int(replicate_id),
                        base_seed=base_seed,
                        n_scenarios=n_scenarios,
                    )
                    name = data["metadata"]["instance_name"]
                    path = write_instance(data, root_dir=root_dir, instance_name=name)
                    written.append(str(path))

    return {"num_instances": len(written), "paths": written}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LandS benchmark instances.")
    # parser.add_argument("--root_dir", default="Everything_lands")
    parser.add_argument("--root_dir", default="lands_instances")
    parser.add_argument("--classes", nargs="+", default=["small", "med", "hard"])
    parser.add_argument("--num_replicates", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=12345)
    parser.add_argument(
        "--scenario_refs",
        nargs="*",
        type=int,
        default=[1],
        help="Scenario-count reference ids Y. Default: 1.",
    )
    args = parser.parse_args(argv)
    summary = generate_benchmark_suite(
        root_dir=args.root_dir,
        size_classes=tuple(args.classes),
        num_replicates=args.num_replicates,
        base_seed=args.base_seed,
        scenario_reference_ids=tuple(args.scenario_refs),
    )
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
