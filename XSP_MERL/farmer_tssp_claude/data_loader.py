"""
farmer_tssp/data_loader.py
──────────────────────────
Loads and validates farmer TSSP data from a JSON file.
All downstream model code works only with the typed dataclass, never raw dicts.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ── Typed containers ──────────────────────────────────────────────────────────

@dataclass
class ScenarioData:
    sid:         int
    name:        str
    probability: float
    yields:      Dict[int, float]   # crop_id → yield (T/acre)


@dataclass
class FarmerData:
    # Sets
    crop_ids:     List[int]
    purch_ids:    List[int]
    sell_ids:     List[int]
    scenario_ids: List[int]

    # Human-readable names (for reporting)
    crop_names:  Dict[int, str]
    sell_names:  Dict[int, str]
    scen_names:  Dict[int, str]

    # Scalar parameters
    budget:     float
    beet_quota: float

    # Vector parameters  (all keyed by int ids)
    planting_cost:   Dict[int, float]   # crop_id   → $/acre
    purchase_price:  Dict[int, float]   # purch_id  → $/T
    sell_price:      Dict[int, float]   # sell_id   → $/T
    min_requirement: Dict[int, float]   # crop_id   → T

    # Scenario data
    scenarios: Dict[int, ScenarioData]

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def probabilities(self) -> Dict[int, float]:
        return {s: self.scenarios[s].probability for s in self.scenario_ids}

    def yield_of(self, scenario: int, crop: int) -> float:
        return self.scenarios[scenario].yields[crop]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_data(json_path: str | Path) -> FarmerData:
    """
    Parse *json_path* and return a fully-validated :class:`FarmerData`.

    Raises
    ------
    ValueError
        If any required field is missing or a value is out of range.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with path.open() as fh:
        raw = json.load(fh)

    # ── helpers ───────────────────────────────────────────────────────────────
    def int_keyed(d: dict) -> dict:
        """Convert string keys that look like ints to actual ints."""
        return {int(k): v for k, v in d.items()}

    # ── sets ──────────────────────────────────────────────────────────────────
    crop_ids     = [int(i) for i in raw["crops"]["ids"]]
    purch_ids    = [int(i) for i in raw["purchase_crops"]["ids"]]
    sell_ids     = [int(i) for i in raw["sell_slots"]["ids"]]
    scenario_ids = [int(i) for i in raw["scenarios"]["ids"]]

    # ── names ─────────────────────────────────────────────────────────────────
    crop_names = int_keyed(raw["crops"]["names"])
    sell_names = int_keyed(raw["sell_slots"]["names"])
    scen_names = int_keyed(raw["scenarios"]["names"])

    # ── scalar params ─────────────────────────────────────────────────────────
    p = raw["parameters"]
    budget     = float(p["budget"])
    beet_quota = float(p["beet_quota"])

    # ── vector params ─────────────────────────────────────────────────────────
    planting_cost   = int_keyed(p["planting_cost"])
    purchase_price  = int_keyed(p["purchase_price"])
    sell_price      = int_keyed(p["sell_price"])
    min_requirement = int_keyed(p["min_requirement"])

    # ── scenarios ─────────────────────────────────────────────────────────────
    sc_raw  = raw["scenarios"]
    sc_prob = int_keyed(sc_raw["probability"])
    sc_yld  = {int(s): int_keyed(ymap) for s, ymap in sc_raw["yield"].items()}

    scenarios: Dict[int, ScenarioData] = {}
    for sid in scenario_ids:
        scenarios[sid] = ScenarioData(
            sid         = sid,
            name        = scen_names[sid],
            probability = float(sc_prob[sid]),
            yields      = {c: float(sc_yld[sid][c]) for c in crop_ids},
        )

    data = FarmerData(
        crop_ids      = crop_ids,
        purch_ids     = purch_ids,
        sell_ids      = sell_ids,
        scenario_ids  = scenario_ids,
        crop_names    = crop_names,
        sell_names    = sell_names,
        scen_names    = scen_names,
        budget        = budget,
        beet_quota    = beet_quota,
        planting_cost  = planting_cost,
        purchase_price = purchase_price,
        sell_price     = sell_price,
        min_requirement= min_requirement,
        scenarios      = scenarios,
    )

    _validate(data)
    return data


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(d: FarmerData) -> None:
    errors: List[str] = []

    if d.budget <= 0:
        errors.append(f"budget must be positive, got {d.budget}")
    if d.beet_quota <= 0:
        errors.append(f"beet_quota must be positive, got {d.beet_quota}")

    prob_sum = sum(d.probabilities.values())
    if not abs(prob_sum - 1.0) < 1e-6:
        errors.append(f"Scenario probabilities sum to {prob_sum:.6f}, expected 1.0")

    for c in d.crop_ids:
        if d.planting_cost[c] <= 0:
            errors.append(f"planting_cost[{c}] must be positive")

    for j in d.purch_ids:
        if d.purchase_price[j] <= 0:
            errors.append(f"purchase_price[{j}] must be positive")

    for k in d.sell_ids:
        if d.sell_price[k] < 0:
            errors.append(f"sell_price[{k}] must be non-negative")

    for s in d.scenario_ids:
        for c in d.crop_ids:
            if d.yield_of(s, c) < 0:
                errors.append(f"yield[s={s}, c={c}] must be non-negative")

    if errors:
        raise ValueError("Data validation failed:\n  " + "\n  ".join(errors))