"""
main.py
───────
Entry point: load data → solve → print full solution report.
"""

from pathlib import Path
from farmer_tssp.data_loader import load_data
from farmer_tssp.solver import compute_metrics

# from data_loader import load_data
# from solver import compute_metrics

# DATA_PATH = Path(__file__).parent / "farmer_tssp" / "data" / "farmer_data.json"
# DATA_PATH = Path(__file__).parent / "data.json"
from pathlib import Path
from .data_loader import load_data
from .solver import compute_metrics

# DATA_PATH = Path(__file__).parent / "data.json"
DATA_PATH = Path(__file__).parent / "data" / "farmer_data.json"

def main():
    data    = load_data(DATA_PATH)
    metrics = compute_metrics(data, solver="glpk", integer_x=True)

    rp  = metrics["RP"]
    ws  = metrics["WS"]
    ev  = metrics["EV"]
    eev = metrics["EEV"]
    evpi = metrics["EVPI"]
    vss  = metrics["VSS"]

    rp_d  = metrics["rp_detail"]
    ws_d  = metrics["ws_detail"]
    eev_d = metrics["eev_detail"]

    W = 62
    def hr(c="─"): print(c * W)
    def sect(title): hr("═"); print(f"  {title}"); hr("═")

    sect("FARMER TWO-STAGE STOCHASTIC PROGRAM — SOLUTION REPORT")

    # ── Stage-1 decisions ─────────────────────────────────────────────────────
    print("\n  STAGE 1  (Here-and-Now: plant before yield is known)")
    hr()
    print(f"  {'Crop':<20} {'Acres Devoted':>14}  {'Planting Cost':>14}")
    hr()
    total_plant = rp_d["first_stage_cost"]
    x = rp_d["x"]
    for cid, cname in data.crop_names.items():
        acres = x[cid]
        cost  = data.planting_cost[cid] * acres
        print(f"  {cname:<20} {acres:>14.1f}  ${cost:>13,.0f}")
    hr()
    print(f"  {'TOTAL':20} {sum(x.values()):>14.1f}  ${total_plant:>13,.0f}")

    # ── Stage-2 decisions per scenario ───────────────────────────────────────
    print("\n\n  STAGE 2  (Recourse: react after yield is revealed)")
    for s in data.scenario_ids:
        sc = data.scenarios[s]
        print(f"\n  Scenario {s}: {sc.name}  "
              f"(p = {sc.probability:.4f}  |  yields = "
              + ", ".join(f"{data.crop_names[c]}:{sc.yields[c]}" for c in data.crop_ids)
              + ")")
        hr()

        print(f"  {'Crop':<20} {'Harvested T':>12} {'Purchased T':>13} "
              f"{'Sold (sub) T':>13} {'Sold (sup) T':>13}")
        hr()
        for cid, cname in data.crop_names.items():
            harvested = sc.yields[cid] * x[cid]
            purchased = rp_d["y"].get((s, cid), 0.0)
            # sell slots: wheat=1,corn=2,beets_under=3,beets_over=4
            sold_sub = rp_d["w"].get((s, cid), 0.0)
            sold_sup = rp_d["w"].get((s, 4), 0.0) if cid == 3 else 0.0
            print(f"  {cname:<20} {harvested:>12.1f} {purchased:>13.1f} "
                  f"{sold_sub:>13.1f} {sold_sup:>13.1f}")
        hr()
        sc_cost = rp_d["scenario_costs"][s]
        print(f"  Scenario total cost: ${sc_cost:,.2f}")

    # ── Objective breakdown ───────────────────────────────────────────────────
    sect("OBJECTIVE BREAKDOWN")
    print(f"  {'First-stage (planting) cost':.<44} ${total_plant:>10,.2f}")
    for s in data.scenario_ids:
        p    = data.scenarios[s].probability
        sc_c = rp_d["scenario_costs"][s]
        recc = sc_c - total_plant
        print(f"  {'  Scenario '+str(s)+' ('+data.scen_names[s]+') recourse × prob':.<44} "
              f"${p*recc:>10,.2f}")
    print(f"  {'':.<44} {'──────────':>11}")
    print(f"  {'Expected Total Cost  (RP)':.<44} ${rp:>10,.2f}")

    # ── Stochastic metrics ────────────────────────────────────────────────────
    sect("STOCHASTIC PROGRAMMING METRICS")
    print(f"  {'RP  – Recourse Problem (stochastic solution)':.<48} ${rp:>10,.2f}")
    print(f"  {'WS  – Wait and See (perfect information)':.<48} ${ws:>10,.2f}")
    print(f"  {'EV  – Expected Value (det. mean-yield model)':.<48} ${ev:>10,.2f}")
    print(f"  {'EEV – Expected cost of EV solution':.<48} ${eev:>10,.2f}")
    hr()
    print(f"  {'EVPI = RP − WS  (worth of perfect information)':.<48} ${evpi:>10,.2f}")
    print(f"  {'VSS  = EEV − RP (worth of stochastic model)':.<48} ${vss:>10,.2f}")
    hr()
    print()
    print("  Interpretation:")
    print(f"    • You save ${abs(vss):,.2f} by using the stochastic model (RP)")
    print(f"      instead of the deterministic average-yield model (EEV).")
    print(f"    • Even with perfect future knowledge you could only save")
    print(f"      ${abs(evpi):,.2f} more (EVPI) — this is the theoretical ceiling.")

    # ── WS per-scenario ───────────────────────────────────────────────────────
    sect("WAIT-AND-SEE SOLUTIONS (per scenario)")
    for s in data.scenario_ids:
        sol = ws_d["per_scenario"][s]
        xs  = ", ".join(
            f"{data.crop_names[i]}={sol['x'][i]:.0f} ac"
            for i in data.crop_ids
        )
        print(f"  Scen {s} ({data.scen_names[s]:<8})  cost=${sol['cost']:>10,.2f}  |  {xs}")
    hr()
    print(f"  {'WS (probability-weighted average)':.<44} ${ws:>10,.2f}")
    print()

if __name__ == "__main__":
    main()