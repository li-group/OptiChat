from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, Set, Any, List, Sequence, Union

import json
import pyomo.environ as pyo
from pyomo.core.base.componentuid import ComponentUID

# NOTE: relative import to work inside the packaged project
from .extractor import LDRExtractor

__all__ = [
    "LDRPrimalDualCore",
    "LDR_solve",
    "UncertaintySpec",
    "ExtractorOptions",
    "BuildOptions",
    "SolverOptions",
]

# ----------------------------- Constants -----------------------------
CONST_TOKEN: str = "const"


# ----------------------------- Module-level helpers (private) -----------------------------
def _uid(obj: Any) -> str:
    """Stable string id for Pyomo Var/Param(Data) via ComponentUID; pass-through if already str."""
    if isinstance(obj, str):
        return obj
    return str(ComponentUID(obj))


def _all_var_ids(md: Any) -> Set[str]:
    """Union of declared var_ids, any vars seen in rows, and any bound names."""
    explicit = set(getattr(md, "var_ids", []) or [])
    from_rows = {v for r in md.constraints for v in r.var_coefs.keys()}
    from_bounds = set(md.var_bounds.keys())
    return explicit | from_rows | from_bounds


def _active_xi_for_row(
    row: Any,
    var_map: Dict[str, Dict[int, pyo.Var]],
    uncertain_params: List[pyo.Param],
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    For a canonical row, compute the active ξ blocks:
      Returns (active blocks, var-driven blocks, rhs-driven blocks) as ints.
      Block 1 is the intercept; blocks 2.. correspond to uncertain parameters.
    """
    var_xi = {
        int(xi)
        for vk in row.var_coefs
        if vk in var_map
        for xi in var_map[vk]
        if int(xi) > 1
    }
    rhs_xi = {
        2 + i
        for i, p in enumerate(uncertain_params)
        if row.param_coefs.get(_uid(p), 0.0) != 0.0
    }
    active = {1} | var_xi | rhs_xi
    return active, var_xi, rhs_xi


# =====================================================================
#                             Friendly option bundles
# =====================================================================
@dataclass
class UncertaintySpec:
    """Minimal uncertainty description."""
    params: Sequence[pyo.Param]                           # ξ₂…ξ_{k+1} in this order
    box: Sequence[Tuple[float, float]]                    # same length/order as params

    def xi_set(self) -> List[int]:
        # We always include intercept block=1; blocks 2..k+1 correspond to params
        return [1] + list(range(2, len(self.params) + 2))

    def bounds(self) -> Sequence[Tuple[float, float]]:
        # For BN slacks and bound rows; default to the same box
        return list(self.box)


@dataclass
class ExtractorOptions:
    """How to build variable-ξ reachability."""
    primal_cfg: Optional[Dict[str, Any]] = None
    dual_cfg: Optional[Dict[str, Any]] = None
    k: int = 0
    khop_temporal: bool = True


@dataclass
class BuildOptions:
    """How to assemble the LDR models."""
    reduced_primal: bool = False
    reduced_dual: bool = False
    M_primal: Optional[Dict[Tuple[int, int], float]] = None
    M_dual: Optional[Dict[Tuple[int, int], float]] = None


@dataclass
class SolverOptions:
    name: str = "gurobi"
    options: Optional[Dict[str, Any]] = field(default_factory=dict)
    tee: bool = False


# =====================================================================
#                             Public Core
# =====================================================================
class LDRPrimalDualCore:
    """
    Facade for building, solving, and inspecting LDR models for both primal & dual.

    Public APIs:
      • build_extract_solve_both(...)  # full control (original)
      • ldr_expression(...)            # print/evaluate learned affine policies
      • LDR_solve(...)                 # simple one-call entrypoint (new)
      • solve(...), solve_reduced(...) # optional nice wrappers (new)
    """

    # ---- class-level cache (for ldr_expression) ----
    _last_primal_md = None
    _last_dual_md = None
    _last_primal_var_map: Dict[str, Dict[int, pyo.Var]] = None
    _last_dual_var_map: Dict[str, Dict[int, pyo.Var]] = None
    _last_primal_ldr: Optional[pyo.ConcreteModel] = None
    _last_dual_ldr: Optional[pyo.ConcreteModel] = None
    _last_uncertain_params: Optional[List[pyo.Param]] = None
    _last_xi_list: Optional[List[int]] = None

    # ---- instance state for this build ----
    def __init__(self) -> None:
        self._primal_md = None
        self._dual_md = None
        self._primal_var_map: Dict[str, Dict[int, pyo.Var]] = {}
        self._dual_var_map: Dict[str, Dict[int, pyo.Var]] = {}
        self._primal_ldr: Optional[pyo.ConcreteModel] = None
        self._dual_ldr: Optional[pyo.ConcreteModel] = None
        self._uncertain_params: List[pyo.Param] = []
        self._xi_list: List[int] = []
        self._bounds: Sequence[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_second_moment_matrix(bounds: Sequence[Tuple[float, float]]) -> Dict[Tuple[int, int], float]:
        """
        Build M = E[ ξ̃ ξ̃ᵀ ] for ξ̃ = (1, ξ₁,…,ξ_k) assuming independent Uniform(a_i, b_i).
        Blocks are indexed 1..k+1 (1 = intercept).
        """
        k = len(bounds)
        mu = [0.5 * (a + b) for (a, b) in bounds]
        m2 = [(a * a + a * b + b * b) / 3.0 for (a, b) in bounds]

        M: Dict[Tuple[int, int], float] = {}
        M[(1, 1)] = 1.0
        for i in range(k):
            bi = i + 2
            M[(1, bi)] = mu[i]
            M[(bi, 1)] = mu[i]
            for j in range(k):
                bj = j + 2
                M[(bi, bj)] = m2[i] if i == j else mu[i] * mu[j]
        return M

    def _build_affine_vars(
        self,
        md: Any,
        ldr: pyo.ConcreteModel,
        xi_set: Iterable[int],  # {1,…,k+1}; 1 is intercept
        *,
        reduced: bool,
        var_to_xi: Optional[Dict[Any, Iterable[Any]]] = None,
    ) -> Tuple[Dict[str, Dict[int, pyo.Var]], Dict[str, Dict[int, pyo.Var]]]:
        """
        Create α-variables for each decision variable in `md`.

        - Continuous vars: α-blocks over xi_set (or pruned by var_to_xi if reduced=True).
        - Discrete vars (binary/integer): intercept-only (block 1); no ξ-dependence.

        Returns:
            (var_map, readable_map): { var_uid_str : {xi_idx : VarData} }
        """
        # normalize mapping tokens (accept CUID/Var/Param or strings)
        tok_map: Dict[str, Set[str]] = {}
        if reduced and var_to_xi:
            for k, seq in var_to_xi.items():
                vk = _uid(k)
                tok_map[vk] = {CONST_TOKEN if (t == CONST_TOKEN) else _uid(t) for t in seq}

        # normalize xi indices to ints
        xi_list = [int(i) for i in xi_set]
        if not xi_list or (1 not in xi_list):
            raise ValueError("xi_set must contain 1 (intercept).")

        # map param uid → block index
        uid_to_blk = {u: 2 + i for i, u in enumerate(md.uncertain_uids)}

        def tokens_to_blocks(tokens: Set[str]) -> Set[int]:
            blocks = set()
            if CONST_TOKEN in tokens:
                blocks.add(1)
            for u in md.uncertain_uids:
                if u in tokens:
                    blocks.add(uid_to_blk[u])
            return blocks

        var_map: Dict[str, Dict[int, pyo.Var]] = {}
        readable: Dict[str, Dict[int, pyo.Var]] = {}

        for vk in _all_var_ids(md):
            dom = getattr(md, "var_domain", {}).get(vk, "cont")  # default to continuous if absent

            # Discrete: intercept-only with proper domain
            if dom in ("binary", "integer"):
                comp_name = f"ldr_{vk.replace(':','__').replace('[','_').replace(']','_')}"
                if dom == "binary":
                    alpha = pyo.Var([1], domain=pyo.Binary)
                else:
                    alpha = pyo.Var([1], domain=pyo.Integers)
                setattr(ldr, comp_name, alpha)
                var_map[vk] = {1: alpha[1]}
                readable[vk] = var_map[vk]
                continue

            # Continuous
            if reduced:
                toks = tok_map.get(vk, set())
                if not toks:
                    continue
                xi_idx = sorted({1} | tokens_to_blocks(toks))  # keep intercept if kept
            else:
                xi_idx = list(xi_list)

            if not xi_idx:
                continue

            xi_idx = [int(i) for i in xi_idx]
            comp_name = f"ldr_{vk.replace(':','__').replace('[','_').replace(']','_')}"
            alpha = pyo.Var(xi_idx, domain=pyo.Reals)
            setattr(ldr, comp_name, alpha)
            var_map[vk] = {int(i): alpha[int(i)] for i in xi_idx}
            readable[vk] = var_map[vk]

        return var_map, readable

    def _build_constraints(
        self,
        md: Any,
        ldr: pyo.ConcreteModel,
        xi_set: Iterable[int],  # {1,…,k+1}; 1 = intercept
        var_map: Dict[str, Dict[int, pyo.Var]],
        bounds: Sequence[Tuple[float, float]],  # [(lb, ub)] for ξ₂ … ξ_{k+1}
        uncertain_params: Sequence[pyo.Param],  # ξ₂ … ξ_{k+1} in the SAME order
        *,
        reduced: bool = False,
    ) -> None:
        """
        Adds:
        • robustified row constraints for every row in md.constraints
        • robustified deterministic bounds for variables that appear in var_map

        Works for **primal** md and **dual** md identically.
        Uses Ben-Tal/Nemirovski linearization with shared slack tensors.
        """
        # normalize xi indices to ints
        xi_list = [int(i) for i in xi_set]
        if 1 not in xi_list:
            raise ValueError("xi_set must contain 1 (intercept).")
        if len(xi_list) != len(uncertain_params) + 1:
            raise ValueError("xi_set length must equal 1 + len(uncertain_params).")

        # ------------------------- (A) Row slacks -------------------------
        ineq_info: List[Tuple[str, List[int]]] = []
        for i, r in enumerate(md.constraints):
            if r.sense != "<=":
                continue
            if reduced and not any(vk in var_map for vk in r.var_coefs):
                continue

            key = f"{r.name}_{r.index}_{i}"
            active, _, _ = _active_xi_for_row(r, var_map, list(uncertain_params))
            used_pairs = {0, 1}
            for u in (active - {1}):
                e = 2 * (int(u) - 1)
                used_pairs.add(e)
                used_pairs.add(e + 1)
            ineq_info.append((key, sorted(used_pairs)))

        ineq_keys = [key for key, _ in ineq_info]
        ineq_keyset = set(ineq_keys)

        if ineq_keys:
            ldr.slack = pyo.Var(
                [(key, j) for key, jset in ineq_info for j in jset],
                domain=pyo.NonNegativeReals,
            )
            ldr.slack_pair = pyo.Constraint(
                ineq_keys, rule=lambda m, key: m.slack[key, 0] - m.slack[key, 1] >= 0
            )

        # ------------------------- (B) Rows themselves --------------------
        for i, row in enumerate(md.constraints):
            if reduced and not any(vk in var_map for vk in row.var_coefs):
                continue

            active, var_xi, rhs_xi = _active_xi_for_row(row, var_map, list(uncertain_params))
            idx = sorted(active)
            cname = f"ldr_row_{i}_{row.name}_{row.index}_{row.sense}"
            blk = pyo.Constraint(idx)
            setattr(ldr, cname, blk)

            key = f"{row.name}_{row.index}_{i}"

            for b in idx:
                # LHS: Σ_j a_ij * α_{j,b}
                lhs = 0.0
                for vk, aij in row.var_coefs.items():
                    xblock = var_map.get(vk, {}).get(int(b), None)
                    if xblock is not None and aij != 0.0:
                        lhs += aij * xblock

                # BN slacks for ≤ rows
                if row.sense == "<=" and key in ineq_keyset:
                    if b == 1:
                        lhs += ldr.slack[key, 0] - ldr.slack[key, 1]
                        for u in (var_xi | rhs_xi):
                            e = 2 * (u - 1)
                            lb, ub = bounds[u - 2]
                            lhs += (-lb) * ldr.slack[key, e]
                            lhs += (ub) * ldr.slack[key, e + 1]
                    else:
                        e = 2 * (b - 1)
                        lhs += ldr.slack[key, e] - ldr.slack[key, e + 1]

                # RHS block
                if b == 1:
                    rhs = row.const
                else:
                    pk = _uid(uncertain_params[b - 2])
                    rhs = row.param_coefs.get(pk, 0.0)

                blk[b] = lhs == rhs

        # ------------------------- (C) Bound rows -------------------------
        bound_rows: List[Tuple[str, str, float, float]] = []
        for vk, (lb, ub) in md.var_bounds.items():
            if vk not in var_map:
                continue

            # skip binaries entirely — domain already enforces 0/1
            dom = getattr(md, "var_domain", {}).get(vk, "cont")
            if dom == "binary":
                continue

            if lb != float("-inf"):
                bound_rows.append((vk, f"{vk}_lb", -1.0, -lb))  # -x ≤ -lb
            if ub != float("+inf"):
                bound_rows.append((vk, f"{vk}_ub", +1.0, +ub))  # +x ≤  ub

        if not bound_rows:
            return

        bkey_to_pairs: Dict[str, List[int]] = {}
        for vk, key, _, _ in bound_rows:
            active = {1} | {int(b) for b in var_map[vk] if int(b) > 1}
            used_pairs = {0, 1}
            for u in (active - {1}):
                e = 2 * (u - 1)
                used_pairs.add(e)
                used_pairs.add(e + 1)
            bkey_to_pairs[key] = sorted(used_pairs)

        bkeys = list(bkey_to_pairs.keys())
        ldr.slack_b = pyo.Var(
            [(key, j) for key in bkeys for j in bkey_to_pairs[key]],
            domain=pyo.NonNegativeReals,
        )
        ldr.slack_b_pair = pyo.Constraint(
            bkeys, rule=lambda m, key: m.slack_b[key, 0] - m.slack_b[key, 1] >= 0
        )

        for vk, key, sign, rhs0 in bound_rows:
            active = {1} | {int(b) for b in var_map[vk] if int(b) > 1}
            idx = sorted(active)
            blk = pyo.Constraint(idx)
            cname = f"ldr_bound_{key.replace(':','__')}"
            setattr(ldr, cname, blk)

            for b in idx:
                coeff = sign * var_map[vk].get(int(b), 0)
                if b == 1:
                    lhs = coeff + (ldr.slack_b[key, 0] - ldr.slack_b[key, 1])
                    for u in (active - {1}):
                        e = 2 * (u - 1)
                        # prefer instance-stored bounds when available (mirrors BN usage above)
                        lb_u, ub_u = self._bounds[u - 2] if self._bounds else bounds[u - 2]
                        lhs += (-lb_u) * ldr.slack_b[key, e]
                        lhs += (ub_u) * ldr.slack_b[key, e + 1]
                    rhs = rhs0
                else:
                    e = 2 * (b - 1)
                    lhs = coeff + ldr.slack_b[key, e] - ldr.slack_b[key, e + 1]
                    rhs = 0.0
                blk[b] = lhs == rhs

    def _build_objective(
        self,
        md: Any,
        ldr: pyo.ConcreteModel,
        xi_set: Iterable[int],  # {1,…,k+1}; 1 = intercept
        var_map: Dict[str, Dict[int, pyo.Var]],
        uncertain_params: Sequence[pyo.Param],  # ξ₂ … ξ_{k+1} in same order
        *,
        M: Optional[Dict[Tuple[int, int], float]] = None,
        reduced: bool = False,
    ) -> pyo.Objective:
        """
        E[ c(ξ)^T x(ξ) ] + E[standalone-ξ terms] + offset, with ξ̃_1 ≡ 1.
        If M is None, uses the uniform second-moment matrix implied by md.param_box.
        """
        xi_list = [int(i) for i in xi_set]
        if 1 not in xi_list:
            raise ValueError("xi_set must contain 1 (intercept).")
        if len(xi_list) != len(uncertain_params) + 1:
            raise ValueError("xi_set length must equal 1 + len(uncertain_params).")

        # Default M: Uniform(a,b) independent
        if M is None:
            M = self._uniform_second_moment_matrix(md.param_box)

        blk_tok: Dict[int, str] = {1: CONST_TOKEN}
        for pos, p in enumerate(uncertain_params, start=2):
            blk_tok[pos] = _uid(p)

        def M_(i: int, j: int) -> float:
            return float(M[i, j])

        expr = 0.0

        # Σ_v Σ_i (Σ_j M_ij C_v[j]) * α_v[i]
        for vk in _all_var_ids(md):
            if reduced and vk not in var_map:
                continue
            C = [md.obj_var_coef.get(vk, {}).get(blk_tok[b], 0.0) for b in xi_list]
            if all(c == 0.0 for c in C):
                continue
            MC = [
                sum(M_(bi, bj) * C[jpos] for jpos, bj in enumerate(xi_list))
                for bi in xi_list
            ]
            for ipos, bi in enumerate(xi_list):
                alph = var_map.get(vk, {}).get(int(bi), None)
                if alph is not None and MC[ipos] != 0.0:
                    expr += MC[ipos] * alph

        # stand-alone β_i ξ_i terms → β_i * E[ξ_i] = β_i * M[1, i_block]
        for pk, beta in md.obj_param_coef.items():
            for b in xi_list[1:]:
                if blk_tok[b] == pk:
                    expr += beta * M_(1, b)
                    break

        expr += md.obj_offset
        sense = pyo.minimize if (md.obj_sense == pyo.minimize) else pyo.maximize
        ldr.obj = pyo.Objective(expr=expr, sense=sense)
        return ldr.obj

    @staticmethod
    def _solve(model: pyo.ConcreteModel, solver_name: str, solver_options: Optional[Dict[str, Any]], tee: bool) -> Dict[str, Any]:
        """
        Thin solver wrapper: identical decision logic for objective extraction/None.
        """
        solver = pyo.SolverFactory(solver_name)
        if solver is None:
            return {"status": "no_solver", "termination": "no_solver", "obj": None, "raw": None}
        if solver_options:
            for kopt, vopt in solver_options.items():
                solver.options[kopt] = vopt
        res = solver.solve(model, tee=tee)
        stat = res.solver.status
        term = res.solver.termination_condition

        if term in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
            try:
                val = pyo.value(model.obj)
            except Exception:
                val = None
        elif term in (
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.unbounded,
            pyo.TerminationCondition.infeasibleOrUnbounded,
            pyo.TerminationCondition.maxTimeLimit,
            pyo.TerminationCondition.maxIterations,
            pyo.TerminationCondition.error,
            pyo.TerminationCondition.other,
        ):
            val = None
        else:
            try:
                val = pyo.value(model.obj)
            except Exception:
                val = None

        return {"status": str(stat), "termination": str(term), "obj": val, "raw": res}

    # ------------------------------------------------------------------
    # Public API 1: Build both, solve, return results (full control)
    # ------------------------------------------------------------------
    @classmethod
    def build_extract_solve_both(
        cls,
        *,
        base_model: pyo.ConcreteModel,
        uncertain_params: Sequence[pyo.Param],
        param_box: Sequence[Tuple[float, float]],
        xi_set: Iterable[int],                           # must contain 1 and have length = 1 + len(uncertain_params)
        bounds: Sequence[Tuple[float, float]],           # same length/order as uncertain_params
        # extractor knobs
        primal_cfg: Optional[Dict[str, Any]] = None,
        dual_cfg: Optional[Dict[str, Any]] = None,
        k: int = 0,
        khop_temporal: bool = True,
        # LDR building knobs
        reduced_primal: bool = False,
        reduced_dual: bool = False,
        M_primal: Optional[Dict[Tuple[int, int], float]] = None,
        M_dual: Optional[Dict[Tuple[int, int], float]] = None,
        # solver knobs
        solver_name: str = "gurobi",
        solver_options: Optional[Dict[str, Any]] = None,
        tee: bool = True,
        # return extras
        return_models: bool = False,
    ) -> Dict[str, Any]:
        """
        End-to-end helper:
          1) Extract primal & dual ModelData via LDRExtractor
          2) Build LDR variables, constraints, objective for both sides
          3) Solve both (Gurobi by default)
          4) Return objective values and a primal–dual gap (if both numeric)
        """
        # create a small instance to hold state and avoid repeating long arg lists
        self = cls()

        # --- basic checks on xi_set / bounds ---
        xi_list = list(xi_set)
        if 1 not in xi_list:
            raise ValueError("xi_set must include 1 (the intercept block).")
        if len(xi_list) != len(uncertain_params) + 1:
            raise ValueError("xi_set length must be 1 + len(uncertain_params).")
        if len(bounds) != len(uncertain_params):
            raise ValueError("bounds and uncertain_params must have the same length.")

        self._xi_list = [int(i) for i in xi_list]
        self._uncertain_params = list(uncertain_params)
        self._bounds = bounds

        # --- 1) Extract primal & dual ModelData and incidence maps ---
        extractor = LDRExtractor(primal_cfg=primal_cfg or {}, dual_cfg=dual_cfg or {})
        primal_md, primal_map, dual_md, dual_map = extractor.extract_data(
            base=base_model,
            uncertain_params=uncertain_params,
            param_box=param_box,
            k=k,
            khop_temporal=khop_temporal,
        )
        self._primal_md, self._dual_md = primal_md, dual_md

        # --- 2) Build LDR models (Pyomo containers) ---
        self._primal_ldr = pyo.ConcreteModel()
        self._dual_ldr = pyo.ConcreteModel()

        # α-vars (use reduced mapping iff reduced_* = True)
        self._primal_var_map, _ = self._build_affine_vars(
            md=self._primal_md,
            ldr=self._primal_ldr,
            xi_set=self._xi_list,
            reduced=reduced_primal,
            var_to_xi=(primal_map if reduced_primal else None),
        )
        self._dual_var_map, _ = self._build_affine_vars(
            md=self._dual_md,
            ldr=self._dual_ldr,
            xi_set=self._xi_list,
            reduced=reduced_dual,
            var_to_xi=(dual_map if reduced_dual else None),
        )

        # constraints (robust rows + robust bounds)
        self._build_constraints(
            md=self._primal_md,
            ldr=self._primal_ldr,
            xi_set=self._xi_list,
            var_map=self._primal_var_map,
            bounds=bounds,
            uncertain_params=self._uncertain_params,
            reduced=reduced_primal,
        )
        self._build_constraints(
            md=self._dual_md,
            ldr=self._dual_ldr,
            xi_set=self._xi_list,
            var_map=self._dual_var_map,
            bounds=bounds,
            uncertain_params=self._uncertain_params,
            reduced=reduced_dual,
        )

        # objectives (default M = uniform second-moment from md.param_box)
        self._build_objective(
            md=self._primal_md,
            ldr=self._primal_ldr,
            xi_set=self._xi_list,
            var_map=self._primal_var_map,
            uncertain_params=self._uncertain_params,
            M=M_primal,
            reduced=reduced_primal,
        )
        self._build_objective(
            md=self._dual_md,
            ldr=self._dual_ldr,
            xi_set=self._xi_list,
            var_map=self._dual_var_map,
            uncertain_params=self._uncertain_params,
            M=M_dual,
            reduced=reduced_dual,
        )

        # --- update class-level cache for ldr_expression() compatibility ---
        cls._last_primal_md = self._primal_md
        cls._last_dual_md = self._dual_md
        cls._last_primal_var_map = self._primal_var_map
        cls._last_dual_var_map = self._dual_var_map
        cls._last_primal_ldr = self._primal_ldr
        cls._last_dual_ldr = self._dual_ldr
        cls._last_uncertain_params = self._uncertain_params
        cls._last_xi_list = self._xi_list

        # --- 3) Solve both ---
        p_res = cls._solve(self._primal_ldr, solver_name, solver_options, tee)
        d_res = cls._solve(self._dual_ldr, solver_name, solver_options, tee)

        # gap (primal - dual) only if both are numbers
        gap = None
        if (p_res["obj"] is not None) and (d_res["obj"] is not None):
            gap = float(p_res["obj"]) - float(d_res["obj"])

        out = {
            "primal_obj": p_res["obj"],
            "dual_obj": d_res["obj"],
            "gap": gap,
            "primal_status": {"status": p_res["status"], "termination": p_res["termination"]},
            "dual_status": {"status": d_res["status"], "termination": d_res["termination"]},
        }
        if return_models:
            out.update({
                "primal_ldr": self._primal_ldr,
                "dual_ldr": self._dual_ldr,
            })
        return out

    # ------------------------------------------------------------------
    # Public API 2: Expression printer/evaluator
    # ------------------------------------------------------------------
    @classmethod
    def ldr_expression(
        cls,
        side: str = "primal",                          # which side to use for `variables`
        variables=None,                                # Optional[Any | Iterable[Any]]
        *,
        coeff: str = "values",                         # "values" | "names"
        digits: int = 6,
        zero_tol: float = 1e-12,
        use_base_names: bool = False,
        # Numeric evaluation of LDR at a given ξ vector (order = uncertain_params)
        xi_values: Optional[Sequence[float]] = None,   # length == len(uncertain_params)
        output: str = "expr",                          # "expr" | "value" | "both"
        validate_bounds: bool = True,
        # Ask for the **dual** of given primal constraints (in addition to `variables`)
        # Accepts: ConstraintData, (name, index|None), or "name" / "name[...]" strings
        dual_of: Optional[Union[Any, Iterable[Any]]] = None,
    ) -> Dict[str, Union[str, float, Dict[str, Union[str, float]]]]:
        """
        Build LDR expressions (as strings) and/or evaluate them at a supplied ξ vector.
        Behavior identical to the original implementation.
        """
        import pyomo.environ as pyo  # local for clarity in helper typing

        # -------- helpers (local, self-contained) -----------------
        def _uid_local(obj) -> str:
            if isinstance(obj, str):
                return obj
            return str(ComponentUID(obj))

        def _pname(p) -> str:
            u = _uid_local(p)
            return u.split("[", 1)[0] if use_base_names else u

        def _coef_repr(b: int, var: pyo.Var):
            if coeff == "values":
                try:
                    val = pyo.value(var)
                    if val is None:
                        return None
                    if abs(val) <= zero_tol and b != 1:
                        return None
                    return round(float(val), digits)
                except Exception:
                    return None
            elif coeff == "names":
                return str(var)
            else:
                raise ValueError("coeff must be 'values' or 'names'.")

        def _fmt_index(idx) -> str:
            if idx is None:
                return ""
            tup = idx if isinstance(idx, tuple) else (idx,)
            return "[" + ",".join(json.dumps(v) for v in tup) + "]"

        def _normalize_iter(x):
            if x is None:
                return []
            if isinstance(x, (str, pyo.Var)):
                return [x]
            try:
                return list(x)
            except Exception:
                return [x]

        def _parse_dual_ref(item) -> Optional[str]:
            """
            Convert a primal constraint reference to its dual y uid:
                y:{row.name}{_fmt_index(row.index)}
            """
            # ConstraintData
            if hasattr(item, "parent_component") and hasattr(item, "index"):
                try:
                    name = item.parent_component().name
                    idx = item.index()  # None for scalar
                    return f"y:{name}{_fmt_index(idx)}"
                except Exception:
                    return None
            # (name, index)
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                name = item[0]
                idx = item[1] if len(item) > 1 else None
                return f"y:{name}{_fmt_index(idx)}"
            # "name" or "name[...]" string
            if isinstance(item, str):
                if "[" in item and item.endswith("]"):
                    name, rest = item.split("[", 1)
                    rest = rest[:-1]  # strip trailing ]
                    toks = [t.strip() for t in rest.split(",")] if rest else []
                    try:
                        idx = tuple(int(t) if t.lstrip("-").isdigit() else t for t in toks) if toks else None
                    except Exception:
                        idx = tuple(toks) if toks else None
                    return f"y:{name}{_fmt_index(idx)}"
                else:
                    return f"y:{item}"
            return None

        def _expr_string_for(vk: str, blocks: Dict[int, pyo.Var], blk_label: Dict[int, str]) -> str:
            if not blocks:
                return "0"
            terms: List[str] = []
            # intercept
            if 1 in blocks:
                c = _coef_repr(1, blocks[1])
                if c is not None:
                    terms.append(str(c))
            # ξ terms
            for b in sorted(k for k in blocks.keys() if k > 1):
                c = _coef_repr(b, blocks[b])
                if c is None:
                    continue
                xlbl = blk_label.get(b, f"xi[{b-1}]")
                if coeff == "values" and isinstance(c, (int, float)):
                    if c < 0:
                        terms.append(f"- {abs(c)}*{xlbl}")
                    else:
                        if terms:
                            terms.append(f"+ {c}*{xlbl}")
                        else:
                            terms.append(f"{c}*{xlbl}")
                else:
                    terms.append(f"{c}*{xlbl}")
            if not terms:
                return "0"
            s = " ".join(terms)
            if s.startswith("+ "):
                s = s[2:]
            return s

        def _value_for(blocks: Dict[int, pyo.Var], xi_vals: Sequence[float]) -> float:
            """Evaluate α⋅ξ̃ with ξ̃_1=1 and ξ̃_{2+i}=xi_vals[i]."""
            total = 0.0
            if 1 in blocks:
                a1 = pyo.value(blocks[1])
                if a1 is not None:
                    total += float(a1)
            for i, xi in enumerate(xi_vals, start=2):
                if i in blocks:
                    ai = pyo.value(blocks[i])
                    if ai is None:
                        continue
                    total += float(ai) * float(xi)
            return float(round(total, digits))

        # -------- pick cached side data -----------------------------------
        side = (side or "").lower()
        if side not in ("primal", "dual"):
            raise ValueError("side must be 'primal' or 'dual'.")

        if side == "primal":
            md = cls._last_primal_md
            var_map = cls._last_primal_var_map
        else:
            md = cls._last_dual_md
            var_map = cls._last_dual_var_map

        uncertain_params = cls._last_uncertain_params
        xi_blocks = cls._last_xi_list

        if md is None or var_map is None or uncertain_params is None or xi_blocks is None:
            raise RuntimeError("No cached LDR build found. Call build_extract_solve_both(...) or LDR_solve(...) first.")

        # -------- set up ξ labels and validate xi_values -------------------
        blk_label: Dict[int, str] = {1: "1"}
        for i, p in enumerate(uncertain_params, start=0):
            blk_label[2 + i] = _uid(p).split("[", 1)[0] if use_base_names else _uid(p)

        # If user provided xi_values but didn't request 'value', assume they want value.
        if xi_values is not None and output == "expr":
            output = "value"

        if xi_values is not None:
            if len(xi_values) != len(uncertain_params):
                raise ValueError(
                    f"xi_values length {len(xi_values)} must equal number of uncertain params {len(uncertain_params)}."
                )
            if validate_bounds and hasattr(md, "param_box") and md.param_box:
                for i, (x, (a, b)) in enumerate(zip(xi_values, md.param_box)):
                    if not (a <= float(x) <= b):
                        raise ValueError(f"xi_values[{i}]={x} is outside bounds [{a}, {b}].")

        var_items = variables if isinstance(variables, (list, tuple, set)) else ([variables] if variables is not None else [])
        dual_items = dual_of if isinstance(dual_of, (list, tuple, set)) else ([dual_of] if dual_of is not None else [])

        # Convert dual_of constraint refs → dual y uids (always from **dual** side)
        def _fmt_index(idx) -> str:
            if idx is None:
                return ""
            tup = idx if isinstance(idx, tuple) else (idx,)
            return "[" + ",".join(json.dumps(v) for v in tup) + "]"

        def _parse_dual_ref(item) -> Optional[str]:
            if hasattr(item, "parent_component") and hasattr(item, "index"):
                try:
                    name = item.parent_component().name
                    idx = item.index()
                    return f"y:{name}{_fmt_index(idx)}"
                except Exception:
                    return None
            if isinstance(item, tuple) and item and isinstance(item[0], str):
                name = item[0]
                idx = item[1] if len(item) > 1 else None
                return f"y:{name}{_fmt_index(idx)}"
            if isinstance(item, str):
                if "[" in item and item.endswith("]"):
                    name, rest = item.split("[", 1)
                    rest = rest[:-1]
                    toks = [t.strip() for t in rest.split(",")] if rest else []
                    try:
                        idx = tuple(int(t) if t.lstrip("-").isdigit() else t for t in toks) if toks else None
                    except Exception:
                        idx = tuple(toks) if toks else None
                    return f"y:{name}{_fmt_index(idx)}"
                else:
                    return f"y:{item}"
            return None

        dual_y_uids: List[str] = []
        for it in dual_items:
            yuid = _parse_dual_ref(it)
            if yuid is not None:
                dual_y_uids.append(yuid)

        dual_md = cls._last_dual_md
        dual_var_map = cls._last_dual_var_map

        # -------- build the answer ----------------------------------------
        result: Dict[str, Union[str, float, Dict[str, Union[str, float]]]] = {}

        # (A) Variables on requested `side`
        for v in var_items:
            vk = _uid(v)
            blocks = var_map.get(vk, {})
            if output == "expr":
                result[vk] = _expr_string_for(vk, blocks, blk_label)
            elif output == "value":
                if xi_values is None:
                    raise ValueError("Provide xi_values=... when requesting output='value'.")
                result[vk] = _value_for(blocks, xi_values)
            elif output == "both":
                if xi_values is None:
                    raise ValueError("Provide xi_values=... when requesting output='both'.")
                result[vk] = {
                    "expr": _expr_string_for(vk, blocks, blk_label),
                    "value": _value_for(blocks, xi_values),
                }
            else:
                raise ValueError("output must be 'expr', 'value', or 'both'.")

        # (B) Dual of primal constraints (always use **dual** side)
        for yuid in dual_y_uids:
            blocks = (dual_var_map or {}).get(yuid, {})
            if output == "expr":
                result[yuid] = _expr_string_for(yuid, blocks, blk_label)
            elif output == "value":
                if xi_values is None:
                    raise ValueError("Provide xi_values=... when requesting output='value'.")
                result[yuid] = _value_for(blocks, xi_values)
            elif output == "both":
                if xi_values is None:
                    raise ValueError("Provide xi_values=... when requesting output='both'.")
                result[yuid] = {
                    "expr": _expr_string_for(yuid, blocks, blk_label),
                    "value": _value_for(blocks, xi_values),
                }

        return result

    # ------------------------------------------------------------------
    # New simple entrypoints
    # ------------------------------------------------------------------
    @classmethod
    def solve(
        cls,
        *,
        model: pyo.ConcreteModel,
        uncertainty: UncertaintySpec,
        extractor: ExtractorOptions = ExtractorOptions(),
        build: BuildOptions = BuildOptions(),
        solver: SolverOptions = SolverOptions(),
        return_models: bool = False,
    ) -> Dict[str, Any]:
        """
        Easiest path: 'full' LDR on both primal and dual (no α-pruning) unless build specifies otherwise.
        Users provide model + uncertainty box. Everything else has sane defaults.
        """
        xi_set = uncertainty.xi_set()
        bounds = uncertainty.bounds()
        return cls.build_extract_solve_both(
            base_model=model,
            uncertain_params=uncertainty.params,
            param_box=uncertainty.box,
            xi_set=xi_set,
            bounds=bounds,
            primal_cfg=extractor.primal_cfg,
            dual_cfg=extractor.dual_cfg,
            k=extractor.k,
            khop_temporal=extractor.khop_temporal,
            reduced_primal=build.reduced_primal,
            reduced_dual=build.reduced_dual,
            M_primal=build.M_primal,
            M_dual=build.M_dual,
            solver_name=solver.name,
            solver_options=solver.options,
            tee=solver.tee,
            return_models=return_models,
        )

    @classmethod
    def solve_reduced(
        cls,
        *,
        model: pyo.ConcreteModel,
        uncertainty: UncertaintySpec,
        extractor: ExtractorOptions = ExtractorOptions(),
        solver: SolverOptions = SolverOptions(),
        return_models: bool = False,
    ) -> Dict[str, Any]:
        """
        Same as `solve` but forces α-pruning on both sides.
        """
        build = BuildOptions(reduced_primal=True, reduced_dual=True)
        return cls.solve(
            model=model,
            uncertainty=uncertainty,
            extractor=extractor,
            build=build,
            solver=solver,
            return_models=return_models,
        )

    # ------------------------------------------------------------------
    # Single "friendly" API name you requested
    # ------------------------------------------------------------------
    @classmethod
    def LDR_solve(
        cls,
        *,
        model: pyo.ConcreteModel,
        uncertainty: Union[
            Tuple[Sequence[pyo.Param], Sequence[Tuple[float, float]]],  # (params, box)
            UncertaintySpec,
        ],
        reduced: bool = False,
        # extractor knobs
        primal_cfg: Optional[Dict[str, Any]] = None,
        dual_cfg: Optional[Dict[str, Any]] = None,
        k: int = 0,
        khop_temporal: bool = True,
        # moments
        M_primal: Optional[Dict[Tuple[int, int], float]] = None,
        M_dual: Optional[Dict[Tuple[int, int], float]] = None,
        # solver
        solver: str = "gurobi",
        solver_options: Optional[Dict[str, Any]] = None,
        tee: bool = False,
        # extras
        return_models: bool = False,
    ) -> Dict[str, Any]:
        """
        Simple one-call entrypoint. Accepts either:
          • uncertainty = (params, box)
          • uncertainty = UncertaintySpec(params=..., box=...)

        Example:
            res = LDRPrimalDualCore.LDR_solve(
                model=m,
                uncertainty=([m.xi1, m.xi2], [(0,1), (-2,3)]),
                reduced=True,
                k=1,
                primal_cfg=pcfg,
                dual_cfg=dcfg,
                solver="gurobi",
                solver_options={"Threads": 8},
            )
        """
        # normalize uncertainty
        if isinstance(uncertainty, UncertaintySpec):
            unc = uncertainty
        else:
            try:
                params, box = uncertainty  # type: ignore[misc]
                unc = UncertaintySpec(params=params, box=box)
            except Exception:
                raise TypeError("uncertainty must be (params, box) or an UncertaintySpec instance")

        return cls.solve(
            model=model,
            uncertainty=unc,
            extractor=ExtractorOptions(
                primal_cfg=primal_cfg,
                dual_cfg=dual_cfg,
                k=k,
                khop_temporal=khop_temporal,
            ),
            build=BuildOptions(
                reduced_primal=reduced,
                reduced_dual=reduced,
                M_primal=M_primal,
                M_dual=M_dual,
            ),
            solver=SolverOptions(name=solver, options=solver_options or {}, tee=tee),
            return_models=return_models,
        )


# ----------------------------------------------------------------------
# Module-level friendly function name (so users can `from ... import LDR_solve`)
# ----------------------------------------------------------------------
def LDR_solve(
    *,
    model: pyo.ConcreteModel,
    uncertainty: Union[
        Tuple[Sequence[pyo.Param], Sequence[Tuple[float, float]]],  # (params, box)
        UncertaintySpec,
    ],
    reduced: bool = False,
    # extractor knobs
    primal_cfg: Optional[Dict[str, Any]] = None,
    dual_cfg: Optional[Dict[str, Any]] = None,
    k: int = 0,
    khop_temporal: bool = True,
    # moments
    M_primal: Optional[Dict[Tuple[int, int], float]] = None,
    M_dual: Optional[Dict[Tuple[int, int], float]] = None,
    # solver
    solver: str = "gurobi",
    solver_options: Optional[Dict[str, Any]] = None,
    tee: bool = False,
    # extras
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Top-level thin wrapper so users can call LDR_solve(...) without referencing the class.
    """
    return LDRPrimalDualCore.LDR_solve(
        model=model,
        uncertainty=uncertainty,
        reduced=reduced,
        primal_cfg=primal_cfg,
        dual_cfg=dual_cfg,
        k=k,
        khop_temporal=khop_temporal,
        M_primal=M_primal,
        M_dual=M_dual,
        solver=solver,
        solver_options=solver_options,
        tee=tee,
        return_models=return_models,
    )
