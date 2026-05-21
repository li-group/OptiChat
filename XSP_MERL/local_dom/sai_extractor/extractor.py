from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Set, Union, Any, Iterable

from collections import defaultdict, deque
import json

import pyomo.environ as pyo
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.param import Param, ParamData
from pyomo.core.base.var import VarData
from pyomo.core.expr.visitor import identify_variables, replace_expressions
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    SumExpression,
    ProductExpression,
    LinearExpression,
    DivisionExpression,
    NPV_DivisionExpression,
)

# ============================ Constants & Type Aliases ============================
CONST_TOKEN: str = "const"
INF: float = float("+inf")
NINF: float = float("-inf")

UIDStr = str
Bounds = Tuple[float, float]
CoefMap = Dict[UIDStr, float]
VarXiBlocks = Dict[UIDStr, Dict[UIDStr, float]]


# ----------------------------- dataclasses (string IDs) -----------------------------
@dataclass(frozen=True)
class ConstraintRow:
    name: str
    index: Optional[Tuple]
    sense: str  # '==' or '<='
    const: float  # RHS-style constant
    var_coefs: Dict[UIDStr, float]  # keys: var uid strings
    param_coefs: Dict[UIDStr, float]  # keys: param uid strings


@dataclass(frozen=True)
class ModelData:
    # (1) Variables and bounds (string IDs)
    var_ids: List[UIDStr]  # can be empty for primal/dual, but we now fill dual too
    var_bounds: Dict[UIDStr, Bounds]  # uid -> (lb, ub)
    var_domain: Dict[UIDStr, str]  # uid -> {"cont","binary","integer"}

    # (2) Canonical rows
    constraints: List[ConstraintRow]

    # (3) Objective
    obj_sense: int
    obj_var_coef: VarXiBlocks  # var uid -> {CONST_TOKEN or ξ uid -> coef}
    obj_param_coef: Dict[UIDStr, float]  # stand-alone ξ in objective (rare)
    obj_offset: float

    # (4) Uncertainty
    uncertain_uids: List[UIDStr]
    param_box: Sequence[Bounds]


# ============================ string UID helpers ============================
def uid(obj: Any) -> UIDStr:
    """Stable string id for Pyomo Var/Param(Data) via ComponentUID; pass-through if already str."""
    if isinstance(obj, str):
        return obj
    return str(ComponentUID(obj))


def base_name(uid_str: UIDStr) -> str:
    return uid_str.split("[", 1)[0]


def split_name_and_index(uid_str: UIDStr) -> Tuple[str, Tuple]:
    if "[" not in uid_str:
        return uid_str, ()
    name, tail = uid_str.split("[", 1)
    tail = tail.rstrip("]")
    parts = [p.strip() for p in tail.split(",")] if tail else []

    def cast(z: str) -> Union[int, str]:
        try:
            return int(z)
        except ValueError:
            return z

    return name, tuple(cast(p) for p in parts)


def time_of(uid_str: UIDStr, tpos: Optional[int]) -> Optional[Any]:
    if tpos is None:
        return None
    _, idx = split_name_and_index(uid_str)
    return idx[tpos] if tpos < len(idx) else None


def fmt_index(idx: Optional[Union[Tuple, Any]]) -> str:
    if idx is None:
        return ""
    tup = idx if isinstance(idx, tuple) else (idx,)
    return "[" + ",".join(json.dumps(v) for v in tup) + "]"


def all_var_uids(md: ModelData) -> Set[UIDStr]:
    """Union of declared var_ids, any vars seen in rows, and any bound-dual names."""
    explicit = set(md.var_ids)
    from_rows = {v for r in md.constraints for v in r.var_coefs.keys()}
    from_bounds = set(md.var_bounds.keys())
    return explicit | from_rows | from_bounds

def _domain_tag(v: VarData) -> str:
    """
    Domain tagging (binary vs. cont) using modern Pyomo APIs.

    - 'binary' if VarData.is_binary() is available and returns True
    - 'cont'   otherwise (includes integers and continuous)

    Note: Integers are deliberately treated as continuous for LDR purposes,
    since we cannot have affine decision rules that guarantee integer values.
    Only binary variables get special treatment (constant decisions).

    If the expected modern helpers are not present, raise a clear error suggesting
    a more recent Pyomo version, rather than guessing from legacy domain internals.
    """
    # Check for modern VarData helpers
    has_is_binary = callable(getattr(v, "is_binary", None))
    has_is_integer = callable(getattr(v, "is_integer", None))      # not used for tag, but signals modern API
    has_is_continuous = callable(getattr(v, "is_continuous", None)) # not used for tag, but signals modern API

    if not (has_is_binary or has_is_integer or has_is_continuous):
        raise RuntimeError(
            "Cannot determine variable domain using modern Pyomo APIs "
            "(VarData.is_binary / is_integer / is_continuous are not available). "
            "Please use a more recent Pyomo version."
        )

    # Only binaries get the 'binary' tag
    if has_is_binary and v.is_binary():
        return "binary"

    # Everything else (including integers) is treated as continuous
    return "cont"


# ============================ extractor class (string-based) ============================
class LDRExtractor:
    """
    String-UID based extractor for primal/dual linearized representations with uncertainty.
    Public API:
        - extract_primal_md
        - build_dual_md
        - build_var_xi_map
        - extract_data
    """

    def __init__(self, primal_cfg: Optional[Dict[str, Any]] = None, dual_cfg: Optional[Dict[str, Any]] = None):
        self.primal_cfg: Dict[str, Any] = primal_cfg or {}
        # Dual config defaults to primal config unless explicitly provided
        self.dual_cfg: Dict[str, Any] = dual_cfg if dual_cfg is not None else self.primal_cfg

    # ---------- objective split (returns string-keyed dicts) ----------
    def _extract_obj(self, expr: Any, uncertain_params_flat: Sequence[ParamData]) -> Tuple[VarXiBlocks, Dict[UIDStr, float], float]:
        """
        Decompose objective expression into:
          - var_param_coef: per-variable blocks keyed by CONST_TOKEN or ξ uid
          - obj_param_coef: stand-alone uncertain parameter terms
          - offset: constant
        Logic preserved; only imports consolidated and minor style cleanups.
        """
        var_param_coef: VarXiBlocks = {}
        obj_param_coef: Dict[UIDStr, float] = {}
        offset = 0.0
        uids = {uid(p) for p in uncertain_params_flat}

        def rec(e: Any, mult: float = 1.0) -> float:
            # numbers or any Pyomo-constant expression
            if isinstance(e, (int, float)):
                return float(e) * mult
            if hasattr(e, "is_constant") and e.is_constant():
                return float(pyo.value(e)) * mult

            # single variable
            if isinstance(e, VarData):
                k = uid(e)
                bucket = var_param_coef.setdefault(k, {})
                bucket[CONST_TOKEN] = bucket.get(CONST_TOKEN, 0.0) + mult
                return 0.0

            # single parameter (maybe uncertain)
            if isinstance(e, ParamData):
                k = uid(e)
                if k in uids:
                    obj_param_coef[k] = obj_param_coef.get(k, 0.0) + mult
                else:
                    return float(e.value) * mult
                return 0.0

            # unary minus
            if isinstance(e, NegationExpression):
                return rec(e.arg(0), -mult)

            # sum
            if isinstance(e, SumExpression):
                s = 0.0
                for a in e.args:
                    s += rec(a, mult)
                return s

            # division: treat as linear expression (will be handled by LinearRepnVisitor)
            if isinstance(e, (DivisionExpression, NPV_DivisionExpression)):
                numerator = e.arg(0)
                denominator = e.arg(1)

                # Helper to check for uncertain parameters or variables in an expression tree
                def has_uncertain_or_var(expr):
                    """Recursively check if expression contains uncertain parameters or variables."""
                    if isinstance(expr, VarData):
                        return True  # Division by variable
                    if isinstance(expr, ParamData):
                        k = uid(expr)
                        if k in uids:
                            return True  # Division by uncertain parameter
                    if hasattr(expr, 'args'):
                        # Recurse into expression arguments
                        for arg in expr.args:
                            if has_uncertain_or_var(arg):
                                return True
                    return False

                if has_uncertain_or_var(denominator):
                    raise NotImplementedError(
                        f"Division by uncertain parameter or variable is not supported in LDR. "
                        "Only division by constants or non-uncertain parameters is allowed."
                    )

                # Now safe to use LinearRepnVisitor since denominator is constant
                repn = LinearRepnVisitor({}).walk_expression(e)
                id_to_var = {id(v): v for v in identify_variables(e)}

                total = mult * float(repn.constant)
                for vid, vcoef in repn.linear.items():
                    v = id_to_var.get(vid, None)
                    if v is None:
                        continue
                    total += rec(v, mult * float(vcoef))
                return total

            # product
            if isinstance(e, ProductExpression):
                # flatten product factors
                fs: List[Any] = []

                def flat(p: Any) -> None:
                    if isinstance(p, ProductExpression):
                        for a in p.args:
                            flat(a)
                    else:
                        fs.append(p)

                flat(e)

                coef = mult
                nvar = npar = 0
                vobj: Optional[VarData] = None
                pobj: Optional[ParamData] = None
                lin_term: Optional[Any] = None  # one Linear/Sum factor allowed

                for f in fs:
                    if isinstance(f, (int, float)):
                        coef *= float(f)
                    elif hasattr(f, "is_constant") and f.is_constant():
                        coef *= float(pyo.value(f))
                    elif isinstance(f, VarData):
                        nvar += 1
                        vobj = f
                    elif isinstance(f, ParamData):
                        k = uid(f)
                        if k in uids:
                            npar += 1
                            pobj = f
                        else:
                            coef *= float(f.value)
                    elif isinstance(f, (SumExpression, LinearExpression, DivisionExpression, NPV_DivisionExpression)):
                        # For division, check if denominator contains uncertain parameters
                        if isinstance(f, (DivisionExpression, NPV_DivisionExpression)):
                            denominator = f.arg(1)

                            # Helper to check for uncertain parameters or variables
                            def has_uncertain_or_var(expr):
                                """Recursively check if expression contains uncertain parameters or variables."""
                                if isinstance(expr, VarData):
                                    return True
                                if isinstance(expr, ParamData):
                                    k = uid(expr)
                                    if k in uids:
                                        return True
                                if hasattr(expr, 'args'):
                                    for arg in expr.args:
                                        if has_uncertain_or_var(arg):
                                            return True
                                return False

                            if has_uncertain_or_var(denominator):
                                raise NotImplementedError(
                                    f"Division by uncertain parameter or variable is not supported in LDR. "
                                    "Only division by constants or non-uncertain parameters is allowed."
                                )

                        # Treat division/sum/linear as a potential linear term
                        if lin_term is None:
                            lin_term = f
                        else:
                            raise NotImplementedError("Product of two linear/sum terms is not supported.")
                    else:
                        raise NotImplementedError(f"Unsupported factor {type(f)}")

                # Distribute over a single linear/sum factor
                if lin_term is not None:
                    # Real bilinear patterns (linear * variable) are not allowed
                    if nvar >= 1:
                        raise NotImplementedError("Bilinear term (linear * variable) is not supported.")

                    # Build linear repn of the linear factor
                    repn = LinearRepnVisitor({}).walk_expression(lin_term)
                    id_to_var = {id(v): v for v in identify_variables(lin_term)}

                    # Case A: one uncertain param times linear → expand to sum of (param*var) + (param*const)
                    if npar == 1 and pobj is not None:
                        pk = uid(pobj)
                        # constant part * param → stand-alone uncertain term in objective
                        cst = coef * float(repn.constant)
                        if cst != 0.0:
                            obj_param_coef[pk] = obj_param_coef.get(pk, 0.0) + cst
                        # variable parts → (param * var) coefficients
                        for vid, vcoef in repn.linear.items():
                            v = id_to_var.get(vid, None)
                            if v is None:
                                continue
                            vk = uid(v)
                            bucket = var_param_coef.setdefault(vk, {})
                            bucket[pk] = bucket.get(pk, 0.0) + coef * float(vcoef)
                        return 0.0

                    # Case B: no uncertain param (just constants * linear) → distribute
                    if npar == 0:
                        total = coef * float(repn.constant)
                        for vid, vcoef in repn.linear.items():
                            v = id_to_var.get(vid, None)
                            if v is None:
                                continue
                            total += rec(v, coef * float(vcoef))
                        return total

                    # Otherwise (e.g., multiple uncertain params) → not supported
                    raise NotImplementedError("Product with >1 uncertain parameter is not supported.")

                # No linear factor: handle simple cases
                if nvar > 1 or npar > 1:
                    raise NotImplementedError("Nonlinear term (var*var or ξ*ξ) is not supported.")
                if nvar == 1 and npar == 1:
                    vk = uid(vobj)  # type: ignore[arg-type]
                    pk = uid(pobj)  # type: ignore[arg-type]
                    bucket = var_param_coef.setdefault(vk, {})
                    bucket[pk] = bucket.get(pk, 0.0) + coef
                    return 0.0
                if nvar == 1:
                    vk = uid(vobj)  # type: ignore[arg-type]
                    bucket = var_param_coef.setdefault(vk, {})
                    bucket[CONST_TOKEN] = bucket.get(CONST_TOKEN, 0.0) + coef
                    return 0.0
                if npar == 1:
                    pk = uid(pobj)  # type: ignore[arg-type]
                    obj_param_coef[pk] = obj_param_coef.get(pk, 0.0) + coef
                    return 0.0

                # only constants remain
                return coef

            # unknown node
            raise NotImplementedError(f"Unknown expr node {type(e)}")

        offset += rec(expr)
        return var_param_coef, obj_param_coef, offset

    # ---------- primal extraction → string ModelData ----------
    def extract_primal_md(
        self,
        base: pyo.ConcreteModel,
        uncertain_params: Sequence[Union[Param, ParamData]],
        param_box: Sequence[Bounds],
    ) -> ModelData:
        """
        Build a string-keyed ModelData from a concrete Pyomo model.
        NOTE: Relies on a module-level `_domain_tag(v: VarData) -> str` helper
        to classify variables as 'cont' | 'binary'.
        """
        # -------- (0) Flatten uncertain params in the user-provided order --------
        flat: List[ParamData] = []
        for p in uncertain_params:
            if isinstance(p, Param):
                flat.extend(list(p.values()))
            elif isinstance(p, ParamData):
                flat.append(p)
            else:
                flat.append(p)  # allow already-flat inputs

        if len(flat) != len(param_box):
            raise ValueError("param_box length must match uncertain_params length.")

        # -------- (1) Variables, IDs, bounds, domains --------
        var_datas = list(base.component_data_objects(pyo.Var, descend_into=True))

        # Deterministic IDs for reproducibility
        var_ids = sorted(uid(v) for v in var_datas)

        # Bounds map: uid → (lb, ub) with ±inf where absent
        var_bounds: Dict[UIDStr, Bounds] = {
            uid(v): (v.lb if v.has_lb() else NINF, v.ub if v.has_ub() else INF)
            for v in var_datas
        }

        # Domains via module-level helper `_domain_tag`
        var_domain: Dict[UIDStr, str] = {uid(v): _domain_tag(v) for v in var_datas}

        # -------- (2) Constraints → canonical rows using dummy ξ substitution --------
        rows: List[ConstraintRow] = []
        tmp = pyo.Var(range(len(flat)), initialize=0.0)
        base.add_component("_scratch_tmp_params", tmp)
        try:
            sub_map = {id(p): tmp[i] for i, p in enumerate(flat)}
            dummy_ids = {id(dv) for dv in tmp.values()}

            for block in base.component_objects(pyo.Constraint, active=True, descend_into=True):
                for idx in block:
                    c = block[idx]
                    body = c.body
                    if c.equality:
                        sense, lhs, rhs, sign = "==", body, c.lower, +1
                    elif c.has_lb() and not c.has_ub():
                        sense, lhs, rhs, sign = "<=", body, c.lower, -1  # ≥ becomes ≤ after flip
                    elif c.has_ub() and not c.has_lb():
                        sense, lhs, rhs, sign = "<=", body, c.upper, +1
                    else:
                        raise ValueError("Two-sided range constraints not supported.")

                    # Shift to LHS with optional sign flip for ≥
                    expr_shift = sign * (lhs - rhs)
                    # Substitute uncertain params with temporary Vars to read linear repn
                    expr_sub = replace_expressions(expr_shift, sub_map)
                    repn_sub = LinearRepnVisitor({}).walk_expression(expr_sub)

                    # Collect variable coefficients (skip dummy ξ vars)
                    var_coefs: CoefMap = {}
                    for v in identify_variables(expr_sub):
                        if id(v) in dummy_ids:
                            continue
                        coef = repn_sub.linear.get(id(v), 0.0)
                        if coef:
                            var_coefs[uid(v)] = float(coef)

                    # Collect ξ coefficients by reading the dummy coefficients and negating
                    param_coefs: CoefMap = {}
                    for i, p in enumerate(flat):
                        coef = -repn_sub.linear.get(id(tmp[i]), 0.0)
                        if coef:
                            param_coefs[uid(p)] = float(coef)

                    # Constant term (RHS-style): -repn.constant
                    const = -float(repn_sub.constant)

                    rows.append(
                        ConstraintRow(
                            name=block.name,
                            index=idx if block.is_indexed() else None,
                            sense=sense,
                            const=const,
                            var_coefs=var_coefs,
                            param_coefs=param_coefs,
                        )
                    )
        finally:
            base.del_component("_scratch_tmp_params")

        # -------- (3) Objective decomposition --------
        if not hasattr(base, "obj"):
            raise RuntimeError("Model must contain an active objective named 'obj'.")

        obj_var_coef, obj_param_coef, obj_offset = self._extract_obj(base.obj.expr, flat)

        # -------- (4) Assemble ModelData --------
        return ModelData(
            var_ids=var_ids,
            var_bounds=var_bounds,
            constraints=rows,
            obj_sense=base.obj.sense,
            obj_var_coef=obj_var_coef,
            obj_param_coef=obj_param_coef,
            obj_offset=obj_offset,
            uncertain_uids=[uid(p) for p in flat],
            param_box=list(param_box),
            var_domain=var_domain,
        )

    # ---------- dual MD from primal MD (all strings) ----------
    def build_dual_md(self, primal: ModelData) -> ModelData:
        """
        Build the dual ModelData from a primal ModelData. Preserves all original logic.
        """
        is_primal_max = (primal.obj_sense == pyo.maximize)
        dual_sense = pyo.minimize if is_primal_max else pyo.maximize

        def row_dual_bounds(sense: str) -> Bounds:
            if sense == "==":
                return (NINF, INF)
            # primal row is '<='
            return (0.0, INF) if is_primal_max else (NINF, 0.0)

        def bound_dual_bounds() -> Bounds:
            return (0.0, INF) if is_primal_max else (NINF, 0.0)

        var_bounds_dual: Dict[UIDStr, Bounds] = {}
        obj_var_coef_dual: VarXiBlocks = {}
        dual_var_ids: List[UIDStr] = []

        # helper for naming row duals
        def yname(row: ConstraintRow) -> str:
            return f'y:{row.name}{fmt_index(row.index)}'

        # 1) Row dual variables (always present)
        for r in primal.constraints:
            yn = yname(r)
            dual_var_ids.append(yn)
            var_bounds_dual[yn] = row_dual_bounds(r.sense)
            block: Dict[UIDStr, float] = {CONST_TOKEN: float(r.const)}
            for puid, coef in r.param_coefs.items():
                block[puid] = block.get(puid, 0.0) + float(coef)
            obj_var_coef_dual[yn] = block

        # helpers for bound-dual names
        def ylb(vk: UIDStr) -> str:
            return f"y_lb:{vk}"

        def yub(vk: UIDStr) -> str:
            return f"y_ub:{vk}"

        # 2) Dual constraints (one per primal variable/column)
        #    First gather A^T coefficients from row-duals
        rows_by_var: Dict[UIDStr, List[Tuple[str, float]]] = {}
        for r in primal.constraints:
            yn = yname(r)
            for vk, aij in r.var_coefs.items():
                rows_by_var.setdefault(vk, []).append((yn, float(aij)))

        dual_rows: List[ConstraintRow] = []

        for vk, (lb, ub) in primal.var_bounds.items():
            # LHS from row duals only (we'll add nonzero bound-duals if needed)
            lhs: CoefMap = {}
            for yn, aij in rows_by_var.get(vk, []):
                lhs[yn] = lhs.get(yn, 0.0) + aij

            # Detect bound type
            has_nonzero_lb = (lb != NINF) and (lb != 0.0)
            has_nonzero_ub = (ub != INF) and (ub != 0.0)
            is_free = (lb == NINF) and (ub == INF)
            is_nonneg = (lb == 0.0) and (ub == INF)
            is_nonpos = (ub == 0.0) and (lb == NINF)

            # Add bound-dual columns ONLY for nonzero finite bounds
            if has_nonzero_ub:
                yn_ub = yub(vk)
                dual_var_ids.append(yn_ub)
                var_bounds_dual[yn_ub] = bound_dual_bounds()
                obj_var_coef_dual[yn_ub] = {CONST_TOKEN: float(ub)}
                lhs[yn_ub] = lhs.get(yn_ub, 0.0) + 1.0

            if has_nonzero_lb:
                yn_lb = ylb(vk)
                dual_var_ids.append(yn_lb)
                var_bounds_dual[yn_lb] = bound_dual_bounds()
                obj_var_coef_dual[yn_lb] = {CONST_TOKEN: float(-lb)}
                lhs[yn_lb] = lhs.get(yn_lb, 0.0) - 1.0

            # RHS blocks from c_j(ξ) = c^0 + Σ C_{j,ξ} ξ
            blocks = primal.obj_var_coef.get(vk, {})
            const = float(blocks.get(CONST_TOKEN, 0.0))
            rhs_xi = {pk: float(c) for pk, c in blocks.items() if pk != CONST_TOKEN}

            # Decide the sense and possibly flip signs to keep '<=' canonicalization
            if has_nonzero_lb or has_nonzero_ub or is_free:
                # Equality if we used any nonzero finite bound-dual, or var is free
                sense = "=="
            else:
                # Zero-bound cases: encode inequality by variable sign
                # For primal MAX:
                #   x ≥ 0 → Aᵀy ≥ c  (write as -Aᵀy ≤ -c)
                #   x ≤ 0 → Aᵀy ≤ c  (already ≤)
                # For primal MIN: directions flip.
                if is_nonneg:
                    if is_primal_max:
                        lhs = {k: -v for k, v in lhs.items()}
                        const = -const
                        rhs_xi = {k: -v for k, v in rhs_xi.items()}
                    sense = "<="
                elif is_nonpos:
                    if not is_primal_max:
                        lhs = {k: -v for k, v in lhs.items()}
                        const = -const
                        rhs_xi = {k: -v for k, v in rhs_xi.items()}
                    sense = "<="
                else:
                    # Fallback to equality (should not occur given the above cases)
                    sense = "=="

            dual_rows.append(
                ConstraintRow(
                    name="dual_col",
                    index=(vk,),
                    sense=sense,
                    const=const,
                    var_coefs=lhs,
                    param_coefs=rhs_xi,
                )
            )

        dual_var_domain = {vk: "cont" for vk in dual_var_ids}

        return ModelData(
            var_ids=dual_var_ids,  # only the variables we actually created
            var_bounds=var_bounds_dual,
            constraints=dual_rows,
            obj_sense=dual_sense,
            obj_var_coef=obj_var_coef_dual,
            obj_param_coef={},  # dual has no stand-alone ξ terms
            obj_offset=0.0,
            uncertain_uids=list(primal.uncertain_uids),
            param_box=list(primal.param_box),
            var_domain=dual_var_domain,
        )

    # ---------- unified ξ-map for either primal or dual (string-based) ----------
    def build_var_xi_map(
        self,
        md: ModelData,
        *,
        manual_map: Optional[Dict[Any, Iterable[Any]]] = None,
        cfg: Optional[Dict[str, Any]] = None,
        knn: int = 0,
        khop_temporal: bool = False,
    ) -> Dict[UIDStr, Set[UIDStr]]:
        """
        Mapping logic (works the same for primal & dual):

        - No cfg → every variable → all uncertain ξ.
        - With cfg:
            - Only ξ whose base name appears in cfg["parameters"] are considered.
            - Temporal ξ (parameters[p]["t_pos"] is int):
                * If khop_temporal=False: window by the variable's own time index using variables[v]["t_pos"] and "window".
                * If khop_temporal=True: also use k-hop reachability (same as non-temporal).
            - Non-temporal ξ (parameters[p]["t_pos"] is None): k-hop reachability.
              Seeding for k-hop uses BOTH row param_coefs AND md.obj_var_coef[var] (important for the dual).
        """
        cfg = self.primal_cfg if cfg is None else cfg

        # Manual override
        if manual_map:
            all_xi = set(md.uncertain_uids)
            out_manual: Dict[UIDStr, Set[UIDStr]] = {}
            for v_key, plist in manual_map.items():
                vk = v_key if isinstance(v_key, str) else uid(v_key)
                if isinstance(plist, str) and plist == "__ALL__":
                    out_manual[vk] = set(all_xi)
                    continue
                sel: Set[UIDStr] = set()
                for p in plist:
                    pk = p if isinstance(p, str) else uid(p)
                    if pk in all_xi:
                        sel.add(pk)
                out_manual[vk] = sel
            return out_manual

        # No cfg + knn<0 → every var → all ξ (legacy all-to-all behavior)
        # No cfg + knn>=0 → use k-hop on constraint graph with all params permitted
        if not cfg and knn < 0:
            xi = set(md.uncertain_uids)
            return {v: set(xi) for v in all_var_uids(md)}

        vcfg = cfg.get("variables", {}) if cfg else {}
        pcfg = cfg.get("parameters", {}) if cfg else {}

        # Helpers that respect base names like "y:inv_bal"
        def v_tpos(vname: str) -> Optional[int]:
            return vcfg.get(vname, {}).get("t_pos", None)

        def p_tpos(pname: str) -> Optional[int]:
            return pcfg.get(pname, {}).get("t_pos", None)

        def v_window(vname: str) -> Tuple[Optional[int], Optional[int]]:
            w = vcfg.get(vname, {}).get("window", {})
            return w.get("past", None), w.get("future", None)

        if cfg:
            # ξ universe when cfg is present: ONLY those listed in cfg["parameters"]
            pcfg_names = set(pcfg.keys())  # base names
            permitted_xi = {p for p in md.uncertain_uids if base_name(p) in pcfg_names}
            # Classify ξ by temporal/non-temporal via cfg["parameters"][...]["t_pos"]
            temporal_xi = {p for p in permitted_xi if p_tpos(base_name(p)) is not None}
            non_temporal_xi = permitted_xi - temporal_xi
        else:
            # No cfg: all params permitted, treat all as non-temporal (knn graph drives coupling)
            permitted_xi = set(md.uncertain_uids)
            temporal_xi = set()
            non_temporal_xi = set(permitted_xi)

        # Rows & graph (used for k-hop)
        row_vars = [list(r.var_coefs.keys()) for r in md.constraints]
        row_params = [list(r.param_coefs.keys()) for r in md.constraints]

        # Build variable adjacency from constraints
        var_adj: Dict[UIDStr, Set[UIDStr]] = defaultdict(set)
        for vs in row_vars:
            for i, v1 in enumerate(vs):
                for v2 in vs[i + 1 :]:
                    var_adj[v1].add(v2)
                    var_adj[v2].add(v1)

        # Direct seeding for k-hop:
        # - from rows: param_coefs on the same row as the variable
        # - from objective: md.obj_var_coef[var] (important for dual)
        var_to_ti_direct: Dict[UIDStr, Set[UIDStr]] = defaultdict(set)
        var_to_tt_direct: Dict[UIDStr, Set[UIDStr]] = defaultdict(set)

        # From rows
        for rid, vs in enumerate(row_vars):
            ps = row_params[rid]
            for v in vs:
                for p in ps:
                    if p in permitted_xi:
                        if p in non_temporal_xi:
                            var_to_ti_direct[v].add(p)
                        else:
                            var_to_tt_direct[v].add(p)

        # From objective (per-var ξ costs)
        for v, blocks in md.obj_var_coef.items():
            for pk, coef in blocks.items():
                if pk == CONST_TOKEN:
                    continue
                if pk in permitted_xi:
                    if pk in non_temporal_xi:
                        var_to_ti_direct[v].add(pk)
                    else:
                        var_to_tt_direct[v].add(pk)

        # k-hop collector
        def khop_collect(start_v: UIDStr, k_: int, direct: Dict[UIDStr, Set[UIDStr]]) -> Set[UIDStr]:
            if k_ <= 0:
                return set(direct[start_v])
            seen = {start_v}
            q = deque([(start_v, 0)])
            got: Set[UIDStr] = set()
            while q:
                v, d = q.popleft()
                got |= direct[v]
                if d == k_:
                    continue
                for nb in var_adj[v]:
                    if nb not in seen:
                        seen.add(nb)
                        q.append((nb, d + 1))
            return got

        # Compute mapping
        out: Dict[UIDStr, Set[UIDStr]] = {}
        all_vs = list(all_var_uids(md))  # include y_lb:/y_ub: etc.

        for v in all_vs:
            vb = base_name(v)

            # Non-temporal via k-hop
            ti = khop_collect(v, knn, var_to_ti_direct)

            # Temporal: windowing by default; optionally k-hop if requested
            if khop_temporal:
                tt = khop_collect(v, knn, var_to_tt_direct)
            else:
                tt = set()
                vtpos = v_tpos(vb)
                if vtpos is not None:
                    tv = time_of(v, vtpos)
                    past, future = v_window(vb)
                    for p in temporal_xi:
                        ptpos = p_tpos(base_name(p))
                        if ptpos is None:
                            continue
                        tp = time_of(p, ptpos)
                        if tp is None or tv is None:
                            continue
                        if (past is not None and tp < tv - past) or (future is not None and tp > tv + future):
                            continue
                        tt.add(p)
                # if var has no t_pos in cfg, it simply doesn't get temporal ξ by windowing

            out[v] = (ti | tt)

        # Final filter (should be no-op because we used permitted_xi)
        allowed = set(md.uncertain_uids)
        for v in out:
            out[v] = {p for p in out[v] if p in allowed and p in permitted_xi}

        return out

    # ---------- end-to-end (returns both sides) ----------
    def extract_data(
        self,
        base: pyo.ConcreteModel,
        uncertain_params: Sequence[Union[Param, ParamData]],
        param_box: Sequence[Bounds],
        *,
        primal_manual_map: Optional[Dict[Any, Iterable[Any]]] = None,
        dual_manual_map: Optional[Dict[Any, Iterable[Any]]] = None,
        knn: int = 1,
        khop_temporal: bool = False,
    ) -> Tuple[ModelData, Dict[UIDStr, Set[UIDStr]], ModelData, Dict[UIDStr, Set[UIDStr]]]:
        """
        Run full pipeline:
            - extract primal ModelData
            - build dual ModelData
            - build var→ξ maps for both (with optional separate configs)
        """
        primal_md = self.extract_primal_md(base, uncertain_params, param_box)
        dual_md = self.build_dual_md(primal_md)

        primal_map = self.build_var_xi_map(
            primal_md,
            manual_map=primal_manual_map,
            cfg=self.primal_cfg,
            knn=knn,
            khop_temporal=khop_temporal,
        )
        dual_map = self.build_var_xi_map(
            dual_md,
            manual_map=dual_manual_map,
            cfg=self.dual_cfg,  # you can pass a different config for dual here
            knn=knn,
            khop_temporal=khop_temporal,
        )
        return primal_md, primal_map, dual_md, dual_map
