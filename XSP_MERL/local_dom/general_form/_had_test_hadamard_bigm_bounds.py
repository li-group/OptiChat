"""Unit tests for the pure Hadamard-Cramer Big-M calculations.

Run from local_dom/general_form/ with:
    pytest -q test_hadamard_bigm_bounds.py
"""

import math
import numpy as np

from _had_local_dominance_radius_hadamard_bigm import derive_hadamard_big_m_constants


def test_rank_two_hadamard_equals_factorial_factor():
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([3.0, 5.0])
    out = derive_hadamard_big_m_constants(W, d, Bmax=7.0)
    assert math.isclose(out["hadamard_factor"], 2.0)
    assert math.isclose(out["factorial_factor"], 2.0)
    assert math.isclose(out["M_primal"], out["factorial_M_primal"])
    assert math.isclose(out["M_mu"], out["factorial_M_mu"])


def test_hadamard_factor_is_strictly_smaller_from_rank_three():
    W = np.eye(3)
    d = np.ones(3)
    out = derive_hadamard_big_m_constants(W, d, Bmax=2.0)
    assert out["hadamard_factor"] < out["factorial_factor"]
    assert out["M_primal"] < out["factorial_M_primal"]
    assert out["M_mu"] < out["factorial_M_mu"]


def test_componentwise_reduced_cost_bounds_do_not_exceed_dense_uniform_bound():
    W = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    d = np.array([2.0, 5.0, 7.0, 11.0])
    out = derive_hadamard_big_m_constants(W, d, Bmax=4.0)
    rc = np.asarray(out["M_reduced_cost_componentwise"])
    uniform = float(out["M_dual_slack_uniform_writeup"])
    assert np.all(rc <= uniform + 1e-12)


def test_wbar_max_includes_identity_slack_block():
    W = np.zeros((3, 2))
    d = np.array([2.0, 4.0])
    out = derive_hadamard_big_m_constants(W, d, Bmax=1.0)
    assert out["Wbar_max_abs"] == 1.0
