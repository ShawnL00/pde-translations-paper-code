#!/usr/bin/env python
"""
Verification for the difference between compressed L2L coefficients and
full L2L coefficients for Yukawa kernel in 2D.
"""

import math
import numpy as np
import scipy.special as spsp
import sympy as sp
import pyopencl as cl
import sumpy.toys as t
from sumpy.kernel import YukawaKernel
from sumpy.expansion.local import (
    LinearPDEConformingVolumeTaylorLocalExpansion,
    VolumeTaylorLocalExpansion
)
from sumpy.array_context import _acf
from sumpy.tools import build_matrix


def to_scalar(val):
    if hasattr(val, 'evalf'):
        val = val.evalf()
    if hasattr(val, 'item'):
        val = val.item()
    return complex(val)


def setup_contexts(dim, extra_kwargs):
    kernel = YukawaKernel(dim)
    actx = _acf()

    toy_ctx = t.ToyContext(
        actx.context,
        kernel=kernel,
        local_expn_class=LinearPDEConformingVolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    toy_ctx_full = t.ToyContext(
        actx.context,
        kernel=kernel,
        local_expn_class=VolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    return kernel, toy_ctx, toy_ctx_full


def compute_expansions(toy_ctx, toy_ctx_full, source, strength, c1, c2, order):
    """Compute P2L and L2L expansions for both compressed and full."""
    p = t.PointSources(toy_ctx, source, weights=strength)
    p_full = t.PointSources(toy_ctx_full, source, weights=strength)

    p2l = t.local_expand(p, c1, order, rscale=1.0)
    p2l2l = t.local_expand(p2l, c2, order, rscale=1.0)

    p2l_full = t.local_expand(p_full, c1, order, rscale=1.0)
    p2l2l_full = t.local_expand(p2l_full, c2, order, rscale=1.0)

    return p2l, p2l2l, p2l_full, p2l2l_full


def build_projection_matrix(kernel, order, repl_dict):
    """Build numeric projection matrix from symbolic representation
    for LinearPDEConformingVolumeTaylorLocalExpansion."""
    p2l2l_expn = LinearPDEConformingVolumeTaylorLocalExpansion(kernel, order)
    wrangler = p2l2l_expn.expansion_terms_wrangler
    M_symbolic = wrangler.get_projection_matrix(rscale=1.0)

    class NumericMatVecOperator:
        def __init__(self, symbolic_op, repl_dict):
            self.symbolic_op = symbolic_op
            self.repl_dict = repl_dict
            self.shape = symbolic_op.shape

        def matvec(self, vec):
            result = self.symbolic_op.matvec(vec)
            numeric_result = []
            for expr in result:
                if hasattr(expr, 'xreplace'):
                    numeric_result.append(
                        complex(expr.xreplace(self.repl_dict).evalf())
                    )
                else:
                    numeric_result.append(complex(expr))
            return np.array(numeric_result)

    numeric_op = NumericMatVecOperator(M_symbolic, repl_dict)
    return build_matrix(numeric_op, dtype=np.complex128), p2l2l_expn


def verify_l2l_coefficients(kernel, order, M, p2l_full, p2l2l, p2l2l_full,
                            mu_c, c1, c2, dim):
    """Verify L2L coefficients by comparing formula vs direct computation."""
    p2l2l_expn = LinearPDEConformingVolumeTaylorLocalExpansion(kernel, order)
    stored_identifiers = list(p2l2l_expn.get_coefficient_identifiers())
    full_identifiers = list(p2l2l_expn.get_full_coefficient_identifiers())
    
    h = c2 - c1
    global_const = 1j / 4

    print(f'L2L Coefficient Verification for {type(kernel).__name__}:')
    print(f'c1 = {c1}')
    print(f'c2 = {c2}')
    print(f'h = c2 - c1 = {h}')
    print()
    print(f"{'i':>3s} | {'ν(i)':>15s} | {'|ν(i)|':6s} | "
          f"{'difference by formula':>31s} | {'difference by direct computation':>31s} | "
          f"{'abs err':>10s}")
    print("-" * 104)

    results = []
    lexpn_idx = VolumeTaylorLocalExpansion(kernel, order).get_full_coefficient_identifiers()

    for i, nu_i in enumerate(full_identifiers):
        i_card = sum(np.array(nu_i))

        error = 0.0 + 0.0j
        for k, nu_jk in enumerate(stored_identifiers):
            jk_card = sum(np.array(nu_jk))
            if jk_card >= i_card:
                continue

            start_idx = math.comb(order - i_card + dim, dim)
            end_idx = math.comb(order - jk_card + dim, dim)

            for q_idx in range(start_idx, end_idx):
                nu_q = full_identifiers[q_idx]
                nu_sum = tuple(map(sum, zip(nu_q, nu_jk)))
                if nu_sum not in full_identifiers:
                    continue

                deriv_idx = full_identifiers.index(nu_sum)
                gamma_deriv = to_scalar(p2l_full.coeffs[deriv_idx])
                h_pow = np.prod(h ** np.array(nu_q))
                fact_nu_q = np.prod(spsp.factorial(nu_q))

                error += -M[i, k] * gamma_deriv * h_pow / fact_nu_q

        error /= np.prod(spsp.factorial(nu_i))

        true_i_idx = lexpn_idx.index(nu_i)
        mu_full = to_scalar(p2l2l_full.coeffs[true_i_idx])
        direct_diff = (mu_full - mu_c[i]) / np.prod(spsp.factorial(nu_i))

        error *= global_const
        direct_diff *= global_const

        abs_err = abs(error - direct_diff)
        results.append((i, nu_i, error, direct_diff, abs_err))

        print(f"{i:3d} | {str(nu_i):>15s} | {i_card:6d} | "
              f"{error.real: .8e}{error.imag:+.8e}j | "
              f"{direct_diff.real: .8e}{direct_diff.imag:+.8e}j | "
              f"{abs_err:9.2e}")

    max_abs_error = max(r[4] for r in results)
    print(f"\nMaximum absolute error: {max_abs_error:.2e}")

    return max_abs_error, results


def main():
    dim = 2
    order = 7
    extra_kwargs = {"lam": 0.1}
    repl_dict = {sp.Symbol('lam'): 0.1}

    source = np.array([[5.0], [5.0]])
    strength = np.array([1.0])
    c1 = np.array([0.0, 0.0])
    c2_c1_dist = 1.0
    c2 = c1 + c2_c1_dist * np.array([-0.5, 1.0])

    kernel, toy_ctx, toy_ctx_full = setup_contexts(dim, extra_kwargs)
    
    p2l, p2l2l, p2l_full, p2l2l_full = compute_expansions(
        toy_ctx, toy_ctx_full, source, strength, c1, c2, order
    )

    M, p2l2l_expn = build_projection_matrix(kernel, order, repl_dict)
    
    mu_c_symbolic = (
        p2l2l_expn.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
            p2l2l.coeffs, rscale=1.0
        )
    )
    mu_c = []
    for coeff in mu_c_symbolic:
        if hasattr(coeff, 'xreplace'):
            mu_c.append(to_scalar(coeff.xreplace(repl_dict)))
        else:
            mu_c.append(to_scalar(coeff))

    max_abs_error, results = verify_l2l_coefficients(
        kernel, order, M, p2l_full, p2l2l, p2l2l_full, mu_c, c1, c2, dim
    )

    return max_abs_error, results


if __name__ == "__main__":
    main()