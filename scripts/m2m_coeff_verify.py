#!/usr/bin/env python
"""
Verification of compressed M2M translation for Helmholtz kernel in 2D.

Compares two approaches:
1. Compress coefficients -> embed -> M2M translate
2. M2M translate with full coefficients
"""

import math
import numpy as np
import scipy.special as spsp
import sympy as sp
import pyopencl as cl
import sumpy.toys as t
from sumpy.kernel import HelmholtzKernel
from sumpy.expansion.multipole import (
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
    VolumeTaylorMultipoleExpansion
)
from sumpy.expansion.local import LinearPDEConformingVolumeTaylorLocalExpansion
from sumpy.array_context import _acf
from sumpy.tools import build_matrix


def to_scalar(val):
    if hasattr(val, 'evalf'):
        val = val.evalf()
    if hasattr(val, 'item'):
        val = val.item()
    return complex(val)


def setup_contexts(dim, extra_kwargs):
    """Create toy contexts for compressed and full multipole expansions."""
    kernel = HelmholtzKernel(dim)
    actx = _acf()

    toy_ctx_full = t.ToyContext(
        actx.context,
        kernel=kernel,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    toy_ctx_local = t.ToyContext(
        actx.context,
        kernel=kernel,
        local_expn_class=LinearPDEConformingVolumeTaylorLocalExpansion,
        extra_kernel_kwargs=extra_kwargs
    )

    return kernel, toy_ctx_full, toy_ctx_local


def build_projection_matrix(kernel, order, repl_dict):
    """Build numeric projection matrix from symbolic representation
    for LinearPDEConformingVolumeTaylorMultipoleExpansion."""
    mexpn = LinearPDEConformingVolumeTaylorMultipoleExpansion(kernel, order)
    wrangler = mexpn.expansion_terms_wrangler
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
    return build_matrix(numeric_op, dtype=np.complex128), mexpn


def verify_m2m_coefficients(kernel, order, M, mexpn, p2m_full, coeffs_full,
                            m_center1, m_center2, target, dim):
    """Verify M2M coefficients by comparing formula vs direct computation."""
    stored_identifiers = list(mexpn.get_coefficient_identifiers())
    full_identifiers = list(mexpn.get_full_coefficient_identifiers())
    
    is_stored = [mi in stored_identifiers for mi in full_identifiers]
    stored_indices = [i for i, st in enumerate(is_stored) if st]
    
    h = m_center2 - m_center1

    print(f'M2M Coefficient Verification for {type(kernel).__name__}:')
    print(f'm_center1 = {m_center1}')
    print(f'm_center2 = {m_center2}')
    print(f'h = m_center2 - m_center1 = {h}')
    print()
    print(f"{'k':>3s} | {'ν(k)':>15s} | {'|ν(k)|':6s} | "
          f"{'difference by formula':>31s} | {'difference by direct computation':>31s} | "
          f"{'abs err':>10s}")
    print("-" * 120)

    results = []
    mexpn_full_idx = VolumeTaylorMultipoleExpansion(kernel, order).get_full_coefficient_identifiers()

    for k, nu_k in enumerate(full_identifiers):
        k_card = sum(np.array(nu_k))
        alpha_k = 1

        true_k_idx = mexpn_full_idx.index(nu_k)
        
        basis_full = np.zeros(len(mexpn_full_idx), dtype=np.complex128)
        basis_full[true_k_idx] = alpha_k
        p2m_full_k = p2m_full.with_coeffs(basis_full)

        # M^T @ alpha
        basis_cmp = np.zeros(M.shape[0], dtype=np.complex128)
        basis_cmp[stored_indices] = M[k, :] * alpha_k
        
        # Embed back into full basis
        basis_cmp_full = np.zeros(len(mexpn_full_idx), dtype=np.complex128)
        for i, nu_i in enumerate(full_identifiers):
            if basis_cmp[i] != 0:
                true_i_idx = mexpn_full_idx.index(nu_i)
                basis_cmp_full[true_i_idx] = basis_cmp[i]
        
        p2m_cmp_k = p2m_full.with_coeffs(basis_cmp_full)

        p2m2m_cmp = t.multipole_expand(p2m_cmp_k, m_center2, order).eval(target)
        p2m2m_full = t.multipole_expand(p2m_full_k, m_center2, order).eval(target)
        direct_diff = (p2m2m_cmp - p2m2m_full)[0]

        error = 0.0 + 0.0j
        for s, nu_js in enumerate(stored_identifiers):
            nu_js_card = sum(np.array(nu_js))
            inner_sum = 0.0 + 0.0j
            
            if nu_js_card <= k_card:
                start_idx = math.comb(order - k_card + dim, dim)
                end_idx = math.comb(order - nu_js_card + dim, dim)
                
                for idx in range(start_idx, end_idx):
                    nu_l = full_identifiers[idx]
                    nu_sum = tuple(a + b for a, b in zip(nu_l, nu_js))
                    
                    if nu_sum not in full_identifiers:
                        continue
                    
                    derivative_idx = full_identifiers.index(nu_sum)
                    h_pow = np.prod(h ** np.array(nu_l))
                    fact_nu_l = np.prod(spsp.factorial(nu_l))
                    
                    inner_sum += coeffs_full[derivative_idx] * h_pow / fact_nu_l
            
            error += inner_sum * M[k, s]

        abs_err = abs(error - direct_diff)
        results.append((k, nu_k, error, direct_diff, abs_err))

        print(f"{k:3d} | {str(nu_k):>15s} | {k_card:6d} | "
              f"{error.real: .8e}{error.imag:+.8e}j | "
              f"{direct_diff.real: .8e}{direct_diff.imag:+.8e}j | "
              f"{abs_err:9.2e}")

    max_abs_error = max(r[4] for r in results)
    print(f"\nMaximum absolute error: {max_abs_error:.2e}")

    return max_abs_error, results


def main():
    dim = 2
    order = 7
    extra_kwargs = {"k": 0.1}
    repl_dict = {sp.Symbol('k'): 0.1}

    source = np.array([[0.0], [0.1]])
    strength = np.array([1.0])
    m_center1 = np.array([0.0, 0.0])
    offset_direction = np.array([-0.5, 0.25])
    c2_c1_dist = 0.1
    m_center2 = m_center1 + c2_c1_dist * offset_direction
    target = np.array([[2.0], [2.0]])

    kernel, toy_ctx_full, toy_ctx_local = setup_contexts(dim, extra_kwargs)
    
    p_full = t.PointSources(toy_ctx_full, source, weights=strength)
    p2m_full = t.multipole_expand(p_full, m_center1, order, rscale=1.0)

    M, mexpn = build_projection_matrix(kernel, order, repl_dict)
    
    # Compute derivatives at m_center2
    p_local = t.PointSources(toy_ctx_local, m_center2.reshape(2, 1), weights=strength)
    p2l = t.local_expand(p_local, target, order)
    coeffs_full = (M @ p2l.coeffs) * (1j / 4)

    max_abs_error, results = verify_m2m_coefficients(
        kernel, order, M, mexpn, p2m_full, coeffs_full, m_center1, m_center2, target, dim
    )

    return max_abs_error, results


if __name__ == "__main__":
    main()