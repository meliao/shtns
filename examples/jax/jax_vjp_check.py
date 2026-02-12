"""Checks the vjp implementation of synth_jax and analys_jax"""

import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

import shtns

from jax_integration_checks import inner_product_gridspace

jax.config.update("jax_enable_x64", True)


def _parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Compare JVP to centered finite differences for SHTns JAX bindings."
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=20,
        help="Maximum spherical harmonic degree.",
    )
    return parser.parse_args()

def inner_product_coeffspace_real(
    qlm: jax.Array, plm: jax.Array, m_vals: np.ndarray
) -> jax.Array:
    mask = jnp.asarray(m_vals > 0)
    base = jnp.vdot(qlm, plm)
    extra = jnp.vdot(qlm * mask, plm)
    return jnp.real(base + extra)

def main():
    args = _parse_args()
    sh = shtns.sht(args.lmax, args.lmax)
    # theta is latitude, phi is longitude
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")



    # Define a test function in the spectral domain
    rng = np.random.default_rng(0)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j 

    # Define another test function in the spectral domain.
    qlm_v = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm_v[sh.m == 0] = qlm_v[sh.m == 0].real + 0j
    v = sh.synth_jax(qlm_v)

    # Let F be the spectral -> spat transform. Implemented in synth_jax.
    # Let J be its Jacobian. F is linear, so F = J.
    # Let F^* be the adjoint of F. We can compute F^* v using jax.vjp applied to synth_jax.
    # We want to check that <F qlm, v> = <qlm, F^* v> for some test vector v

    F_qlm = sh.synth_jax(qlm)
    _, vjp_fun = jax.vjp(sh.synth_jax, qlm)
    F_star_v = vjp_fun(v)[0]

    # Compute LHS
    lhs = inner_product_gridspace(F_qlm, v, phi_grid, theta_grid, sh)

    print("LHS = <F qlm, v>:", lhs)
    # Compute RHS. This is an inner product in spectral domain.
    rhs = inner_product_coeffspace_real(qlm, F_star_v, sh.m)

    print("RHS = <qlm, F^* v>:", rhs)

    diff = lhs - rhs
    print("Difference:", diff)

if __name__ == "__main__":
    main()
