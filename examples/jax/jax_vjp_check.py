"""Checks the vjp implementation of synth_jax and analys_jax"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp

import shtns

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


def grid_weights(sht: shtns.sht) -> np.ndarray:
    wts = np.array(sht.gauss_wts(), dtype=np.float64)
    if wts.shape[0] * 2 == sht.nlat:
        wts = np.hstack((wts, np.flip(wts)))
    elif wts.shape[0] != sht.nlat:
        wts = np.ones((sht.nlat,), dtype=np.float64)

    wts = wts * (2.0 * np.pi) / sht.nphi
    return wts


def inner_product_gridspace(
    f: np.ndarray,
    g: np.ndarray,
    phi_grid: np.ndarray,
    theta_grid: np.ndarray,
    sht: shtns.sht,
) -> float:
    """
    Approximate the integral of f * g over the sphere using the quadrature weights from shtns.

    f and g are represented on the points defined by phi_grid and theta_grid.
    """
    if f.shape != g.shape:
        raise ValueError("f and g must have the same shape.")
    if f.shape[-2:] != sht.spat_shape:
        raise ValueError("Input arrays must end with the grid shape from set_grid().")

    wts = grid_weights(sht)

    # Match weights to the active grid layout.
    if sht.spat_shape == (sht.nlat, sht.nphi):
        wts = wts.reshape((sht.nlat, 1))
    elif sht.spat_shape == (sht.nphi, sht.nlat):
        wts = wts.reshape((1, sht.nlat))

    wts = jnp.asarray(wts)
    return jnp.sum(jnp.conj(f) * g * wts)


def main():
    args = _parse_args()
    sh = shtns.sht(args.lmax, args.lmax)
    # theta is latitude, phi is longitude
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    f_const = np.full(phi_grid.shape, 3.0, dtype=np.float64)
    ones = np.ones_like(f_const)
    const_integral = inner_product_gridspace(f_const, ones, phi_grid, theta_grid, sh)
    expected_integral = 3.0 * 4.0 * np.pi

    print("Integral of f(theta, phi)=3 over sphere:", const_integral)
    print("Expected integral:", expected_integral)

    qlm_const = sh.analys_jax(jnp.array(f_const, dtype=jnp.float64))
    const_l2 = jnp.linalg.norm(qlm_const)
    print("L2 norm of spectral coefficients for f=3:", const_l2)

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

    F_qlm = sh.synth_jax(jnp.copy(qlm))
    _, vjp_fun = jax.vjp(sh.synth_jax, jnp.copy(qlm))
    F_star_v = vjp_fun(jnp.copy(v))[0]

    F_star_v_np = sh.adjoint_synth(np.array(v))

    # Compute LHS
    lhs = inner_product_gridspace(F_qlm, v, phi_grid, theta_grid, sh)

    print("LHS = <F qlm, v>:", lhs)
    # Compute RHS. This is an inner product in spectral domain.
    rhs = jnp.vdot(qlm, jnp.conj(F_star_v_np))

    print("RHS = <qlm, F^* v>:", rhs)

    diff = lhs - rhs
    print("Difference:", diff)
    rel_err = np.abs(diff) / max(1.0, np.abs(lhs), np.abs(rhs))
    print("Relative error:", rel_err)


if __name__ == "__main__":
    main()
