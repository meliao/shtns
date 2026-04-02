import argparse
import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import lineax as lx
import shtns


def _parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Check numerical integration in gridspace and coefficient space."
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=20,
        help="Maximum spherical harmonic degree.",
    )
    return parser.parse_args()

def apply_laplace_beltrami_diagonal(
    coeffs: jax.Array, lvals: jax.Array, mean_val_coeff: float = 0.0
) -> jax.Array:
    nrm_vals = -lvals * (lvals + 1)
    applied_coeffs = (
        coeffs * nrm_vals[:, None] if coeffs.ndim == 2 else coeffs * nrm_vals
    )
    applied_coeffs = applied_coeffs.at[0].set(mean_val_coeff * coeffs[0])
    return applied_coeffs

@partial(jax.jit, static_argnames=["sh"])
def matvec_screened_poisson(x: jax.Array, sh: shtns.sht, c_gridspace: jax.Array, lvals_jax: jax.Array) -> jax.Array:
    lap_coeffs = apply_laplace_beltrami_diagonal(x, lvals_jax)

    gridspace = sh.synth_jax(jnp.copy(x))
    mult_gridspace = c_gridspace * gridspace
    mult_coeffs = sh.analys_jax(jnp.copy(mult_gridspace))

    return lap_coeffs - mult_coeffs

def c(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """cos(sin(theta) * cos(phi) + 3) + 1.1"""
    # return np.zeros_like(theta)
    x_vals = np.cos(np.sin(theta) * np.cos(phi) + 3)
    return x_vals + 1.1


def main() -> None:
    args = _parse_args()
    sh = shtns.sht(args.lmax, args.lmax)
    # theta is latitude, phi is longitude
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    c_gridspace = c(theta_grid, phi_grid)
    c_gridspace_jax = jnp.array(c_gridspace, dtype=jnp.float64)
    lvals_jax = jnp.array(sh.l, dtype=jnp.float64)

    u_true = np.cos(theta_grid) + 0.2 * np.sin(theta_grid) * np.cos(phi_grid)
    u_true_jax = jnp.array(u_true, dtype=jnp.float64)
    qlm_true = sh.analys_jax(u_true_jax)

    matvec = partial(
        matvec_screened_poisson,
        sh=sh,
        c_gridspace=c_gridspace_jax,
        lvals_jax=lvals_jax,
    )
    rhs = matvec(qlm_true)

    operator = lx.FunctionLinearOperator(
        matvec, jax.eval_shape(lambda: qlm_true)
    )
    solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)
    solution = lx.linear_solve(
        operator,
        rhs,
        solver=solver,
        options={"y0": jnp.zeros_like(qlm_true)},
    )
    qlm_sol = solution.value

    u_sol = sh.synth_jax(qlm_sol)
    rel_err = jnp.linalg.norm(u_sol - u_true_jax) / jnp.linalg.norm(u_true_jax)
    print("GMRES result=%s stats=%s", solution.result, solution.stats)
    print("Relative error (gridspace) = %.3e", float(rel_err))

if __name__ == "__main__":
    main()
