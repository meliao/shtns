"""Compare numerical integration in gridspace vs coefficient space."""

import argparse
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import shtns

jax.config.update("jax_enable_x64", True)


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


def integral_coeffspace(qlm: jnp.ndarray, sht: shtns.sht) -> jnp.ndarray:
    qlm00 = qlm[sht.idx(0, 0)]
    return (4.0 * np.pi) * jnp.real(qlm00) / sht.sh00_1()


def inner_product_coeffspace(qlm_a: jnp.ndarray, qlm_b: jnp.ndarray) -> jnp.ndarray:
    return jnp.vdot(qlm_a, qlm_b)


def check_integral(
    label: str,
    f_grid: np.ndarray,
    phi_grid: np.ndarray,
    theta_grid: np.ndarray,
    sht: shtns.sht,
    expected: float,
) -> None:
    ones = jnp.ones_like(f_grid)
    grid_integral = inner_product_gridspace(
        f_grid, ones, phi_grid, theta_grid, sht
    )
    qlm = sht.analys_jax(jnp.array(f_grid, dtype=jnp.float64))
    coeff_integral = integral_coeffspace(qlm, sht)

    diff_gridspace = jnp.abs(grid_integral - expected) / jnp.abs(expected)
    diff_coeffspace = jnp.abs(coeff_integral - expected) / jnp.abs(expected)

    print(
        f"{label} gridspace integral:",
        grid_integral,
        f"(relative error: {diff_gridspace:.2e})",
    )
    print(
        f"{label} coefficient-space integral:",
        coeff_integral,
        f"(relative error: {diff_coeffspace:.2e})",
    )


def check_inner_product_self(
    label: str,
    f_grid: np.ndarray,
    phi_grid: np.ndarray,
    theta_grid: np.ndarray,
    sht: shtns.sht,
    expected: float,
) -> None:
    grid_inner = inner_product_gridspace(
        f_grid, f_grid, phi_grid, theta_grid, sht
    )
    qlm = sht.analys_jax(jnp.array(f_grid, dtype=jnp.float64))
    coeff_inner = inner_product_coeffspace(qlm, qlm)
    diff_inner = jnp.abs(grid_inner - expected) / jnp.abs(expected)

    diff_coeff = jnp.abs(coeff_inner - expected) / jnp.abs(expected)
    print(
        f"{label} gridspace inner product:",
        grid_inner,
        f"(relative error: {diff_inner:.2e})",
    )
    print(f"{label} coefficient-space inner product:", coeff_inner, f"(relative error: {diff_coeff:.2e})")


def main() -> None:
    args = _parse_args()
    sh = shtns.sht(args.lmax, args.lmax)
    # theta is latitude, phi is longitude
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    f_const = np.full(phi_grid.shape, 3.0, dtype=np.float64)
    check_integral(
        "f(theta, phi)=3",
        f_const,
        phi_grid,
        theta_grid,
        sh,
        expected=3.0 * 4.0 * np.pi,
    )

    f_var = 3.0 * np.cos(theta_grid) + 4.0
    check_integral(
        "f(theta, phi)=3 cos(theta) + 4",
        f_var,
        phi_grid,
        theta_grid,
        sh,
        expected=4.0 * 4.0 * np.pi,
    )
    check_inner_product_self(
        "f(theta, phi)=3 cos(theta) + 4",
        f_var,
        phi_grid,
        theta_grid,
        sh,
        expected=76.0 * np.pi,
    )


if __name__ == "__main__":
    main()
