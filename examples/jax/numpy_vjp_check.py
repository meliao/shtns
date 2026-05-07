"""Checks that synth and analys are numericall adjoint"""

import argparse
import numpy as np

import shtns
import matplotlib.pyplot as plt


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
    """Weights for the grid points"""

    # TODO try numpy leggauss weights instead of shtns
    theta_pos_wts = np.array(sht.gauss_wts(), dtype=np.float64)
    # # Need to flip and concat the weights
    theta_wts = np.concatenate((theta_pos_wts, theta_pos_wts[::-1]))

    # theta_wts = np.ones((sht.nlat,), dtype=np.float64) * (np.pi / sht.nlat)

    phi_weights = np.ones((sht.nphi,), dtype=np.float64) * (2.0 * np.pi) / sht.nphi

    wts = theta_wts[None, :] * phi_weights[:, None]
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
    print("Weights shape:", wts.shape)
    print("f shape:", f.shape)
    print("g shape:", g.shape)

    return np.sum(np.conj(f) * g * wts)


def main():
    args = _parse_args()
    sh = shtns.sht(args.lmax, args.lmax)
    # theta is latitude, phi is longitude
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)

    # Plot cos(theta) values and uniform-spaced cos(theta) values to compare
    theta_vals = np.linspace(0, np.pi, ntheta + 1, endpoint=False)[1:]

    #
    thetas = np.arccos(sh.cos_theta)
    pts, _ = np.polynomial.legendre.leggauss(ntheta)
    pts = pts + 1.0
    pts = pts * (np.pi / 2.0)
    # TODO: Why are thetas not G-L spaced?
    plt.plot(thetas, np.zeros_like(thetas), "o")
    plt.plot(pts, np.zeros_like(pts), "x")
    plt.show()
    plt.close()
    # Display range of thetas
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    # Function is cos(theta) + 3. The integral of cos(theta) over the sphere is 0, so the integral of f should be 3 * area of sphere = 3 * 4 * pi
    f_example = 4 * np.cos(theta_grid) + 3.0
    const_integral = inner_product_gridspace(
        f_example, f_example, phi_grid, theta_grid, sh
    )
    expected_integral = (4.0 * 2 / 3) + 9.0 * 4.0 * np.pi

    print("Integral of f(theta, phi)=3 over sphere:", const_integral)
    print("Expected integral:", expected_integral)

    qlm_example = sh.analys(np.copy(f_example))
    l_vals = sh.l
    l_nrm_factor = np.sqrt((2.0 * l_vals + 1.0) / (4.0 * np.pi))
    qlm_example = qlm_example / l_nrm_factor
    const_l2 = np.linalg.norm(qlm_example)
    print("L2 norm of spectral coefficients for f=3:", const_l2)

    ########################################################################
    # Evaluate the ISHT of a sparse coeff vector which gives us cos(theta)
    # in gridspace. Then plot the result for a single phi slice.

    qlm_cos_theta = np.zeros(sh.nlm, dtype=np.complex128)
    qlm_cos_theta[sh.idx(l=1, m=0)] = 2.0 * np.sqrt(np.pi / 3.0)

    # Compute the inverse spherical harmonic transform (ISHT)
    f_gridspace = sh.synth(np.copy(qlm_cos_theta))
    # Plot the result for a single phi slice
    plt.plot(f_gridspace[0, :], "x-", label="ISHT of cos(theta) coeffs")
    plt.plot(np.cos(theta_vals), "o-", label="cos(theta) values")
    plt.xlabel("theta")
    plt.ylabel("f(theta)")
    plt.legend()
    plt.title("ISHT of cos(theta) coefficients vs cos(theta) values")
    plt.show()
    plt.close()

    # Define a test function in the spectral domain
    rng = np.random.default_rng(0)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j

    # Define another test function in the spectral domain.
    qlm_v = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm_v[sh.m == 0] = qlm_v[sh.m == 0].real + 0j
    v = sh.synth(np.copy(qlm_v))

    # Let F be the spectral -> spat transform. Implemented in synth_jax.
    # Let J be its Jacobian. F is linear, so F = J.
    # Let F^* be the adjoint of F. We can compute F^* v using analys
    # We want to check that <F qlm, v> = <qlm, F^* v> for some test vector v

    F_qlm = sh.synth(np.copy(qlm))
    F_star_v = sh.analys(np.copy(v))

    # Compute LHS
    lhs = inner_product_gridspace(F_qlm, v, phi_grid, theta_grid, sh)

    print("LHS = <F qlm, v>:", lhs)
    # Compute RHS. This is an inner product in spectral domain.
    # Scale qlm by the norm factor
    qlm /= l_nrm_factor
    rhs = np.vdot(qlm, np.conj(F_star_v))

    print("RHS = <qlm, F^* v>:", rhs)

    diff = lhs - rhs
    print("Difference:", diff)
    rel_err = np.abs(diff) / max(1.0, np.abs(lhs), np.abs(rhs))
    print("Relative error:", rel_err)


if __name__ == "__main__":
    main()
