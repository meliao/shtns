import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import shtns

jax.config.update("jax_enable_x64", True)


def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(nl_order=2)
    return sh


def _real_field_spectral(sh, seed=0):
    # For real spatial fields, m=0 coefficients must be real.
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    try:
        qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j
    except Exception:
        pass
    return qlm


def _finite_diff(f, x, v, eps):
    return (f(x + eps * v) - f(x - eps * v)) / (2.0 * eps)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare JVP to centered finite differences for SHTns JAX bindings."
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=8,
        help="Maximum spherical harmonic degree (default: 8).",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=1e-8,
        help="Minimum epsilon for FD sweep (default: 1e-8).",
    )
    parser.add_argument(
        "--eps-max",
        type=float,
        default=1e-2,
        help="Maximum epsilon for FD sweep (default: 1e-2).",
    )
    parser.add_argument(
        "--eps-steps",
        type=int,
        default=13,
        help="Number of eps values in logspace (default: 13).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="jvp_fd_convergence.png",
        help="Output plot path (default: jvp_fd_convergence.png).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    sh = _make_cfg(args.lmax, args.lmax, 1)
    eps_values = np.logspace(
        np.log10(args.eps_min), np.log10(args.eps_max), args.eps_steps
    )

    # ---- synth_jax: spectral -> spatial (complex128 -> float64) ----
    qlm = _real_field_spectral(sh, seed=1)
    v_qlm = _real_field_spectral(sh, seed=2)

    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
    v_qlm_jax = jnp.array(v_qlm, dtype=jnp.complex128)

    f_synth = sh.synth_jax
    _, jvp_synth = jax.jvp(f_synth, (qlm_jax,), (v_qlm_jax,))
    jvp_synth_np = np.array(jvp_synth)

    rel_err_synth = []
    for eps in eps_values:
        fd_synth = _finite_diff(f_synth, qlm_jax, v_qlm_jax, eps)
        fd_synth_np = np.array(fd_synth)
        err = np.linalg.norm(jvp_synth_np - fd_synth_np)
        ref = np.linalg.norm(fd_synth_np)
        rel_err_synth.append(err / (ref + 1e-30))

    print("synth_jax JVP vs FD:")
    print("  eps:", eps_values)
    print("  rel err:", rel_err_synth)

    # ---- analys_jax: spatial -> spectral (float64 -> complex128) ----
    rng = np.random.default_rng(3)
    spat = rng.standard_normal((sh.nlat, sh.nphi))
    v_spat = rng.standard_normal((sh.nlat, sh.nphi))

    spat_jax = jnp.array(spat, dtype=jnp.float64)
    v_spat_jax = jnp.array(v_spat, dtype=jnp.float64)

    f_analys = sh.analys_jax
    _, jvp_analys = jax.jvp(f_analys, (spat_jax,), (v_spat_jax,))
    jvp_analys_np = np.array(jvp_analys)

    rel_err_analys = []
    for eps in eps_values:
        fd_analys = _finite_diff(f_analys, spat_jax, v_spat_jax, eps)
        fd_analys_np = np.array(fd_analys)
        err = np.linalg.norm(jvp_analys_np - fd_analys_np)
        ref = np.linalg.norm(fd_analys_np)
        rel_err_analys.append(err / (ref + 1e-30))

    print("analys_jax JVP vs FD:")
    print("  eps:", eps_values)
    print("  rel err:", rel_err_analys)

    plt.figure(figsize=(6, 4))
    plt.loglog(eps_values, rel_err_synth, marker="o", label="synth_jax")
    plt.loglog(eps_values, rel_err_analys, marker="s", label="analys_jax")
    plt.xlabel("eps (FD step size)")
    plt.ylabel("relative error")
    plt.title(f"JVP vs FD convergence (lmax={args.lmax})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
