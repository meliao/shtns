import numpy as np
import jax
import jax.numpy as jnp

import shtns

RTOL = 1e-10
ATOL = 1e-10

def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)
    return sh


def _random_spectral_data(sh: shtns.sht, batch: int, seed: int=0):
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal((batch, sh.nlm)) + 1j * rng.standard_normal((batch, sh.nlm))
    # For real spatial fields, m=0 coefficients must be real.
    try:
        m = sh.m
        qlm[:, m == 0] = qlm[:, m == 0].real + 0j
    except Exception:
        pass
    return qlm


def _random_spatial_data(sh: shtns.sht, batch: int, seed: int=0):
    rng = np.random.default_rng(seed)
    spat = rng.standard_normal((batch, sh.nlat, sh.nphi))
    return spat

def test_jit_vmap_synth_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    qlm = _random_spectral_data(sh, batch, seed=1)

    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)

    f = jax.jit(jax.vmap(sh.synth_jax))
    out_jax = f(qlm_jax)

    out_np = np.stack([sh.synth(qlm[i]) for i in range(batch)], axis=0)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)


def test_vmap_jit_synth_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    qlm = _random_spectral_data(sh, batch, seed=2)

    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)

    f = jax.vmap(jax.jit(sh.synth_jax))
    out_jax = f(qlm_jax)

    out_np = np.stack([sh.synth(qlm[i]) for i in range(batch)], axis=0)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)


def test_jit_vmap_analys_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    qlm = _random_spectral_data(sh, batch, seed=3)
    # Transform to spatial domain using synth_jax
    spat_jax = sh.synth_jax(qlm)

    f = jax.jit(jax.vmap(sh.analys_jax))
    qlm_from_jax = np.array(f(spat_jax)).reshape(batch, sh.nlm)

    assert np.allclose(qlm_from_jax, qlm, rtol=RTOL, atol=ATOL)


def test_vmap_jit_analys_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    qlm = _random_spectral_data(sh, batch, seed=4)

    spat_jax = sh.synth_jax(qlm)

    f = jax.vmap(jax.jit(sh.analys_jax))
    qlm_from_jax = np.array(f(spat_jax)).reshape(batch, sh.nlm)

    assert np.allclose(qlm_from_jax, qlm, rtol=RTOL, atol=ATOL)


def test_jvp_synth_equals_apply_to_tangent():
    sh = _make_cfg(8, 8, 1)
    qlm = _random_spectral_data(sh, 1, seed=10)[0]
    v_qlm = _random_spectral_data(sh, 1, seed=11)[0]

    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
    v_qlm_jax = jnp.array(v_qlm, dtype=jnp.complex128)

    _, jvp_out = jax.jvp(sh.synth_jax, (qlm_jax,), (v_qlm_jax,))
    direct = sh.synth_jax(v_qlm_jax)

    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


def test_jvp_analys_equals_apply_to_tangent():
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(12)
    spat = rng.standard_normal((sh.nlat, sh.nphi))
    v_spat = rng.standard_normal((sh.nlat, sh.nphi))

    spat_jax = jnp.array(spat, dtype=jnp.float64)
    v_spat_jax = jnp.array(v_spat, dtype=jnp.float64)

    _, jvp_out = jax.jvp(sh.analys_jax, (spat_jax,), (v_spat_jax,))
    direct = sh.analys_jax(v_spat_jax)

    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


def test_vjp_synth_runs_without_error():
    sh = _make_cfg(8, 8, 1)
    qlm = _random_spectral_data(sh, 1, seed=13)[0]
    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)

    out, pullback = jax.vjp(sh.synth_jax, qlm_jax)
    cotangent = jnp.ones_like(out)
    (cot_in,) = pullback(cotangent)

    assert cot_in.shape == qlm_jax.shape
    assert cot_in.dtype == qlm_jax.dtype
