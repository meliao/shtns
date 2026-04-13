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
    direct = sh.analys_jax(v_spat_jax)

    _, jvp_out = jax.jvp(sh.analys_jax, (spat_jax,), (v_spat_jax,))

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


def test_vjp_analys_runs_without_error():
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(14)
    spat = rng.standard_normal((sh.nlat, sh.nphi))
    spat_jax = jnp.array(spat, dtype=jnp.float64)

    out, pullback = jax.vjp(sh.analys_jax, spat_jax)
    cotangent = jnp.ones_like(out)
    (cot_in,) = pullback(cotangent)

    assert cot_in.shape == spat_jax.shape
    assert cot_in.dtype == spat_jax.dtype


# ---- Complex transform tests ----

def _random_cplx_spectral_data(sh: shtns.sht, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    nlm_cplx = sh.nlm_cplx
    return rng.standard_normal((batch, nlm_cplx)) + 1j * rng.standard_normal((batch, nlm_cplx))


def _random_cplx_spatial_data(sh: shtns.sht, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((batch, sh.nlat, sh.nphi))
            + 1j * rng.standard_normal((batch, sh.nlat, sh.nphi)))


def test_jit_vmap_synth_cplx_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    alm = _random_cplx_spectral_data(sh, batch, seed=20)
    alm_jax = jnp.array(alm, dtype=jnp.complex128)

    f = jax.jit(jax.vmap(sh.synth_cplx_jax))
    out_jax = f(alm_jax)

    out_np = np.stack([sh.synth_cplx(alm[i]) for i in range(batch)], axis=0)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)


def test_vmap_jit_synth_cplx_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    alm = _random_cplx_spectral_data(sh, batch, seed=21)
    alm_jax = jnp.array(alm, dtype=jnp.complex128)

    f = jax.vmap(jax.jit(sh.synth_cplx_jax))
    out_jax = f(alm_jax)

    out_np = np.stack([sh.synth_cplx(alm[i]) for i in range(batch)], axis=0)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)


def test_jit_vmap_analys_cplx_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    alm = _random_cplx_spectral_data(sh, batch, seed=22)
    # Synthesise first to get valid complex spatial data
    z_jax = jax.vmap(sh.synth_cplx_jax)(jnp.array(alm, dtype=jnp.complex128))

    f = jax.jit(jax.vmap(sh.analys_cplx_jax))
    alm_from_jax = np.array(f(z_jax)).reshape(batch, sh.nlm_cplx)

    assert np.allclose(alm_from_jax, alm, rtol=RTOL, atol=ATOL)


def test_vmap_jit_analys_cplx_matches_numpy():
    sh = _make_cfg(8, 8, 1)
    batch = 4
    alm = _random_cplx_spectral_data(sh, batch, seed=23)
    z_jax = jax.vmap(sh.synth_cplx_jax)(jnp.array(alm, dtype=jnp.complex128))

    f = jax.vmap(jax.jit(sh.analys_cplx_jax))
    alm_from_jax = np.array(f(z_jax)).reshape(batch, sh.nlm_cplx)

    assert np.allclose(alm_from_jax, alm, rtol=RTOL, atol=ATOL)


def test_jvp_synth_cplx_equals_apply_to_tangent():
    sh = _make_cfg(8, 8, 1)
    alm = _random_cplx_spectral_data(sh, 1, seed=24)[0]
    v_alm = _random_cplx_spectral_data(sh, 1, seed=25)[0]

    alm_jax = jnp.array(alm, dtype=jnp.complex128)
    v_alm_jax = jnp.array(v_alm, dtype=jnp.complex128)

    _, jvp_out = jax.jvp(sh.synth_cplx_jax, (alm_jax,), (v_alm_jax,))
    direct = sh.synth_cplx_jax(v_alm_jax)

    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


def test_jvp_analys_cplx_equals_apply_to_tangent():
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(26)
    z = rng.standard_normal((sh.nlat, sh.nphi)) + 1j * rng.standard_normal((sh.nlat, sh.nphi))
    v_z = rng.standard_normal((sh.nlat, sh.nphi)) + 1j * rng.standard_normal((sh.nlat, sh.nphi))

    z_jax = jnp.array(z, dtype=jnp.complex128)
    v_z_jax = jnp.array(v_z, dtype=jnp.complex128)

    _, jvp_out = jax.jvp(sh.analys_cplx_jax, (z_jax,), (v_z_jax,))
    direct = sh.analys_cplx_jax(v_z_jax)

    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


def test_vjp_synth_cplx_runs_without_error():
    sh = _make_cfg(8, 8, 1)
    alm = _random_cplx_spectral_data(sh, 1, seed=27)[0]
    alm_jax = jnp.array(alm, dtype=jnp.complex128)

    out, pullback = jax.vjp(sh.synth_cplx_jax, alm_jax)
    cotangent = jnp.ones_like(out)
    (cot_in,) = pullback(cotangent)

    assert cot_in.shape == alm_jax.shape
    assert cot_in.dtype == alm_jax.dtype


def test_vjp_analys_cplx_runs_without_error():
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(28)
    z = rng.standard_normal((sh.nlat, sh.nphi)) + 1j * rng.standard_normal((sh.nlat, sh.nphi))
    z_jax = jnp.array(z, dtype=jnp.complex128)

    out, pullback = jax.vjp(sh.analys_cplx_jax, z_jax)
    cotangent = jnp.ones_like(out)
    (cot_in,) = pullback(cotangent)

    assert cot_in.shape == z_jax.shape
    assert cot_in.dtype == z_jax.dtype