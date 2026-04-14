import numpy as np
import pytest
import jax
import jax.numpy as jnp

import shtns

RTOL = 1e-10
ATOL = 1e-10


def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)
    return sh


def _spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j
    return jnp.array(qlm, dtype=jnp.complex128)


def _spatial_real_input(sh, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((sh.nlat, sh.nphi)), dtype=jnp.float64)


def _cplx_spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    alm = rng.standard_normal(sh.nlm_cplx) + 1j * rng.standard_normal(sh.nlm_cplx)
    return jnp.array(alm, dtype=jnp.complex128)


def _cplx_spatial_input(sh, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sh.nlat, sh.nphi)) + 1j * rng.standard_normal((sh.nlat, sh.nphi))
    return jnp.array(z, dtype=jnp.complex128)


TRANSFORMS = [
    pytest.param("synth_jax", _spectral_input, id="synth"),
    pytest.param("analys_jax", _spatial_real_input, id="analys"),
    pytest.param("synth_cplx_jax", _cplx_spectral_input, id="synth_cplx"),
    pytest.param("analys_cplx_jax", _cplx_spatial_input, id="analys_cplx"),
]


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_jit(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    expected = fn(x)
    out = jax.jit(fn)(x)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_vmap(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    batch = 4
    x = jnp.stack([make_input(sh, seed=i) for i in range(batch)])
    out = jax.vmap(fn)(x)
    expected_elem = fn(make_input(sh, seed=0))
    assert out.shape == (batch,) + expected_elem.shape
    assert out.dtype == expected_elem.dtype


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_jvp(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    v = make_input(sh, seed=1)
    _, jvp_out = jax.jvp(fn, (x,), (v,))
    direct = fn(v)
    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_vjp(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    out, pullback = jax.vjp(fn, x)
    cotangent = jnp.ones_like(out)
    (cot_in,) = pullback(cotangent)
    assert cot_in.shape == x.shape
    assert cot_in.dtype == x.dtype
