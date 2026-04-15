import numpy as np
import pytest
import jax
import jax.numpy as jnp

import shtns

RTOL = 1e-10
ATOL = 1e-10


try:
    CUDA_DEVICES = jax.devices("cuda")
except RuntimeError:
    CUDA_DEVICES = []
GPU_AVAILABLE = len(CUDA_DEVICES) > 0

def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)
    return sh


def _spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j
    return jnp.array(qlm, dtype=jnp.complex128)


def _spatial_real_input(sh, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((sh.nphi, sh.nlat)), dtype=jnp.float64)


def _cplx_spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    alm = rng.standard_normal(sh.nlm_cplx) + 1j * rng.standard_normal(sh.nlm_cplx)
    return jnp.array(alm, dtype=jnp.complex128)


def _cplx_spatial_input(sh, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sh.nphi, sh.nlat)) + 1j * rng.standard_normal((sh.nphi, sh.nlat))
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
    v_cp = v.copy()
    _, jvp_out = jax.jvp(fn, (x,), (v_cp,))
    direct = fn(v)
    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_vjp(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    print("x shape: ", x.shape)
    print("x dtype: ", x.dtype)
    out, pullback = jax.vjp(fn, x)
    print("output shape:", out.shape)
    print("output dtype:", out.dtype)
    cotangent = jnp.ones_like(out)
    print("cotangent shape:", cotangent.shape)
    print("cotangent dtype:", cotangent.dtype)
    (cot_in,) = pullback(cotangent)
    assert cot_in.shape == x.shape
    assert cot_in.dtype == x.dtype

@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_no_input_modification(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    x_copy = x.copy()
    _ = fn(x)

    assert jnp.array_equal(x, x_copy), f"{fn_name} modified its input array."


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_cuda_implementation(fn_name, make_input):
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    x_cuda = jax.device_put(x, device=CUDA_DEVICES[0])
    y_cuda = fn(x_cuda)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
def test_theta_contiguous_cuda():
    # This one asserts that an error is thrown
    sh = shtns.sht(8, 8)
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_PHI_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")
    f_const = np.full(phi_grid.shape, 3.0, dtype=np.float64)
    
    # Check that the analys_jax and synth_jax perform as expected on this grid.
    with pytest.raises(ValueError, match="SHT_PHI_CONTIGUOUS"):
        _ = sh.analys_jax(f_const)

    qlm_jax = jnp.zeros(sh.nlm, dtype=jnp.complex128)

    with pytest.raises(ValueError, match="SHT_PHI_CONTIGUOUS"):
        _ = sh.synth_jax(qlm_jax)