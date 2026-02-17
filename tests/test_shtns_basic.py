import numpy as np
import pytest
import logging

import shtns


import jax
import jax.numpy as jnp

RTOL = 1e-10
ATOL = 1e-10
CUDA_DEVICES =  jax.devices("cuda")
GPU_AVAILABLE = len(CUDA_DEVICES) > 0

def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_PHI_CONTIGUOUS)
    return sh


def test_nlm_calc_matches_cfg():
    sh = _make_cfg(8, 5, 1)
    assert shtns.nlm_calc(sh.lmax, sh.mmax, sh.mres) == sh.nlm
    assert shtns.nlm_cplx_calc(sh.lmax, sh.mmax, sh.mres) == sh.nlm_cplx


def test_synth_analys_roundtrip_scalar():
    sh = _make_cfg(10, 10, 1)
    rng = np.random.default_rng(0)
    qlm = rng.standard_normal(sh.nlm) + 0j

    spat = sh.synth(qlm)
    qlm_back = sh.analys(spat)

    assert np.allclose(qlm_back, qlm, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(GPU_AVAILABLE, reason="CPU-only test")
def test_synth_jax_complex_matches_numpy_cpu(caplog):
    caplog.set_level(logging.INFO)
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(1)
    qlm = rng.standard_normal(sh.nlm) + 0j


    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
    qlm_jax = jax.device_put(qlm_jax, device=jax.devices("cpu")[0])
    logging.info("test_synth_jax_complex_matches_numpy_cpu: qlm_jax device: %s", qlm_jax.devices())

    out_np = sh.synth(qlm)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-only test")
def test_synth_jax_complex_matches_numpy_cuda(caplog):
    caplog.set_level(logging.INFO)
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(1)
    qlm = rng.standard_normal(sh.nlm) + 0j

    qlm_jax = jnp.array(qlm, dtype=jnp.complex128)
    qlm_jax = jax.device_put(qlm_jax, device=jax.devices("cuda")[0])
    logging.info("test_synth_jax_complex_matches_numpy_cuda: qlm_jax device: %s", qlm_jax.devices())
    out_jax = sh.synth_jax(qlm_jax)

    out_np = sh.synth(qlm)
    assert np.allclose(np.array(out_jax), out_np, rtol=RTOL, atol=ATOL)

@pytest.mark.skipif(GPU_AVAILABLE, reason="CPU-only test")
def test_analys_jax_matches_numpy_cpu():

    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(2)
    qlm = rng.standard_normal(sh.nlm) + 0j

    spat = sh.synth(qlm)
    spat_jax = jnp.array(spat, dtype=jnp.float64)
    spat_jax = jax.device_put(spat_jax, device=jax.devices("cpu")[0])
    qlm_jax = sh.analys_jax(spat_jax)

    qlm_from_jax = np.array(qlm_jax)


    assert qlm_from_jax.shape == (sh.nlm,)
    assert np.allclose(qlm_from_jax, qlm, rtol=RTOL, atol=ATOL)

@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA-only test")
def test_analys_jax_matches_numpy_cuda():
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(22)
    qlm = rng.standard_normal(sh.nlm) + 0j

    spat = sh.synth(qlm)
    spat_cuda = jax.device_put(jnp.array(spat, dtype=jnp.float64), device=jax.devices("cuda")[0])
    qlm_from_jax = np.array(jax.jit(sh.analys_jax, backend="cuda")(spat_cuda))

    assert qlm_from_jax.shape == (sh.nlm,)
    assert np.allclose(qlm_from_jax, qlm, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(GPU_AVAILABLE, reason="CPU-only test")
def test_theta_contiguous_cpu():
    # This one asserts the basic functionality on the CPU.
    sh = shtns.sht(8, 8)
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")
    f_const = np.full(phi_grid.shape, 3.0, dtype=np.float64)
    
    # Check that the analys_jax and synth_jax perform as expected on this grid.
    qlm = sh.analys_jax(f_const)
    assert qlm.shape == (sh.nlm,)

    f_const_back = sh.synth_jax(qlm)
    assert f_const_back.shape == sh.spat_shape

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
def test_theta_contiguous_cuda():
    # This one asserts that an error is thrown
    sh = shtns.sht(8, 8)
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")
    f_const = np.full(phi_grid.shape, 3.0, dtype=np.float64)
    
    # Check that the analys_jax and synth_jax perform as expected on this grid.
    with pytest.raises(ValueError, match="SHT_THETA_CONTIGUOUS"):
        _ = sh.analys_jax(f_const)

    qlm_jax = jnp.zeros(sh.nlm, dtype=jnp.complex128)

    with pytest.raises(ValueError, match="SHT_THETA_CONTIGUOUS"):
        _ = sh.synth_jax(qlm_jax)


def test_jax_no_input_modification(caplog):
    """Tests that synth_jax() and analys_jax() do not modify their input arrays in-place."""

    caplog.set_level(logging.INFO)
    sh = _make_cfg(8, 8, 1)
    rng = np.random.default_rng(3)
    qlm = rng.standard_normal(sh.nlm) + 0j
    qlm_copy = qlm.copy()

    spat = sh.synth(qlm)
    spat_copy = spat.copy()

    # Call the JAX versions
    _ = sh.synth_jax(qlm)
    _ = sh.analys_jax(spat)

    # Check that the original arrays were not modified
    assert jnp.all(qlm == qlm_copy), "synth_jax modified its input array"
    assert jnp.all(spat == spat_copy), "analys_jax modified its input array"