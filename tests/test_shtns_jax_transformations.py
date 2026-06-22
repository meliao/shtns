"""Run these tests from the root of the repository with
```
python -m pytest tests/
```
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import shtns
import shtns_jax

RTOL = 1e-10
ATOL = 1e-10


try:
    CUDA_DEVICES = jax.devices("cuda")
except RuntimeError:
    CUDA_DEVICES = []
GPU_AVAILABLE = len(CUDA_DEVICES) > 0
CPU_DEVICES = jax.devices("cpu")


def _make_cfg(lmax=8, mmax=8, mres=1):
    sh = shtns_jax.sht(lmax, mmax, mres)
    sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS)
    return sh


def _spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal(sh.nlm) + 1j * rng.standard_normal(sh.nlm)
    qlm[sh.m == 0] = qlm[sh.m == 0].real + 0j
    return jnp.array(qlm, dtype=jnp.complex128)


def _spectral_vec_input(sh, seed):
    rng = np.random.default_rng(seed)
    qlm_vec = rng.standard_normal((3, sh.nlm)) + 1j * rng.standard_normal((3, sh.nlm))
    qlm_vec[:, sh.m == 0] = qlm_vec[:, sh.m == 0].real + 0j
    return jnp.array(qlm_vec, dtype=jnp.complex128)


def _spatial_real_input(sh, seed):
    rng = np.random.default_rng(seed)
    return jnp.array(rng.standard_normal((sh.nphi, sh.nlat)), dtype=jnp.float64)


def _spatial_vec_input(sh, seed):
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal((3, sh.nphi, sh.nlat))
    return jnp.array(vec, dtype=jnp.float64)


def _cplx_spectral_input(sh, seed):
    rng = np.random.default_rng(seed)
    alm = rng.standard_normal(sh.nlm_cplx) + 1j * rng.standard_normal(sh.nlm_cplx)
    return jnp.array(alm, dtype=jnp.complex128)


def _cplx_spatial_input(sh, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((sh.nphi, sh.nlat)) + 1j * rng.standard_normal(
        (sh.nphi, sh.nlat)
    )
    return jnp.array(z, dtype=jnp.complex128)


def _cplx_spectral_vec_input(sh, seed):
    rng = np.random.default_rng(seed)
    alm_vec = rng.standard_normal((3, sh.nlm_cplx)) + 1j * rng.standard_normal(
        (3, sh.nlm_cplx)
    )
    return jnp.array(alm_vec, dtype=jnp.complex128)

def _cplx_spatial_vec_input(sh, seed):
    rng = np.random.default_rng(seed)
    z_vec = rng.standard_normal((3, sh.nphi, sh.nlat)) + 1j * rng.standard_normal(
        (3, sh.nphi, sh.nlat)
    )
    return jnp.array(z_vec, dtype=jnp.complex128)

SCALAR_TRANSFORMS = [
    pytest.param("synth_jax", _spectral_input, id="synth"),
    pytest.param("analys_jax", _spatial_real_input, id="analys"),
    # pytest.param("synth_cplx_jax", _cplx_spectral_input, id="synth_cplx"),
    # pytest.param("analys_cplx_jax", _cplx_spatial_input, id="analys_cplx"),
]

VECTOR_TRANSFORMS = [
    pytest.param("synth_vec_jax", _spectral_vec_input, id="synth_vec"),
    pytest.param("analys_vec_jax", _spatial_vec_input, id="analys_vec"),
    pytest.param("synth_vec_cplx_jax", _cplx_spectral_vec_input, id="synth_vec_cplx"),
    pytest.param("analys_vec_cplx_jax", _cplx_spatial_vec_input, id="analys_vec_cplx"),
]

TRANSFORMS = SCALAR_TRANSFORMS + VECTOR_TRANSFORMS


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


@pytest.mark.parametrize("fn_name,make_input", SCALAR_TRANSFORMS)
def test_against_numpy_scalar(fn_name, make_input):
    sh = _make_cfg()
    fn_jax = getattr(sh, fn_name)
    # Get the name of the reference implementation by stripping the "_jax" suffix
    fn_numpy = getattr(sh, fn_name.replace("_jax", ""))
    x = make_input(sh, seed=0)
    x_cp = x.copy()
    y_numpy = fn_numpy(np.array(x))
    y_jax = fn_jax(x_cp)
    print(f"y_jax shape: {y_jax.shape}, y_numpy shape: {y_numpy.shape}")
    print(y_jax[:5])
    print(y_numpy[:5])
    diffs = np.abs(np.array(y_jax) - np.array(y_numpy))
    print("Max difference:", np.max(diffs))
    assert np.allclose(y_jax, y_numpy, rtol=RTOL, atol=ATOL), (
        f"{fn_name} output differs from numpy implementation."
    )


@pytest.mark.parametrize("fn_name,make_input", VECTOR_TRANSFORMS)
def test_against_numpy_vector(fn_name, make_input):
    sh = _make_cfg()
    fn_jax = getattr(sh, fn_name)
    # Get the name of the reference implementation by stripping the "_jax" suffix
    contains_cplx = "cplx" in fn_name
    if fn_name.startswith("synth"):
        if contains_cplx:
            fn_numpy = sh.synth_cplx
        else:
            fn_numpy = sh.synth
    elif fn_name.startswith("analys"):
        if contains_cplx:
            fn_numpy = sh.analys_cplx
        else:
            fn_numpy = sh.analys
    else:
        raise ValueError(f"Unexpected function name: {fn_name}")
    x = make_input(sh, seed=0)
    x_cp = x.copy()
    in_0 = np.array(x[0])
    in_1 = np.array(x[1])
    in_2 = np.array(x[2])
    print("Input dtypes:", in_0.dtype, in_1.dtype, in_2.dtype)
    y_numpy = fn_numpy(np.array(x[0]), np.array(x[1]), np.array(x[2]))
    y_numpy = np.array(y_numpy)
    y_jax = fn_jax(x_cp)
    print(f"y_jax shape: {y_jax.shape}, y_numpy shape: {y_numpy.shape}")
    print(y_jax[:5])
    print(y_numpy[:5])
    diffs = np.abs(np.array(y_jax) - np.array(y_numpy))
    print("Max difference:", np.max(diffs))
    assert np.allclose(y_jax, y_numpy, rtol=RTOL, atol=ATOL), (
        f"{fn_name} output differs from numpy implementation."
    )


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_cuda_implementation(fn_name, make_input):
    """Make sure the CUDA implementation works when CUDA is available"""
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    x_cuda = jax.device_put(x, device=CUDA_DEVICES[0])
    y_cuda = fn(x_cuda)  # noqa: F841


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_cpu_implementation(fn_name, make_input):
    """Make sure the CPU implementation works when CUDA is available"""
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    x_cpu = jax.device_put(x, device=CPU_DEVICES[0])
    y_cpu = fn(x_cpu)  # noqa: F841


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
def test_theta_contiguous_cuda():
    # This one asserts that an error is thrown
    sh = shtns_jax.sht(8, 8)
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


def _cplx_qst_input(sh, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        c = rng.standard_normal(sh.nlm_cplx) + 1j * rng.standard_normal(sh.nlm_cplx)
        out.append(c.astype(np.complex128))
    return out  # [Qlm, Slm, Tlm]


@pytest.mark.parametrize("lmax", [8, 16])
@pytest.mark.parametrize("seed", [1, 2])
def test_SHqst_to_point_cplx(lmax, seed):
    """SHqst_to_point_cplx at grid nodes must match the full-grid complex vector
    synthesis SHqst_to_spat_cplx (exposed as synth_cplx with 3 args)."""
    sh = shtns_jax.sht(lmax, lmax, 1)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    Qlm, Slm, Tlm = _cplx_qst_input(sh, seed)

    vr, vt, vp = sh.synth_cplx(Qlm, Slm, Tlm)  # ground truth, shape (nphi, nlat)
    cost = sh.cos_theta
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    # interior latitudes only (avoid the poles where sin(theta) -> 0)
    for i in range(nphi):
        for j in range(1, nlat - 1):
            r, t, p = sh.SHqst_to_point_cplx(Qlm, Slm, Tlm, cost[j], phi[i])
            assert np.allclose(r, vr[i, j], rtol=RTOL, atol=ATOL)
            assert np.allclose(t, vt[i, j], rtol=RTOL, atol=ATOL)
            assert np.allclose(p, vp[i, j], rtol=RTOL, atol=ATOL)


def test_SHqst_to_point_cplx_vectorized():
    """Array (cost, phi) input returns arrays equal to looping scalar calls."""
    sh = shtns_jax.sht(12, 12, 1)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)
    Qlm, Slm, Tlm = _cplx_qst_input(sh, seed=5)
    cost = sh.cos_theta[1:-1]
    phi = np.full_like(cost, 0.7)
    vr, vt, vp = sh.SHqst_to_point_cplx(Qlm, Slm, Tlm, cost, phi)
    for k in range(cost.size):
        r, t, p = sh.SHqst_to_point_cplx(Qlm, Slm, Tlm, float(cost[k]), float(phi[k]))
        assert np.allclose([vr[k], vt[k], vp[k]], [r, t, p], rtol=RTOL, atol=ATOL)
