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


def _cplx_qst_input(sh, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(3):
        c = rng.standard_normal(sh.nlm_cplx) + 1j * rng.standard_normal(sh.nlm_cplx)
        out.append(c.astype(np.complex128))
    return out  # [Qlm, Slm, Tlm]


SCALAR_TRANSFORMS = [
    pytest.param("synth_jax", _spectral_input, id="synth"),
    pytest.param("analys_jax", _spatial_real_input, id="analys"),
    # pytest.param("synth_cplx_jax", _cplx_spectral_input, id="synth_cplx"),
    # pytest.param("analys_cplx_jax", _cplx_spatial_input, id="analys_cplx"),
]

VECTOR_TRANSFORMS = [
    pytest.param("synth_vec_jax", _spectral_vec_input, id="synth_vec"),
    pytest.param("analys_vec_jax", _spatial_vec_input, id="analys_vec"),
    # pytest.param("synth_vec_cplx_jax", _cplx_spectral_vec_input, id="synth_vec_cplx"),
    # pytest.param("analys_vec_cplx_jax", _cplx_spatial_vec_input, id="analys_vec_cplx"),
]


TRANSFORMS = SCALAR_TRANSFORMS + VECTOR_TRANSFORMS

# Subset of TRANSFORMS whose vjp is defined via jax.custom_jvp/custom_transpose
# (excludes synth_vec_cplx_jax/analys_vec_cplx_jax, which have no custom vjp rule).
# Each entry pairs an input-space generator with an output-space generator, so that
# both sides respect the real-SH convention that m=0 coefficients carry no imaginary
# part (as already enforced by _spectral_input/_spectral_vec_input).
ADJOINT_IDENTITY_TRANSFORMS = [
    pytest.param("synth_jax", _spectral_input, _spatial_real_input, id="synth"),
    pytest.param("analys_jax", _spatial_real_input, _spectral_input, id="analys"),
    pytest.param(
        "synth_vec_jax", _spectral_vec_input, _spatial_vec_input, id="synth_vec"
    ),
    pytest.param(
        "analys_vec_jax", _spatial_vec_input, _spectral_vec_input, id="analys_vec"
    ),
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


@pytest.mark.parametrize("fn_name,make_input,make_output", ADJOINT_IDENTITY_TRANSFORMS)
def test_vjp_adjoint_identity(fn_name, make_input, make_output):
    """The vjp pullback must be the adjoint of the jvp push-forward:
    Re(<jvp(dx), y>) == Re(<dx, vjp(y)>) for all dx, y.

    This is the check that actually catches a numerically wrong adjoint
    (unlike test_vjp above, which only checks shape/dtype).
    """
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    x = make_input(sh, seed=0)
    dx = make_input(sh, seed=1)
    _, pullback = jax.vjp(fn, x)
    y = make_output(sh, seed=2)
    _, jvp_out = jax.jvp(fn, (x,), (dx,))
    (cot_in,) = pullback(y)
    lhs = jnp.real(jnp.vdot(jvp_out, y))
    rhs = jnp.real(jnp.vdot(dx, cot_in))
    assert np.allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU-only test")
@pytest.mark.parametrize("fn_name,make_input,make_output", ADJOINT_IDENTITY_TRANSFORMS)
def test_vjp_adjoint_identity_cuda(fn_name, make_input, make_output):
    """Same check as test_vjp_adjoint_identity, but with inputs placed on the
    CUDA device, to verify the GPU-composed adjoint FFI targets are correct
    (not just registered)."""
    sh = _make_cfg()
    fn = getattr(sh, fn_name)
    device = CUDA_DEVICES[0]
    x = jax.device_put(make_input(sh, seed=0), device=device)
    dx = jax.device_put(make_input(sh, seed=1), device=device)
    _, pullback = jax.vjp(fn, x)
    y = jax.device_put(make_output(sh, seed=2), device=device)
    _, jvp_out = jax.jvp(fn, (x,), (dx,))
    (cot_in,) = pullback(y)
    lhs = jnp.real(jnp.vdot(jvp_out, y))
    rhs = jnp.real(jnp.vdot(dx, cot_in))
    assert np.allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


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

    for i in range(nphi):
        for j in range(nlat):
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


@pytest.mark.parametrize("lmax", [8, 16, 32])
def test_SHqst_to_point_cplx_analytic(lmax: int) -> None:
    """
    Analytic test case.
    """
    sh = shtns_jax.sht(lmax, lmax)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)

    Qlm = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    Slm = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    Tlm = np.zeros(sh.nlm_cplx, dtype=np.complex128)

    # Set some non-zero coefficients.
    Qlm[sh.zidx(3, 1)] = 1.0 + 1j
    Slm[sh.zidx(2, -2)] = 1.0 + 1j
    Tlm[sh.zidx(2, -1)] = 1.0 + 1j

    # Create a uniform grid in phi theta space for sampling
    phi_vals = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    # Don't want to sample either pole.
    theta_vals = np.linspace(0, np.pi, 101, endpoint=False)[1:]

    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

    cost_vec = np.cos(theta_grid.flatten())
    sint_vec = np.sin(theta_grid.flatten())
    phi_vec = phi_grid.flatten()
    print("cost_vec dtype: ", cost_vec.dtype)
    print("phi_vec dtype: ", phi_vec.dtype)

    vr, vtheta, vphi = sh.SHqst_to_point_cplx(Qlm, Slm, Tlm, cost_vec, phi_vec)

    #####################################################
    # Spherical harmonic evals for computing reference solution.
    # l = 3, m = 1
    coeffs_3_1 = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    coeffs_3_1[sh.zidx(3, 1)] = 1.0
    Y_3_1_evals = np.zeros_like(cost_vec, dtype=np.complex128)
    for i in range(cost_vec.size):
        Y_3_1_evals[i] = sh.SH_to_point_cplx(coeffs_3_1, cost_vec[i], phi_vec[i])
    # l = 2, m = -2
    coeffs_2_m2 = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    coeffs_2_m2[sh.zidx(2, -2)] = 1.0
    Y_2_m2_evals = np.zeros_like(cost_vec, dtype=np.complex128)
    for i in range(cost_vec.size):
        Y_2_m2_evals[i] = sh.SH_to_point_cplx(coeffs_2_m2, cost_vec[i], phi_vec[i])
    # l = 2, m = -1
    coeffs_2_m1 = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    coeffs_2_m1[sh.zidx(2, -1)] = 1.0
    Y_2_m1_evals = np.zeros_like(cost_vec, dtype=np.complex128)
    for i in range(cost_vec.size):
        Y_2_m1_evals[i] = sh.SH_to_point_cplx(coeffs_2_m1, cost_vec[i], phi_vec[i])
    # l=2, m=0
    coeffs_2_0 = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    coeffs_2_0[sh.zidx(2, 0)] = 1.0
    Y_2_0_evals = np.zeros_like(cost_vec, dtype=np.complex128)
    for i in range(cost_vec.size):
        Y_2_0_evals[i] = sh.SH_to_point_cplx(coeffs_2_0, cost_vec[i], phi_vec[i])

    ####################################################
    # Construct expected solutions
    vr_expected = (1.0 + 1j) * Y_3_1_evals

    # vt_expected
    term1 = 2 * np.exp(-1j * phi_vec) * Y_2_m1_evals
    term2 = -2 * cost_vec / sint_vec * Y_2_m2_evals
    term3 = -1j / sint_vec * Y_2_m1_evals
    vt_expected = (1.0 + 1j) * (term1 + term2 + term3)

    # vp_expected
    term1 = np.sqrt(6) * np.exp(-1j * phi_vec) * Y_2_0_evals
    term2 = -cost_vec / sint_vec * Y_2_m1_evals
    term3 = 2j / sint_vec * Y_2_m2_evals
    vp_expected = -1 * (1.0 + 1j) * (term1 + term2 + term3)

    ####################################################
    # Compare computed vs expected

    assert np.allclose(vr, vr_expected, rtol=RTOL, atol=ATOL)
    assert np.allclose(vtheta, vt_expected, rtol=RTOL, atol=ATOL)

    # print("vphi: ", vphi[:5])
    # print("vp_expected: ", vp_expected[:5])
    # print("diffs: ", np.abs(vphi - vp_expected)[:5])

    assert np.allclose(vphi, vp_expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("lmax", [8, 16, 32])
def test_SHqst_to_point_cplx_against_real(lmax: int) -> None:
    """
    Test the complex function against the one for real-valued signals.
    """
    sh = shtns_jax.sht(lmax, lmax)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)

    Qlm = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    Slm = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    Tlm = np.zeros(sh.nlm_cplx, dtype=np.complex128)

    # Set some non-zero coefficients.
    # c[l,-m] = (-1)^m * conj(c[l,m])
    Qlm[sh.zidx(3, 1)] = 1.0 + 1j
    Qlm[sh.zidx(3, -1)] = -1.0 + 1j
    Slm[sh.zidx(2, 2)] = 1.0 + 1j
    Slm[sh.zidx(2, -2)] = 1.0 - 1j
    Tlm[sh.zidx(2, 1)] = 1.0 + 1j
    Tlm[sh.zidx(2, -1)] = -1.0 + 1j

    # Do the same for real-valued signals.
    Qlm_real = np.zeros(sh.nlm, dtype=np.complex128)
    Slm_real = np.zeros(sh.nlm, dtype=np.complex128)
    Tlm_real = np.zeros(sh.nlm, dtype=np.complex128)
    Qlm_real[sh.idx(3, 1)] = 1.0 + 1j
    Slm_real[sh.idx(2, 2)] = 1.0 + 1j
    Tlm_real[sh.idx(2, 1)] = 1.0 + 1j

    # Check that I've set up the coeffs correctly by transforming to the GL grid.
    vr_real, vtheta_real, vphi_real = sh.synth(Qlm_real, Slm_real, Tlm_real)
    vr_cplx, vtheta_cplx, vphi_cplx = sh.synth_cplx(Qlm, Slm, Tlm)

    print("vr_real: ", vr_real.flatten()[:5])
    print("vr_cplx: ", vr_cplx.flatten()[:5])
    assert np.allclose(vr_real, vr_cplx, rtol=RTOL, atol=ATOL)
    assert np.allclose(vtheta_real, vtheta_cplx, rtol=RTOL, atol=ATOL)
    assert np.allclose(vphi_real, vphi_cplx, rtol=RTOL, atol=ATOL)

    # Create a uniform grid in phi theta space for sampling
    phi_vals = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    # Don't want to sample either pole.
    theta_vals = np.linspace(0, np.pi, 101, endpoint=False)[1:]

    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)

    cost_vec = np.cos(theta_grid.flatten())
    phi_vec = phi_grid.flatten()
    print("cost_vec dtype: ", cost_vec.dtype)
    print("phi_vec dtype: ", phi_vec.dtype)

    vr = np.zeros_like(cost_vec, dtype=np.complex128)
    vtheta = np.zeros_like(cost_vec, dtype=np.complex128)
    vphi = np.zeros_like(cost_vec, dtype=np.complex128)

    vr, vtheta, vphi = sh.SHqst_to_point_cplx(Qlm, Slm, Tlm, cost_vec, phi_vec)

    # Test against the output of SHqst_to_point
    vr_real = np.zeros_like(cost_vec, dtype=np.float64)
    vtheta_real = np.zeros_like(cost_vec, dtype=np.float64)
    vphi_real = np.zeros_like(cost_vec, dtype=np.float64)
    for i in range(cost_vec.size):
        vr_real[i], vtheta_real[i], vphi_real[i] = sh.SHqst_to_point(
            Qlm_real, Slm_real, Tlm_real, cost_vec[i], phi_vec[i]
        )
    # vr_real, vtheta_real, vphi_real = sh.SHqst_to_point(Qlm_real, Slm_real, Tlm_real, cost_vec, phi_vec)

    print("vr: ", vr[:5])
    print("vr_real: ", vr_real[:5])
    print("max abs diffs: ", np.max(np.abs(vr - vr_real)))
    print("vr max imag part: ", np.max(np.abs(vr.imag)))
    assert np.allclose(vr, vr_real, rtol=RTOL, atol=ATOL)
    assert np.allclose(vtheta, vtheta_real, rtol=RTOL, atol=ATOL)
    assert np.allclose(vphi, vphi_real, rtol=RTOL, atol=ATOL)

    assert np.max(vr.imag) < ATOL
    assert np.max(vtheta.imag) < ATOL
    assert np.max(vphi.imag) < ATOL


@pytest.mark.parametrize("lmax", [8, 16, 32])
def test_SHqst_to_lat_jax_against_numpy(lmax: int) -> None:
    """
    Test the SHqst_to_lat_jax function against the numpy implementation.
    """
    sh = shtns_jax.sht(lmax, lmax)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)

    Qlm = _spectral_input(sh, seed=1)
    Slm = _spectral_input(sh, seed=2)
    Tlm = _spectral_input(sh, seed=3)

    cost = np.cos(2.0)
    print("cost dtype: ", cost.dtype)
    nphi = 45

    # Call the JAX implementation
    vr_jax, vtheta_jax, vphi_jax = sh.SHqst_to_lat_jax(Qlm, Slm, Tlm, cost, nphi)

    # Call the numpy implementation
    # First need to define the empty arrays for output
    vr_numpy = np.empty(nphi)
    vtheta_numpy = np.empty(nphi)
    vphi_numpy = np.empty(nphi)
    Qlm_np = np.array(Qlm)
    Slm_np = np.array(Slm)
    Tlm_np = np.array(Tlm)
    sh.SHqst_to_lat(Qlm_np, Slm_np, Tlm_np, cost, vr_numpy, vtheta_numpy, vphi_numpy)

    assert np.allclose(vr_jax, vr_numpy, rtol=RTOL, atol=ATOL)
    assert np.allclose(vtheta_jax, vtheta_numpy, rtol=RTOL, atol=ATOL)
    assert np.allclose(vphi_jax, vphi_numpy, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("lmax", [8, 16, 32])
def test_SHqst_to_lat_jax_transforms(lmax: int) -> None:
    """
    Tests jax.jit and jax.vmap for SHqst_to_lat_jax."""
    sh = shtns_jax.sht(lmax, lmax)
    nlat, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS)

    Qlm = _spectral_input(sh, seed=1)
    Slm = _spectral_input(sh, seed=2)
    Tlm = _spectral_input(sh, seed=3)

    cost = np.cos(2.0)
    print("cost dtype: ", cost.dtype)
    nphi = 45

    # JIT the function.
    jit_fn = jax.jit(sh.SHqst_to_lat_jax, static_argnames=("nphi",))
    vr_jit, vtheta_jit, vphi_jit = jit_fn(Qlm, Slm, Tlm, cost, nphi)

    # Test vmap over the cost argument.
    # nphi must be closed over rather than passed as an argument: passing it through
    # jax.jit(jax.vmap(...)) would make the outer jit trace it as a dynamic integer,
    # which the inner jit cannot hash as a static arg.
    batch_size = 3
    cost_batch = np.cos(np.linspace(0.1, 2.0, batch_size))
    jit_fn_vmap = jax.jit(
        jax.vmap(
            lambda Qlm, Slm, Tlm, cost: sh.SHqst_to_lat_jax(Qlm, Slm, Tlm, cost, nphi),
            in_axes=(None, None, None, 0),
        )
    )
    vr_vmap, vtheta_vmap, vphi_vmap = jit_fn_vmap(Qlm, Slm, Tlm, cost_batch)
    # Check output shapes.
    assert vr_vmap.shape == (batch_size, nphi)
    assert vtheta_vmap.shape == (batch_size, nphi)
    assert vphi_vmap.shape == (batch_size, nphi)

    # Compare the outputs of the vmap with the individual calls.
    for i in range(batch_size):
        vr_single, vtheta_single, vphi_single = sh.SHqst_to_lat_jax(
            Qlm, Slm, Tlm, cost_batch[i], nphi
        )
        assert np.allclose(vr_vmap[i], vr_single, rtol=RTOL, atol=ATOL)
        assert np.allclose(vtheta_vmap[i], vtheta_single, rtol=RTOL, atol=ATOL)
        assert np.allclose(vphi_vmap[i], vphi_single, rtol=RTOL, atol=ATOL)
