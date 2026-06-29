import numpy as np
import pytest
import jax
import jax.numpy as jnp

import shtns_jax

RTOL = 1e-10
ATOL = 1e-10


try:
    CUDA_DEVICES = jax.devices("cuda")
except RuntimeError:
    CUDA_DEVICES = []
GPU_AVAILABLE = len(CUDA_DEVICES) > 0
CPU_DEVICES = jax.devices("cpu")


def _spectral_input(rot: shtns_jax.rotation, seed: int):
    rng = np.random.default_rng(seed)
    qlm = rng.standard_normal(rot._nlm) + 1j * rng.standard_normal(rot._nlm)

    qlm[: rot.mmax] = qlm[: rot.mmax].real + 0j
    return jnp.array(qlm, dtype=jnp.complex128)


def _cplx_spectral_input(rot: shtns_jax.rotation, seed: int):
    rng = np.random.default_rng(seed)
    alm = rng.standard_normal(rot._nlm_cplx) + 1j * rng.standard_normal(rot._nlm_cplx)
    return jnp.array(alm, dtype=jnp.complex128)


def _make_rotation_obj(lmax=8, mmax=8, theta=0.4, Vx=0.1, Vy=0.2, Vz=0.3):
    # Init shtns.rotation object
    rot = shtns_jax.rotation(lmax, mmax)
    # Normalize the vector [Vx, Vy, Vz]
    norm = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    Vx /= norm
    Vy /= norm
    Vz /= norm
    # Set the rotation parameters
    rot.set_angle_axis(theta, Vx, Vy, Vz)
    return rot


TRANSFORMS = [
    pytest.param("apply_real_jax", _spectral_input, id="apply_real_jax"),
    pytest.param("apply_cplx_jax", _cplx_spectral_input, id="apply_cplx_jax"),
]


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_jit(fn_name, make_input):
    rot = _make_rotation_obj()
    fn = getattr(rot, fn_name)
    x = make_input(rot, seed=0)
    expected = fn(x)
    out = jax.jit(fn)(x)
    assert out.shape == expected.shape
    assert out.dtype == expected.dtype


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_vmap(fn_name, make_input):
    rot = _make_rotation_obj()
    fn = getattr(rot, fn_name)
    batch = 4
    x = jnp.stack([make_input(rot, seed=i) for i in range(batch)])
    out = jax.vmap(fn)(x)
    expected_elem = fn(make_input(rot, seed=0))
    assert out.shape == (batch,) + expected_elem.shape
    assert out.dtype == expected_elem.dtype


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_jvp(fn_name, make_input):
    rot = _make_rotation_obj()
    fn = getattr(rot, fn_name)
    x = make_input(rot, seed=0)
    v = make_input(rot, seed=1)
    v_cp = v.copy()
    _, jvp_out = jax.jvp(fn, (x,), (v_cp,))
    direct = fn(v)
    assert np.allclose(np.array(jvp_out), np.array(direct), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("fn_name,make_input", TRANSFORMS)
def test_vjp(fn_name, make_input):
    rot = _make_rotation_obj()
    fn = getattr(rot, fn_name)
    x = make_input(rot, seed=0)
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
    rot = _make_rotation_obj()
    fn = getattr(rot, fn_name)
    x = make_input(rot, seed=0)
    x_copy = x.copy()
    _ = fn(x)

    assert jnp.array_equal(x, x_copy), f"{fn_name} modified its input array."
