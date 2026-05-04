"""JAX bindings for SHTns.

Import this module instead of shtns to get a sht class with JAX autodiff
support (synth_jax, analys_jax, synth_cplx_jax, analys_cplx_jax).

Example::

    import shtns_jax
    sh = shtns_jax.sht(lmax)
    sh.set_grid()
    alm = sh.analys_jax(spatial_array)   # supports jit, vmap, jvp, vjp
"""

import ctypes
import os

import jax
from jax.custom_transpose import custom_transpose
import jax.numpy as jnp

import shtns

jax.config.update("jax_enable_x64", True)

###################################
# FFI library loading

_this_dir = os.path.dirname(__file__)


def _load_jax_lib(*names):
    last_error = None
    for name in names:
        try:
            return ctypes.cdll.LoadLibrary(os.path.join(_this_dir, name))
        except OSError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise OSError("No library names provided for loading.")


_shtns_jax_lib_cpu = _load_jax_lib("libshtns_jax_cpu.so")
_cpu_lib_members = [
    ("shtns_synth", _shtns_jax_lib_cpu.synth_cpu),
    ("shtns_analys", _shtns_jax_lib_cpu.analys_cpu),
    ("shtns_synth_cplx", _shtns_jax_lib_cpu.synth_cplx_cpu),
    ("shtns_analys_cplx", _shtns_jax_lib_cpu.analys_cplx_cpu),
]
for _name, _func in _cpu_lib_members:
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="cpu")
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="Host")

CUDA_AVAILABLE = False
try:
    _shtns_jax_lib_cuda = _load_jax_lib("libshtns_jax_cuda.so")
    _gpu_lib_members = [
        ("shtns_synth_gpu", _shtns_jax_lib_cuda.synth_gpu),
        ("shtns_analys_gpu", _shtns_jax_lib_cuda.analys_gpu),
    ]
    for _name, _func in _gpu_lib_members:
        jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="CUDA")
    CUDA_AVAILABLE = True
except Exception as e:
    print("Could not find GPU implementation for JAX:", e)


###################################
# Re-define sht class with JAX support

class sht(shtns.sht):
    """SHTns sht with JAX autodiff support.

    Inherits all NumPy methods from shtns.sht and adds:
      synth_jax, analys_jax, synth_cplx_jax, analys_cplx_jax
    """

    def _check_jax_gpu_grid_compat(self):
        if CUDA_AVAILABLE and (self._grid_flags & shtns.SHT_PHI_CONTIGUOUS):
            raise ValueError(
                "JAX GPU backend does not support SHT_PHI_CONTIGUOUS grids. "
                "Call set_grid() with SHT_THETA_CONTIGUOUS or run with CPU backend."
            )

    def _check_shape_dtype(self, x: jax.Array, shape: tuple, dtype) -> None:
        n = len(shape)
        if x.shape[-n:] != shape:
            raise ValueError(f"Input array must end with shape {shape}. Got {x.shape}.")
        if x.dtype != dtype:
            raise ValueError(f"Input array must have dtype {dtype}. Got {x.dtype}.")

    def _grid_weights(self) -> jax.Array:
        w = self.gauss_wts()  # shape (nlat/2,)
        full_w = jnp.concatenate([w, w[::-1]])  # shape (nlat,)
        return full_w.reshape(1, -1) * 2 * jnp.pi / self.nphi

    def synth_jax(self, x: jax.Array) -> jax.Array:
        """Inverse SHT: complex128 spectral (nlm,) -> float64 spatial (spat_shape)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (self.nlm,), jnp.complex128)

        @jax.custom_jvp
        def _synth_impl(x_in: jax.Array) -> jax.Array:
            """Defines the forward pass"""
            orig_shape = x_in.shape
            out_shape = (
                self.spat_shape
                if len(orig_shape) == 1
                else (*orig_shape[:-1], *self.spat_shape)
            )

            def get_impl(target_name):
                return lambda x: jax.ffi.ffi_call(
                    target_name,
                    jax.ShapeDtypeStruct(out_shape, jnp.float64),
                    vmap_method="broadcast_all",
                )(x, cfg=int(self.this))

            return jax.lax.platform_dependent(
                x_in, cpu=get_impl("shtns_synth"), cuda=get_impl("shtns_synth_gpu")
            )

        @custom_transpose
        def _synth_tangent(residuals, x_tan: jax.Array) -> jax.Array:
            """Re-define the forward pass, to allow
            for custom transpose definition."""
            return _synth_impl(x_tan)

        @_synth_tangent.def_transpose
        def _synth_impl_vjp(residuals, ct_out: jax.Array):
            """This defines the VJP."""
            prefix_shape = ct_out.shape[:-2]
            out_shape = (
                (self.nlm,) if len(ct_out.shape) == 2 else (*prefix_shape, self.nlm)
            )
            weights = self._grid_weights()

            def get_impl(target_name):
                return lambda x: jax.ffi.ffi_call(
                    target_name,
                    jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                    vmap_method="broadcast_all",
                )(x, cfg=int(self.this))

            scaled = ct_out / weights

            result = jax.lax.platform_dependent(
                scaled,
                cpu=get_impl("shtns_analys"),
                cuda=get_impl("shtns_analys_gpu"),
            )
            if self.orthonormal:
                result = result.at[self.lmax + 1 :].multiply(2.0)
            return result

        @_synth_impl.defjvp
        def _synth_impl_jvp(primals, tangents):
            """This defines the JVP."""
            (x_in,) = primals
            (x_tan,) = tangents
            y = _synth_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _synth_tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _synth_impl(x)

    def analys_jax(self, x: jax.Array) -> jax.Array:
        """Forward SHT: float64 spatial (spat_shape) -> complex128 spectral (nlm,)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, self.spat_shape, jnp.float64)

        @jax.custom_jvp
        def _analys_impl(x_in: jax.Array) -> jax.Array:
            prefix_shape = x_in.shape[:-2]
            out_shape = (
                (self.nlm,) if len(x_in.shape) == 2 else (*prefix_shape, self.nlm)
            )

            def get_impl(target_name):
                return lambda x: jax.ffi.ffi_call(
                    target_name,
                    jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                    vmap_method="broadcast_all",
                )(x, cfg=int(self.this))

            return jax.lax.platform_dependent(
                x_in, cpu=get_impl("shtns_analys"), cuda=get_impl("shtns_analys_gpu")
            )

        @custom_transpose
        def _analys_tangent(residuals, x_tan: jax.Array) -> jax.Array:
            return _analys_impl(x_tan)

        @_analys_tangent.def_transpose
        def _analys_vjp(residuals, ct_out: jax.Array):
            orig_shape = ct_out.shape
            out_shape = (
                self.spat_shape
                if len(orig_shape) == 1
                else (*orig_shape[:-1], *self.spat_shape)
            )

            def get_impl(target_name):
                return lambda x: jax.ffi.ffi_call(
                    target_name,
                    jax.ShapeDtypeStruct(out_shape, jnp.float64),
                    vmap_method="broadcast_all",
                )(x, cfg=int(self.this))

            if self.orthonormal:
                ct_out = ct_out.at[self.lmax + 1 :].multiply(0.5)
            result = jax.lax.platform_dependent(
                ct_out,
                cpu=get_impl("shtns_synth"),
                cuda=get_impl("shtns_synth_gpu"),
            )
            weights = self._grid_weights()
            return result * weights

        @_analys_impl.defjvp
        def _analys_impl_jvp(primals, tangents):
            (x_in,) = primals
            (x_tan,) = tangents
            y = _analys_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _analys_tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _analys_impl(x)

    def synth_cplx_jax(self, x: jax.Array) -> jax.Array:
        """Complex inverse SHT: complex128 spectral (nlm_cplx,) -> 
        complex128 spatial (spat_shape)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (self.nlm_cplx,), jnp.complex128)

        @jax.custom_jvp
        def _synth_cplx_impl(x_in: jax.Array) -> jax.Array:
            orig_shape = x_in.shape
            out_shape = (
                self.spat_shape
                if len(orig_shape) == 1
                else (*orig_shape[:-1], *self.spat_shape)
            )
            return jax.ffi.ffi_call(
                "shtns_synth_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        @custom_transpose
        def _tangent(residuals, x_tan: jax.Array) -> jax.Array:
            return _synth_cplx_impl(x_tan)

        @_tangent.def_transpose
        def _synth_cplx_vjp(residuals, ct_out: jax.Array):
            prefix_shape = ct_out.shape[:-2]
            out_shape = (
                (self.nlm_cplx,)
                if len(ct_out.shape) == 2
                else (*prefix_shape, self.nlm_cplx)
            )
            weights = self._grid_weights()
            result = jax.ffi.ffi_call(
                "shtns_analys_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(ct_out / weights, cfg=int(self.this))
            if self.orthonormal:
                result = result.at[self.zm != 0].multiply(2.0)
            return result

        @_synth_cplx_impl.defjvp
        def _synth_cplx_jvp(primals, tangents):
            (x_in,) = primals
            (x_tan,) = tangents
            y = _synth_cplx_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _synth_cplx_impl(x)

    def analys_cplx_jax(self, x: jax.Array) -> jax.Array:
        """Complex forward SHT: complex128 spatial (spat_shape) -> 
        complex128 spectral (nlm_cplx,)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, self.spat_shape, jnp.complex128)

        @jax.custom_jvp
        def _analys_cplx_impl(x_in: jax.Array) -> jax.Array:
            prefix_shape = x_in.shape[:-2]
            out_shape = (
                (self.nlm_cplx,)
                if len(x_in.shape) == 2
                else (*prefix_shape, self.nlm_cplx)
            )
            return jax.ffi.ffi_call(
                "shtns_analys_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        @custom_transpose
        def _tangent(residuals, x_tan: jax.Array) -> jax.Array:
            return _analys_cplx_impl(x_tan)

        @_tangent.def_transpose
        def _analys_cplx_vjp(residuals, ct_out: jax.Array):
            orig_shape = ct_out.shape
            out_shape = (
                self.spat_shape
                if len(orig_shape) == 1
                else (*orig_shape[:-1], *self.spat_shape)
            )
            if self.orthonormal:
                ct_out = ct_out.at[self.zm != 0].multiply(0.5)
            result = jax.ffi.ffi_call(
                "shtns_synth_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(ct_out, cfg=int(self.this))
            weights = self._grid_weights()
            return result * weights

        @_analys_cplx_impl.defjvp
        def _analys_cplx_jvp(primals, tangents):
            (x_in,) = primals
            (x_tan,) = tangents
            y = _analys_cplx_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _analys_cplx_impl(x)
