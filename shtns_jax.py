"""JAX bindings for SHTns.

Import this module instead of shtns to get a sht class with JAX autodiff
support (synth_jax, analys_jax, synth_cplx_jax, analys_cplx_jax).

Example::

    import shtns_jax
    sh = shtns_jax.sht(lmax)
    sh.set_grid()
    alm = sh.analys_jax(spatial_array)   # supports jit, vmap, jvp, vjp
"""

import logging
import ctypes
import os

import jax
from jax.custom_transpose import custom_transpose
import jax.numpy as jnp
import numpy as np

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
    ("shtns_synth_vec", _shtns_jax_lib_cpu.synth_vec_cpu),
    ("shtns_analys_vec", _shtns_jax_lib_cpu.analys_vec_cpu),
    ("shtns_adjoint_synth_vec", _shtns_jax_lib_cpu.adjoint_synth_vec_cpu),
    ("shtns_adjoint_analys_vec", _shtns_jax_lib_cpu.adjoint_analys_vec_cpu),
    ("shtns_synth_cplx", _shtns_jax_lib_cpu.synth_cplx_cpu),
    ("shtns_analys_cplx", _shtns_jax_lib_cpu.analys_cplx_cpu),
    ("shtns_synth_vec_cplx", _shtns_jax_lib_cpu.synth_vec_cplx_cpu),
    ("shtns_analys_vec_cplx", _shtns_jax_lib_cpu.analys_vec_cplx_cpu),
]
for _name, _func in _cpu_lib_members:
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="cpu")
    jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="Host")

CUDA_AVAILABLE = False
try:
    _shtns_jax_lib_cuda = _load_jax_lib("libshtns_jax_cuda.so")
    _gpu_lib_members = [
        ("shtns_synth", _shtns_jax_lib_cuda.synth_gpu),
        ("shtns_analys", _shtns_jax_lib_cuda.analys_gpu),
        ("shtns_synth_vec", _shtns_jax_lib_cuda.synth_vec_gpu),
        ("shtns_analys_vec", _shtns_jax_lib_cuda.analys_vec_gpu),
    ]
    for _name, _func in _gpu_lib_members:
        jax.ffi.register_ffi_target(_name, jax.ffi.pycapsule(_func), platform="CUDA")
    CUDA_AVAILABLE = True
except Exception:
    logging.warning("Could not find GPU implementation for JAX:")


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

    def _dtheta_cplx_op(self):
        """Precompute (and cache) the sin(theta) d/dtheta operator on the complex
        SH layout (index k = l(l+1)+m), expressed so that it can be evaluated by
        SH_to_point_cplx.

        Uses the orthonormal-SH recurrence
            sin(theta) dY_l^m/dtheta = l eps_{l+1}^m Y_{l+1}^m - (l+1) eps_l^m Y_{l-1}^m,
            eps_l^m = sqrt((l^2 - m^2) / ((2l-1)(2l+1))),
        which in coefficient form (g = sin(theta) df/dtheta) reads
            d_(l,m) = (l-1) eps_l^m c_(l-1,m) - (l+2) eps_{l+1}^m c_(l+1,m).

        The derivative of a degree-lmax field carries degree lmax+1 content, so the
        output is laid out for a degree-(lmax+1) config ``sh_hi`` (which can still be
        evaluated pointwise without a grid). Returns (sh_hi, idx_dn, w_dn, idx_up,
        w_up): gather indices into the BASE (length nlm_cplx) coefficient array and
        weights, each of length ``sh_hi.nlm_cplx``; out-of-range gathers carry
        weight 0 and index 0.
        """
        cached = getattr(self, "_dtheta_cplx_cache", None)
        if cached is not None:
            return cached

        lmax = self.lmax
        nlm = self.nlm_cplx
        sh_hi = sht(lmax + 1, lmax + 1, 1)  # degree lmax+1; no grid needed for point eval

        l = np.asarray(sh_hi.zl, dtype=np.int64)   # output (hi) layout
        m = np.asarray(sh_hi.zm, dtype=np.int64)

        def eps(ll, mm):
            ll = ll.astype(np.float64)
            mm = mm.astype(np.float64)
            num = ll * ll - mm * mm
            den = (2.0 * ll - 1.0) * (2.0 * ll + 1.0)
            out = np.zeros_like(num)
            ok = (den != 0.0) & (num >= 0.0)
            out[ok] = np.sqrt(num[ok] / den[ok])
            return out

        # Lower neighbour, base coeff (l-1, m): contributes (l-1) eps_l^m c_(l-1,m)
        ld = l - 1
        has_dn = (ld >= np.abs(m)) & (ld >= 0) & (ld <= lmax)
        idx_dn = np.where(has_dn, ld * (ld + 1) + m, 0)      # zidx_base(l-1,m)
        w_dn = np.where(has_dn, ld.astype(np.float64) * eps(l, m), 0.0)

        # Upper neighbour, base coeff (l+1, m): contributes -(l+2) eps_{l+1}^m c_(l+1,m)
        lu = l + 1
        has_up = (lu >= np.abs(m)) & (lu <= lmax)
        idx_up = np.where(has_up, lu * (lu + 1) + m, 0)      # zidx_base(l+1,m)
        w_up = np.where(has_up, -(l + 2).astype(np.float64) * eps(lu, m), 0.0)

        assert idx_dn.max(initial=0) < nlm and idx_up.max(initial=0) < nlm
        cache = (sh_hi, idx_dn, w_dn, idx_up, w_up)
        self._dtheta_cplx_cache = cache
        return cache

    def SHqst_to_point_cplx(self, Qlm, Slm, Tlm, cost, phi):
        """Evaluate a complex-valued 3D vector field, given by its complex
        radial/spheroidal/toroidal (Q/S/T) spectral coefficients, at the point
        cost=cos(theta), phi.  Returns complex (vr, vt, vp).

        Complex analogue of the (real) shtns.SHqst_to_point, built purely in
        Python on top of the inherited scalar complex point evaluator
        SH_to_point_cplx.  Q/S/T are complex arrays of length nlm_cplx.

        cost, phi may be scalars (returns 3 complex scalars) or matching 1-D
        arrays (returns 3 complex arrays); the coefficient transforms are formed
        once and only the scalar C point-eval is looped over the points.
        """
        n = self.nlm_cplx
        Qlm = np.ascontiguousarray(Qlm, dtype=np.complex128)
        Slm = np.ascontiguousarray(Slm, dtype=np.complex128)
        Tlm = np.ascontiguousarray(Tlm, dtype=np.complex128)
        for name, arr in (("Qlm", Qlm), ("Slm", Slm), ("Tlm", Tlm)):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},). Got {arr.shape}.")

        zm = np.asarray(self.zm, dtype=np.float64)
        sh_hi, idx_dn, w_dn, idx_up, w_up = self._dtheta_cplx_op()

        def stdt_hi(c):
            # sin(theta) d c/d theta, laid out for the degree-(lmax+1) config sh_hi
            return w_dn * c[idx_dn] + w_up * c[idx_up]

        # transformed coefficient arrays (formed once, independent of the point)
        S_dt = stdt_hi(Slm)           # sin(theta) dS/dtheta (sh_hi layout)
        T_dt = stdt_hi(Tlm)           # sin(theta) dT/dtheta (sh_hi layout)
        S_dp = 1j * zm * Slm          # dS/dphi = i m S      (base layout)
        T_dp = 1j * zm * Tlm          # dT/dphi = i m T      (base layout)

        cost_arr = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=np.float64))
        if cost_arr.shape != phi_arr.shape:
            raise ValueError("cost and phi must have the same shape.")

        vr = np.empty(cost_arr.shape, dtype=np.complex128)
        vt = np.empty(cost_arr.shape, dtype=np.complex128)
        vp = np.empty(cost_arr.shape, dtype=np.complex128)
        sp = self.SH_to_point_cplx          # base (degree lmax)
        sp_hi = sh_hi.SH_to_point_cplx      # degree lmax+1, for the theta-derivative
        for i in range(cost_arr.size):
            ct = float(cost_arr.flat[i])
            ph = float(phi_arr.flat[i])
            sint = np.sqrt((1.0 - ct) * (1.0 + ct))
            vr.flat[i] = sp(Qlm, ct, ph)
            dSdt = sp_hi(S_dt, ct, ph) / sint
            dTdt = sp_hi(T_dt, ct, ph) / sint
            imS = sp(S_dp, ct, ph) / sint
            imT = sp(T_dp, ct, ph) / sint
            vt.flat[i] = dSdt + imT      # dS/dtheta + (i m / sin) T
            vp.flat[i] = imS - dTdt      # (i m / sin) S - dT/dtheta

        if np.isscalar(cost) and np.isscalar(phi):
            return complex(vr[0]), complex(vt[0]), complex(vp[0])
        return vr, vt, vp

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

            return jax.ffi.ffi_call(
                "shtns_synth",
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

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

            scaled = ct_out / weights

            result = jax.ffi.ffi_call(
                "shtns_analys",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(scaled, cfg=int(self.this))
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

            return jax.ffi.ffi_call(
                "shtns_analys",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

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

            if self.orthonormal:
                ct_out = ct_out.at[self.lmax + 1 :].multiply(0.5)
            result = jax.ffi.ffi_call(
                "shtns_synth",
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(ct_out, cfg=int(self.this))
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

    def synth_vec_jax(self, x: jax.Array) -> jax.Array:
        """
        Vector inverse SHT: complex128 spectral (3, nlm,) -> float64 spatial (3, *spat_shape).
        """
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (3, self.nlm), jnp.complex128)

        @jax.custom_jvp
        def _synth_vec_impl(x_in: jax.Array) -> jax.Array:
            orig_shape = x_in.shape
            out_shape = (
                (3, *self.spat_shape)
                if len(orig_shape) == 2
                else (*orig_shape[:-2], 3, *self.spat_shape)
            )
            return jax.ffi.ffi_call(
                "shtns_synth_vec",
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        @custom_transpose
        def _synth_vec_tangent(residuals, x_tan: jax.Array) -> jax.Array:
            return _synth_vec_impl(x_tan)

        @_synth_vec_tangent.def_transpose
        def _synth_vec_vjp(residuals, ct_out: jax.Array):
            n_vec_spat_dims = 1 + len(self.spat_shape)
            prefix_shape = ct_out.shape[:-n_vec_spat_dims]
            out_shape = (
                (3, self.nlm)
                if len(ct_out.shape) == n_vec_spat_dims
                else (*prefix_shape, 3, self.nlm)
            )
            return jax.ffi.ffi_call(
                "shtns_adjoint_synth_vec",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(ct_out, cfg=int(self.this))

        @_synth_vec_impl.defjvp
        def _synth_vec_jvp(primals, tangents):
            (x_in,) = primals
            (x_tan,) = tangents
            y = _synth_vec_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _synth_vec_tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _synth_vec_impl(x)

    def analys_vec_jax(self, x: jax.Array) -> jax.Array:
        """Vector forward SHT: float64 spatial (3, spat_shape) -> complex128 spectral (3, nlm)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (3, *self.spat_shape), jnp.float64)

        @jax.custom_jvp
        def _analys_vec_impl(x_in: jax.Array) -> jax.Array:
            prefix_shape = x_in.shape[:-3]
            out_shape = (
                (3, self.nlm) if len(x_in.shape) == 3 else (*prefix_shape, 3, self.nlm)
            )
            return jax.ffi.ffi_call(
                "shtns_analys_vec",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        @custom_transpose
        def _analys_vec_tangent(residuals, x_tan: jax.Array) -> jax.Array:
            return _analys_vec_impl(x_tan)

        @_analys_vec_tangent.def_transpose
        def _analys_vec_vjp(residuals, ct_out: jax.Array):
            orig_shape = ct_out.shape
            out_shape = (
                (3, *self.spat_shape)
                if len(orig_shape) == 2
                else (*orig_shape[:-2], 3, *self.spat_shape)
            )
            return jax.ffi.ffi_call(
                "shtns_adjoint_analys_vec",
                jax.ShapeDtypeStruct(out_shape, jnp.float64),
                vmap_method="broadcast_all",
            )(ct_out, cfg=int(self.this))

        @_analys_vec_impl.defjvp
        def _analys_vec_jvp(primals, tangents):
            (x_in,) = primals
            (x_tan,) = tangents
            y = _analys_vec_impl(x_in)
            tan_out_types = jax.typeof(y).to_tangent_aval()
            y_tan = _analys_vec_tangent(tan_out_types, None, x_tan)
            return y, y_tan

        return _analys_vec_impl(x)

    def synth_vec_cplx_jax(self, x: jax.Array) -> jax.Array:
        """Complex vector inverse SHT: complex128 spectral (3, nlm_cplx) ->
        complex128 spatial (3, spat_shape)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (3, self.nlm_cplx), jnp.complex128)

        def _synth_vec_cplx_impl(x_in: jax.Array) -> jax.Array:
            orig_shape = x_in.shape
            out_shape = (
                (3, *self.spat_shape)
                if len(orig_shape) == 2
                else (*orig_shape[:-2], 3, *self.spat_shape)
            )
            return jax.ffi.ffi_call(
                "shtns_synth_vec_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        return _synth_vec_cplx_impl(x)

    def analys_vec_cplx_jax(self, x: jax.Array) -> jax.Array:
        """Complex vector forward SHT: complex128 spatial (3, spat_shape) ->
        complex128 spectral (3, nlm_cplx)."""
        self._check_jax_gpu_grid_compat()
        self._check_shape_dtype(x, (3, *self.spat_shape), jnp.complex128)

        def _analys_vec_cplx_impl(x_in: jax.Array) -> jax.Array:
            prefix_shape = x_in.shape[:-3]
            out_shape = (
                (3, self.nlm_cplx)
                if len(x_in.shape) == 3
                else (*prefix_shape, 3, self.nlm_cplx)
            )
            return jax.ffi.ffi_call(
                "shtns_analys_vec_cplx",
                jax.ShapeDtypeStruct(out_shape, jnp.complex128),
                vmap_method="broadcast_all",
            )(x_in, cfg=int(self.this))

        return _analys_vec_cplx_impl(x)

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
