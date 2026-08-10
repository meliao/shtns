// compile with:
// 	g++ -O2 -fpic -lfftw3 -fopenmp -shared -o libshtns_jax.so shtns_jax.cpp
// or in the Makefile:
// libshtns_jax.so : Makefile shtns_jax.cpp $(objs)
//		g++ -O2 -fpic -fopenmp -shared  -lfftw3_omp -lfftw3 -lm -o libshtns_jax.so $(objs) shtns_jax.cpp


#include "sht_private.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#include <cstdint>

ffi::Error synth_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                       ffi::ResultBuffer<ffi::F64> y) {
      // Inverse spherical harmonic transform. Spectral -> spatial transform.
      // Expects complex128 inputs, returns float64 outputs.
						   
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != sh->nlm)  return ffi::Error::InvalidArgument("shtns: synth input array has wrong size");
  
  long n_other = x.element_count() / nlm;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)  return ffi::Error::InvalidArgument("shtns: synth output array has wrong size");

  for (int64_t n = 0; n < n_other; n ++) {	// loop over other dimensions
	  SH_to_spat(sh, (cplx*) &(x.typed_data()[n*nlm]), &(y->typed_data()[n*n_spat]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_cpu, synth_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()  // qlm
        .Ret<ffi::Buffer<ffi::F64>>()  // q
);

ffi::Error analys_jax_cpu(int64_t cfg, ffi::Buffer<ffi::F64> x,
                       ffi::ResultBuffer<ffi::C128> y) {
  // Forward spherical harmonic transform. Spatial -> spectral transform.
  // Expects float64 inputs, returns complex128 outputs.
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_elem = x.element_count();
  if ((n_spat == 0) || (n_elem % n_spat != 0)) {
	  return ffi::Error::InvalidArgument("shtns: analys input array has wrong size");
  }
  long n_other = n_elem / n_spat;
  long nlm = sh->nlm;
  if (y->element_count() != n_other * nlm) {
	  return ffi::Error::InvalidArgument("shtns: analys output array has wrong size");
  }

  // spat_to_SH uses the spatial buffer as FFT scratch space (in-place transforms),
  // so we must copy before passing to avoid corrupting JAX's immutable Arg buffer.
  std::vector<double> x_copy(x.typed_data(), x.typed_data() + n_elem);
  for (int64_t n = 0; n < n_other; n++) {
      spat_to_SH(sh, &(x_copy[n * n_spat]), (cplx*) &(y->typed_data()[n * nlm]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_cpu, analys_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // q
        .Ret<ffi::Buffer<ffi::C128>>()  // qlm
);

// Adjoint of synthesis: F64 spatial (nlat*nphi) -> C128 spectral (nlm)
// Used as VJP of synth.
ffi::Error adjoint_synth_jax_cpu(int64_t cfg, ffi::Buffer<ffi::F64> x,
                                  ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_elem = x.element_count();
  if ((n_spat == 0) || (n_elem % n_spat != 0))
    return ffi::Error::InvalidArgument("shtns: adjoint_synth input array has wrong size");
  long n_other = n_elem / n_spat;
  long nlm = sh->nlm;
  if (y->element_count() != n_other * nlm)
    return ffi::Error::InvalidArgument("shtns: adjoint_synth output array has wrong size");
  // adjoint_SH_to_spat uses the spatial buffer as FFT scratch space (in-place transforms),
  // so we must copy before passing to avoid corrupting JAX's immutable Arg buffer.
  std::vector<double> x_copy(x.typed_data(), x.typed_data() + n_elem);
  for (int64_t n = 0; n < n_other; n++) {
    adjoint_SH_to_spat(sh, &(x_copy[n * n_spat]), (cplx*) &(y->typed_data()[n * nlm]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_synth_cpu, adjoint_synth_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()   // q: real spatial
        .Ret<ffi::Buffer<ffi::C128>>()  // qlm: complex spectral
);

// Adjoint of analysis: C128 spectral (nlm) -> F64 spatial (nlat*nphi)
// Used as VJP of analys.
ffi::Error adjoint_analys_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                   ffi::ResultBuffer<ffi::F64> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != sh->nlm)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys input array has wrong size");
  long n_other = x.element_count() / nlm;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys output array has wrong size");
  for (int64_t n = 0; n < n_other; n++) {
    adjoint_spat_to_SH(sh, (cplx*) &(x.typed_data()[n * nlm]), &(y->typed_data()[n * n_spat]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_analys_cpu, adjoint_analys_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()  // qlm: complex spectral
        .Ret<ffi::Buffer<ffi::F64>>()   // q: real spatial
);

// Vector transforms

// Vector synthesis: C128 spectral (3, nlm) -> F64 spatial (3, nlat*nphi)
// Input stacking: [qlm (nlm), slm (nlm), tlm (nlm)]
// Output stacking: [vr (n_spat), vt (n_spat), vp (n_spat)]
ffi::Error synth_vec_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                               ffi::ResultBuffer<ffi::F64> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != sh->nlm)
    return ffi::Error::InvalidArgument("shtns: synth_vec input array has wrong size");
  long n_total = x.element_count();
  if (n_total % (3 * nlm) != 0)
    return ffi::Error::InvalidArgument("shtns: synth_vec input second-to-last dim must be 3");
  long n_other = n_total / (3 * nlm);
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * 3 * n_spat)
    return ffi::Error::InvalidArgument("shtns: synth_vec output array has wrong size");
  for (int64_t n = 0; n < n_other; n++) {
    cplx* qlm = (cplx*) &(x.typed_data()[n * 3 * nlm + 0 * nlm]);
    cplx* slm = (cplx*) &(x.typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(x.typed_data()[n * 3 * nlm + 2 * nlm]);
    double* vr = &(y->typed_data()[n * 3 * n_spat + 0 * n_spat]);
    double* vt = &(y->typed_data()[n * 3 * n_spat + 1 * n_spat]);
    double* vp = &(y->typed_data()[n * 3 * n_spat + 2 * n_spat]);
    SHqst_to_spat(sh, qlm, slm, tlm, vr, vt, vp);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_vec_cpu, synth_vec_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked spectral (3, nlm)
        .Ret<ffi::Buffer<ffi::F64>>()    // [vr, vt, vp]: stacked spatial (3, n_spat)
);

// Vector analysis: F64 spatial (3, nlat*nphi) -> C128 spectral (3, nlm)
// Input stacking: [vr (n_spat), vt (n_spat), vp (n_spat)]
// Output stacking: [qlm (nlm), slm (nlm), tlm (nlm)]
ffi::Error analys_vec_jax_cpu(int64_t cfg, ffi::Buffer<ffi::F64> x,
                               ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_total = x.element_count();
  if ((n_spat == 0) || (n_total % (3 * n_spat) != 0))
    return ffi::Error::InvalidArgument("shtns: analys_vec input array has wrong size");
  long n_other = n_total / (3 * n_spat);
  long nlm = sh->nlm;
  if (y->element_count() != n_other * 3 * nlm)
    return ffi::Error::InvalidArgument("shtns: analys_vec output array has wrong size");
  // spat_to_SHqst uses spatial buffers as FFT scratch space, so copy first
  std::vector<double> x_copy(x.typed_data(), x.typed_data() + n_total);
  for (int64_t n = 0; n < n_other; n++) {
    double* vr = &(x_copy[n * 3 * n_spat + 0 * n_spat]);
    double* vt = &(x_copy[n * 3 * n_spat + 1 * n_spat]);
    double* vp = &(x_copy[n * 3 * n_spat + 2 * n_spat]);
    cplx* qlm = (cplx*) &(y->typed_data()[n * 3 * nlm + 0 * nlm]);
    cplx* slm = (cplx*) &(y->typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(y->typed_data()[n * 3 * nlm + 2 * nlm]);
    spat_to_SHqst(sh, vr, vt, vp, qlm, slm, tlm);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_vec_cpu, analys_vec_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()    // [vr, vt, vp]: stacked spatial (3, n_spat)
        .Ret<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked spectral (3, nlm)
);

// Adjoint of vector synthesis: F64 spatial (3, n_spat) -> C128 spectral (3, nlm)
// Used as VJP of synth_vec.
ffi::Error adjoint_synth_vec_jax_cpu(int64_t cfg, ffi::Buffer<ffi::F64> x,
                                      ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_total = x.element_count();
  if ((n_spat == 0) || (n_total % (3 * n_spat) != 0))
    return ffi::Error::InvalidArgument("shtns: adjoint_synth_vec input array has wrong size");
  long n_other = n_total / (3 * n_spat);
  long nlm = sh->nlm;
  if (y->element_count() != n_other * 3 * nlm)
    return ffi::Error::InvalidArgument("shtns: adjoint_synth_vec output array has wrong size");
  // Make a copy of the input
  std::vector<double> x_copy(x.typed_data(), x.typed_data() + n_total);
  for (int64_t n = 0; n < n_other; n++) {
    double* vr = &(x_copy[n * 3 * n_spat ]);
    double* vt = &(x_copy[n * 3 * n_spat + 1 * n_spat]);
    double* vp = &(x_copy[n * 3 * n_spat + 2 * n_spat]);
    cplx* qlm = (cplx*) &(y->typed_data()[n * 3 * nlm ]);
    cplx* slm = (cplx*) &(y->typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(y->typed_data()[n * 3 * nlm + 2 * nlm]);
    adjoint_SHqst_to_spat(sh, vr, vt, vp, qlm, slm, tlm);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_synth_vec_cpu, adjoint_synth_vec_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()    // [vr, vt, vp]: stacked spatial (3, n_spat)
        .Ret<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked spectral (3, nlm)
);

// Adjoint of vector analysis: C128 spectral (3, nlm) -> F64 spatial (3, n_spat)
// Used as VJP of analys_vec.
ffi::Error adjoint_analys_vec_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                       ffi::ResultBuffer<ffi::F64> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != sh->nlm)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys_vec input array has wrong size");
  long n_total = x.element_count();
  if (n_total % (3 * nlm) != 0)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys_vec input second-to-last dim must be 3");
  long n_other = n_total / (3 * nlm);
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * 3 * n_spat)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys_vec output array has wrong size");
  for (int64_t n = 0; n < n_other; n++) {
    cplx* qlm = (cplx*) &(x.typed_data()[n * 3 * nlm ]);
    cplx* slm = (cplx*) &(x.typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(x.typed_data()[n * 3 * nlm + 2 * nlm]);
    double* vr = &(y->typed_data()[n * 3 * n_spat ]);
    double* vt = &(y->typed_data()[n * 3 * n_spat + 1 * n_spat]);
    double* vp = &(y->typed_data()[n * 3 * n_spat + 2 * n_spat]);
    adjoint_spat_to_SHqst(sh, qlm, slm, tlm, vr, vt, vp);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_analys_vec_cpu, adjoint_analys_vec_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked spectral (3, nlm)
        .Ret<ffi::Buffer<ffi::F64>>()    // [vr, vt, vp]: stacked spatial (3, n_spat)
);

// Complex synthesis: C128 spectral (nlm_cplx) -> C128 spatial (nlat*nphi)
ffi::Error synth_cplx_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                               ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm_cplx = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm_cplx != (long)sh->nlm_cplx)
    return ffi::Error::InvalidArgument("shtns: synth_cplx input array has wrong size");
  long n_other = x.element_count() / nlm_cplx;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)
    return ffi::Error::InvalidArgument("shtns: synth_cplx output array has wrong size");
  for (int64_t n = 0; n < n_other; n++)
    SH_to_spat_cplx(sh, (cplx*) &(x.typed_data()[n*nlm_cplx]),
                        (cplx*) &(y->typed_data()[n*n_spat]));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_cplx_cpu, synth_cplx_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // alm: complex spectral
        .Ret<ffi::Buffer<ffi::C128>>()   // z:   complex spatial
);

// Complex analysis: C128 spatial (nlat*nphi) -> C128 spectral (nlm_cplx)
ffi::Error analys_cplx_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_elem = x.element_count();
  if ((n_spat == 0) || (n_elem % n_spat != 0))
    return ffi::Error::InvalidArgument("shtns: analys_cplx input array has wrong size");
  long n_other = n_elem / n_spat;
  long nlm_cplx = sh->nlm_cplx;
  if (y->element_count() != n_other * nlm_cplx)
    return ffi::Error::InvalidArgument("shtns: analys_cplx output array has wrong size");
  std::vector<cplx> x_copy(x.typed_data(), x.typed_data() + n_elem);
  for (int64_t n = 0; n < n_other; n++)
    spat_cplx_to_SH(sh, (cplx*) &(x_copy[n*n_spat]),
                        (cplx*) &(y->typed_data()[n*nlm_cplx]));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_cplx_cpu, analys_cplx_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // z:   complex spatial
        .Ret<ffi::Buffer<ffi::C128>>()   // alm: complex spectral
);

// Complex vector synthesis: C128 spectral (3, nlm_cplx) -> C128 spatial (3, nlat*nphi)
// Input stacking:  [qlm (nlm_cplx), slm (nlm_cplx), tlm (nlm_cplx)]
// Output stacking: [vr (n_spat),    vt (n_spat),    vp (n_spat)]
ffi::Error synth_vec_cplx_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                   ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm_cplx = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm_cplx != (long)sh->nlm_cplx)
    return ffi::Error::InvalidArgument("shtns: synth_vec_cplx input array has wrong size");
  long n_total = x.element_count();
  if (n_total % (3 * nlm_cplx) != 0)
    return ffi::Error::InvalidArgument("shtns: synth_vec_cplx input second-to-last dim must be 3");
  long n_other = n_total / (3 * nlm_cplx);
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * 3 * n_spat)
    return ffi::Error::InvalidArgument("shtns: synth_vec_cplx output array has wrong size");
  for (int64_t n = 0; n < n_other; n++) {
    cplx* qlm = (cplx*) &(x.typed_data()[n * 3 * nlm_cplx + 0 * nlm_cplx]);
    cplx* slm = (cplx*) &(x.typed_data()[n * 3 * nlm_cplx + 1 * nlm_cplx]);
    cplx* tlm = (cplx*) &(x.typed_data()[n * 3 * nlm_cplx + 2 * nlm_cplx]);
    cplx* vr  = (cplx*) &(y->typed_data()[n * 3 * n_spat + 0 * n_spat]);
    cplx* vt  = (cplx*) &(y->typed_data()[n * 3 * n_spat + 1 * n_spat]);
    cplx* vp  = (cplx*) &(y->typed_data()[n * 3 * n_spat + 2 * n_spat]);
    SHqst_to_spat_cplx(sh, qlm, slm, tlm, vr, vt, vp);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_vec_cplx_cpu, synth_vec_cplx_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked complex spectral (3, nlm_cplx)
        .Ret<ffi::Buffer<ffi::C128>>()   // [vr, vt, vp]:   stacked complex spatial  (3, n_spat)
);

// Complex vector analysis: C128 spatial (3, nlat*nphi) -> C128 spectral (3, nlm_cplx)
// Input stacking:  [vr (n_spat),    vt (n_spat),    vp (n_spat)]
// Output stacking: [qlm (nlm_cplx), slm (nlm_cplx), tlm (nlm_cplx)]
ffi::Error analys_vec_cplx_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                    ffi::ResultBuffer<ffi::C128> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_total = x.element_count();
  if ((n_spat == 0) || (n_total % (3 * n_spat) != 0))
    return ffi::Error::InvalidArgument("shtns: analys_vec_cplx input array has wrong size");
  long n_other = n_total / (3 * n_spat);
  long nlm_cplx = sh->nlm_cplx;
  if (y->element_count() != n_other * 3 * nlm_cplx)
    return ffi::Error::InvalidArgument("shtns: analys_vec_cplx output array has wrong size");
  // spat_cplx_to_SHqst uses spatial buffers as FFT scratch space, so copy first
  std::vector<cplx> x_copy(x.typed_data(), x.typed_data() + n_total);
  for (int64_t n = 0; n < n_other; n++) {
    cplx* vr  = &(x_copy[n * 3 * n_spat + 0 * n_spat]);
    cplx* vt  = &(x_copy[n * 3 * n_spat + 1 * n_spat]);
    cplx* vp  = &(x_copy[n * 3 * n_spat + 2 * n_spat]);
    cplx* qlm = (cplx*) &(y->typed_data()[n * 3 * nlm_cplx + 0 * nlm_cplx]);
    cplx* slm = (cplx*) &(y->typed_data()[n * 3 * nlm_cplx + 1 * nlm_cplx]);
    cplx* tlm = (cplx*) &(y->typed_data()[n * 3 * nlm_cplx + 2 * nlm_cplx]);
    spat_cplx_to_SHqst(sh, vr, vt, vp, qlm, slm, tlm);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_vec_cplx_cpu, analys_vec_cplx_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // [vr, vt, vp]:   stacked complex spatial  (3, n_spat)
        .Ret<ffi::Buffer<ffi::C128>>()   // [qlm, slm, tlm]: stacked complex spectral (3, nlm_cplx)
);


// SHqst_to_lat: C128 spectral (..., 3, nlm) + F64 cost (..., 1) -> F64 spatial (..., 3, nphi)
// NOT thread-safe: SHqst_to_lat caches Legendre functions in shtns_cfg. Batch loop is serial.
ffi::Error SHqst_to_lat_jax_cpu(int64_t cfg, int64_t nphi_attr, int64_t ltr, int64_t mtr,
                                  ffi::Buffer<ffi::C128> spec,
                                  ffi::Buffer<ffi::F64> cost_buf,
                                  ffi::ResultBuffer<ffi::F64> out) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  int nphi = (int)nphi_attr;
  long nlm = sh->nlm;
  long n_total = spec.element_count();
  if (nlm <= 0 || n_total % (3 * nlm) != 0)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_lat: spectral input bad size");
  long n_other = n_total / (3 * nlm);
  if (out->element_count() != n_other * 3 * nphi)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_lat: output bad size");
  for (int64_t n = 0; n < n_other; n++) {
    double cost = cost_buf.typed_data()[n];
    cplx* qlm = (cplx*)&spec.typed_data()[n * 3 * nlm];
    cplx* slm = (cplx*)&spec.typed_data()[n * 3 * nlm + nlm];
    cplx* tlm = (cplx*)&spec.typed_data()[n * 3 * nlm + 2 * nlm];
    double* vr = &out->typed_data()[n * 3 * nphi];
    double* vt = &out->typed_data()[n * 3 * nphi + nphi];
    double* vp = &out->typed_data()[n * 3 * nphi + 2 * nphi];
    SHqst_to_lat(sh, qlm, slm, tlm, cost, vr, vt, vp, nphi, (int)ltr, (int)mtr);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SHqst_to_lat_cpu, SHqst_to_lat_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Attr<int64_t>("nphi")
        .Attr<int64_t>("ltr")
        .Attr<int64_t>("mtr")
        .Arg<ffi::Buffer<ffi::C128>>()   // stacked spectral (..., 3, nlm): [Qlm, Slm, Tlm]
        .Arg<ffi::Buffer<ffi::F64>>()    // cost (..., 1)
        .Ret<ffi::Buffer<ffi::F64>>()    // stacked spatial (..., 3, nphi): [Vr, Vt, Vp]
);




// Rotation apply_real: C128 spectral (..., nlm) -> C128 spectral (..., nlm)
// This is for a REAL signal on the sphere.
ffi::Error rotation_apply_real_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                        ffi::ResultBuffer<ffi::C128> y) {
  shtns_rot r = reinterpret_cast<shtns_rot>(cfg);
  long nlm = x.dimensions().back();
  long n_other = (nlm > 0) ? x.element_count() / nlm : 0;
  if (y->element_count() != x.element_count())
    return ffi::Error::InvalidArgument("shtns: rotation_apply_real: output size mismatch");
  for (int64_t n = 0; n < n_other; n++)
    shtns_rotation_apply_real(r,
        (cplx*)&x.typed_data()[n * nlm],
        (cplx*)&y->typed_data()[n * nlm]);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rotation_apply_real_cpu, rotation_apply_real_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // alm: real SH spectral coeffs (..., nlm)
        .Ret<ffi::Buffer<ffi::C128>>()   // rlm: rotated coeffs (..., nlm)
);

// Rotation apply_cplx: C128 spectral (..., nlm_cplx) -> C128 spectral (..., nlm_cplx)
// This is for a COMPLEX signal on the sphere.
ffi::Error rotation_apply_cplx_jax_cpu(int64_t cfg, ffi::Buffer<ffi::C128> x,
                                        ffi::ResultBuffer<ffi::C128> y) {
  shtns_rot r = reinterpret_cast<shtns_rot>(cfg);
  long nlm_cplx = x.dimensions().back();
  long n_other = (nlm_cplx > 0) ? x.element_count() / nlm_cplx : 0;
  if (y->element_count() != x.element_count())
    return ffi::Error::InvalidArgument("shtns: rotation_apply_cplx: output size mismatch");
  for (int64_t n = 0; n < n_other; n++)
    shtns_rotation_apply_cplx(r,
        (cplx*)&x.typed_data()[n * nlm_cplx],
        (cplx*)&y->typed_data()[n * nlm_cplx]);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rotation_apply_cplx_cpu, rotation_apply_cplx_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // alm: complex SH spectral coeffs (..., nlm_cplx)
        .Ret<ffi::Buffer<ffi::C128>>()   // rlm: rotated coeffs (..., nlm_cplx)
);
