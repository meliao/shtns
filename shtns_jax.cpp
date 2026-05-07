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
  // std::vector<double> x_copy(x.typed_data(), x.typed_data() + n_elem);
  for (int64_t n = 0; n < n_other; n++) {
      spat_to_SH(sh, &(x.typed_data()[n * n_spat]), (cplx*) &(y->typed_data()[n * nlm]));
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
