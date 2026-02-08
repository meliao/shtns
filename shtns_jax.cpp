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
  long nlm2 = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm2 != 2*sh->nlm)  return ffi::Error::InvalidArgument("shtns: synth input array has wrong size");
  
  long n_other = x.element_count() / nlm2;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)  return ffi::Error::InvalidArgument("shtns: synth output array has wrong size");

  for (int64_t n = 0; n < n_other; n ++) {	// loop over other dimensions
	  SH_to_spat(sh, (cplx*) &(x.typed_data()[n*nlm2]), &(y->typed_data()[n*n_spat]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_cpu, synth_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // qlm
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
  long nlm2 = 2 * sh->nlm;
  if (y->element_count() != n_other * nlm2) {
	  return ffi::Error::InvalidArgument("shtns: analys output array has wrong size");
  }

  for (int64_t n = 0; n < n_other; n ++) {	// loop over other dimensions
	  spat_to_SH(sh, &(x.typed_data()[n*n_spat]), (cplx*) &(y->typed_data()[n*nlm2]));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_cpu, analys_jax_cpu,
    ffi::Ffi::Bind()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // q
        .Ret<ffi::Buffer<ffi::F64>>()  // qlm
);


#if SHTNS_GPU == 1

ffi::Error synth_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::F64> x,
                       ffi::ResultBuffer<ffi::F64> y) {
						   
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm2 = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm2 != 2*sh->nlm)  return ffi::Error::InvalidArgument("shtns: synth input array has wrong size");
  
  long n_other = x.element_count() / nlm2;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)  return ffi::Error::InvalidArgument("shtns: synth output array has wrong size");

  cudaEventRecord( shtns->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( shtns->comp_stream, shtns->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream
  for (int64_t n = 0; n < n_other; n ++) {	// loop over other dimensions
	  cu_SH_to_spat(sh, (cplx*) &(x.typed_data()[n*nlm2]), &(y->typed_data()[n*n_spat]), sh->lmax);
  }
  cudaEventRecord( shtns->sync_evt, shtns->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, shtns->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_gpu, synth_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // qlm
        .Ret<ffi::Buffer<ffi::F64>>()  // q
);

#endif
