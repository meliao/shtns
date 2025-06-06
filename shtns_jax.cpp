// compile with:
// 	g++ -O2 -fpic -lfftw3 -fopenmp -shared -o libshtns_jax.so shtns_jax.cpp
// or in the Makefile:
// libshtns_jax.so : Makefile shtns_jax.cpp $(objs)
//		g++ -O2 -fpic -fopenmp -shared  -lfftw3_omp -lfftw3 -lm -o libshtns_jax.so $(objs) shtns_jax.cpp


#include "shtns.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

ffi::Error synth_jax_cpu(long cfg, ffi::Buffer<ffi::F64> x,
                       ffi::ResultBuffer<ffi::F64> y) {
						   
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
        .Attr<long>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // qlm
        .Ret<ffi::Buffer<ffi::F64>>()  // q
);
