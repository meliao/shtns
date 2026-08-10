#include "sht_private.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#include <cstdint>

ffi::Error synth_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::C128> x,
                       ffi::ResultBuffer<ffi::F64> y) {

  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm2 = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm2 != sh->nlm)  return ffi::Error::InvalidArgument("shtns: synth input array has wrong size");

  long n_other = x.element_count() / nlm2;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)  return ffi::Error::InvalidArgument("shtns: synth output array has wrong size");

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream
  for (int64_t n = 0; n < n_other; n ++) {	// loop over other dimensions
	  cu_SH_to_spat(sh, (cplx*) &(x.typed_data()[n*nlm2]), &(y->typed_data()[n*n_spat]), sh->lmax);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_gpu, synth_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()  // input: spectral x
        .Ret<ffi::Buffer<ffi::F64>>()  // output: spatial y
);

ffi::Error analys_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::F64> x,
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

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  // Copy input to a scratch buffer: cu_spat_to_SH modifies its input in-place.
  double* x_copy;
  cudaMallocAsync(&x_copy, n_elem * sizeof(double), sh->comp_stream);
  cudaMemcpyAsync(x_copy, x.typed_data(), n_elem * sizeof(double),
                  cudaMemcpyDeviceToDevice, sh->comp_stream);
  for (int64_t n = 0; n < n_other; n++)
    cu_spat_to_SH(sh, &x_copy[n * n_spat], (cplx*) &(y->typed_data()[n * nlm]), sh->lmax);
  cudaFreeAsync(x_copy, sh->comp_stream);

  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_gpu, analys_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // input: spatial x
        .Ret<ffi::Buffer<ffi::C128>>()  // output: spectral y
);

// Adjoint of synthesis: F64 spatial (nlat*nphi) -> C128 spectral (nlm)
// Used as VJP of synth.
ffi::Error adjoint_synth_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::F64> x,
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

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  // Copy input to a scratch buffer: cu_adjoint_SH_to_spat modifies its input in-place.
  double* x_copy;
  cudaMallocAsync(&x_copy, n_elem * sizeof(double), sh->comp_stream);
  cudaMemcpyAsync(x_copy, x.typed_data(), n_elem * sizeof(double),
                  cudaMemcpyDeviceToDevice, sh->comp_stream);
  for (int64_t n = 0; n < n_other; n++)
    cu_adjoint_SH_to_spat(sh, &x_copy[n * n_spat], (cplx*) &(y->typed_data()[n * nlm]), sh->lmax);
  cudaFreeAsync(x_copy, sh->comp_stream);

  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_synth_gpu, adjoint_synth_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()   // input: spatial q
        .Ret<ffi::Buffer<ffi::C128>>()  // output: spectral qlm
);

// Adjoint of analysis: C128 spectral (nlm) -> F64 spatial (nlat*nphi)
// Used as VJP of analys. 
ffi::Error adjoint_analys_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::C128> x,
                                   ffi::ResultBuffer<ffi::F64> y) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != sh->nlm)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys input array has wrong size");
  long n_other = x.element_count() / nlm;
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * n_spat)
    return ffi::Error::InvalidArgument("shtns: adjoint_analys output array has wrong size");

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream
  for (int64_t n = 0; n < n_other; n++) {
    cu_SH_to_spat(sh, (cplx*) &(x.typed_data()[n * nlm]), &(y->typed_data()[n * n_spat]),
                  sh->lmax | SHTNS_ADJOINT);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_analys_gpu, adjoint_analys_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()  // input: spectral qlm
        .Ret<ffi::Buffer<ffi::F64>>()   // output: spatial q
);


// Vector transforms
ffi::Error synth_vec_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::C128> x,
                            ffi::ResultBuffer<ffi::F64> y) {
  // Vector synthesis: C128 spectral (3, nlm) -> F64 spatial (3, n_spat)
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = (x.dimensions().size() == 0) ? 0 : x.dimensions().back();
  if (nlm != (long)sh->nlm)
    return ffi::Error::InvalidArgument("shtns: synth_vec input array has wrong size");
  long n_other = x.element_count() / (3 * nlm);
  long n_spat = sh->nlat * sh->nphi;
  if (y->element_count() != n_other * 3 * n_spat)
    return ffi::Error::InvalidArgument("shtns: synth_vec output array has wrong size");

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream
  for (int64_t n = 0; n < n_other; n++) {
    cplx* qlm = (cplx*) &(x.typed_data()[n * 3 * nlm ]);
    cplx* slm = (cplx*) &(x.typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(x.typed_data()[n * 3 * nlm + 2 * nlm]);
    double* vr  = (double*) &(y->typed_data()[n * 3 * n_spat + 0 * n_spat]);
    double* vt  = (double*) &(y->typed_data()[n * 3 * n_spat + 1 * n_spat]);
    double* vp  = (double*) &(y->typed_data()[n * 3 * n_spat + 2 * n_spat]);
    cu_SHqst_to_spat(sh, qlm, slm, tlm, vr, vt, vp, sh->lmax);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    synth_vec_gpu, synth_vec_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()  // input: spectral x
        .Ret<ffi::Buffer<ffi::F64>>()  // output: spatial y
);

ffi::Error analys_vec_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::F64> x,
                            ffi::ResultBuffer<ffi::C128> y) {
  // Vector analysis: F64 spatial (3, n_spat) -> C128 spectral (3, nlm)
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long n_spat = sh->nlat * sh->nphi;
  long n_total = x.element_count();
  if ((n_spat == 0) || (n_total % (3 * n_spat) != 0))
    return ffi::Error::InvalidArgument("shtns: analys_vec input array has wrong size");
  long n_other = n_total / (3 * n_spat);
  long nlm = sh->nlm;
  if (y->element_count() != n_other * 3 * nlm)
    return ffi::Error::InvalidArgument("shtns: analys_vec output array has wrong size");

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  // Make a copy of the input array
  double* x_copy;
  cudaMallocAsync(&x_copy, n_total * sizeof(double), sh->comp_stream);
  cudaMemcpyAsync(x_copy, x.typed_data(), n_total * sizeof(double), cudaMemcpyDeviceToDevice, sh->comp_stream);
  for (int64_t n = 0; n < n_other; n++) {
    double* vr = (double*) &(x_copy[n * 3 * n_spat]);
    double* vt = (double*) &(x_copy[n * 3 * n_spat + 1 * n_spat]);
    double* vp = (double*) &(x_copy[n * 3 * n_spat + 2 * n_spat]);
    cplx* qlm = (cplx*) &(y->typed_data()[n * 3 * nlm]);
    cplx* slm = (cplx*) &(y->typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(y->typed_data()[n * 3 * nlm + 2 * nlm]);
    cu_spat_to_SHqst(sh, vr, vt, vp, qlm, slm, tlm, sh->lmax);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  cudaFreeAsync(x_copy, sh->comp_stream);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analys_vec_gpu, analys_vec_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()  // input: spatial x
        .Ret<ffi::Buffer<ffi::C128>>()  // output: spectral y
);

// Adjoint of vector synthesis: F64 spatial (3, n_spat) -> C128 spectral (3, nlm)
// Used as VJP of synth_vec.
ffi::Error adjoint_synth_vec_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::F64> x,
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

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  // Copy input to a scratch buffer: cu_adjoint_SHqst_to_spat modifies its input in-place.
  double* x_copy;
  cudaMallocAsync(&x_copy, n_total * sizeof(double), sh->comp_stream);
  cudaMemcpyAsync(x_copy, x.typed_data(), n_total * sizeof(double), cudaMemcpyDeviceToDevice, sh->comp_stream);
  for (int64_t n = 0; n < n_other; n++) {
    double* vr = (double*) &(x_copy[n * 3 * n_spat]);
    double* vt = (double*) &(x_copy[n * 3 * n_spat + 1 * n_spat]);
    double* vp = (double*) &(x_copy[n * 3 * n_spat + 2 * n_spat]);
    cplx* qlm = (cplx*) &(y->typed_data()[n * 3 * nlm]);
    cplx* slm = (cplx*) &(y->typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(y->typed_data()[n * 3 * nlm + 2 * nlm]);
    cu_adjoint_SHqst_to_spat(sh, vr, vt, vp, qlm, slm, tlm, sh->lmax);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  cudaFreeAsync(x_copy, sh->comp_stream);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_synth_vec_gpu, adjoint_synth_vec_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::F64>>()    // input: stacked spatial (3, n_spat) [vr, vt, vp]
        .Ret<ffi::Buffer<ffi::C128>>()   // output: stacked spectral (3, nlm) [qlm, slm, tlm]
);

// Adjoint of vector analysis: C128 spectral (3, nlm) -> F64 spatial (3, n_spat)
// Used as VJP of analys_vec. No native cu_adjoint_spat_to_SHqst export exists, but
// (as with the scalar case) cu_SHqst_to_spat already applies weights on-device
// when SHTNS_ADJOINT is OR'd into ltr.
ffi::Error adjoint_analys_vec_jax_gpu(cudaStream_t jax_strm, int64_t cfg, ffi::Buffer<ffi::C128> x,
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

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream
  for (int64_t n = 0; n < n_other; n++) {
    cplx* qlm = (cplx*) &(x.typed_data()[n * 3 * nlm]);
    cplx* slm = (cplx*) &(x.typed_data()[n * 3 * nlm + 1 * nlm]);
    cplx* tlm = (cplx*) &(x.typed_data()[n * 3 * nlm + 2 * nlm]);
    double* vr = (double*) &(y->typed_data()[n * 3 * n_spat]);
    double* vt = (double*) &(y->typed_data()[n * 3 * n_spat + 1 * n_spat]);
    double* vp = (double*) &(y->typed_data()[n * 3 * n_spat + 2 * n_spat]);
    cu_SHqst_to_spat(sh, qlm, slm, tlm, vr, vt, vp, sh->lmax | SHTNS_ADJOINT);
  }
  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    adjoint_analys_vec_gpu, adjoint_analys_vec_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // input: stacked spectral (3, nlm) [qlm, slm, tlm]
        .Ret<ffi::Buffer<ffi::F64>>()    // output: stacked spatial (3, n_spat) [vr, vt, vp]
);
