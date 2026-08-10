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

// SHqst_to_point GPU: C128 spectral (n_fields, 3, nlm) [Qlm,Slm,Tlm] + F64 cost (npts,)
// + F64 phi (npts,) -> F64 (3, npts) [Vr,Vt,Vp]. n_fields must be 1 (a single field shared
// by every point) or npts (one field per point, zipped with cost/phi) -- the latter is what
// vmap_method="broadcast_all" produces when this op is vmapped. When n_fields>1, the JAX
// buffer is interleaved per point ([Q,S,T,Q,S,T,...]) but cu_SHqst_to_point (sht_gpu_local.cu)
// expects 3 planar per-field blocks, so this wrapper repacks via cudaMemcpy2DAsync first
// (same technique as SHqst_to_lat below); the n_fields==1 case needs no repacking.
ffi::Error SHqst_to_point_jax_gpu(cudaStream_t jax_strm, int64_t cfg,
                                   ffi::Buffer<ffi::C128> spec,
                                   ffi::Buffer<ffi::F64> cost_buf,
                                   ffi::Buffer<ffi::F64> phi_buf,
                                   ffi::ResultBuffer<ffi::F64> out) {
  shtns_cfg sh = reinterpret_cast<shtns_cfg>(cfg);
  long nlm = sh->nlm;
  long n_total = spec.element_count();
  if (nlm <= 0 || n_total % (3 * nlm) != 0)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_point: spectral input bad size");
  long n_fields = n_total / (3 * nlm);
  long npts = cost_buf.element_count();
  if (phi_buf.element_count() != npts)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_point: cost/phi size mismatch");
  if (n_fields != 1 && n_fields != npts)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_point: spectral batch must be 1 or match npts");
  if (out->element_count() != 3 * npts)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_point: output bad size");

  double* vr = &out->typed_data()[0];
  double* vt = &out->typed_data()[npts];
  double* vp = &out->typed_data()[2 * npts];

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  const cplx *qlm, *slm, *tlm;
  cplx *d_Qlm = nullptr, *d_Slm = nullptr, *d_Tlm = nullptr;
  if (n_fields == 1) {
    qlm = (const cplx*) &spec.typed_data()[0];
    slm = (const cplx*) &spec.typed_data()[nlm];
    tlm = (const cplx*) &spec.typed_data()[2 * nlm];
  } else {
    cplx* spec_ptr = (cplx*) spec.typed_data();
    const size_t celem = sizeof(cplx);
    cudaMallocAsync((void**)&d_Qlm, n_fields*nlm*celem, sh->comp_stream);
    cudaMallocAsync((void**)&d_Slm, n_fields*nlm*celem, sh->comp_stream);
    cudaMallocAsync((void**)&d_Tlm, n_fields*nlm*celem, sh->comp_stream);
    cudaMemcpy2DAsync(d_Qlm, nlm*celem, spec_ptr + 0*nlm, 3*nlm*celem, nlm*celem, n_fields, cudaMemcpyDeviceToDevice, sh->comp_stream);
    cudaMemcpy2DAsync(d_Slm, nlm*celem, spec_ptr + 1*nlm, 3*nlm*celem, nlm*celem, n_fields, cudaMemcpyDeviceToDevice, sh->comp_stream);
    cudaMemcpy2DAsync(d_Tlm, nlm*celem, spec_ptr + 2*nlm, 3*nlm*celem, nlm*celem, n_fields, cudaMemcpyDeviceToDevice, sh->comp_stream);
    qlm = d_Qlm; slm = d_Slm; tlm = d_Tlm;
  }

  cu_SHqst_to_point(sh, qlm, slm, tlm, n_fields, cost_buf.typed_data(), phi_buf.typed_data(),
                     vr, vt, vp, npts, sh->lmax, sh->mmax);

  if (n_fields != 1) {
    cudaFreeAsync(d_Qlm, sh->comp_stream);
    cudaFreeAsync(d_Slm, sh->comp_stream);
    cudaFreeAsync(d_Tlm, sh->comp_stream);
  }

  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SHqst_to_point_gpu, SHqst_to_point_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Arg<ffi::Buffer<ffi::C128>>()   // stacked spectral (3, nlm): [Qlm, Slm, Tlm] (one shared field)
        .Arg<ffi::Buffer<ffi::F64>>()    // cost (npts,)
        .Arg<ffi::Buffer<ffi::F64>>()    // phi (npts,)
        .Ret<ffi::Buffer<ffi::F64>>()    // stacked (3, npts): [Vr, Vt, Vp]
);

// SHqst_to_lat GPU: C128 spectral (n_other, 3, nlm) interleaved [Qlm,Slm,Tlm] per batch
// + F64 cost (n_other,) -> F64 spatial (n_other, 3, nphi) interleaved [Vr,Vt,Vp] per batch.
// cu_SHqst_to_lat (sht_gpu_local.cu) works on PLANAR per-field buffers (one contiguous block
// per field, spanning all n_other batches), so this wrapper repacks via strided
// device-to-device copies (cudaMemcpy2DAsync) around the call, keeping the already-validated
// kernel/VkFFT code untouched.
ffi::Error SHqst_to_lat_jax_gpu(cudaStream_t jax_strm, int64_t cfg, int64_t nphi_attr,
                                 int64_t ltr, int64_t mtr,
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
  if (cost_buf.element_count() != n_other)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_lat: cost size mismatch");
  if (out->element_count() != n_other * 3 * nphi)
    return ffi::Error::InvalidArgument("shtns: SHqst_to_lat: output bad size");

  cplx* spec_ptr = (cplx*) spec.typed_data();
  double* out_ptr = out->typed_data();
  const size_t celem = sizeof(cplx), delem = sizeof(double);

  cudaEventRecord( sh->sync_evt, jax_strm );		// record an event on the JAX stream
  cudaStreamWaitEvent( sh->comp_stream, sh->sync_evt, 0 );	// make the SHTns stream wait for the JAX stream

  cplx *d_Qlm, *d_Slm, *d_Tlm;
  cudaMallocAsync((void**)&d_Qlm, n_other*nlm*celem, sh->comp_stream);
  cudaMallocAsync((void**)&d_Slm, n_other*nlm*celem, sh->comp_stream);
  cudaMallocAsync((void**)&d_Tlm, n_other*nlm*celem, sh->comp_stream);
  // interleaved (n_other,3,nlm) -> planar (n_other,nlm) per field
  cudaMemcpy2DAsync(d_Qlm, nlm*celem, spec_ptr + 0*nlm, 3*nlm*celem, nlm*celem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);
  cudaMemcpy2DAsync(d_Slm, nlm*celem, spec_ptr + 1*nlm, 3*nlm*celem, nlm*celem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);
  cudaMemcpy2DAsync(d_Tlm, nlm*celem, spec_ptr + 2*nlm, 3*nlm*celem, nlm*celem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);

  double *d_vr, *d_vt, *d_vp;
  cudaMallocAsync((void**)&d_vr, n_other*nphi*delem, sh->comp_stream);
  cudaMallocAsync((void**)&d_vt, n_other*nphi*delem, sh->comp_stream);
  cudaMallocAsync((void**)&d_vp, n_other*nphi*delem, sh->comp_stream);

  cu_SHqst_to_lat(sh, d_Qlm, d_Slm, d_Tlm, cost_buf.typed_data(), d_vr, d_vt, d_vp,
                   n_other, nphi, (int)ltr, (int)mtr);

  // planar (n_other,nphi) per field -> interleaved (n_other,3,nphi)
  cudaMemcpy2DAsync(out_ptr + 0*nphi, 3*nphi*delem, d_vr, nphi*delem, nphi*delem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);
  cudaMemcpy2DAsync(out_ptr + 1*nphi, 3*nphi*delem, d_vt, nphi*delem, nphi*delem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);
  cudaMemcpy2DAsync(out_ptr + 2*nphi, 3*nphi*delem, d_vp, nphi*delem, nphi*delem, n_other, cudaMemcpyDeviceToDevice, sh->comp_stream);

  cudaFreeAsync(d_Qlm, sh->comp_stream);
  cudaFreeAsync(d_Slm, sh->comp_stream);
  cudaFreeAsync(d_Tlm, sh->comp_stream);
  cudaFreeAsync(d_vr, sh->comp_stream);
  cudaFreeAsync(d_vt, sh->comp_stream);
  cudaFreeAsync(d_vp, sh->comp_stream);

  cudaEventRecord( sh->sync_evt, sh->comp_stream );		// record an event on the SHTns stream
  cudaStreamWaitEvent( jax_strm, sh->sync_evt, 0 );	// make the JAX stream wait for the SHTns stream

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SHqst_to_lat_gpu, SHqst_to_lat_jax_gpu,
    ffi::Ffi::Bind()
		.Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("cfg")
        .Attr<int64_t>("nphi")
        .Attr<int64_t>("ltr")
        .Attr<int64_t>("mtr")
        .Arg<ffi::Buffer<ffi::C128>>()   // stacked spectral (..., 3, nlm): [Qlm, Slm, Tlm]
        .Arg<ffi::Buffer<ffi::F64>>()    // cost (..., 1)
        .Ret<ffi::Buffer<ffi::F64>>()    // stacked spatial (..., 3, nphi): [Vr, Vt, Vp]
);
