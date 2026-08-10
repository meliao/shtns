/*
 * Copyright (c) 2010-2024 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 */

/** \file sht_gpu_local.cu
 * \brief GPU (CUDA) implementations of the "local" point/latitude evaluation functions:
 * \ref SHqst_to_point and \ref SHqst_to_lat. See vector_to_point_to_lat.md at the repo
 * root for a full description of the CPU algorithms this file ports.
 *
 * Unlike the main grid transforms (sht_gpu.cu / cuda_legendre.gen.cu), these kernels are
 * compiled directly by nvcc (not NVRTC): SHqst_to_point/SHqst_to_lat already take their
 * truncation degree/order (ltr/mtr) as runtime parameters on CPU, so there is no need for
 * compile-time-baked LMAX/MRES constants the way the grid kernels use them.
 *
 * The CPU functions evaluate the associated-Legendre recurrence on the fly using the
 * "plain" recurrence coefficients shtns->alm (sht_legendre.c). Those were never mirrored
 * to the GPU (the grid transforms use the separate Ishioka-recurrence coefficients
 * d_clm/d_xlm/d_x2lm instead), so this file's host launchers rely on a new device buffer
 * shtns->d_alm, uploaded once in cushtns_init_gpu (sht_gpu.cu).
 */

#include "sht_private.h"

// Defined (non-static) in sht_gpu.cu; reused here for VkFFT buffer-stride sizing.
extern int ncplx_align(int nphi);

/* ================= device-side Legendre recurrence, fused with accumulation ================= */

/// Port of sint_pow_n_ext (sht_legendre.c): sin(theta)^n = val * SHT_SCALE_FACTOR^nval.
__device__ __forceinline__ double dev_sint_pow_n_ext(double cost, int n, int *nval)
{
	double s2 = 1. - cost*cost;
	int ns2 = 0;
	int nv = 0;
	double val = 1.0;
	if (n & 1) val *= sqrt(s2);
	while (n >>= 1) {
		if (n & 1) {
			if (val < 1.0/SHT_SCALE_FACTOR) { nv--; val *= SHT_SCALE_FACTOR; }
			val *= s2; nv += ns2;
		}
		s2 *= s2; ns2 += ns2;
		if (s2 < 1.0/SHT_SCALE_FACTOR) { ns2--; s2 *= SHT_SCALE_FACTOR; }
	}
	while ((nv < 0) && (val > 1.0/SHT_SCALE_FACTOR)) { ++nv; val *= 1.0/SHT_SCALE_FACTOR; }
	*nval = nv;
	return val;
}

/// Port of the alm_im(shtns,im) offset macro (sht_legendre.c:36), operating on lmax/mres directly.
__device__ __host__ __forceinline__ long dev_alm_im_offset(int lmax, int mres, int im)
{
	return (long)im * (2*(lmax+1) - ((long)im - 1)*mres);
}

/// Port of the LiM(shtns,l,im) macro (shtns.h:110), operating on lmax/mres directly.
__device__ __host__ __forceinline__ long dev_LiM(int lmax, int mres, int l, int im)
{
	return (((long)im * (2*lmax + 2 - (im+1)*mres)) >> 1) + l;
}

/// m=0 special case (real-only): accumulates vr0 += yl[l]*Re(Ql[l]), vtt += dtyl[l]*Re(Sl[l]),
/// vpp -= dtyl[l]*Re(Tl[l]) for l=0..ltr. Shared verbatim between SHqst_to_point and
/// SHqst_to_lat, since both have an identical m=0 block on CPU
/// (sht_func.c:704-710 and sht_func.c:804-810/812-814 respectively).
__device__ void dev_legendre_deriv_m0(const double* alm, int mres, int ltr, double x, double sint,
                                       const double2* Ql, const double2* Sl, const double2* Tl,
                                       double* vr0, double* vtt, double* vpp)
{
	const double* al = alm;	// dev_alm_im_offset(lmax,mres,0) == 0
	double y0 = al[0];
	double dy0 = 0.0;
	const double st = sint;

	*vr0 += y0 * Ql[0].x;
	*vtt += dy0 * Sl[0].x;
	*vpp -= dy0 * Tl[0].x;
	if (ltr == 0) return;

	double y1 = al[1] * (x * y0);
	double dy1 = al[1] * (x*dy0 - st*y0);
	*vr0 += y1 * Ql[1].x;
	*vtt += dy1 * Sl[1].x;
	*vpp -= dy1 * Tl[1].x;
	if (ltr == 1) return;

	al += 2;
	for (int l = 2; l <= ltr; l++) {
		const double a = al[0], b = al[1];
		const double y_l  = b*(x*y1) + a*y0;
		const double dy_l = b*(x*dy1 - y1*st) + a*dy0;
		*vr0 += y_l * Ql[l].x;
		*vtt += dy_l * Sl[l].x;
		*vpp -= dy_l * Tl[l].x;
		y0 = y1; y1 = y_l;
		dy0 = dy1; dy1 = dy_l;
		al += 2;
	}
}

/// Complex accumulators for the m>0 case, shared between SHqst_to_point and SHqst_to_lat.
struct LegAccum5 { double2 qm, dsdt, dtdt, dsdp, dtdp; };

__device__ __forceinline__ void leg_accum5(LegAccum5* acc, double y, double dy,
                                            double2 Q, double2 S, double2 T)
{
	acc->qm.x   += y*Q.x;   acc->qm.y   += y*Q.y;
	acc->dsdt.x += dy*S.x;  acc->dsdt.y += dy*S.y;
	acc->dtdt.x += dy*T.x;  acc->dtdt.y += dy*T.y;
	acc->dsdp.x += y*S.x;   acc->dsdp.y += y*S.y;
	acc->dtdp.x += y*T.x;   acc->dtdp.y += y*T.y;
}

/// m>0 case: port of legendre_sphPlm_deriv_array (sht_legendre.c:236-307), fused with the
/// l-sum accumulation done separately on CPU (e.g. sht_func.c:722-728). Ql/Sl/Tl must
/// already be shifted so that Ql[l] is the (l,m) coefficient (i.e. Ql = Qlm + dev_LiM(lmax,mres,0,im)).
/// Unlike the CPU version (which unrolls the recurrence by 2 for performance and stores
/// yl[]/dtyl[] arrays), this walks one l at a time and accumulates immediately -- same
/// math, no per-thread array storage (which would blow local memory at large lmax).
__device__ void dev_legendre_deriv_m(const double* alm, int lmax, int mres, int ltr, int im,
                                      double x, double sint,
                                      const double2* Ql, const double2* Sl, const double2* Tl,
                                      LegAccum5* acc)
{
	const int m = im * mres;
	const double* al = alm + dev_alm_im_offset(lmax, mres, im);

	int ny = 0;
	double y0 = al[0] * dev_sint_pow_n_ext(x, m - 1, &ny);
	double dy0 = x * m * y0;
	const double st = sint * sint;

	double y1 = al[1] * (x * y0);
	double dy1 = al[1] * (x*dy0 - st*y0);
	al += 2;

	if (ny >= 0) leg_accum5(acc, y0, dy0, Ql[m], Sl[m], Tl[m]);
	if (ltr == m) return;

	if (ny >= 0) leg_accum5(acc, y1, dy1, Ql[m+1], Sl[m+1], Tl[m+1]);
	if (ltr == m+1) return;

	for (int l = m+2; l <= ltr; l++) {
		const double a = al[0], b = al[1];
		const double y_l  = b*(x*y1) + a*y0;
		const double dy_l = b*(x*dy1 - y1*st) + a*dy0;
		y0 = y1; y1 = y_l;
		dy0 = dy1; dy1 = dy_l;
		al += 2;

		if (ny < 0) {
			if (fabs(y1) > 1.0) {
				const double inv = 1.0/SHT_SCALE_FACTOR;
				y0 *= inv; y1 *= inv; dy0 *= inv; dy1 *= inv;
				ny++;
			}
			if (ny < 0) continue;	// still negligible at this l: no contribution
		}
		leg_accum5(acc, y1, dy1, Ql[l], Sl[l], Tl[l]);
	}
}

/* ================================= SHqst_to_point kernel ================================= */

/// One CUDA thread per evaluation point (grid-stride loop). Ports SHqst_to_point
/// (sht_func.c:694-740) verbatim: Qlm/Slm/Tlm are ONE shared field evaluated at `npts`
/// independent (cost,phi) points.
__global__ void shqst_point_kernel(const double* alm, int lmax, int mres, int ltr, int mtr,
                                    const double2* Qlm, const double2* Slm, const double2* Tlm,
                                    long field_stride,	// 0 = single field shared by all points; nlm = one field per point
                                    const double* cost, const double* phi,
                                    double* vr, double* vt, double* vp, long npts)
{
	for (long i = (long)blockIdx.x*blockDim.x + threadIdx.x; i < npts; i += (long)blockDim.x*gridDim.x) {
		const double2* Ql = Qlm + i*field_stride;
		const double2* Sl = Slm + i*field_stride;
		const double2* Tl = Tlm + i*field_stride;
		const double x = cost[i];
		const double ph = phi[i];
		const double sint = sqrt((1.0-x)*(1.0+x));

		double vr0 = 0.0, vtt = 0.0, vpp = 0.0, vrm = 0.0;
		dev_legendre_deriv_m0(alm, mres, ltr, x, sint, Ql, Sl, Tl, &vr0, &vtt, &vpp);

		for (int im = 1; im <= mtr; im++) {
			const int m = im * mres;
			const long off = dev_LiM(lmax, mres, 0, im);
			LegAccum5 acc = {};
			dev_legendre_deriv_m(alm, lmax, mres, ltr, im, x, sint, Ql+off, Sl+off, Tl+off, &acc);

			const double eimp_re = 2.0*cos(m*ph), eimp_im = 2.0*sin(m*ph);
			// imeimp = I*m*eimp = -m*eimp_im + i*m*eimp_re
			const double imeimp_re = -m*eimp_im, imeimp_im = m*eimp_re;

			vrm += acc.qm.x*eimp_re - acc.qm.y*eimp_im;
			vtt += (acc.dtdp.x*imeimp_re - acc.dtdp.y*imeimp_im) + (acc.dsdt.x*eimp_re - acc.dsdt.y*eimp_im);
			vpp += (acc.dsdp.x*imeimp_re - acc.dsdp.y*imeimp_im) - (acc.dtdt.x*eimp_re - acc.dtdt.y*eimp_im);
		}
		vr0 += vrm * sint;

		vr[i] = vr0;
		vt[i] = vtt;
		vp[i] = vpp;
	}
}

/// Qlm/Slm/Tlm hold either a single shared field (n_fields==1, broadcast to every point)
/// or one field per point (n_fields==npts, zipped with cost/phi) -- the latter is what a
/// vmap over (Qlm,Slm,Tlm,cost,phi) with vmap_method="broadcast_all" produces, since
/// broadcast_all materializes a batch dimension even on arguments that are logically
/// constant across the batch.
extern "C"
void cu_SHqst_to_point(shtns_cfg shtns, const cplx *Qlm, const cplx *Slm, const cplx *Tlm,
                       long n_fields, const double *cost, const double *phi,
                       double *vr, double *vt, double *vp, long npts, int ltr, int mtr)
{
	if (npts <= 0) return;
	if (ltr > shtns->lmax) ltr = shtns->lmax;
	if (mtr > shtns->mmax) mtr = shtns->mmax;
	if (mtr*shtns->mres > ltr) mtr = ltr/shtns->mres;

	const long field_stride = (n_fields == 1) ? 0 : (long)shtns->nlm;

	const int block = 128;
	long grid = (npts + block - 1) / block;
	if (grid > 65535) grid = 65535;		// grid-stride loop covers any remainder

	shqst_point_kernel<<<(unsigned)grid, block, 0, shtns->comp_stream>>>(
		shtns->d_alm, shtns->lmax, shtns->mres, ltr, mtr,
		(const double2*) Qlm, (const double2*) Slm, (const double2*) Tlm, field_stride,
		cost, phi, vr, vt, vp, npts);
}

/* ================================= SHqst_to_lat kernels ================================= */

/// One CUDA thread per batch element (grid-stride loop over n_other). Ports the
/// Legendre+l-sum stage of SHqst_to_lat (sht_func.c:771-828) for THAT thread's own
/// Qlm/Slm/Tlm/cost slice (matching the existing SHqst_to_lat_jax_cpu batching convention,
/// shtns_jax.cpp:402-431, where spec and cost share one leading batch dim). Writes complex
/// per-m Fourier coefficients into a scratch buffer for the caller to expand via VkFFT.
/// Unlike CPU's SHqst_to_lat, this touches no cross-call cached state (no ylm_lat/ct_lat
/// equivalent) -- each thread computes fresh into its own scratch slice.
__global__ void shqst_lat_fourier_kernel(const double* alm, int lmax, int mres, int ltr, int mtr,
                                          const double2* Qlm, const double2* Slm, const double2* Tlm,
                                          const double* cost, long nlm, long n_other, int nphc_stride,
                                          double2* vrc, double2* vtc, double2* vpc)
{
	for (long n = (long)blockIdx.x*blockDim.x + threadIdx.x; n < n_other; n += (long)blockDim.x*gridDim.x) {
		const double2* Ql = Qlm + n*nlm;
		const double2* Sl = Slm + n*nlm;
		const double2* Tl = Tlm + n*nlm;
		double2* vrc_n = vrc + n*(long)nphc_stride;
		double2* vtc_n = vtc + n*(long)nphc_stride;
		double2* vpc_n = vpc + n*(long)nphc_stride;

		const double x = cost[n];
		const double sint = sqrt((1.0-x)*(1.0+x));

		double vrr = 0.0, vtt = 0.0, vpp = 0.0;
		dev_legendre_deriv_m0(alm, mres, ltr, x, sint, Ql, Sl, Tl, &vrr, &vtt, &vpp);
		vrc_n[0] = make_double2(vrr, 0.0);
		vtc_n[0] = make_double2(vtt, 0.0);	// Vt = dS/dt
		vpc_n[0] = make_double2(vpp, 0.0);	// Vp = -dT/dt (sign already folded in by dev_legendre_deriv_m0)

		for (int im = 1; im <= mtr; im++) {
			const int m = im * mres;
			const long off = dev_LiM(lmax, mres, 0, im);
			LegAccum5 acc = {};
			dev_legendre_deriv_m(alm, lmax, mres, ltr, im, x, sint, Ql+off, Sl+off, Tl+off, &acc);

			// vrc[m] = qm * sint  (undo the 1/sint pre-division baked into yl for m>0)
			vrc_n[m] = make_double2(acc.qm.x*sint, acc.qm.y*sint);
			// vtc[m] = I*m*dtdp + dsdt   (I*m*(a+bi) = -m*b + i*m*a)
			vtc_n[m] = make_double2(acc.dsdt.x - m*acc.dtdp.y, acc.dsdt.y + m*acc.dtdp.x);
			// vpc[m] = I*m*dsdp - dtdt
			vpc_n[m] = make_double2(-m*acc.dsdp.y - acc.dtdt.x, m*acc.dsdp.x - acc.dtdt.y);
		}
	}
}

extern "C"
void cu_SHqst_to_lat(shtns_cfg shtns, const cplx *Qlm, const cplx *Slm, const cplx *Tlm,
                     const double *cost, double *vr, double *vt, double *vp,
                     long n_other, int nphi, int ltr, int mtr)
{
	if (n_other <= 0 || nphi <= 0) return;
	if (ltr > shtns->lmax) ltr = shtns->lmax;
	if (mtr > shtns->mmax) mtr = shtns->mmax;
	if (mtr*shtns->mres > ltr) mtr = ltr/shtns->mres;
	if (mtr*2*shtns->mres >= nphi) mtr = (nphi-1)/(2*shtns->mres);

	const long nlm = shtns->nlm;
	const int nphc_stride = ncplx_align(nphi);		// complex buffer stride expected by VkFFT below

	// The plan's numberBatches is baked in at build time, so it must be rebuilt whenever
	// EITHER nphi or the batch size n_other changes -- not just nphi.
	if (nphi != shtns->nphi_lat_gpu || n_other != shtns->n_other_lat_gpu) {
		if (shtns->nphi_lat_gpu) deleteVkFFT(&shtns->vkfft_plan_lat);
		memset(&shtns->vkfft_plan_lat, 0, sizeof(VkFFTApplication));	// deleteVkFFT doesn't fully reset the struct

		VkFFTConfiguration config = {};
		int device_id = -1;
		cudaGetDevice(&device_id);
		CUdevice vkfft_device_struct;
		cuDeviceGet(&vkfft_device_struct, device_id);

		config.FFTdim = 1;
		config.size[0] = nphi;
		config.performR2C = 1;
		config.isInputFormatted = 1;
		config.inverseReturnToInputBuffer = 1;
		config.inputBufferStride[0] = nphi;
		config.bufferStride[0] = nphc_stride;
		config.numberBatches = n_other;
		config.doublePrecision = 1;
		config.device = &vkfft_device_struct;
		config.stream = &shtns->comp_stream;
		config.num_streams = 1;

		VkFFTResult vk_res = initializeVkFFT(&shtns->vkfft_plan_lat, config);
		if (vk_res != VKFFT_SUCCESS) {
			printf("[cu_SHqst_to_lat] vkFFT init failed (error %d)\n", vk_res);
			shtns->nphi_lat_gpu = 0;
			shtns->n_other_lat_gpu = 0;
			return;
		}
		shtns->nphi_lat_gpu = nphi;
		shtns->n_other_lat_gpu = n_other;
	}

	double2 *vrc = 0, *vtc = 0, *vpc = 0;
	const size_t bytes = (size_t) n_other * nphc_stride * sizeof(double2);
	cudaMallocAsync((void**)&vrc, bytes, shtns->comp_stream);
	cudaMallocAsync((void**)&vtc, bytes, shtns->comp_stream);
	cudaMallocAsync((void**)&vpc, bytes, shtns->comp_stream);
	cudaMemsetAsync(vrc, 0, bytes, shtns->comp_stream);
	cudaMemsetAsync(vtc, 0, bytes, shtns->comp_stream);
	cudaMemsetAsync(vpc, 0, bytes, shtns->comp_stream);

	const int block = 128;
	long grid = (n_other + block - 1) / block;
	if (grid > 65535) grid = 65535;

	shqst_lat_fourier_kernel<<<(unsigned)grid, block, 0, shtns->comp_stream>>>(
		shtns->d_alm, shtns->lmax, shtns->mres, ltr, mtr,
		(const double2*) Qlm, (const double2*) Slm, (const double2*) Tlm,
		cost, nlm, n_other, nphc_stride, vrc, vtc, vpc);

	double2* cbufs[3]  = { vrc, vtc, vpc };
	double*  rbufs[3]  = { vr,  vt,  vp  };
	for (int f = 0; f < 3; f++) {
		void* cbuf = (void*) cbufs[f];
		void* rbuf = (void*) rbufs[f];
		VkFFTLaunchParams lp = {};
		lp.buffer = &cbuf;
		lp.inputBuffer = &rbuf;
		VkFFTAppend(&shtns->vkfft_plan_lat, 1, &lp);
	}

	cudaFreeAsync(vrc, shtns->comp_stream);
	cudaFreeAsync(vtc, shtns->comp_stream);
	cudaFreeAsync(vpc, shtns->comp_stream);
}
