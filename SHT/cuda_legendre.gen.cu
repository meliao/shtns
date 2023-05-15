/*
 * Copyright (c) 2010-2023 Centre National de la Recherche Scientifique.
 * written by Nathanael Schaeffer (CNRS, ISTerre, Grenoble, France).
 * 
 * nathanael.schaeffer@univ-grenoble-alpes.fr
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

// CUDA kernels for SHTns that require run-time compiling using nvrtc
// at compilation, the following must be defined
// WARPSZE : the warpsize (32 for nvidia, 64 for amd)
// LMAX : max degree of spherical harmonics. used for address calculation
// MRES : order multiplicity (often 1). used for address calculation.
// HI_LLIM : 0 for small values of lmax, 1 for large (rescaling needed)
// M0_ONLY : 0 for all m's, 1 for axisymmetric
// ROBERT_FORM : 0 for regular transform, 1 for the Robert form transform (applies to S=1 only)
// BLKSZE_S : blocksize for synthesis (leg_m_kernel)
// BLKSZE_A : blocksize for analysis (ileg_m_kernel)
// BLKSZE_SH2ISH : 0 for separate ishioka pre-computation, or blocksize for in-kernel pre-computation -- for scalar (S=0) synthesis only.
// NF_S : number of fields treated together for synthesis
// NF_A : number of fields treated together for analysis
// LSPAN_A : number of SH degrees treated together (analysis)
// NW_S : number of spatial points per thread (synthesis)
// MPOS_SCALE : scale factor for analysis (used once)
// NLAT_2 : half the number of latidunal points, should be equal to the nlat_2 kernel parameter

// TODO: some parameters can be made compile-time constants!
// 		nlat_2, nphi, m_inc, mpos_scale

// define our own suffle macros, to accomodate cuda<9 and cuda>=9
#if __CUDACC_VER_MAJOR__ < 9
	#define shfl_xor(...) __shfl_xor(__VA_ARGS__)
	#define shfl_down(...) __shfl_down(__VA_ARGS__)
	#define shfl(...) __shfl(__VA_ARGS__)
	#define _any(p) __any(p)
	#define _all(p) __all(p)
	#define _ballot(p) __ballot(p)
	#define _syncwarp 0
	#define _syncwarp_fence __threadfence_block()
#else
	#define shfl_xor(...) __shfl_xor_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl_down(...) __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl(...) __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
	#define _any(p) __any_sync(0xFFFFFFFF, p)
	#define _all(p) __all_sync(0xFFFFFFFF, p)
	#define _ballot(p) __ballot_sync(0xFFFFFFFF, p)
	#define _syncwarp __syncwarp()
	#define _syncwarp_fence __syncwarp()
#endif

#ifdef __gfx90a__
	#define atomicAdd_sht unsafeAtomicAdd
#else	/* NOT __gfx90a__ */
#if WARPSZE == 64
	// AMD HIP
	#define __forceinline__ inline
#endif

#if (__CUDACC_VER_MAJOR__ < 8) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
__device__ __forceinline__ void atomicAdd_sht(double* address, double val)
{
	unsigned long long* address_as_ull = (unsigned long long*)address;
	unsigned long long old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
}
__device__ __forceinline__ void atomicAdd_sht(float* address, float val) {
	atomicAdd(adress,val);
}
#else
	#define atomicAdd_sht atomicAdd
#endif
#endif

__device__ __forceinline__ bool polar_skip_sint(double sint, int llim, int m)
{
	// polar optimization (see Reinecke 2013, section 3.3)
	int x = sint * llim;
	#if LMAX > 10350
		return (m - max(80, llim>>7) > x);
	#else
		return (m - 80 > x);
	#endif
}
__device__ __forceinline__ bool polar_skip_sint(float sint, int llim, int m) {
	return false;
}

__device__ __forceinline__ bool polar_skip_sint2(double sint2, int llim, int m)
{
	// polar optimization (see Reinecke 2013, section 3.3) -- squared
	int mm = m - ((LMAX > 10350) ? max(80, llim>>7) : 80);
	return (mm>0) && (mm*mm > (int) (sint2*(llim*llim)));
}
__device__ __forceinline__ bool polar_skip_sint2(float sint2, int llim, int m) {
	return false;
}

#if BLKSZE_SH2ISH > 0
__device__ real qish(const real* __restrict__ xlm, const real* __restrict__ ql, const int llim_m, int ll)
{
	real q = ql[ll];
	const int x_ofs = 3*(ll >> 2) + (ll&2);
	q *= xlm[x_ofs];
	if (((ll&2)==0) && (ll+2 <2*llim_m))	// l-m even
		q += ql[ll+4] * xlm[x_ofs + 1];		// contribution of l+2
	return q;
}
#endif

/// requirements : blockSize must be 1 in the y- and z-direction and BLKSZE_S in the x-direction.
/// llim MUST BE <= 1800, unless HI_LLIM=1
template<int S> __global__
#if NLAT_2<=64 && defined(__gfx90a__)
__launch_bounds__(64, 1)	// leads to better performance for small transforms on MI250
#endif
void leg_m_kernel(
	const real* __restrict__ al, const real* __restrict__ ct, const real* __restrict__ ql, real *q,
	const int llim, const int nlat_2, const int nphi, const int m_inc,
	const int ql_dist, const int q_dist
#if BLKSZE_SH2ISH > 0
	,const real* __restrict__ xlm
#endif
)

{
	const int BLOCKSIZE = (BLKSZE_SH2ISH>0 && S==0) ? BLKSZE_SH2ISH : BLKSZE_S;
	const int NW=NW_S;
	const int NFIELDS=NF_S;
	const int im = (M0_ONLY) ? 0 : blockIdx.z;
	const int j = threadIdx.x;
	const int b = blockIdx.y;		// position in batch
	//const int m_inc = 2*nlat_2;
	const int k_inc = 1;

	const int LSPAN = (WARPSZE==32 && BLOCKSIZE >= 2*WARPSZE) ? BLOCKSIZE/2 : WARPSZE;		// always WARPSZE for amd
	static_assert(LSPAN <= BLOCKSIZE, "LSPAN must not exceed BLOCKSIZE");
	static_assert(LSPAN % 4 == 0, "LSPAN must be a multiple of 4");
	__shared__ real ak[LSPAN];
	__shared__ real qk[NFIELDS][(M0_ONLY) ? LSPAN : LSPAN*2];

	static_assert( (!HI_LLIM) || ( NW==1 || (NW&1)==0 ), "high llim works with NW=1 or NW even" );

	#define COST_CACHE		// optional: store cos(theta) into shared memory to reduce register pressure
	real y0[NW];
	real y1[NW];
	real ct2[NW];
	#ifndef COST_CACHE
	real cost_[NW];
	#define COST(i,j) cost_[i]
	#else
	__shared__ real cost_[NW][BLOCKSIZE];
	#define COST(i,j) cost_[i][j]
	#endif
	#pragma unroll
	for (int i=0; i<NW; i++) {
		const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
		ct2[i] = (it < nlat_2) ? ct[it] : 0.0;
	}

	if (im==0) {
		if ((LSPAN==BLOCKSIZE || j<LSPAN) && (j<=llim)) {
			ak[j] = al[j+2];
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) 	{
				#if BLKSZE_SH2ISH > 0
					if (S==0)	qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*j);
					else
				#endif
						qk[f][j] = ql[j + (b*NFIELDS+f)*ql_dist];		// keep only real part
			}
		}

		#pragma unroll
		for (int i=0; i<NW; i++) {	COST(i,j) = ct2[i];		ct2[i] *= ct2[i];	}	// cos(theta)^2

		real re[NFIELDS][NW], ro[NFIELDS][NW];
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++) {
				re[f][i] = 0.0;
				ro[f][i] = 0.0;
			}
		}
		int l = 0;
		#pragma unroll
		for (int i=0; i<NW; i++) y0[i] = (S==1 && !ROBERT_FORM) ? rsqrt(1.0 - ct2[i]) : 1.0;    // for vectors, divide by sin(theta) -- except in Robert form
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i] + al[0])*y0[i];

		al+=2;
		if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }

		while (l<=llim - LSPAN) {	// compute even and odd parts
			for (int k = 0; k<LSPAN; k+=4) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						re[f][i] += y0[i] * qk[f][k];		// real
						ro[f][i] += y0[i] * qk[f][k+1];		// real
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y0[i] += (ak[k+1]*ct2[i] + ak[k]) * y1[i];
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						re[f][i] += y1[i] * qk[f][k+2];		// real
						ro[f][i] += y1[i] * qk[f][k+3];		// real
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y1[i] += (ak[k+3]*ct2[i] + ak[k+2]) * y0[i];
			}
			al += LSPAN;
			l  += LSPAN;
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if ((l+j <= llim) && (BLOCKSIZE==LSPAN || j<LSPAN)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#if BLKSZE_SH2ISH > 0
					if (S==0)	qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*(l+j));
					else
					#endif
						qk[f][j] = ql[l+j + (b*NFIELDS+f)*ql_dist];
				}
				ak[j] = al[j];
			}
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
		int k=0;
		while (l<llim) {	// compute even and odd parts
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					re[f][i] += y0[i] * qk[f][k];	// real
					ro[f][i] += y0[i] * qk[f][k+1];	// real
				}
			}
			#pragma unroll
			for (int i=0; i<NW; i++) {
				real tmp = (ak[k+1]*ct2[i] + ak[k]) * y1[i] + y0[i];
				y0[i] = y1[i];
				y1[i] = tmp;
			}
			l+=2;	k+=2;
		}
		if (l==llim) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					re[f][i] += y0[i] * qk[f][k];		// real
				}
			}
		}

		#pragma unroll
		for (int i=0; i<NW; i++) {
			const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
			if (it < nlat_2) {
				// store mangled for complex fft
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					q[it*k_inc              + (b*NFIELDS+f)*q_dist] = re[f][i]+ro[f][i]*COST(i,j);
					q[(nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = re[f][i]-ro[f][i]*COST(i,j);
				}
			}
		}
	}
#if M0_ONLY==0
	else { 	// m>0
		real rer[NFIELDS][NW], ror[NFIELDS][NW], rei[NFIELDS][NW], roi[NFIELDS][NW];
		const int m = im*MRES;
		int l = (im*(2*(LMAX+1)-MRES-m))>>1;
		#if BLKSZE_SH2ISH > 0
			if (S==0)	xlm += 3*im*(2*(LMAX+4)+MRES-m)/4;
		#endif
		al += l+m;
		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1

		if ((LSPAN==BLOCKSIZE || j<LSPAN) && (m+j<=llim)) 	ak[j] = al[j+2];
			if ((m+j/2 <= llim) && (2*LSPAN>=BLOCKSIZE || j<2*LSPAN)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#if BLKSZE_SH2ISH > 0
					if (S==0)		qk[f][j] = qish(xlm, ql+2*m+(b*NFIELDS+f)*ql_dist, llim-m, j);
					else
					#endif
						qk[f][j] = ql[2*m+j + (b*NFIELDS+f)*ql_dist];
				}
			}
			if ((BLOCKSIZE < 2*LSPAN) && (m+j/2+BLOCKSIZE/2 <= llim) && (2*BLOCKSIZE<=2*LSPAN || j+BLOCKSIZE < 2*LSPAN)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#if BLKSZE_SH2ISH > 0
					if (S==0)	qk[f][j+BLOCKSIZE] = qish(xlm, ql+2*m+(b*NFIELDS+f)*ql_dist, llim-m, j+BLOCKSIZE);
					else
					#endif
						qk[f][j+BLOCKSIZE] = ql[2*m+j+BLOCKSIZE + (b*NFIELDS+f)*ql_dist];	
				}
			}

		#pragma unroll
		for (int i=0; i<NW; i++) {	COST(i,j) = ct2[i];		ct2[i] *= ct2[i];	}	// cos(theta)^2
		#pragma unroll
		for (int i=0; i<NW; i++) 	y1[i] = 1.0 - ct2[i];		// y1 = sin(theta)^2
		#pragma unroll
		for (int i=0; i<NW; i++) 	y0[i] = 1.0;

		#pragma unroll
		for (int i=0; i<NW; i++) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				ror[f][i] = 0.0;		roi[f][i] = 0.0;
				rer[f][i] = 0.0;		rei[f][i] = 0.0;
			}
		}

		bool skip_block = false;
		if (NLAT_2 > BLOCKSIZE*NW) {	// polar optimization
			if (j == BLOCKSIZE-1)	skip_block = polar_skip_sint2(y1[NW-1], llim, m);
			#if WARPSZE==32
			if (BLOCKSIZE == WARPSZE) skip_block = _any(skip_block);	// get largest value in block/warp
			#else
			if (BLOCKSIZE == WARPSZE) skip_block = shfl(skip_block,WARPSZE-1);	// get largest value in block/warp
			#endif
			else {
				__shared__ volatile int xx;
				if (j == BLOCKSIZE-1) xx = skip_block;	// one thread writes its value (the largest one)
				__syncthreads();
				skip_block = xx;	// everyone reads that value
			}
		} else if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		// at this point, block is in sync (consistent view of shared memory).
	if (!skip_block) {
		#if HI_LLIM==1
		int ny = 0;		// only used for HI_LLIM
		#else
		constexpr int ny = 0;
		#endif
		{	// compute sin(theta)^(m-S)
			l = (S==1 && ROBERT_FORM) ? m : m-S;		// multiply vectors by sin(theta) with robert_form
			if (l&1) {	// square-root needed
				#pragma unroll
				for (int i=0; i<NW; i++)	y0[i] = sqrt(y1[i]);
			}
			l >>= 1;
			#if HI_LLIM==1
			int nsint = 0;
			#endif
			do {
				if (l&1) {
					#pragma unroll
					for (int i=0; i<NW; i++) y0[i] *= y1[i];
					#if HI_LLIM==1
						ny += nsint;
						if (y0[NW-1] < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
							#pragma unroll
							for (int i=0; i<NW; i++) y0[i] *= SHT_SCALE_FACTOR;
							ny--;
						}
					#endif
				}
				#pragma unroll
				for (int i=0; i<NW; i++) y1[i] *= y1[i];
				#if HI_LLIM==1
					nsint += nsint;
					if (y1[NW-1] < 1.0/SHT_SCALE_FACTOR) {
						nsint--;
						#pragma unroll
						for (int i=0; i<NW; i++) y1[i] *= SHT_SCALE_FACTOR;
					}
				#endif
			} while(l >>= 1);
		}

		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i] + al[0])*y0[i];

		l=m;		al+=2;
		while (l<=llim - LSPAN) {	// compute even and odd parts
			for (int k = 0; k<LSPAN; k+=4) {
				real tmp[NW];
				#pragma unroll
				for (int i=0; i<NW; i++)	tmp[i] = ak[k+1]*ct2[i] + ak[k];
				if ((!HI_LLIM) || (ny==0)) {
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						#pragma unroll
						for (int i=0; i<NW; i++) {
							rer[f][i] += y0[i] * qk[f][2*k];	// real
							rei[f][i] += y0[i] * qk[f][2*k+1];	// imag
							ror[f][i] += y0[i] * qk[f][2*k+2];	// real
							roi[f][i] += y0[i] * qk[f][2*k+3];	// imag
						}
					}
				}
				#pragma unroll
				for (int i=0; i<NW; i++)	y0[i] += tmp[i] * y1[i];
				#pragma unroll
				for (int i=0; i<NW; i++)	tmp[i] = ak[k+3]*ct2[i] + ak[k+2];
				if ((!HI_LLIM) || (ny==0)) {
					#pragma unroll
					for (int f=0; f<NFIELDS; f++) {
						#pragma unroll
						for (int i=0; i<NW; i++) {
							rer[f][i] += y1[i] * qk[f][2*k+4];	// real
							rei[f][i] += y1[i] * qk[f][2*k+5];	// imag
							ror[f][i] += y1[i] * qk[f][2*k+6];	// real
							roi[f][i] += y1[i] * qk[f][2*k+7];	// imag
						}
					}
				}
				#if HI_LLIM==1
				else if (fabs(y0[NW-1]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
				{	// rescale when value is significant
					++ny;
					#pragma unroll
					for (int i=0; i<NW; i++) {
						y0[i] *= 1.0/SHT_SCALE_FACTOR;
						y1[i] *= 1.0/SHT_SCALE_FACTOR;
					}
				}
				#endif
				#pragma unroll
				for (int i=0; i<NW; i++)	y1[i] += tmp[i] * y0[i];
			}
			al += LSPAN;
			l  += LSPAN;
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if ((l+j/2 <= llim) && (BLOCKSIZE<=2*LSPAN || j<2*LSPAN)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#if BLKSZE_SH2ISH > 0
					if (S==0)	qk[f][j] = qish(xlm, ql+2*m+(b*NFIELDS+f)*ql_dist, llim-m, 2*(l-m)+j);
					else
					#endif
						qk[f][j] = ql[2*l+j + (b*NFIELDS+f)*ql_dist];
				}
			}
			if ((BLOCKSIZE < 2*LSPAN) && (l+j/2+BLOCKSIZE/2 <= llim) && (2*BLOCKSIZE<=2*LSPAN || j+BLOCKSIZE < 2*LSPAN)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#if BLKSZE_SH2ISH > 0
					if (S==0)	qk[f][j+BLOCKSIZE] = qish(xlm, ql+2*m+(b*NFIELDS+f)*ql_dist, llim-m, 2*(l-m)+j+BLOCKSIZE);
					else
					#endif
						qk[f][BLOCKSIZE+j] = ql[2*l+BLOCKSIZE+j + (b*NFIELDS+f)*ql_dist];
				}
			}
			if ((l+j <= llim) && (LSPAN==BLOCKSIZE || j<LSPAN))	 ak[j] = al[j];
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
		int k=0;
		while (l<llim) {	// compute even and odd parts
			real tmp[NW];
			#pragma unroll
			for (int i=0; i<NW; i++)	tmp[i] = ak[k+1]*ct2[i] + ak[k];
			if ((!HI_LLIM) || (ny==0)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						rer[f][i] += y0[i] * qk[f][2*k];	// real
						rei[f][i] += y0[i] * qk[f][2*k+1];	// imag
						ror[f][i] += y0[i] * qk[f][2*k+2];	// real
						roi[f][i] += y0[i] * qk[f][2*k+3];	// imag
					}
				}
			}
			#if HI_LLIM==1
			else if (fabs(y1[NW-1]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
			{	// rescale when value is significant
				++ny;
				#pragma unroll
				for (int i=0; i<NW; i++) {
					y0[i] *= 1.0/SHT_SCALE_FACTOR;
					y1[i] *= 1.0/SHT_SCALE_FACTOR;
				}
			}
			#endif
			#pragma unroll
			for (int i=0; i<NW; i++) tmp[i] = tmp[i] * y1[i] + y0[i];
			#pragma unroll
			for (int i=0; i<NW; i++) y0[i] = y1[i];
			l+=2;	k+=2;
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] = tmp[i];
		}
		if (l==llim) {
			if ((!HI_LLIM) || (ny==0)) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					#pragma unroll
					for (int i=0; i<NW; i++) {
						rer[f][i] += y0[i] * qk[f][2*k];	// real
						rei[f][i] += y0[i] * qk[f][2*k+1];	// imag
					}
				}
			}
		}

		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++) {
				y0[i]     = rer[f][i]+ror[f][i]*COST(i,j);	// recycle y0 as temporary value
				rer[f][i] = rer[f][i]-ror[f][i]*COST(i,j);
				ror[f][i] = rei[f][i]-roi[f][i]*COST(i,j);
				rei[f][i] = rei[f][i]+roi[f][i]*COST(i,j);
				roi[f][i] = y0[i];
			}
		}

		/// store mangled for complex fft
		if ((!HI_LLIM) || (NW==1)) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				#pragma unroll
				for (int i=0; i<NW; i++) {
					ror[f][i] = shfl_xor(ror[f][i], 1);
					rei[f][i] = shfl_xor(rei[f][i], 1);
				}
			}
		}
	}

		#pragma unroll
		for (int i=0; i<NW; i++) {
			const real sgn = (HI_LLIM && NW>1) ? (i^1)-i : (j^1)-j; 	//(it^1) - it;	// 1 - 2*(j&1);		// 1 for even j, -1 for odd j.
			const int it = BLOCKSIZE*NW * blockIdx.x + ((HI_LLIM) ? NW*j+i : j+i*BLOCKSIZE);
			const int i2 = (HI_LLIM && NW>1) ? i^1 : i;
			if (it < nlat_2) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					q[im*m_inc        + it*k_inc              + (b*NFIELDS+f)*q_dist] = roi[f][i] - rei[f][i2]*sgn;
					q[(nphi-im)*m_inc + it*k_inc              + (b*NFIELDS+f)*q_dist] = roi[f][i] + rei[f][i2]*sgn;
					q[im*m_inc        + (nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = rer[f][i] + ror[f][i2]*sgn;
					q[(nphi-im)*m_inc + (nlat_2*2-1-it)*k_inc + (b*NFIELDS+f)*q_dist] = rer[f][i] - ror[f][i2]*sgn;
				}
			}
		}
	}
#endif
}


#if LMAX >= 1000
	#undef HI_LLIM
	#define HI_LLIM 1
#endif

template<int S> __global__
#ifdef __gfx90a__
__launch_bounds__(64,1)
#endif
void ileg_m_kernel(const real* __restrict__ al, const real* __restrict__ ct, const real* __restrict__ q, real *ql, const int llim, 
	const int nlat_2, const int nphi, const int m_inc, const int q_dist, const int ql_dist)
{
	const int BLOCKSIZE=BLKSZE_A;
	const int NFIELDS=NF_A;
	const int LSPAN=LSPAN_A;
	const int it = BLOCKSIZE * blockIdx.x + threadIdx.x;
	const int j = threadIdx.x;
	const int im = (M0_ONLY) ? 0 : blockIdx.z;
	const int b = blockIdx.y;
	//const int m_inc = 2*nlat_2;
	const int f0 = (NFIELDS==1) ? 0 : j / (BLOCKSIZE/NFIELDS);			// assign each thread a field f0

	static_assert((BLOCKSIZE % (((M0_ONLY)?1:2)*LSPAN*NFIELDS)) == 0, "BLOCKSIZE must be a multiple of 2*LSPAN*NFIELDS");
	static_assert( ((WARPSZE >= BLOCKSIZE/LSPAN) ? (WARPSZE % (BLOCKSIZE/LSPAN)) : ((BLOCKSIZE/LSPAN) % WARPSZE)) == 0, "WARPSZE and BLOCKSIZE/LSPAN must be multiples");
	static_assert((LSPAN % 4) == 0, "LSPAN must be a multiple of 4");

	const int padding = 2;		// padding = 0 is very bad for performance (shared-memory bank conflicts).
	const int l_inc = BLOCKSIZE+padding;
	__shared__ real ak[LSPAN+2];	// cache
	const int NROWS = M0_ONLY ? ( (LSPAN>4*NFIELDS) ? LSPAN/2 : 2*NFIELDS ) : ( (LSPAN>8*NFIELDS) ? LSPAN/2 : 4*NFIELDS );
	__shared__ real yl[NROWS*l_inc - padding];		// yl is also used for even/odd computation.

	real cost = (it < nlat_2) ? ct[it] : 0.0;
	real y0, y1;

	if (im == 0) {
		const int NW = NFIELDS*LSPAN;
		// re-assign each thread an l (transposed view)
		const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);
		real my_reo[NW];			// in registers

		q += b*NFIELDS*q_dist;
		if (j < LSPAN+2) ak[j] = al[j];

		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			real x0 = (it < nlat_2) ? q[it              + f*q_dist] : 0.0;	// north
			real x1 = (it < nlat_2) ? q[nlat_2*2-1 - it + f*q_dist] : 0.0;	// south
			yl[f*2*l_inc +j]     = x0+x1;			// even
			yl[(f*2+1)*l_inc +j] = (x0-x1)*cost;	// odd
		}
		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }

		y0 = (it < nlat_2) ? ct[it + nlat_2] : 0.0;		// weights are stored just after ct.
		cost *= cost;	// ct2
		
		// transpose reo to my_reo
		#pragma unroll
		for (int k=0; k<NW; k++) {
			int it = j % (BLOCKSIZE/NW) + k*(BLOCKSIZE/NW);
			my_reo[k] = yl[(2*f0  + (ll&1))*l_inc + it];
		}
		if (S==1) y0 *= (ROBERT_FORM) ? 1.0/(1.0-cost) : rsqrt(1.0 - cost);
		y1 = (ak[1]*cost + ak[0]) * y0;
		if (WARPSZE < LSPAN+2  &&  j<LSPAN+2-WARPSZE)	ak[WARPSZE+j] = al[WARPSZE+j];		// sometimes a bit more than a warp is needed

		al+=2;
		int l = 0;
		do {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
				#pragma unroll
				for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
					real c0 = ak[2*k+3]*cost + ak[2*k+2];
					real c1 = ak[2*k+5]*cost + ak[2*k+4];
					yl[k*l_inc +j]     = y0;		// l and l+1
					yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
					y0 += c0 * y1;
					y1 += c1 * y0;
				}
			// now re-assign each thread an l (transpose)
			const int itl = (ll >> 1)*l_inc + j % (BLOCKSIZE/NW);

			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }

			const int NACC = 4;		// number of independent accumulators per NFIELD. 4 is good for V100
			real qll[NACC];		// accumulators
			#pragma unroll
			for (int a=0; a<NACC; a++) 	qll[a] = my_reo[a] * yl[itl + a*(BLOCKSIZE/NW)];	// first element of sum
			const int ql_ofs = (l+ll) + (b*NFIELDS+f0)*ql_dist;		// compute destination ofset in parallel with reduce!
			#pragma unroll
			for (int k=NACC; k<NW; k+=NACC) {
				#pragma unroll
				for (int a=0; a<NACC; a++) 	qll[a] += my_reo[a+k] * yl[itl + (k+a)*(BLOCKSIZE/NW)];
			}

			al += LSPAN;
			if (j<LSPAN) ak[j+2] = al[j];

			if (NACC > 1) {		// reduce the NACC independent accumulators
				#pragma unroll
				for (int a=0; a<NACC; a+=2) {
					qll[a] += qll[a+1];
				}
				for (int a=2; a<NACC; a+=2) {
					qll[0] += qll[a];
				}				
			}

			static_assert(BLOCKSIZE/NW <= WARPSZE, "Block size must not exceed LSPAN*NFIELDS*WARPSZE");
			// reduce_add within same l is in same warp too:
				#pragma unroll
				for (int ofs = BLOCKSIZE/(NW*2); ofs > 0; ofs>>=1) {
					qll[0] += shfl_down(qll[0], ofs, BLOCKSIZE/NW);
				}
				if ( ((j % (BLOCKSIZE/NW)) == 0) && ((l+ll)<=llim) ) {	// write result
					#if NLAT_2 <= BLKSZE_A
						// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
						ql[ql_ofs] = qll[0];
					#else
						atomicAdd_sht(ql+ql_ofs, qll[0]);		// VERY slow atomic add on Kepler.
					#endif
				}

			l+=LSPAN;
		} while (l <= llim);
	}
#if M0_ONLY==0
	else {	// im > 0
		const int NW = NFIELDS*LSPAN*2;
		// re-assign each thread an l (transposed view)
		const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);		// actualy ll = 2*l + (imag ? 1 : 0)
		real my_reo[NW];			// in registers
		const int m = im*MRES;
		y0 = cost * cost;			// cos(theta)^2
		int l = (im*(2*(LMAX+1)-MRES-m))>>1;
		y1 = 1.0 - y0;		// sin(theta)^2
		al += l+m;
		if (j < LSPAN+2) ak[j] = al[j];
		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1

		#if NLAT_2 > BLKSZE_A
		{	// polar optimization
			bool skip_block = (j == BLOCKSIZE-1) ? polar_skip_sint2(y1, llim, m) : false;
			#if WARPSZE == 32
			if (BLOCKSIZE == WARPSZE) skip_block = _any(skip_block);	// get largest value in block/warp
			#else
			if (BLOCKSIZE == WARPSZE) skip_block = shfl(skip_block,WARPSZE-1);	// get largest value in block/warp
			#endif
			else {
				__shared__ volatile int xx;
				if (j == BLOCKSIZE-1) xx = skip_block;	// one thread writes its value (the largest one)
				__syncthreads();
				skip_block = xx;	// everyone reads that value
			}
			// at this point, block is in sync (consistent view of shared memory)
			if (skip_block)  return;
		}
		#endif

		q += b*NFIELDS*q_dist;
		const real sgn = (j^1)-j;	//	1-2*(j&1);	// +/-
		const real costx = shfl_xor(cost, 1)*sgn;		// neighboor cost for "reverse" exchange
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			real qer = (it < nlat_2) ? q[im*m_inc        + it            + f*q_dist] : 0.0;	// north imag (ani)
			real t0  = (it < nlat_2) ? q[(nphi-im)*m_inc + it            + f*q_dist] : 0.0;	// north real (an)
			real qor = (it < nlat_2) ? q[im*m_inc        + nlat_2*2-1-it + f*q_dist] : 0.0;	// south imag (asi)
			real t1  = (it < nlat_2) ? q[(nphi-im)*m_inc + nlat_2*2-1-it + f*q_dist] : 0.0;	// south real (as)
			real qei = t0-qer;		qer += t0;		// ani = -qei[lane+1],   bni = qei[lane-1]
			real qoi = t1-qor;		qor += t1;		// bsi = -qoi[lane-1],   asi = qoi[lane+1];

			yl[(f*4+3)*l_inc +(j^1)] = (qei + qoi)*costx;	// roi, exchange even and odd lanes
			yl[(f*4+2)*l_inc + j]    = (qer - qor)*cost;	// ror
			yl[(f*4+1)*l_inc +(j^1)] = (qei - qoi)*sgn;		// rei, exchange evend and odd lanes
			yl[f*4*l_inc     + j]    =  qer + qor;			// rer
		}

		const int ofs = (4*f0+(ll&3))*l_inc + j % (BLOCKSIZE/NW);
		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			// transpose yl to my_reo (registers)
			#pragma unroll
			for (int k=0; k<NW; k++) {
				my_reo[k] = yl[ofs + k*(BLOCKSIZE/NW)];
			}

		cost = y0;		// cos(theta)^2
		#if HI_LLIM==1
		int ny = 0;
		#endif
		{	// compute sin(theta)^(m-S)
			y0 = MPOS_SCALE;	// y0
			l = m - S;		// exponent of sin(theta)
			if (ROBERT_FORM && S==1) {
				if (MRES==1 && l==0) {
					y0 *= rsqrt(y1);	// division by sin(theta) only for m=1 in Robert form
				} else --l;		// otherwise we just reduce the exponent of sin(theta)^l
			}
			if (l&1) y0 *= sqrt(y1);	// sqrt only computed when needed
			l>>=1;
			#if HI_LLIM==1
			int nsint = 0;
			#endif
			do {		// sin(theta)^(m-S)
				if (l&1) {
					y0 *= y1;
					#if HI_LLIM==1
						ny += nsint;
						if (y0 < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
							ny--;
							y0 *= SHT_SCALE_FACTOR;
						}
					#endif
				}
				y1 *= y1;
				#if HI_LLIM==1
					nsint += nsint;
					if (y1 < 1.0/SHT_SCALE_FACTOR) {
						nsint--;
						y1 *= SHT_SCALE_FACTOR;
					}
				#endif
			} while(l >>= 1);
		}

		if (it < nlat_2)     y0 *= ct[it + nlat_2];		// include quadrature weights.
		y1 = (ak[1]*cost + ak[0]) * y0;

		l=m;		al+=2+LSPAN;
		const int itl = (ll>>2)*l_inc + (j % (BLOCKSIZE/NW));		// transposed work (at given l)
	#if HI_LLIM==1
		static_assert(BLOCKSIZE == WARPSZE, "with HI_LLIM, block size must equal warp size");
		#if WARPSZE == 32
		unsigned int y_zero = _ballot(ny);
		#else
		unsigned long long y_zero = _ballot(ny);
		#endif
		if (ny) for (int k=0; k<LSPAN/2; k++)  yl[k*l_inc +j] = 0.0;
		while (y_zero && l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			#pragma unroll 4
			for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real c0 = ak[2*k+3]*cost + ak[2*k+2];
				real c1 = ak[2*k+5]*cost + ak[2*k+4];
					if (fabs(y0) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
					{	// rescale when value is significant
						++ny;
						y0 *= 1.0/SHT_SCALE_FACTOR;
						y1 *= 1.0/SHT_SCALE_FACTOR;
					}
				if (ny==0) yl[k*l_inc +j]     = y0;		// l and l+1
				if (ny==0) yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}

			y_zero = _ballot(ny);	// at this point block is in sync (consistent view of shared memory).

			if (j<LSPAN) ak[j+2] = al[j];

			if (y_zero + 1 != 0) {		// when all y are zero (all bits set -- independent of size), we can skip this.
				#ifndef __gfx90a__
				const int NACC = 2;		// number of independent accumulators (2 is the sweetspot for V100).
				#else
				const int NACC = 4;		// number of independent accumulators (4 is the sweetspot for MI200 / CDNA2).
				#endif
				real qlri[NACC];		// accumulators

				#pragma unroll
				for (int a=0; a<NACC; a++) {	// NACC independent accumulators
					qlri[a]   = my_reo[a]   * yl[itl + a*(BLOCKSIZE/NW)];
				}
				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination ofset in parallel with reduce!
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					#pragma unroll
					for (int a=0; a<NACC; a++) {	// NACC independent accumulators
						qlri[a]   += my_reo[k+a]   * yl[itl + (k+a)*(BLOCKSIZE/NW)];
					}
				}
				if (NACC>1) {	// reduce the NACC independent accumulators
					#pragma unroll
					for (int a=0; a<NACC; a+=2) 	qlri[a] += qlri[a+1];
					#pragma unroll
					for (int a=2; a<NACC; a+=2) 	qlri[0] += qlri[a];
				}

				static_assert(BLOCKSIZE/NW <= WARPSZE, "Blocksize must not exceed 2*LSPAN*WARPSZE");
					// reduce_add within same l is in same warp too:
					#pragma unroll
					for (int ofs = BLOCKSIZE/(NW*2); ofs > 0; ofs>>=1) {
						qlri[0] += shfl_down(qlri[0], ofs, BLOCKSIZE/NW);
					}
					if ( ((j % (BLOCKSIZE/NW)) == 0) && ((l+(ll>>1))<=llim) ) {	// write result
						#if NLAT_2 <= BLKSZE_A
							// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
							ql[ql_ofs]   = qlri[0];
						#else
							atomicAdd_sht(ql+ql_ofs, qlri[0]);
						#endif
					}
			}

			l+=LSPAN;
			al += LSPAN;
		}
	#endif

		while (l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			#pragma unroll 4
			for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
				real c0 = ak[2*k+3]*cost + ak[2*k+2];
				real c1 = ak[2*k+5]*cost + ak[2*k+4];
				yl[k*l_inc +j]     = y0;		// l and l+1
				yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
				y0 += c0 * y1;
				y1 += c1 * y0;
			}

			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp_fence; }
			// at this point block is in sync (consistent view of shared memory).
				#if WARPSZE==64
				if (j<LSPAN) ak[j+2] = al[j];	// on AMD, loading after this point kills performance
				#endif

				#ifndef __gfx90a__
				const int NACC = 2;		// number of independent accumulators (2 is the sweetspot for V100).
				#else
				const int NACC = 4;		// number of independent accumulators (4 is the sweetspot for MI200 / CDNA2).
				#endif
				real qlri[NACC];		// accumulators

				#pragma unroll
				for (int a=0; a<NACC; a++) {	// NACC independent accumulators
					qlri[a]   = my_reo[a]   * yl[itl + a*(BLOCKSIZE/NW)];
				}
				const int ql_ofs = 2*l+ll + (b*NFIELDS+f0)*ql_dist;		// compute destination ofset in parallel with reduce!
				#pragma unroll
				for (int k=NACC; k<NW; k+=NACC) {		// accumulate in NACC separate accumulators
					#pragma unroll
					for (int a=0; a<NACC; a++) {	// NACC independent accumulators
						qlri[a]   += my_reo[k+a]   * yl[itl + (k+a)*(BLOCKSIZE/NW)];
					}
				}

				#if WARPSZE==32
				if (j<LSPAN) ak[j+2] = al[j];	// on nvidia, loading after accumulation is more efficient
				#endif
				if (NACC>1) {	// reduce the NACC independent accumulators
					#pragma unroll
					for (int a=0; a<NACC; a+=2) 	qlri[a] += qlri[a+1];
					#pragma unroll
					for (int a=2; a<NACC; a+=2) 	qlri[0] += qlri[a];
				}

				static_assert(BLOCKSIZE/NW <= WARPSZE, "Blocksize must not exceed 2*LSPAN*WARPSZE");
					// reduce_add within same l is in same warp too:
					#pragma unroll
					for (int ofs = BLOCKSIZE/(NW*2); ofs > 0; ofs>>=1) {
						qlri[0] += shfl_down(qlri[0], ofs, BLOCKSIZE/NW);
					}
					if ( ((j % (BLOCKSIZE/NW)) == 0) && ((l+(ll>>1))<=llim) ) {	// write result
						#if NLAT_2 <= BLKSZE_A
							// no atomicAdd needed if (nlat_2 <= BLOCKSIZE), which can be decided before compilation
							ql[ql_ofs]   = qlri[0];
						#else
							atomicAdd_sht(ql+ql_ofs, qlri[0]);
						#endif
					}

			l+=LSPAN;
			al += LSPAN;
		}

	}
  #endif
}
