/*
 * Copyright (c) 2010-2020 Centre National de la Recherche Scientifique.
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

// Various CUDA kernels for SHTns

// adjustment for cuda
#undef SHT_L_RESCALE_FLY
#undef SHT_ACCURACY
#undef SHT_SCALE_FACTOR

#define SHT_L_RESCALE_FLY 1800
#define SHT_ACCURACY 1.0e-40
#define SHT_SCALE_FACTOR 2.0370359763344860863e+90

#if (__CUDACC_VER_MAJOR__ < 8) || ( defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 )
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
							 (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
	old = atomicCAS(address_as_ull, assumed,
						__double_as_longlong(val +
							   __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

// define our own suffle macros, to accomodate cuda<9 and cuda>=9
#if __CUDACC_VER_MAJOR__ < 9
	#define shfl_xor(...) __shfl_xor(__VA_ARGS__)
	#define shfl_down(...) __shfl_down(__VA_ARGS__)
	#define shfl(...) __shfl(__VA_ARGS__)
	#define _any(p) __any(p)
	#define _all(p) __all(p)
	#define _syncwarp 0
#else
	#define shfl_xor(...) __shfl_xor_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl_down(...) __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)
	#define shfl(...) __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
	#define _any(p) __any_sync(0xFFFFFFFF, p)
	#define _all(p) __all_sync(0xFFFFFFFF, p)
	#define _syncwarp __syncwarp()
#endif

/*
__device__ __forceinline__ int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId) );
  return laneId;
}

__device__ __forceinline__ void namedBarrierWait(int name, int numThreads) {
  asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

__device__ __forceinline__ void namedBarrierArrived(int name, int numThreads) {
  asm volatile("bar.arrive %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}
*/

/// Macro to check for cuda error and print details
#define CUDA_ERROR_CHECK cuda_error_check(__FILE__, __LINE__)
bool cuda_error_check(const char* fname, int l)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA ERROR %s:%d : %s!\n", fname, l, cudaGetErrorString(err));
		return true;
	}
	return false;
}

/// dim0, dim1 : size in complex numbers !
/// BLOCK_DIM_Y must be between 1 and 16
template<int BLOCK_DIM_Y> __global__ void
transpose_cplx_kernel(const double* in, double* out, const int dim0, const int dim1)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ double shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

	const int lx = threadIdx.x >> 1;
	const int ly = threadIdx.y;
	const int ri = threadIdx.x & 1;		// real/imag index

	const int bx = TILE_DIM * blockIdx.x;
	const int by = TILE_DIM * blockIdx.y;

	int gx = lx + bx;
	int gy = ly + by;
	#pragma unroll
	for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
		int gy_ = gy+repeat;
		shrdMem[ly + repeat][lx][ri] = in[2*(gy_ * dim0 + gx) + ri];
	}

	// transpose tiles:
	gx = lx + by;
	gy = ly + bx;

	__syncthreads();
	// transpose within tile:
	#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
		int gy_ = gy+repeat;
		out[2*(gy_ * dim1 + gx) + ri] = shrdMem[lx][ly + repeat][ri];
	}
}

/// dim0, dim1 : size in complex numbers !
/// BLOCK_DIM_Y must be a power of 2 between 1 and 16
template<int BLOCK_DIM_Y> __global__ void
transpose_cplx_zero_kernel(const double* in, double* out, const int dim0, const int dim1, const int mmax)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ double shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

	const int ly = threadIdx.y;
	const int lx = threadIdx.x >> 1;
	const int ri = threadIdx.x & 1;		// real/imag index

	const int by = TILE_DIM * blockIdx.y;
	const int bx = TILE_DIM * blockIdx.x;

	int gy = ly + by;
	int gx = lx + bx;

	if ((gy+(TILE_DIM-BLOCK_DIM_Y) <= mmax) || (gy >= dim1 - mmax)) {		// SAFE, no zero to insert
		#pragma unroll
		for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			shrdMem[ly + repeat][lx][ri] = in[2*(gy_ * dim0 + gx) + ri];
		}
	} else {
		for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			if ((gy_ <= mmax) || (gy_ >= dim1 - mmax)) {
				shrdMem[ly + repeat][lx][ri] = in[2*(gy_ * dim0 + gx) + ri];
			} else {
				shrdMem[ly + repeat][lx][ri] = 0.0;
			}
		}
	}

	// transpose tiles:
	gy = ly + bx;
	gx = lx + by;

	__syncthreads();
	// transpose within tile:
	#pragma unroll
	for (unsigned repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
		int gy_ = gy+repeat;
		out[2*(gy_ * dim1 + gx) + ri] = shrdMem[lx][ly + repeat][ri];
	}
}

/// dim0, dim1 : size in complex numbers !
/// BLOCK_DIM_Y must be a power of 2 between 1 and 16
template<int BLOCK_DIM_Y> __global__ void
transpose_cplx_skip_kernel(const double* in, double* out, const int dim0, const int dim1, const int mmax)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ double shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

	const int lx = threadIdx.x >> 1;
	const int ly = threadIdx.y;
	const int ri = threadIdx.x & 1;		// real/imag index

	const int bx = TILE_DIM * blockIdx.x;
	const int by = TILE_DIM * blockIdx.y;

	int gx = lx + bx;
	int gy = ly + by;

	if ((gx <= mmax) || (gx >= dim0 - mmax)) {		// read only data if m<=mmax
		#pragma unroll
		for (int repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			shrdMem[ly + repeat][lx][ri] = in[2*(gy_ * dim0 + gx) + ri];
		}
	}

	// transpose tiles:
	gy = ly + bx;
	gx = lx + by;

	__syncthreads();
	// transpose within tile:
	if ((gy <= mmax) || (gy+(TILE_DIM-BLOCK_DIM_Y) >= dim0 - mmax)) {		// write all useful data (+a bit more), the rest is ignored anyway
		#pragma unroll
		for (unsigned repeat = 0; repeat < TILE_DIM; repeat += BLOCK_DIM_Y) {
			int gy_ = gy+repeat;
			out[2*(gy_ * dim1 + gx) + ri] = shrdMem[lx][ly + repeat][ri];
		}
	}
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx(cudaStream_t stream, const double* in, double* out, const int dim0, const int dim1)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	transpose_cplx_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>(in, out, dim0, dim1);
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_zero(cudaStream_t stream, const double* in, double* out, const int dim0, const int dim1, const int mmax)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	transpose_cplx_zero_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>(in, out, dim0, dim1, mmax);
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_skip(cudaStream_t stream, const double* in, double* out, const int dim0, const int dim1, const int mmax)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	transpose_cplx_skip_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>(in, out, dim0, dim1, mmax);
}



__device__ double qish(const double* __restrict__ xlm, const double* __restrict__ ql, const int llim_m, int ll)
{
	const int l = ll >> 1;
	const int x_ofs = 3*(ll >> 2);

	double q = 0.0;
	if (l <= llim_m) {
		q = ql[ll] * xlm[x_ofs + (ll&2)];
		if (((ll&2)==0) && (l+2 <=llim_m)) {	// l-m even
			q += ql[ll+4] * xlm[x_ofs + 1];		// contribution of l+2
		}
	}
	return q;
}

__global__ void
sh2ishioka_kernel_alt(const double* __restrict__ xlm, const double* __restrict__ ql, double* ql_ish, 
		const int llim, const int lmax, const int mres, const int S, const int ql_dist=0, const int ql_ish_dist=0)
{
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	const int l0 = ((blockDim.x-4) * blockIdx.x) >> 1;              // some overlap needed
	const int l  = l0 + (j >> 1);
	const int m = im*mres;
	const int llim_m = llim-m;
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres);

	xlm += 3*im*(2*(lmax+4) -m+mres)/4;
	ql     += q_ofs;
	ql_ish += q_ofs;
	
	if (im==0) { if (l<=lmax) ql_ish[2*l0+j  + b*ql_ish_dist] = ql[2*l0+j + b*ql_dist];	return; }	// DEBUG: copy

	if ((l<=llim_m) && (j < blockDim.x-4)) {
		double q = qish(xlm, ql + b*ql_dist, llim-m, 2*l0 + j);
		if (im>0) {
			ql_ish[2*l0 +j + b*ql_ish_dist] = q;   // coalesced store
		} else if ((j&1)==0) {
			ql_ish[l0 +(j>>1) + b*ql_ish_dist] = q;   // coalesced store
		}
	}
}

__global__ void
sh2ishioka_kernel(const double* __restrict__ xlm, const double* __restrict__ ql, double* ql_ish, 
	const int llim, const int lmax, const int mres, const int S, const int ql_dist=0, const int ql_ish_dist=0)
{
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	const int l0 = ((blockDim.x-4) * blockIdx.x) >> 1;		// some overlap needed

	const int l  = l0 + (j >> 1);
	const int m = im*mres;
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres) + 2*l0;
	const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4 + 3*(l0 >> 1);
	const int llim_m = llim-m;

	extern __shared__ double ql_[];			// size blockDim.x
	double* const xl_ = ql_ + blockDim.x;	// size blockDim.x/4*3 - 3

	double q = 0.0;
	if (l <= llim_m) {
		if (j<(blockDim.x>>2)*3-3) xl_[j] = xlm[x_ofs +j];
		q = ql[q_ofs +j + b*ql_dist];
	}
	if ((l-2 <= llim_m) && ((j&2) == 0)) ql_[(j>>1)+(j&1)] = q;

	__syncthreads();

	if ((l<=llim_m) && (j < blockDim.x-4)) {
		int ix = 3*(j>>2);		// 3*l/2.
		q *= xl_[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
		if ((j&2)==0) {		// for l-m even
			q += ql_[(j>>1)+(j&1)+2] * xl_[ix+1];			// contribution of l+2
		}
		if (im > 0) {
			ql_ish[q_ofs +j + b*ql_ish_dist] = q;	// coalesced store
		} else if ((j&1)==0) {
			ql_ish[q_ofs -l0 +(j>>1) + b*ql_ish_dist] = q;	// coalesced store
		}
	}
}

/// performs: Ql[2*l] = qq[2*l]*xlm[3*l] + qq[2*l-2]*xlm[3*l+1];   Ql[2*l+1] = qq[2*l+1] * xlm[3*l+2];
/// includes zero-out for unused modes.
__global__ void
ishioka2sh_kernel(const double* __restrict__ xlm, const double* __restrict__ ql_ish, double* ql, 
	const int llim, const int lmax, const int mmax, const int mres, const int S, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	const int l0 = ((blockDim.x-4) * blockIdx.x) >> 1;		// some overlap needed

	const int l  = l0 + (j >> 1);
	const int m = im*mres;
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres) + 2*l0;
	const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4 + 3*(l0 >> 1);
	const int llim_m = llim-m;

	extern __shared__ double ql_[];			// size blockDim.x
	double* const xl_ = ql_ + blockDim.x;	// size blockDim.x/4*3 - 3

	double q = 0.0;
	if ((l-2 <= llim_m) && (im <= mmax)) {
		if ((j<(blockDim.x>>2)*3+3) && (x_ofs+j-3 >= 0)) xl_[j] = xlm[x_ofs +j-3];
		if (l-2 >= 0) {
			if (im>0) {
				q = ql_ish[q_ofs +j-4 + b*ql_ish_dist];		// ql_[4] = ql_ish[0]
			} else if ((j&1) == 0) {
				q = ql_ish[((q_ofs +j-4)>>1) + b*ql_ish_dist];		// ql_[4] = ql_ish[0]
			}
		}
		ql_[j] = q;
	}

	__syncthreads();

	if (j<blockDim.x-4) {
		q = 0.0;
		if ((l<=llim_m) && (im <= mmax)) {
			int ix = 3*(j>>2)+3;		// 3*l/2.
			q = ql_[j+4] * xl_[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
			if ((j&2)==0) {		// for l-m even
				q += ql_[j] * xl_[ix-2];			// contribution of l-2
			}
		}
		if (l<=lmax+S-m)
			ql[q_ofs +j + b*ql_dist] = q;	// coalesced store (including zero-out for llim<l<=lmax) AND zero-out for m>mmax
	}
}


/** \internal convert from vector SH to scalar SH. slm or tlm can be null pointers.
	Vlm =  st*d(Slm)/dtheta + I*m*Tlm
	Wlm = -st*d(Tlm)/dtheta + I*m*Slm
*/
template<int BLOCKSIZE> __global__ void
sphtor2scal_kernel(const double* __restrict__ mx, const double* __restrict__ slm, const double* __restrict__ tlm, double *vlm, double *wlm, const int llim, const int lmax, const int mres)
{
	// indices for overlapping blocks:
	const int ll = (blockDim.x-4) * blockIdx.x + threadIdx.x - 2;		// = 2*l + ((imag) ? 1 : 0)
	const int j = threadIdx.x;
	const int im = blockIdx.y;

	__shared__ double sl[BLOCKSIZE];
	__shared__ double tl[BLOCKSIZE];
	__shared__ double M[BLOCKSIZE];

	const int m = im*mres;
	const int ofs   = im*(((lmax+1)<<1) -m + mres) + ll;

	if ( (ll >= 0) && (ll < 2*(llim+1-m)) ) {
		M[j] = mx[ofs];
		sl[j] = (slm) ? slm[ofs] : 0.0;
		tl[j] = (tlm) ? tlm[ofs] : 0.0;
	} else {
		M[j] = 0.0;
		sl[j] = 0.0;
		tl[j] = 0.0;
	}
	const double mimag = m * (j - (j^1));

	__syncthreads();

	if ((j<BLOCKSIZE-4) && (ll < 2*(llim+1-m))) {
		double ml = M[2*(j>>1)+1];
		double mu = M[2*(j>>1)+2];
		double v = mimag*tl[(j+2)^1]  +  (ml*sl[j] + mu*sl[j+4]);
		double w = mimag*sl[(j+2)^1]  -  (ml*tl[j] + mu*tl[j+4]);
		vlm[ofs+2*im+2] = v;
		wlm[ofs+2*im+2] = w;
	}
}

__global__ void
sphtor2ish_kernel(const double* __restrict__ mx, const double* __restrict__ xlm,
		const double* __restrict__ slm, const double* __restrict__ tlm, double *vlm, double *wlm, 
		const int llim, const int lmax, const int mres, const int ql_dist=0, const int ql_ish_dist=0)
{
	// indices for overlapping blocks:
	const int l0 = (blockDim.x-8) * blockIdx.x;		// some overlap needed
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	int ll = l0 + j - 2;

	extern __shared__ double sl[];			// size blockDim.x
	double* const tl = sl + blockDim.x;		// size blockDim.x
	double* const M  = sl + 2*blockDim.x;	// size blockDim.x
	
	const int m = im*mres;
	const int llim_m_p1 = llim+1-m;
	const int ofs = im*(((lmax+1)<<1) -m + mres) + ll;
	ll >>= 1;

	if ( (ll >= 0) && (ll < llim_m_p1) ) {
		M[j] = mx[ofs];
		sl[j] = (slm) ? slm[ofs + b*ql_dist] : 0.0;
		tl[j] = (tlm) ? tlm[ofs + b*ql_dist] : 0.0;
	} else {
		M[j] = 0.0;
		sl[j] = 0.0;
		tl[j] = 0.0;
	}

	__syncthreads();

	double v = 0.0;
	double w = 0.0;
	const double mimag = m * (j - (j^1));
	if ((j<blockDim.x-4) && (ll < llim_m_p1)) {
		double ml = M[2*(j>>1)+1];
		double mu = M[2*(j>>1)+2];
		v = mimag*tl[(j+2)^1]  +  (ml*sl[j] + mu*sl[j+4]);
		w = mimag*sl[(j+2)^1]  -  (ml*tl[j] + mu*tl[j+4]);
	}

	const int j2 = (j>>1)+(j&1);

	__syncthreads();

	if ((j&2)==0) {
		sl[j2] = v;
		tl[j2] = w;
	}
	if (ll >= llim_m_p1) return;	// nothing else to do.

	const int x_ofs = (3*im*(2*(lmax+4) -m+mres)>>2) + 3*(l0 >> 2);
	if (j<(blockDim.x>>2)*3) M[j] = xlm[x_ofs +j];

	__syncthreads();

	if (j < blockDim.x-8) {
		int ix = 3*(j>>2);		// 3*l/2.
		double x0 = M[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
		v *= x0;
		w *= x0;
		if ((j&2)==0) {		// for l-m even
			double x2 = M[ix+1];			// contribution of l+2
			v += x2 * sl[j2+2];
			w += x2 * tl[j2+2];
		}
		if (im>0) {
			vlm[ofs+2*im+2 + b*ql_ish_dist] = v;
			wlm[ofs+2*im+2 + b*ql_ish_dist] = w;
		} else if ((j&1)==0) {		// compress complex to real (as imaginary part is zero)
			vlm[(ofs>>1)+1 + b*ql_ish_dist] = v;	// coalesced store
			wlm[(ofs>>1)+1 + b*ql_ish_dist] = w;	// coalesced store
		}
	}
}


/** \internal convert from 2 scalar SH to vector SH
	Slm = - (I*m*Wlm + MX*Vlm) / (l*(l+1))
	Tlm = - (I*m*Vlm - MX*Wlm) / (l*(l+1))
**/
template<int BLOCKSIZE> __global__ void
scal2sphtor_kernel(const double* __restrict__ mx, const double* __restrict__ vlm, const double* __restrict__ wlm, double *slm, double *tlm, const int llim, const int lmax, const int mres)
{
	// indices for overlapping blocks:
	int ll = (blockDim.x-4) * blockIdx.x + threadIdx.x - 2;		// = 2*l + ((imag) ? 1 : 0)
	const int j = threadIdx.x;
	const int im = blockIdx.y;

	__shared__ double vl[BLOCKSIZE];
	__shared__ double wl[BLOCKSIZE];
	__shared__ double M[BLOCKSIZE];

	const int m = im * mres;
	int ofs = im*(2*(lmax+1) -m + mres)  + ll;
	const int llim_m_p1 = llim+1-m;
	ll >>= 1;

	if ( (ll >= 0) && (ll < llim_m_p1) ) {
		M[j] = mx[ofs];
	} else M[j] = 0.0;

	if ( (ll >= 0) && (ll <= llim_m_p1) ) {
		vl[j] = vlm[ofs+2*im];
		wl[j] = wlm[ofs+2*im];
	} else {
		vl[j] = 0.0;
		wl[j] = 0.0;
	}

	ll += m + 1;		// +1 because we shift below

	__syncthreads();

	if (j<BLOCKSIZE-4) {
		if ((ll <= llim) && (ll>0)) {
			const double mimag = m * ((j^1) -j);
			double ll_1 = 1.0 / (ll*(ll+1));
			double ml = M[2*(j>>1)+1];
			double mu = M[2*(j>>1)+2];
			double s = mimag*wl[(j+2)^1]  -  (ml*vl[j] + mu*vl[j+4]);
			double t = mimag*vl[(j+2)^1]  +  (ml*wl[j] + mu*wl[j+4]);
			slm[ofs+2] = s * ll_1;
			tlm[ofs+2] = t * ll_1;
		} else if (ll <= lmax) {	// fill with zeros up to lmax (and l=0 too).
			slm[ofs+2] = 0.0;
			tlm[ofs+2] = 0.0;
		}
	}
}

__global__ void
ish2sphtor_kernel(const double* __restrict__ mx, const double* __restrict__ xlm, const double* __restrict__ vlm, const double* __restrict__ wlm, 
	double *slm, double *tlm, const int llim, const int lmax, const int mres, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int j = threadIdx.x;
	const int im = blockIdx.y;
	const int b = blockIdx.z;
	const int l0 = (blockDim.x-8) * blockIdx.x;		// some overlap needed

	int l  = (l0 + j) >> 1;
	const int m = im*mres;
	const int q_ofs = im*(((lmax+1)*2) -m+mres) + l0;
	const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4 + 3*(l0 >> 2);
	const int llim_m_p1 = llim+1-m;

	extern __shared__ double vl[];			// size blockDim.x
	double* const wl = vl + blockDim.x;		// size blockDim.x
	double* const M  = vl + 2*blockDim.x;	// size blockDim.x

	double v = 0.0;		double w = 0.0;
	if (l-2 <= llim_m_p1) {
		if ((j<(blockDim.x>>2)*3+3) && (x_ofs+j-3 >= 0)) M[j] = xlm[x_ofs +j-3];
		if (l-2 >= 0) {
			if (im>0) {
				v = vlm[q_ofs +2*im +j-4 + b*ql_ish_dist];		// vl[4] = vlm[0]
				w = wlm[q_ofs +2*im +j-4 + b*ql_ish_dist];		// vl[4] = vlm[0]
			} else if ((j&1)==0) {
				v = vlm[((q_ofs +j-4)>>1) + b*ql_ish_dist];		// vl[4] = vlm[0]
				w = wlm[((q_ofs +j-4)>>1) + b*ql_ish_dist];		// vl[4] = vlm[0]
			}
		}
	}
	vl[j] = v;
	wl[j] = w;

	__syncthreads();

	if (j<blockDim.x-4) {
		v = 0.0;
		w = 0.0;
		if (l<=llim_m_p1) {
			int ix = 3*(j>>2)+3;		// 3*l/2.
			v = vl[j+4] * M[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
			w = wl[j+4] * M[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
			if ((j&2)==0) {		// for l-m even
				v += vl[j] * M[ix-2];			// contribution of l-2
				w += wl[j] * M[ix-2];			// contribution of l-2
			}
		}
	}

	__syncthreads();
	
	if (j<blockDim.x-4) {
		if (l<=lmax+1-m) {
			vl[j+4] = v;	// vlm[q_ofs + 2*im +j]
			wl[j+4] = w;
		} else {
			vl[j+4] = 0.0;		wl[j+4] = 0.0;
		}
	}

	if ( (l > 0) && (l <= llim_m_p1) ) {
		M[j] = mx[q_ofs+j-2];
	} else M[j] = 0.0;

	l += m;

	__syncthreads();

	if ((j<blockDim.x-6) &&  (j >= ((blockIdx.x == 0) ? 0 : 2))) {
		v = 0.0;	w = 0.0;
		if ((l <= llim) && (l>0)) {
			const double mimag = m * ((j^1) -j);
			double ll_1 = 1.0 / (l*(l+1));
			double ml = M[2*(j>>1)+1];
			double mu = M[2*(j>>1)+2];
			v = mimag*wl[(j+4)^1]  -  (ml*vl[j+2] + mu*vl[j+6]);
			w = mimag*vl[(j+4)^1]  +  (ml*wl[j+2] + mu*wl[j+6]);
			v *= ll_1;
			w *= ll_1;
		}
		if (l <= lmax) {	// fill with zeros up to lmax (and l=0 too).
			slm[q_ofs+j + b*ql_dist] = v;
			tlm[q_ofs+j + b*ql_dist] = w;
		}
	}
}


void sh2ishioka_gpu(shtns_cfg shtns, cplx* d_Qlm, cplx* d_Qlm_ish, int llim, int mmax, int S=0)
{
	int blksze = (((llim+2)*2+WARPSZE-1)/WARPSZE) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(llim+3)+blksze-5)/(blksze-4), mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	sh2ishioka_kernel <<< blocks, threads,(blksze/4*7-3)*sizeof(double), shtns->comp_stream >>>
		(shtns->d_xlm, (double*) d_Qlm, (double*) d_Qlm_ish, llim, shtns->lmax, shtns->mres, S, shtns->spec_dist*2, shtns->nlm_stride);
	CUDA_ERROR_CHECK;
}

void ishioka2sh_gpu(shtns_cfg shtns, cplx* d_Qlm_ish, cplx* d_Qlm, int llim, int mmax, int S=0)
{
	int blksze = (((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-5)/(blksze-4), shtns->mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	ishioka2sh_kernel <<< blocks, threads, (blksze/4*7+3)*sizeof(double), shtns->comp_stream >>>
		(shtns->d_xlm, (double*) d_Qlm_ish, (double*) d_Qlm, llim, shtns->lmax, mmax, shtns->mres, S, shtns->nlm_stride, shtns->spec_dist*2);
	if (CUDA_ERROR_CHECK) return;
}

void sphtor2scal_gpu(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, cplx* d_Vlm, cplx* d_Wlm, int llim, int mmax)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-9)/(WARPSZE-8) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-9)/(blksze-8), mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	sphtor2ish_kernel <<< blocks, threads, blksze*3*sizeof(double), shtns->comp_stream >>>
		(shtns->d_mx_stdt, shtns->d_xlm, (double*) d_Slm, (double*) d_Tlm, (double*) d_Vlm, (double*) d_Wlm, llim, shtns->lmax, shtns->mres, shtns->spec_dist*2, shtns->nlm_stride);
	CUDA_ERROR_CHECK;
}

void scal2sphtor_gpu(shtns_cfg shtns, cplx* d_Vlm, cplx* d_Wlm, cplx* d_Slm, cplx* d_Tlm, int llim)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-9)/(WARPSZE-8) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-9)/(blksze-8), shtns->mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	ish2sphtor_kernel <<< blocks, threads, blksze*3*sizeof(double), shtns->comp_stream >>>
		(shtns->d_mx_van, shtns->d_xlm, (double*) d_Vlm, (double*) d_Wlm, (double*)d_Slm, (double*)d_Tlm, llim, shtns->lmax, shtns->mres, shtns->nlm_stride, shtns->spec_dist*2);
	CUDA_ERROR_CHECK;
}



/// requirements : blockSize must be 1 in the y-direction and THREADS_PER_BLOCK in the x-direction.
/// llim MUST BE <= 1800
/// S can only be 0 (for scalar) or 1 (for spin 1 / vector)
template<int BLOCKSIZE, int S, int NFIELDS, int NW, bool HI_LLIM, bool M0_ONLY=false, bool ROBERT_FORM=false, bool SH2ISH=false>
static __global__ void leg_m_kernel(
	const double* __restrict__ al, const double* __restrict__ ct, const double* __restrict__ ql, double *q,
	const int llim, const int nlat_2, const int lmax, const int mres, const int nphi, const int m_inc,
	const int ql_dist=0, const int q_dist=0, const double* __restrict__ xlm = 0)
{
	const int it = BLOCKSIZE*NW * blockIdx.x + threadIdx.x;
	const int im = (M0_ONLY) ? 0 : blockIdx.y;
	const int j = threadIdx.x;
	const int b = blockIdx.z;		// position in batch
	//const int m_inc = 2*nlat_2;
	const int k_inc = 1;

	__shared__ double ak[BLOCKSIZE];		// size blockDim.x
	__shared__ double qk[NFIELDS][(M0_ONLY) ? BLOCKSIZE : BLOCKSIZE*2];	// size 2*blockDim.x * NFIELDS

	//static_assert( NFIELDS==1, "only NFIELDS=1 is supported" );		// WIP batch

	static_assert( (!HI_LLIM) || ((NW==1) && (BLOCKSIZE == WARPSZE)), "high llim works with NW=1 and BLOCKSIZE=32" );

	double cost[NW];
	double y0[NW];
	double y1[NW];
	#pragma unroll
	for (int i=0; i<NW; i++) {
		const int iit = it+i*BLOCKSIZE;
		cost[i] = (iit < nlat_2) ? ct[iit] : 0.0;
	}
	double ct2[NW];
	#pragma unroll
	for (int i=0; i<NW; i++) ct2[i] = cost[i]*cost[i];		// cos(theta)^2

	if (im==0) {
		ak[j] = al[j+2];
		if (j<2*(llim+1)) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) 	{
				if (!SH2ISH) qk[f][j] = ql[j + (b*NFIELDS+f)*ql_dist];		// keep only real part
				else qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*j);
			}
		}
		double re[NFIELDS][NW], ro[NFIELDS][NW];
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
		for (int i=0; i<NW; i++) y0[i] = 1.0;
		if (S==1 && !ROBERT_FORM) for (int i=0; i<NW; i++) y0[i] = rsqrt(1.0 - ct2[i]);	// for vectors, divide by sin(theta) -- except in Robert form
		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i] + al[0])*y0[i];

		al+=2;
		if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }

		while (l<=llim - BLOCKSIZE) {	// compute even and odd parts
			#pragma unroll
			for (int k = 0; k<BLOCKSIZE; k+=4) {
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
			al += BLOCKSIZE;
			l  += BLOCKSIZE;
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if (l+j <= llim) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	if (!SH2ISH) qk[f][j] = ql[l+j + (b*NFIELDS+f)*ql_dist];
					else qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*(l+j));
			}
			if (l+j <= llim)	 ak[j] = al[j];
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
				double tmp = (ak[k+1]*ct2[i] + ak[k]) * y1[i] + y0[i];
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
			const int iit = it+i*BLOCKSIZE;
			if (iit < nlat_2) {
				// store mangled for complex fft
				const int iit = it+i*BLOCKSIZE;
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
						q[iit*k_inc + (b*NFIELDS+f)*q_dist] = re[f][i]+ro[f][i]*cost[i];
						q[(nlat_2*2-1-iit)*k_inc + (b*NFIELDS+f)*q_dist] = re[f][i]-ro[f][i]*cost[i];
				}
			}
		}
	} else { 	// m>0
		double rer[NFIELDS][NW], ror[NFIELDS][NW], rei[NFIELDS][NW], roi[NFIELDS][NW];
		int m = im*mres;
		int l = (im*(2*(lmax+1)-(m+mres)))>>1;
		if (SH2ISH)	xlm += 3*im*(2*(lmax+4) -m+mres)/4;
			#pragma unroll
			for (int i=0; i<NW; i++) 	y1[i] = sqrt(1.0 - ct2[i]);		// y1 = sin(theta)
			al += l+m;
		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1

		ak[j] = al[j+2];
		if (m+j/2 <= llim) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++)	if (!SH2ISH) qk[f][j] = ql[2*m+j + (b*NFIELDS+f)*ql_dist];
					else qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*m+j);
		}
			if (m+j/2+BLOCKSIZE/2 <= llim) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	if (!SH2ISH) qk[f][j+BLOCKSIZE] = ql[2*m+j+BLOCKSIZE + (b*NFIELDS+f)*ql_dist];
					else qk[f][j+BLOCKSIZE] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*m+j+BLOCKSIZE);
			}

		#pragma unroll
		for (int i=0; i<NW; i++) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++) {
				ror[f][i] = 0.0;		roi[f][i] = 0.0;
				rer[f][i] = 0.0;		rei[f][i] = 0.0;
			}
			y0[i] = 1.0;
		}

	if ((NW>1) || (BLOCKSIZE > WARPSZE) || (_any(m - llim*y1[0] <= max(80, llim>>7))))	// polar optimization (see Reinecke 2013), avoiding warp divergence
	{
		l = m - S;
		if (S==1 && ROBERT_FORM) l = m;		// multiply vectors by sin(theta) with robert_form
		int nsint = 0;
		int ny = 0;
		do {		// sin(theta)^(m-S)
			if (l&1) {
				#pragma unroll
				for (int i=0; i<NW; i++) y0[i] *= y1[i];
				if (HI_LLIM) {
					ny += nsint;
					if (y0[0] < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
						y0[0] *= SHT_SCALE_FACTOR;
						ny--;
					}
				}
			}
			#pragma unroll
			for (int i=0; i<NW; i++) y1[i] *= y1[i];
			if (HI_LLIM) {
				nsint += nsint;
				if (y1[0] < 1.0/SHT_SCALE_FACTOR) {
					nsint--;
					y1[0] *= SHT_SCALE_FACTOR;
				}
			}
		} while(l >>= 1);

		#pragma unroll
		for (int i=0; i<NW; i++) y1[i] = (al[1]*ct2[i] + al[0])*y0[i];

		if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		l=m;		al+=2;

		while (l<=llim - BLOCKSIZE) {	// compute even and odd parts
			#pragma unroll
			for (int k = 0; k<BLOCKSIZE; k+=4) {
				double tmp[NW];
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
				} else if (fabs(y0[0]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
				{	// rescale when value is significant
					++ny;
					y0[0] *= 1.0/SHT_SCALE_FACTOR;
					y1[0] *= 1.0/SHT_SCALE_FACTOR;
				}
				#pragma unroll
				for (int i=0; i<NW; i++)	y1[i] += tmp[i] * y0[i];
			}
			al += BLOCKSIZE;
			l  += BLOCKSIZE;
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
			if (l+j/2 <= llim) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	if (!SH2ISH) qk[f][j] = ql[2*l+j + (b*NFIELDS+f)*ql_dist];
						else qk[f][j] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*l+j);
			}
			if (l+j/2+BLOCKSIZE/2 <= llim) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++)	if (!SH2ISH) qk[f][BLOCKSIZE+j] = ql[2*l+BLOCKSIZE+j + (b*NFIELDS+f)*ql_dist];
						else qk[f][j+BLOCKSIZE] = qish(xlm, ql+(b*NFIELDS+f)*ql_dist, llim, 2*l+j+BLOCKSIZE);
			}
			if (l+j <= llim)	 ak[j] = al[j];
			if (BLOCKSIZE > WARPSZE) { __syncthreads(); } else { _syncwarp; }
		}
		int k=0;
		while (l<llim) {	// compute even and odd parts
			double tmp[NW];
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
			} else if (fabs(y1[0]) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
			{	// rescale when value is significant
				++ny;
				y0[0] *= 1.0/SHT_SCALE_FACTOR;
				y1[0] *= 1.0/SHT_SCALE_FACTOR;
			}
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

		// correct odd part, before fft mangling
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++)	roi[f][i] *= cost[i];		// do roi first, used in shuffle below
		}
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			#pragma unroll
			for (int i=0; i<NW; i++)	ror[f][i] *= cost[i];
		}

		/// store mangled for complex fft
		#pragma unroll
		for (int i=0; i<NW; i++) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++)	rei[f][i] = shfl_xor(rei[f][i], 1);
		}
		#pragma unroll
		for (int i=0; i<NW; i++) {
			#pragma unroll
			for (int f=0; f<NFIELDS; f++)	roi[f][i] = shfl_xor(roi[f][i], 1);
		}
	}

		double nr[NFIELDS][NW];
		const double sgn = (j^1) - j;	// 1 - 2*(j&1);		// 1 for even j, -1 for odd j.
		#pragma unroll
		for (int i=0; i<NW; i++) {
			const int iit = it+i*BLOCKSIZE;
			if (iit < nlat_2) {
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					nr[f][i] =  rer[f][i]+ror[f][i];
					rer[f][i] = rer[f][i]-ror[f][i];
					ror[f][i] = rei[f][i]+roi[f][i];
					rei[f][i] = rei[f][i]-roi[f][i];
				}
				#pragma unroll
				for (int f=0; f<NFIELDS; f++) {
					q[im*m_inc + iit*k_inc + (b*NFIELDS+f)*q_dist]                     = nr[f][i]  - ror[f][i]*sgn;
					q[(nphi-im)*m_inc + iit*k_inc + (b*NFIELDS+f)*q_dist]              = nr[f][i]  + ror[f][i]*sgn;
					q[im*m_inc + (nlat_2*2-1-iit)*k_inc + (b*NFIELDS+f)*q_dist]        = rer[f][i] + rei[f][i]*sgn;
					q[(nphi-im)*m_inc + (nlat_2*2-1-iit)*k_inc + (b*NFIELDS+f)*q_dist] = rer[f][i] - rei[f][i]*sgn;
				}
			}
		}
	}
}

template<int S, int NFIELDS, bool HI_LLIM=false>
static void leg_m(shtns_cfg shtns, const double *ql, double *q, const int llim, const int mmax, long spat_dist=0)
{
	const int lmax = shtns->lmax;
	const int mres = shtns->mres;
	const int nlat_2 = shtns->nlat_2;
	const int nphi = shtns->nphi;
	double *d_alm = shtns->d_clm;
	double *d_ct = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;

	const int BLOCKSIZE = 32;		// 32 allows to use polar optimization; 128 and NW=2 are sometimes better though.
	const int threadsPerBlock = BLOCKSIZE;	// can be from 32 to 1024, we should try to measure the fastest !
	if (spat_dist == 0) spat_dist = shtns->spat_stride;
	
	dim3 threads(threadsPerBlock, 1, 1);
	if (shtns->howmany % 4 == 0) {	// multiple of 4
		const int NW = 1;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany/4);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 4, NW, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 4, NW, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		}
	} else if (shtns->howmany % 2 == 0) {	// multiple of 2
		const int NW = (HI_LLIM) ? 1 : 2;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany/2);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 2, NW, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 2, NW, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		}
	} else {
		const int NW = (HI_LLIM) ? 1 : 2;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 1, NW, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 1, NW, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) ql, (double*) q, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->nlm_stride, spat_dist);
		}
	}
}

template<int S, int NFIELDS>
static void leg_m0(shtns_cfg shtns, const double *ql, double *q, const int llim, long spat_dist = 0)
{
	const int nlat_2 = shtns->nlat_2;
	double *d_ct = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;

	static_assert( NFIELDS == 1, "batched transform requires NFIELDS=1");
	if (spat_dist == 0) spat_dist = shtns->spat_stride;

	const int BLOCKSIZE = 32;		// good value
	const int NW = 4;
	const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);

	// Launch the Legendre CUDA Kernel
	const int threadsPerBlock = BLOCKSIZE;	// can be from 32 to 1024, we should try to measure the fastest !
	if (shtns->howmany % 4 == 0) {
		dim3 threads(threadsPerBlock, 1, 1);
		dim3 blocks(blocksPerGrid, 1, shtns->howmany/4);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 4, NW, false, true, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 4, NW, false, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		}
	} else if (shtns->howmany % 2 == 0) {
		const int NW = 4;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);
		dim3 threads(threadsPerBlock, 1, 1);
		dim3 blocks(blocksPerGrid, 1, shtns->howmany/2);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 2, NW, false, true, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 2, NW, false, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		}
	} else {
		const int NW = 4;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE*NW - 1) / (BLOCKSIZE*NW);
		dim3 threads(threadsPerBlock, 1, 1);
		dim3 blocks(blocksPerGrid, 1, shtns->howmany);
		if (S==1 && shtns->robert_form) {
			leg_m_kernel<BLOCKSIZE, S, 1, NW, false, true, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		} else {
			leg_m_kernel<BLOCKSIZE, S, 1, NW, false, true> <<<blocks, threads, 0, stream>>>
				(shtns->d_clm, d_ct, (double*) ql, (double*) q, llim, nlat_2, llim,1, 1, shtns->nlat_padded, shtns->nlm_stride, spat_dist, shtns->d_xlm);
		}
	}
}


template<int BLOCKSIZE, int LSPAN, int S, int NFIELDS, bool HI_LLIM, bool M0_ONLY=false, bool ROBERT_FORM=false> __global__ void
ileg_m_kernel(const double* __restrict__ al, const double* __restrict__ ct, const double* __restrict__ q, double *ql, const int llim, 
	const int nlat_2, const int lmax, const int mres, const int nphi, const int m_inc, const double mpos_scale, const int q_dist=0, const int ql_dist=0)
{
	const int it = BLOCKSIZE * blockIdx.x + threadIdx.x;
	const int j = threadIdx.x;
	const int im = (M0_ONLY) ? 0 : blockIdx.y;
	const int b = blockIdx.z;
	//const int m_inc = 2*nlat_2;
	const int f0 = (NFIELDS==1) ? 0 : j / (BLOCKSIZE/NFIELDS);			// assign each thread a field f0

	static_assert((BLOCKSIZE % (((M0_ONLY)?1:2)*LSPAN*NFIELDS)) == 0, "BLOCKSIZE must be a multiple of 2*LSPAN*NFIELDS");
	static_assert( ((WARPSZE >= BLOCKSIZE/LSPAN) ? (WARPSZE % (BLOCKSIZE/LSPAN)) : ((BLOCKSIZE/LSPAN) % WARPSZE)) == 0, "WARPSZE and BLOCKSIZE/LSPAN must be multiples");
	static_assert( (!HI_LLIM) || (BLOCKSIZE == WARPSZE), "for high llim, BLOCKSIZE must be 32");
	static_assert((LSPAN % 4) == 0, "LSPAN must be a multiple of 4");

	const int padding = 2;		// padding = 0 is very bad for performance (shared-memory bank conflicts).
	const int l_inc = BLOCKSIZE+padding;
	__shared__ double ak[LSPAN+2];	// cache
	const int NROWS = M0_ONLY ? ( (LSPAN>4*NFIELDS) ? LSPAN/2 : 2*NFIELDS ) : ( (LSPAN>8*NFIELDS) ? LSPAN/2 : 4*NFIELDS );
	__shared__ double yl[NROWS*l_inc - padding];		// yl is also used for even/odd computation.

	double cost = (it < nlat_2) ? ct[it] : 0.0;
	double y0, y1;

	if (im == 0) {
		const int NW = NFIELDS*LSPAN;
		// re-assign each thread an l (transposed view)
		const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);
		double my_reo[NW];			// in registers

		if (j < LSPAN+2) ak[j] = al[j];

		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			y0 = (it < nlat_2) ? q[it + (b*NFIELDS+f)*q_dist] : 0.0;				// north
			y1 = (it < nlat_2) ? q[nlat_2*2-1 - it + (b*NFIELDS+f)*q_dist] : 0.0;	// south
			if (ROBERT_FORM) {
				double st_1 = rsqrt(1.0 - cost*cost);		// 1/sin(theta)
				y0 *= st_1;		y1 *= st_1;
			}
			yl[f*2*l_inc +j]     = y0+y1;			// even
			yl[(f*2+1)*l_inc +j] = (y0-y1)*cost;	// odd
		}
		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }
			// transpose reo to my_reo
			#pragma unroll
			for (int k=0; k<NW; k++) {
				int it = j % (BLOCKSIZE/NW) + k*(BLOCKSIZE/NW);
				my_reo[k] = yl[(2*f0  + (ll&1))*l_inc + it];
			}

		int l = 0;
		y0 = (it < nlat_2) ? ct[it + nlat_2] : 0.0;		// weights are stored just after ct.
		cost *= cost;	// ct2
		if (S==1) y0 *= rsqrt(1.0 - cost);
		y1 = (ak[1]*cost + ak[0]) * y0;

		al+=2;
		while (l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }
				#pragma unroll
				for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
					double c0 = ak[2*k+3]*cost + ak[2*k+2];
					double c1 = ak[2*k+5]*cost + ak[2*k+4];
					yl[k*l_inc +j]     = y0;		// l and l+1
					yl[(k+1)*l_inc +j] = y1;		// l+2 and l+3
					al += 4;
					y0 = c0 * y1 + y0;
					y1 = c1 * y0 + y1;
				}
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }

			const int NACC = 4;		// number of independent accumulators per NFIELD. 4 is good for V100
			double qll[NACC];		// accumulators
			// now re-assign each thread an l (transpose)
			const int itl = (ll >> 1)*l_inc + j % (BLOCKSIZE/NW);
			#pragma unroll
			for (int a=0; a<NACC; a++) 	qll[a] = my_reo[a] * yl[itl + a*(BLOCKSIZE/NW)];	// first element of sum
			#pragma unroll
			for (int k=NACC; k<NW; k+=NACC) {
				#pragma unroll
				for (int a=0; a<NACC; a++) 	qll[a] += my_reo[a+k] * yl[itl + (k+a)*(BLOCKSIZE/NW)];
			}
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
					if ((!HI_LLIM) && (nlat_2 <= BLOCKSIZE)) {		// do we need atomic add or not ?
						ql[(l+ll) + (b*NFIELDS+f0)*ql_dist] = qll[0];
					} else {
						atomicAdd(ql+(l+ll) + (b*NFIELDS+f0)*ql_dist, qll[0]);		// VERY slow atomic add on Kepler.
					}
				}

			if (j<LSPAN) ak[j+2] = al[j];
			l+=LSPAN;
		}
	} else {	// im > 0
		const int NW = NFIELDS*LSPAN*(M0_ONLY ? 1:2);		// the test for M0_ONLY is to silence a warning
		// re-assign each thread an l (transposed view)
		const int ll = (j % (BLOCKSIZE/NFIELDS)) / (BLOCKSIZE/NW);		// actualy ll = 2*l + (imag ? 1 : 0)
		double my_reo[NW];			// in registers

		int m = im*mres;
		int l = (im*(2*(lmax+1)-(m+mres)))>>1;

		y0 = cost * cost;			// cos(theta)^2
		al += l+m;
		y1 = sqrt(1.0 - y0);	// sin(theta)
		if (j < LSPAN+2) ak[j] = al[j];

		// polar optimization (see Reinecke 2013)
		if ( (BLOCKSIZE == WARPSZE) && _all(HI_LLIM ? m - llim*y1 > max(80, llim>>7) : m-80 > llim*y1 ) ) return;

		ql += 2*(l + S*im);	// allow vector transforms where llim = lmax+1
		const double sgn = j - (j^1);	//	2*(j&1) - 1;	// -/+
		#pragma unroll
		for (int f=0; f<NFIELDS; f++) {
			double qer = (it < nlat_2) ? q[im*m_inc        + it            + (b*NFIELDS+f)*q_dist] : 0.0;	// north imag (ani)
			double t0  = (it < nlat_2) ? q[(nphi-im)*m_inc + it            + (b*NFIELDS+f)*q_dist] : 0.0;	// north real (an)
			double qor = (it < nlat_2) ? q[im*m_inc        + nlat_2*2-1-it + (b*NFIELDS+f)*q_dist] : 0.0;	// south imag (asi)
			double t1  = (it < nlat_2) ? q[(nphi-im)*m_inc + nlat_2*2-1-it + (b*NFIELDS+f)*q_dist] : 0.0;	// south real (as)
			double qei = t0-qer;		qer += t0;		// ani = -qei[lane+1],   bni = qei[lane-1]
			double qoi = t1-qor;		qor += t1;		// bsi = -qoi[lane-1],   asi = qoi[lane+1];
			t0 = shfl_xor(qei, 1);	// exchange between adjacent lanes.
			t1 = shfl_xor(qoi, 1);
			if (ROBERT_FORM) {
				double st_1 = rsqrt(1.0 - y0);		// 1/sin(theta)
				t0  *= st_1;	t1 *=  st_1;
				qer *= st_1;	qor *= st_1;
			}

			yl[(f*4+3)*l_inc +j] = (sgn*cost)*(t0 + t1);	// roi, exchange even and odd lanes
			yl[(f*4+2)*l_inc +j] = (qer - qor)*cost;		// ror
			yl[(f*4+1)*l_inc +j]   = sgn*(t0 - t1);	// rei, exchange evend and odd lanes
			yl[f*4*l_inc     +j] 		   = qer + qor;		// rer
		}

		const int ofs = (4*f0+(ll&3))*l_inc + j % (BLOCKSIZE/NW);
		if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }
			// transpose yl to my_reo (registers)
			#pragma unroll
			for (int k=0; k<NW; k++) {
				my_reo[k] = yl[ofs + k*(BLOCKSIZE/NW)];
			}

		cost = y0;		// cos(theta)^2
		y0 = mpos_scale;	// y0
		l = m - S;
		int ny = 0;
		int nsint = 0;
		do {		// sin(theta)^(m-S)
			if (l&1) {
				y0 *= y1;
				if (HI_LLIM) {
					ny += nsint;
					if (y0 < (SHT_ACCURACY+1.0/SHT_SCALE_FACTOR)) {
						ny--;
						y0 *= SHT_SCALE_FACTOR;
					}
				}
			}
			y1 *= y1;
			if (HI_LLIM) {
				nsint += nsint;
				if (y1 < 1.0/SHT_SCALE_FACTOR) {
					nsint--;
					y1 *= SHT_SCALE_FACTOR;
				}
			}
		} while(l >>= 1);
		if (it < nlat_2)     y0 *= ct[it + nlat_2];		// include quadrature weights.
		y1 = (ak[1]*cost + ak[0]) * y0;

		l=m;		al+=2;
		while (l <= llim) {
			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }
			#pragma unroll
			for (int k=0; k<LSPAN/2; k+=2) {		// compute a block of the matrix, write it in shared mem.
				double c0 = ak[2*k+3]*cost + ak[2*k+2];
				double c1 = ak[2*k+5]*cost + ak[2*k+4];
				if ((HI_LLIM) && (ny < 0)) {
					if (fabs(y0) > SHT_ACCURACY*SHT_SCALE_FACTOR + 1.0)
					{	// rescale when value is significant
						++ny;
						y0 *= 1.0/SHT_SCALE_FACTOR;
						y1 *= 1.0/SHT_SCALE_FACTOR;
					}
				}
				al += 4;
				yl[k*l_inc +j]     = (HI_LLIM && (ny<0)) ? 0.0 : y0;		// l and l+1
				yl[(k+1)*l_inc +j] = (HI_LLIM && (ny<0)) ? 0.0 : y1;		// l+2 and l+3
				y0 = c0 * y1 + y0;
				y1 = c1 * y0 + y1;
			}
			const bool y_not_zero = (HI_LLIM) ? _any(ny==0) : true;		// special case where all y are zero.

			if (BLOCKSIZE > WARPSZE) {	__syncthreads(); } else { _syncwarp; }

			if ((!HI_LLIM) || (y_not_zero)) {		// when all y are zero, we can skip this.
				// transposed work (at given l):
				const int NACC = 2;		// number of independent accumulators (2 is the sweetspot for V100).
				double qlri[NACC];		// accumulators

				const int itl = (ll>>2)*l_inc + (j % (BLOCKSIZE/NW));
				#pragma unroll
				for (int a=0; a<NACC; a++) {	// NACC independent accumulators
					qlri[a]   = my_reo[a]   * yl[itl + a*(BLOCKSIZE/NW)];
				}
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
						if ((!HI_LLIM) && (nlat_2 <= BLOCKSIZE)) {		// do we need atomic add or not ?
							ql[2*l+ll + (b*NFIELDS+f0)*ql_dist]   = qlri[0];
						} else {
							atomicAdd(ql+2*l+ll + (b*NFIELDS+f0)*ql_dist, qlri[0]);		// VERY slow atomic add on Kepler.
						}
					}
			}

			if (j<LSPAN) ak[j+2] = al[j];
			l+=LSPAN;
		}
	}
}

template<int S, int ignored, bool HI_LLIM=false>
static void ileg_m(shtns_cfg shtns, const double* q, double *ql, const int llim, int q_dist=0, int ql_dist=0)
{
	const int lmax = shtns->lmax;
	const int mres = shtns->mres;
	const int nlat_2 = shtns->nlat_2;
	const int nphi = shtns->nphi;
	int mmax = shtns->mmax;
	double *d_alm = shtns->d_clm;
	double *d_ct = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;

	const int BLOCKSIZE = WARPSZE;	// on V100, 32 is the best choice, by far.
	const int threadsPerBlock = BLOCKSIZE;	// can be from 32 to 1024, we should try to measure the fastest !
	const int blocksPerGrid = (nlat_2 + BLOCKSIZE - 1) / (BLOCKSIZE);

	if (q_dist == 0) q_dist = shtns->spat_stride;
	if (ql_dist == 0) ql_dist = shtns->nlm_stride;
	if (llim < mmax*mres) mmax = llim / mres;	// truncate mmax too !

	dim3 threads(threadsPerBlock, 1, 1);
	if ((shtns->howmany & 3) == 0) {	// number of transforms is a multiple of 4
		const int NFIELDS = 4;
		const int LSPAN_ = 16/NFIELDS;
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany/NFIELDS);
		if (S==1 && shtns->robert_form) {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		} else {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		}
	} else if ((shtns->howmany & 1) == 0) {	// even number of transforms
		const int NFIELDS = 2;
		const int LSPAN_ = 16/NFIELDS;
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany/NFIELDS);
		if (S==1 && shtns->robert_form) {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		} else {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		}
	} else {	// odd number of transforms
		const int NFIELDS = 1;
		const int LSPAN_ = 16/NFIELDS;
		dim3 blocks(blocksPerGrid, mmax+1, shtns->howmany/NFIELDS);
		if (S==1 && shtns->robert_form) {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		} else {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, HI_LLIM> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, lmax,mres, nphi, shtns->nlat_padded, shtns->mpos_scale_analys, q_dist, ql_dist);
		}
	}
}

template<int S, int ignored>
static void ileg_m0(shtns_cfg shtns, const double* q, double *ql, const int llim, int q_dist=0, int ql_dist=0)
{
	const int nlat_2 = shtns->nlat_2;
	double *d_alm = shtns->d_clm;
	double *d_ct = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;
	if (q_dist == 0) q_dist = shtns->spat_stride;
	if (ql_dist == 0) ql_dist = shtns->nlm_stride;

	if ((shtns->howmany & 1) == 0) {	// even number of transforms
		const int NFIELDS = 2;		// V100: best with NFIELDS=2
		const int BLOCKSIZE = 32;	// V100: best with BLOCKSIZE=32
		const int LSPAN_ = 16;		// V100: best with LSPAN_=16
		const int threadsPerBlock = BLOCKSIZE;
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE - 1) / (BLOCKSIZE);
		dim3 blocks(blocksPerGrid, 1, shtns->howmany/NFIELDS);
		dim3 threads(threadsPerBlock, 1, 1);
		if (S==1 && shtns->robert_form) {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, false, true, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, llim,0, 0, 0, shtns->mpos_scale_analys, q_dist, ql_dist);
		} else {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, llim,0, 0, 0, shtns->mpos_scale_analys, q_dist, ql_dist);
		}
	} else {	// odd number of transforms
		const int NFIELDS = 1;		// V100: best with NFIELDS=1
		const int BLOCKSIZE = 64;	// V100: best with BLOCKSIZE=64
		const int LSPAN_ = 32;		// V100: best with LSPAN_=32
		const int threadsPerBlock = BLOCKSIZE;	// can be from 32 to 1024, we should try to measure the fastest !
		const int blocksPerGrid = (nlat_2 + BLOCKSIZE - 1) / (BLOCKSIZE);
		dim3 blocks(blocksPerGrid, 1, shtns->howmany/NFIELDS);
		dim3 threads(threadsPerBlock, 1, 1);
		if (S==1 && shtns->robert_form) {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, false, true, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, llim,0, 0, 0, shtns->mpos_scale_analys, q_dist, ql_dist);
		} else {
			ileg_m_kernel<BLOCKSIZE, LSPAN_, S, NFIELDS, false, true> <<<blocks, threads, 0, stream>>>
				(d_alm, d_ct, (double*) q, (double*) ql, llim, nlat_2, llim,0, 0, 0, shtns->mpos_scale_analys, q_dist, ql_dist);
		}
	}
}


template<int S, int NFIELDS>
static void legendre(shtns_cfg shtns, const double *ql, double *q, const int llim, const int mmax, long spat_dist = 0)
{
	if (spat_dist == 0) spat_dist = shtns->spat_stride;
	if (mmax==0) {
		leg_m0<S,NFIELDS>(shtns, ql, q, llim, spat_dist);
	} else {
		if (llim <= SHT_L_RESCALE_FLY) {
			leg_m<S,NFIELDS>(shtns, ql, q, llim, mmax, spat_dist);
		} else {
			leg_m<S,NFIELDS,true>(shtns, ql, q, llim, mmax, spat_dist);
		}
	}
}

/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, int NFIELDS>
static void ilegendre(shtns_cfg shtns, const double *q, double* ql, const int llim, long spat_dist = 0)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;

	if (spat_dist == 0) spat_dist = shtns->spat_stride;
	cudaMemsetAsync(ql, 0, sizeof(double) * NFIELDS * shtns->nlm_stride * shtns->howmany, shtns->comp_stream);		// set to zero before we start.
	if (llim < mmax*mres) mmax = llim / mres;	// truncate mmax too !
	if (mmax==0) {
		ileg_m0<S, NFIELDS>(shtns, q, ql, llim, spat_dist, shtns->nlm_stride);
	} else
	if (llim <= SHT_L_RESCALE_FLY) {
		ileg_m<S, NFIELDS>(shtns, q, ql, llim, spat_dist, shtns->nlm_stride);
	} else {
		ileg_m<S, NFIELDS, true>(shtns, q, ql, llim, spat_dist, shtns->nlm_stride);
	}
}
