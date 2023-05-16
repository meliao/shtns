/*
 * Copyright (c) 2010-2022 Centre National de la Recherche Scientifique.
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

/// Maximum number of threads per block that should be used.
#define MAX_THREADS_PER_BLOCK 256

// adjustment for cuda
#undef SHT_L_RESCALE_FLY
#define SHT_L_RESCALE_FLY 1800

// when possible, allows to fuse sh2ishioka into leg_m_kernel, reducing memory traffic
#define SHT_ALLOW_SH2ISH_FUSE 1
// use alternate ishioka conversion without shared mem, relying on cache (always faster on V100)
#define SHT_ISH_ALT 1


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
template<int BLOCK_DIM_Y, typename real=double> __global__ void
transpose_cplx_kernel(const real* in, real* out, const int dim0, const int dim1)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ real shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

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
template<int BLOCK_DIM_Y, typename real=double> __global__ void
transpose_cplx_zero_kernel(const real* in, real* out, const int dim0, const int dim1, const int mmax)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ real shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

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
template<int BLOCK_DIM_Y, typename real=double> __global__ void
transpose_cplx_skip_kernel(const real* in, real* out, const int dim0, const int dim1, const int mmax)
{
	const int TILE_DIM = WARPSZE/2;		// 16 double2 per warp, read as 32 doubles.
	__shared__ real shrdMem[TILE_DIM][TILE_DIM+1][2];		// avoid shared mem conflicts

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
transpose_cplx(cudaStream_t stream, const void* in, void* out, const int dim0, const int dim1, int sizeof_real = 8)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	if (sizeof_real==8)
		transpose_cplx_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1);
	else
		transpose_cplx_kernel<block_dim_y,float> <<<blocks, threads, 0, stream>>>((float*)in, (float*)out, dim0, dim1);
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_zero(cudaStream_t stream, const void* in, void* out, const int dim0, const int dim1, const int mmax, int sizeof_real = 8)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	if (sizeof_real==8)
		transpose_cplx_zero_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax);
	else
		transpose_cplx_zero_kernel<block_dim_y,float> <<<blocks, threads, 0, stream>>>((float*)in, (float*)out, dim0, dim1, mmax);
}

/// dim0, dim1 must be multiple of 16.
static void
transpose_cplx_skip(cudaStream_t stream, const double* in, double* out, const int dim0, const int dim1, const int mmax, int sizeof_real = 8)
{
	const int block_dim_y = 4;		// good performance with 4 (MUST be power of 2 between 1 and 16)
	dim3 blocks(dim0/16, dim1/16);
	dim3 threads(32, block_dim_y);
	if (sizeof_real==8)
		transpose_cplx_skip_kernel<block_dim_y> <<<blocks, threads, 0, stream>>>((double*)in, (double*)out, dim0, dim1, mmax);
	else
		transpose_cplx_skip_kernel<block_dim_y,float> <<<blocks, threads, 0, stream>>>((float*)in, (float*)out, dim0, dim1, mmax);
}


template<typename real> __global__ void
sh2ishioka_kernel_alt(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql, real* ql_ish,
		const int llim, const int lmax, const int mres, const int S, const int ql_dist=0, const int ql_ish_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;
	const int l  = ll >> 1;
	const int llim_m = llim-m;

	if (l>llim_m) return;		// nothing to do

	// first load matrix coefficients into registers
	const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4 + 3*(ll >> 2);
	real x0 = xlm[x_ofs + (ll&2)];
	real x1 = xlm[x_ofs + 1];

	// address calculation
	const int q_ofs = im*(((lmax+1+S)*2) -m+mres);
	const int b = (blockIdx.z*blockDim.z + threadIdx.z)*NFIELDS;
	ql     += q_ofs + b*ql_dist + ll;
	ql_ish += q_ofs + b*ql_ish_dist + ((im>0) ? ll : l);

	const bool write = (im>0 || (ll&1)==0);
	const bool add2 = ((l&1)==0) && (l+1 <llim_m);
	// loop over NFIELDS different fields
	for (int k=NFIELDS-1; k>=0; k--) {
		real q = ql[k*ql_dist] * x0;
		if (add2) {	// l-m even
			q += ql[k*ql_dist +4] * x1;		// contribution of l+2
		}
		if (write) {
			ql_ish[k*ql_ish_dist] = q;   // coalesced store -- for im=0, compacting real parts together without imaginary part (0)
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
template<typename real> __global__ void
ishioka2sh_kernel_alt(const int NFIELDS, const real* __restrict__ xlm, const real* __restrict__ ql_ish, real* ql,
	const int llim, const int lmax, const int mmax, const int mres, const int S, const int ql_ish_dist=0, const int ql_dist=0)
{
	const int im = blockIdx.y;
	const int ll = blockDim.x * blockIdx.x + threadIdx.x;
	const int m = im*mres;

	if ((ll>>1) > lmax+S-m) return;		// be sure to include zero-out for llim<l<=lmax AND zero-out for m>mmax

	// first load matrix coefficients into registers
	xlm += 3*im*(2*(lmax+4) -m+mres)/4;
	const int x_ofs = 3*(ll>>2);
	real x0 = xlm[x_ofs + (ll&2)];
	real x1;
	if (x_ofs>0) x1 = xlm[x_ofs-2];

	const int b = (blockIdx.z*blockDim.z + threadIdx.z) * NFIELDS;
	int q_ofs = ll;
	real q = 0.0;
	if (im==0) {
		ql_ish += b*ql_ish_dist + (ll>>1);
		ql += q_ofs + b*ql_dist;
		const bool read = (ll>>1) <= llim-m && ((ll&1)==0);
		const bool add2 = ((ll&2)==0) && (ll >= 4) && read;
		for (int k=NFIELDS-1; k>=0; k--) {
			if (read)  q = ql_ish[k*ql_ish_dist] * x0;	// only real part (ll&1 == 0)
			if (add2) {	// l-m even && real part (ll&3 == 0)
				q += ql_ish[k*ql_ish_dist -2] * x1;		// contribution of l-2
			}
			ql[k*ql_dist] = q;	// coalesced store
		}
	} else {
		q_ofs += im*(((lmax+1+S)*2) -m+mres);
		ql_ish += b*ql_ish_dist + q_ofs;
		ql += q_ofs + b*ql_dist;
		if (im<=mmax) {
			const bool read = (ll>>1) <= llim-m;
			const bool add2 = ((ll&2)==0) && (ll >= 4) && read;
			for (int k=NFIELDS-1; k>=0; k--) {
				if (read)  q = ql_ish[k*ql_ish_dist] * x0;
				if (add2) {	// l-m even
					q += ql_ish[k*ql_ish_dist -4] * x1;		// contribution of l-2
				}
				ql[k*ql_dist] = q;	// coalesced store
			}
		} else {
			for (int k=NFIELDS-1; k>=0; k--)  ql[k*ql_dist] = 0.0;	// coalesced store
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
	const int llim_m = llim-m;
	const int ofs = im*(((lmax+1)<<1) -m + mres) + ll;
	ll >>= 1;

	double v = 0.0;
	double w = 0.0;
	double mm = 0.0;
	if ( (ll >= 0) && (ll <= llim_m) ) {
		mm = mx[ofs];
		if (slm) v = slm[ofs + b*ql_dist];
		if (tlm) w = tlm[ofs + b*ql_dist];
	}
	M[j] = mm;
	sl[j] = v;
	tl[j] = w;

	__syncthreads();

	const double mimag = m * (j - (j^1));
	if ((j<blockDim.x-4) && (ll <= llim_m)) {
		double ml = M[j|1];
		double mu = M[(j|1)+1];
		v = mimag*tl[(j^1)+2]  +  (ml*v + mu*sl[j+4]);
		w = mimag*sl[(j^1)+2]  -  (ml*w + mu*tl[j+4]);
	}

	__syncthreads();

	const int j2 = j - (j>>1);	//(j>>1)+(j&1);
	if ((j&2)==0) {
		sl[j2] = v;
		tl[j2] = w;
	}

	const int x_ofs = (3*im*(2*(lmax+4) -m+mres)>>2) + 3*((l0+j) >> 2);

	__syncthreads();

	if ((j < blockDim.x-8) && (ll <= llim_m)) {
		double x0 = xlm[x_ofs + (j&2)];   //M[ix + (j&2)];	// ix for l-m even, ix+2 for l-m odd
		v *= x0;
		w *= x0;
		if ((j&2)==0) {		// for l-m even
			double x2 = xlm[x_ofs +1];   //M[ix+1];			// contribution of l+2
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
			double ml = M[2*(j>>1)];
			double mu = M[2*(j>>1)+3];
			double s = mimag*wl[(j+2)^1]  +  (ml*vl[j] + mu*vl[j+4]);
			double t = mimag*vl[(j+2)^1]  -  (ml*wl[j] + mu*wl[j+4]);
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
	const int l0 = (blockDim.x-4) * blockIdx.x;		// some overlap needed

	int l  = (l0 + j) >> 1;
	const int m = im*mres;
	const int q_ofs = im*(((lmax+1)*2) -m+mres);
	const int llim_m_p1 = llim+1-m;

	extern __shared__ double vl[];			// size blockDim.x
	double* const wl = vl + blockDim.x+2;		// size blockDim.x

	double v = 0.0;
	double w = 0.0;
	{
		if (l<=llim_m_p1) {
			const int x_ofs = 3*im*(2*(lmax+4) -m+mres)/4;
			double x = xlm[x_ofs + 3*(l>>1) + (j&2)];
			double x2 = (l>=2) ? xlm[x_ofs + 3*(l>>1) -2] : 0.0;
			if (im!=0) {
				const int i = q_ofs + 2*im + b*ql_ish_dist + l0+(j^1);		// xchg real and imag
				v = vlm[i] * x;
				w = wlm[i] * x;
				if (((j&2)==0) && (l>0)) {	// l-m even and l-m>2
					v += vlm[i-4] * x2;		// contribution of l-2
					w += wlm[i-4] * x2;
				}
			} else if (j&1) {
				const int i = q_ofs + b*ql_ish_dist + l;
				v = vlm[i] * x;
				w = wlm[i] * x;
				if (((j&2)==0) && (l>0)) {	// l-m even and l-m>2
					v += vlm[i-2] * x2;		// contribution of l-2
					w += wlm[i-2] * x2;
				}
			}
		}
		vl[(j^1)+2] = v;
		wl[(j^1)+2] = w;
	}
	if (l==0) {  vl[j] = 0.0;	wl[j] = 0.0; }


	l += m;
	double ml,mu, ll_1;
	if ((l <= llim) && (l>0)) {
		ll_1 = 1.0 / (l*(l+1));
		mu = mx[q_ofs + l0 + (j|1)];    //M[2*(j>>1)+3];
		ml = mx[q_ofs + l0 + (j|1) -3];	//M[2*(j>>1)+0];
	}

	__syncthreads();

	if ((j<blockDim.x-2) &&  (j >= ((blockIdx.x == 0) ? 0 : 2))) {
		double v2 = 0.0;
		double w2 = 0.0;
		if ((l <= llim) && (l>0)) {
			const double mimag = m * ((j^1) -j);
			v2 = mimag*w  +  (ml*vl[j] + mu*vl[j+4]);
			w2 = mimag*v  -  (ml*wl[j] + mu*wl[j+4]);
			v2 *= ll_1;
			w2 *= ll_1;
		}
		if (l <= lmax) {	// fill with zeros up to lmax (and l=0 too).
			slm[q_ofs+l0+j + b*ql_dist] = v2;
			tlm[q_ofs+l0+j + b*ql_dist] = w2;
		}
	}
}

void set_block_size_ish(int n_elem_x, int howmany_z, int& blksze_x, int& blksze_z, int &nblk_z)
{
	blksze_z = 1;		nblk_z = howmany_z;
	if (howmany_z % 4 == 0) {	blksze_z = 4;	nblk_z /= 4;  }
	else if (howmany_z % 3 == 0) {  blksze_z = 3;	nblk_z /= 3;  }
	else if (howmany_z % 2 == 0) {  blksze_z = 2;	nblk_z /= 2;  }

	if (blksze_z > 1) {
		blksze_x = (((n_elem_x+1)/2+WARPSZE-1)/WARPSZE) * WARPSZE;
		if (blksze_x > MAX_THREADS_PER_BLOCK/2) blksze_x = MAX_THREADS_PER_BLOCK/2;
	} else {
		blksze_x = ((n_elem_x+WARPSZE-1)/WARPSZE) * WARPSZE;
		if (blksze_x > MAX_THREADS_PER_BLOCK) blksze_x = MAX_THREADS_PER_BLOCK;
	}
}

template<typename real=double>
void sh2ishioka_gpu(shtns_cfg shtns, std::complex<real>* d_Qlm, std::complex<real>* d_Qlm_ish, int llim, int mmax, int S=0)
{
#ifndef SHT_ISH_ALT
	int blksze = (((llim+2)*2+WARPSZE-1)/WARPSZE) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(llim+3)+blksze-5)/(blksze-4), mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	sh2ishioka_kernel <<< blocks, threads,(blksze/4*7-3)*sizeof(double), shtns->comp_stream >>>
		(shtns->d_xlm, (double*) d_Qlm, (double*) d_Qlm_ish, llim, shtns->lmax, shtns->mres, S, shtns->spec_dist*2, shtns->nlm_stride);
#else
	int blksze, blksze_z, nblk_z;
	const int nelem_max = (llim+1+S)*2;
	int nfields = shtns->howmany;
	if (shtns->nlm > 256*1024 || nfields <= 5) {	// enough work to saturate the GPU with 1 z-block, or only few fields
		set_block_size_ish( nelem_max, 1, blksze, blksze_z, nblk_z);
	} else {
		nblk_z = nfields;	nfields=1;
		for (int k=4; k>=2; k--) {		// first factorization
			if (nblk_z % k == 0) { nblk_z /= k;	nfields=k;	break; }
		}
		set_block_size_ish( nelem_max, nblk_z, blksze, blksze_z, nblk_z);		// second factor into blocks
	}
	dim3 blocks((nelem_max+blksze-1)/blksze, mmax+1, nblk_z);
	dim3 threads(blksze, 1, blksze_z);
	const real* xlm = (real*) shtns->d_xlm;
	sh2ishioka_kernel_alt <<< blocks, threads, 0, shtns->comp_stream >>>
		(nfields, xlm, (real*) d_Qlm, (real*) d_Qlm_ish, llim, shtns->lmax, shtns->mres, S, shtns->spec_dist*2, shtns->nlm_stride);
#endif
	CUDA_ERROR_CHECK;
}

template<typename real=double>
void ishioka2sh_gpu(shtns_cfg shtns, std::complex<real>* d_Qlm_ish, std::complex<real>* d_Qlm, int llim, int mmax, int S=0)
{
#ifndef SHT_ISH_ALT
	int blksze = (((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE) * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-5)/(blksze-4), shtns->mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	ishioka2sh_kernel <<< blocks, threads, (blksze/4*7+3)*sizeof(double), shtns->comp_stream >>>
		(shtns->d_xlm, (double*) d_Qlm_ish, (double*) d_Qlm, llim, shtns->lmax, mmax, shtns->mres, S, shtns->nlm_stride, shtns->spec_dist*2);
#else
	int blksze, blksze_z, nblk_z;
	const int nelem_max = (shtns->lmax+1+S)*2;
	int nfields = shtns->howmany;
	if (shtns->nlm > 256*1024 || nfields <= 5) {	// enough work to saturate the GPU with 1 z-block, or only few fields
		set_block_size_ish( nelem_max, 1, blksze, blksze_z, nblk_z);
	} else {
		nblk_z = nfields;	nfields=1;
		for (int k=4; k>=2; k--) {		// first factorization
			if (nblk_z % k == 0) { nblk_z /= k;	nfields=k;	break; }
		}
		set_block_size_ish( nelem_max, nblk_z, blksze, blksze_z, nblk_z);		// second factor into blocks
	}

	dim3 blocks((nelem_max+blksze-1)/blksze, shtns->mmax+1, nblk_z);
	dim3 threads(blksze, 1, blksze_z);
	const real* xlm = (real*) shtns->d_xlm;
	ishioka2sh_kernel_alt <<< blocks, threads, 0, shtns->comp_stream >>>
		(nfields, xlm, (real*) d_Qlm_ish, (real*) d_Qlm, llim, shtns->lmax, mmax, shtns->mres, S, shtns->nlm_stride, shtns->spec_dist*2);
#endif
	CUDA_ERROR_CHECK;
}

void sphtor2scal_gpu(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, cplx* d_Vlm, cplx* d_Wlm, int llim, int mmax)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-9)/(blksze-8), mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	sphtor2ish_kernel <<< blocks, threads, blksze*3*sizeof(double), shtns->comp_stream >>>
		(shtns->d_mx_stdt, shtns->d_xlm, (double*) d_Slm, (double*) d_Tlm, (double*) d_Vlm, (double*) d_Wlm, llim, shtns->lmax, shtns->mres, shtns->spec_dist*2, shtns->nlm_stride);
	CUDA_ERROR_CHECK;
}

void scal2sphtor_gpu(shtns_cfg shtns, cplx* d_Vlm, cplx* d_Wlm, cplx* d_Slm, cplx* d_Tlm, int llim)
{
	size_t blksze = ((shtns->lmax+3)*2+WARPSZE-1)/WARPSZE * WARPSZE;
	if (blksze > MAX_THREADS_PER_BLOCK) blksze = MAX_THREADS_PER_BLOCK;
	dim3 blocks((2*(shtns->lmax+3)+blksze-5)/(blksze-4), shtns->mmax+1, shtns->howmany);
	dim3 threads(blksze, 1, 1);
	ish2sphtor_kernel <<< blocks, threads, (blksze+2)*2*sizeof(double), shtns->comp_stream >>>
		(shtns->d_mx_van, shtns->d_xlm, (double*) d_Vlm, (double*) d_Wlm, (double*)d_Slm, (double*)d_Tlm, llim, shtns->lmax, shtns->mres, shtns->nlm_stride, shtns->spec_dist*2);
	CUDA_ERROR_CHECK;
}

