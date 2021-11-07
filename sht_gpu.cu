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

/* NOTES:
 * - the cuda transforms are NOT thread-safe. Use cushtns_clone() to clone transforms for each thread.
*/

/* TODO
 * 0) DYNAMIC THREADS/BLOCK, computed on init.
 * 1) use static polar optimization (from constant memory ?)
 * 2) use a for loop in m-direction to re-use threads at larger m's ?
 * 3) implement FFT on host versions. otpimize data transfer (don't transfer zeros): store as complex2real fft ?
 * 4) allow several variants, which may change occupancy for large sizes ?
 * 5) generalize NFIELDS to all kernels
 * 6) find optimal threads/block for minor kernels too (e.g. sphtor2scal)
 * 7) try to use transposed fft, to see if it is faster than my own FFT + transpose...   => No, it's not.
 */

/* Session with S. Chauveau from nvidia:
 * useful metrics = achieved_occupancy, cache_hit
 * for leg_m_lowllim, the "while(l<llim)" loop:
 * 		0) full al, ql load before the while loop.	=> DONE
 * 		1) reduce pointer update by moving ql and al updates into the "if" statement.	=> DONE
 * 	    2a) try to use a double-buffer (indexed by b, switched by b=1-b)		=> only 1 __syncthread() instead of 2.
 * 	OR:	2b) preload al, ql into registers => may reduce the waiting at __syncthreads()
 * 		3) unroll by hand the while loop (with an inner-loop of fixed size)		=> DONE
 * 		4) introduce NWAY (1 to 4) to avoid the need of several blocks in theta => one block for all means al, ql are read only once!	=> DONE
 * 				=> increases register pressure, but may be OK !
 */

// NOTE variables gridDim.x, blockIdx.x, blockDim.x, threadIdx.x, and warpSize are defined in device functions
/* NOTE:
 * 				KEPLER							PASCAL
 * cache-line:  128 bytes (16 doubles)
 * 
 * fetching 1 double/thread: 2 requests/warp
 */

#include "sht_private.h"

/// Maximum number of threads per block that should be used.
#define MAX_THREADS_PER_BLOCK 512
/// The warp size is always 32 on cuda devices (up to Ampere at least)
#define WARPSZE 32

#ifndef SHTNS_ISHIOKA
#error "GPU transform requires SHTNS_ISHIOKA"
#endif

#include "sht_gpu_kernels.cu"

enum cushtns_flags { CUSHT_OFF=0, CUSHT_ON=1, CUSHT_OWN_XFER_STREAM=4};

/* TOOL FUNCTIONS */

extern "C"
void* shtns_malloc(size_t size) {
	void* ptr = 0;
	cudaError_t err = cudaMallocHost(&ptr, size);		// try to allocate pinned memory (for faster transfers !)
	if (err != cudaSuccess) {
		cudaGetLastError();		// clears the error status.
		printf("!WARNING! [shtns_malloc] failed to alloc pinned memory. using regular memory instead.\n");
		ptr = VMALLOC(size);		// return regular memory instead...
	}
	return ptr;
}

extern "C"
void shtns_free(void* p) {
	cudaError_t err = cudaSuccess;
	if (p) err = cudaFreeHost(p);
	if (err != cudaSuccess) {
		cudaGetLastError();		// clears the error status.
		printf("!WARNING! [sntns_free] not page locked memory. trying regular free...\n");
		VFREE(p);
	}
}

void memzero_omp(double* mem, const size_t sze)
{
	#ifdef _OPENMP
	#pragma omp parallel
	{
		int i = omp_get_thread_num();
		int n = omp_get_num_threads();
		int ofs = (i*sze)/n;
		memset(mem + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
	}
	#else
		memset(mem, 0, sze*sizeof(double));
	#endif
}

void memzero_omp(double* mem, double* mem2, const size_t sze)
{
	#ifdef _OPENMP
	#pragma omp parallel
	{
		int i = omp_get_thread_num();
		int n = omp_get_num_threads();
		int ofs = (i*sze)/n;
		memset(mem + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
		memset(mem2 + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
	}
	#else
		memset(mem,  0, sze*sizeof(double));
		memset(mem2, 0, sze*sizeof(double));
	#endif
}

void memzero_omp(double* mem, double* mem2, double* mem3, const size_t sze)
{
	#ifdef _OPENMP
	#pragma omp parallel
	{
		int i = omp_get_thread_num();
		int n = omp_get_num_threads();
		int ofs = (i*sze)/n;
		memset(mem + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
		memset(mem2 + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
		memset(mem3 + ofs, 0, ((i+1)*sze/n - ofs)*sizeof(double));
	}
	#else
		memset(mem,  0, sze*sizeof(double));
		memset(mem2, 0, sze*sizeof(double));
		memset(mem3, 0, sze*sizeof(double));
	#endif
}


static void destroy_cuda_buffer_fft(shtns_cfg shtns)
{
	if (shtns->nphi > 1) cufftDestroy(shtns->cufft_plan);
	if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
	if (shtns->gpu_mem) cudaFree(shtns->gpu_mem);
	if (shtns->gpu_buf_out) cudaFree(shtns->gpu_buf_out);
	if (shtns->gpu_buf_in) cudaFree(shtns->gpu_buf_in);
	if (shtns->xfft_cpu) shtns_free(shtns->xfft_cpu);
}

#ifdef HAVE_LIBCUFFT
int cuda_gpu_id = 0;	// by default, use gpu device 0
#endif
#ifdef VKFFT_BACKEND
CUdevice vkfft_device_struct;
#endif

// WARNING! streams should be set BEFORE this routine is called!!
static int init_cuda_buffer_fft(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	int err_count = 0;

	err = cudaStreamCreateWithFlags(&shtns->xfer_stream, cudaStreamNonBlocking);		// stream for async data transfer.
	shtns->cu_flags |= CUSHT_OWN_XFER_STREAM;		// mark the transfer stream as managed by shtns.
	if (err != cudaSuccess)  err_count ++;

	/* cuFFT init */
	int nfft = shtns->nphi;
	//int nreal = 2*(nfft/2+1);
	if (nfft > 1) {
		// cufftPlanMany(cufftHandle *plan, int rank, int *n,   int *inembed, int istride, int idist,   int *onembed, int ostride, int odist,   cufftType type, int batch);
		cufftResult res = CUFFT_SUCCESS;
		if ((shtns->fft_mode & FFT_PHI_CONTIG_CPLX) && (nfft % 16 == 0) && (shtns->nlat_2 % 16 == 0)) {	// DEPRECATED: use the fastest data-layout for large sizes in CUFFT
			printf("!!! Use phi-contiguous FFT +transpose: WARNING, the spatial data is neither phi-contiguous nor theta-contiguous !!!\n");
			res = cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, 1, shtns->nphi, &nfft, 1, shtns->nphi, CUFFT_Z2Z, shtns->nlat_2);
			//cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, 1, shtns->nphi, &nreal, 1, shtns->nphi, CUFFT_D2Z, shtns->nlat);
		} else if (shtns->fft_mode & FFT_THETA_CONTIG) {
			printf("!!! Use theta-contiguous FFT on GPU !!!\n");
			int howmany = shtns->nlat_2 * shtns->howmany;		// support batched transforms
			int dist = shtns->nlat_padded / 2;
			res = cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, dist, 1, &nfft, dist, 1, CUFFT_Z2Z, howmany);
			#ifdef VKFFT_BACKEND
				VkFFTConfiguration config = {};		//zero-initialize configuration
				config.FFTdim = 2; //FFT dimension: 1D, but we use a second dimension to get non-unit strides.
				config.size[0] = howmany;
				config.size[1] = nfft;
				config.bufferStride[0] = dist;
				config.bufferStride[1] = dist * nfft;
				config.omitDimension[0] = 1;		// no FFT on the first dimension.
				config.doublePrecision = 1;
				if (2*(shtns->mmax+1) <= nfft) {	// let vkFFT perform the zero-padding (saves memory bandwidth)
					config.performZeropadding[1] = 1;
					config.frequencyZeroPadding = 1;
					config.fft_zeropad_left[1] = shtns->mmax + 1;			// first zero element
					config.fft_zeropad_right[1] = nfft - shtns->mmax;		// first non-zero element
				}
				//config.disableReorderFourStep = 1;		// avoids the use of temp buffer for large transforms at the cost of a mangled output.
				cuDeviceGet(&vkfft_device_struct, cuda_gpu_id);
				config.device = &vkfft_device_struct;
				config.stream = &shtns->comp_stream;
				config.num_streams = 1;
				VkFFTResult vk_res = initializeVkFFT(&shtns->vkfft_plan, config);

				const int ver = VkFFTGetVersion();
				printf("=> Using VkFFT v%d.%d.%d\n",ver/10000,(ver%10000)/100,ver%100);
				if (vk_res != VKFFT_SUCCESS) {
					printf("vkfft init FAILED with error code %d\n", vk_res);
					err_count ++;
				}
			#endif
		} else {
			printf("WARNING: layout not available on GPU.\n");
			err_count ++;
			return 1;
		}
		if (res != CUFFT_SUCCESS) {
			printf("cufft init FAILED with error code %d\n", res);
			err_count ++;
		}
		res = cufftSetStream(shtns->cufft_plan, shtns->comp_stream);	// select stream for cufft
		size_t worksize = 0;
		cufftGetSize(shtns->cufft_plan, &worksize);
		printf("cufft work-area size: %ld \t nlat*nphi = %ld\n", worksize/8, shtns->nlat * shtns->nphi);
	}

	// Allocate working arrays for SHT on GPU:
	double* gpu_mem = 0;
	const int nlm2 = shtns->nlm + (shtns->mmax+1);		// one more data per m
	const size_t nlm_stride = ((2*nlm2+WARPSZE-1)/WARPSZE) * WARPSZE;
	const size_t spat_stride = ((shtns->nlat_padded*shtns->nphi+WARPSZE-1)/WARPSZE) * WARPSZE;
	const size_t dual_stride = (spat_stride < nlm_stride) ? nlm_stride : spat_stride;		// we need two spatial buffers to also hold spectral data.
	const int howmany = shtns->howmany;		// batch size

	size_t sze = 2*nlm_stride;		// 2 spectral buffers
	if (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) {
		if (spat_stride > sze) sze = spat_stride;		// one spatial buffer for FFT -OR- 2 spectral buffers should fit in.
	}
	err = cudaMalloc( (void **)&shtns->gpu_buf_in,  sze*sizeof(double) * howmany );
	if (err != cudaSuccess)	err_count++;
	err = cudaMalloc( (void **)&shtns->gpu_buf_out, 2*dual_stride*sizeof(double) * howmany );		// 2 spatial -OR- 2 spectral
	if (err != cudaSuccess)	err_count++;

	err = cudaMalloc( (void **)&gpu_mem, (2*nlm_stride*howmany + 2*dual_stride + spat_stride)*sizeof(double) );		// maximum GPU memory required for SHT
	if (err != cudaSuccess)	err_count++;
	
	if (shtns->fft_mode & FFT_OOP) {
		// we also need a buffer on the CPU when the FFT is out-of-place:
		shtns->xfft_cpu = (double*) shtns_malloc(spat_stride * sizeof(double) * howmany);
	}

	shtns->nlm_stride = nlm_stride;
	shtns->spat_stride = dual_stride;
	shtns->gpu_mem = gpu_mem;

	return err_count;
}


extern "C"
void cushtns_release_gpu(shtns_cfg shtns)
{
	destroy_cuda_buffer_fft(shtns);
	// TODO: arrays possibly shared between different shtns_cfg should be deallocated ONLY if not used by other shtns_cfg.
	if (shtns->d_ct) cudaFree(shtns->d_ct);
	if (shtns->d_alm) cudaFree(shtns->d_alm);
	if (shtns->d_xlm) cudaFree(shtns->d_xlm);
	if (shtns->d_clm) cudaFree(shtns->d_clm);
	if (shtns->d_mx_stdt) cudaFree(shtns->d_mx_stdt);
	if (shtns->d_mx_van) cudaFree(shtns->d_mx_van);
	shtns->d_alm = 0;		// disable gpu.
	shtns->cu_flags = 0;
}

extern "C"
int cushtns_init_gpu(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	const long nlm = shtns->nlm;
	const long nlat_2 = shtns->nlat_2;

	double *d_alm = 0;
	double *d_ct  = 0;
	double *d_mx_stdt = 0;
	double *d_mx_van = 0;
	double *d_xlm = 0;
	double *d_clm = 0;
	int err_count = 0;
	int device_id = -1;

	cudaDeviceProp prop;
	cudaGetDevice(&device_id);
	err = cudaGetDeviceProperties(&prop, device_id);
	if (err != cudaSuccess) return -1;
	#if SHT_VERBOSE > 0
	printf("  cuda GPU #%d \"%s\" found (warp size = %d, compute capabilities = %d.%d).\n", device_id, prop.name, prop.warpSize, prop.major, prop.minor);
	#endif
	if (prop.warpSize != WARPSZE) return -1;		// failure, SHTns requires a warpSize of 32.
	if (prop.major < 3) return -1;			// failure, SHTns requires compute cap. >= 3 (warp shuffle instructions)

	// Allocate the device input vector alm
	err = cudaMalloc((void **)&d_alm, (2*nlm+MAX_THREADS_PER_BLOCK-1)*sizeof(double));	// allow some overflow.
	if (err != cudaSuccess) err_count ++;
	const long nlm0 = nlm_calc(LMAX+4, MMAX, MRES);
	err = cudaMalloc((void **)&d_clm, (nlm0+MAX_THREADS_PER_BLOCK-1)*sizeof(double));	// allow some overflow.
	if (err != cudaSuccess) err_count ++;
	err = cudaMalloc((void **)&d_xlm, (3*nlm0/2+MAX_THREADS_PER_BLOCK-1)*sizeof(double));	// allow some overflow.
	if (err != cudaSuccess) err_count ++;
	if (shtns->mx_stdt) {
		// Allocate the device matrix for d(sin(t))/dt
		err = cudaMalloc((void **)&d_mx_stdt, (2*nlm+MAX_THREADS_PER_BLOCK-1)*sizeof(double));
		if (err != cudaSuccess) err_count ++;
		// Same thing for analysis
		err = cudaMalloc((void **)&d_mx_van, (2*nlm+MAX_THREADS_PER_BLOCK-1)*sizeof(double));
		if (err != cudaSuccess) err_count ++;
	}
	// Allocate the device input vector cos(theta) and gauss weights
	err = cudaMalloc((void **)&d_ct, 2*nlat_2*sizeof(double));
	if (err != cudaSuccess) err_count ++;

	if (err_count == 0) {
		err = cudaMemcpy(d_alm, shtns->alm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)  err_count ++;
		err = cudaMemcpy(d_clm, shtns->clm, nlm0*sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)  err_count ++;
		err = cudaMemcpy(d_xlm, shtns->xlm, 3*nlm0/2*sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)  err_count ++;
		if (shtns->mx_stdt) {
			err = cudaMemcpy(d_mx_stdt, shtns->mx_stdt, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
			if (err != cudaSuccess)  err_count ++;
			err = cudaMemcpy(d_mx_van, shtns->mx_van, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
			if (err != cudaSuccess)  err_count ++;
		}
		err = cudaMemcpy(d_ct, shtns->ct, nlat_2*sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)  err_count ++;
		err = cudaMemcpy(d_ct + nlat_2, shtns->wg, nlat_2*sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)  err_count ++;
	}

	shtns->d_xlm = d_xlm;
	shtns->d_clm = d_clm;
	shtns->d_alm = d_alm;
	shtns->d_ct  = d_ct;
	shtns->d_mx_stdt = d_mx_stdt;
	shtns->d_mx_van = d_mx_van;

	err_count += init_cuda_buffer_fft(shtns);

	if (err_count != 0) {
		cushtns_release_gpu(shtns);
		return -1;	// fail
	}

	return device_id;		// success, return device_id
}

/// \internal Enables parallel transforms on selected GPU device, if available. \see shtns_use_gpu
extern "C"
int cushtns_use_gpu(int device_id)
{
	int count = 0;
	if (device_id >= 0) {
		cudaGetDeviceCount(&count);
		if (count > 0) {
			device_id = device_id % count;		// assign actual gpu in a round-robin fashion
			cudaSetDevice(device_id);
			cuda_gpu_id = device_id;
			return cuda_gpu_id;
		}
	}
	cuda_gpu_id = -1;
	return -1;		// disable gpu.
}

/// WARNING: cushtns_set_streams must be called BEFORE shtns_set_grid
extern "C"
void cushtns_set_streams(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	shtns->comp_stream = compute_stream;
	if (transfer_stream != 0) {
		if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
		shtns->xfer_stream = transfer_stream;
		shtns->cu_flags &= ~((int)CUSHT_OWN_XFER_STREAM);		// we don't manage this stream
	}
}

/*
extern "C"
shtns_cfg cushtns_clone(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	if (shtns->d_alm == 0) return 0;		// do not clone if there is no GPU associated...

	shtns_cfg sht_clone;
	sht_clone = shtns_create_with_grid(shtns, shtns->mmax, 0);		// copy the shtns_cfg, sharing all data.

	// set new buffer and cufft plan (should be unique for each shtns_cfg).
	int err_count = init_cuda_buffer_fft(sht_clone);
	if (err_count > 0) return 0;		// TODO: memory should be properly deallocated here...
	// set new streams (should also be unique).
	cushtns_set_streams(sht_clone, compute_stream, transfer_stream);
	return sht_clone;
}
*/

extern "C"
shtns_cfg cushtns_clone(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	shtns_cfg sht_clone;
	sht_clone = shtns_create_with_grid(shtns, shtns->mmax, 0);		// copy the shtns_cfg, sharing all data.

	int dev_id = cushtns_init_gpu(sht_clone);
	if (dev_id >= 0) {
		cushtns_set_streams(sht_clone, compute_stream, transfer_stream);
		return sht_clone;
	} else {
		shtns_destroy(sht_clone);
		return 0;		// fail
	}
}

void fourier_to_spat_gpu(shtns_cfg shtns, double* q, const int mmax)
{
	const int nphi = shtns->nphi;
	cufftResult res = CUFFT_SUCCESS;
	if (nphi > 1) {
		cufftDoubleComplex* x = (cufftDoubleComplex*) q;
		if (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) {
			double* xfft = shtns->gpu_buf_in;
			transpose_cplx_zero(shtns->comp_stream, (double*) x, xfft, shtns->nlat_2, nphi, mmax);		// zero out m>mmax during transpose
			res = cufftExecZ2Z(shtns->cufft_plan, (cufftDoubleComplex*) xfft, x, CUFFT_INVERSE);
		} else {	// THETA_CONTIGUOUS:
			#ifndef VKFFT_BACKEND
			if (2*(mmax+1) <= nphi) {
				const int nlat = shtns->nlat_padded;
				cudaMemsetAsync( q + (mmax+1)*nlat, 0, sizeof(double)*(nphi-2*mmax-1)*nlat, shtns->comp_stream );		// zero out m>mmax before fft
			}
			res = cufftExecZ2Z(shtns->cufft_plan, x, x, CUFFT_INVERSE);
			#else
				// rely on vkfft to avoid reading the unused Fourier modes.
				VkFFTLaunchParams launchParams = {};
				launchParams.buffer = (void**) &x;
				VkFFTAppend(&shtns->vkfft_plan, 1, &launchParams);
			#endif
		}
		if (res != CUFFT_SUCCESS) printf("cufft error %d\n", res);
	}
}

void spat_to_fourier_gpu(shtns_cfg shtns, double* q, const int mmax)
{
	const int nphi = shtns->nphi;
	cufftResult res = CUFFT_SUCCESS;
	if (nphi > 1) {
		cufftDoubleComplex *x = (cufftDoubleComplex*) q;
		if (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) {
			double* xfft = shtns->gpu_buf_in;
			res = cufftExecZ2Z(shtns->cufft_plan, x, (cufftDoubleComplex*) xfft, CUFFT_FORWARD);
			transpose_cplx_skip(shtns->comp_stream, xfft, (double*) x, nphi, shtns->nlat_2, mmax);		// ignore m > mmax during transpose
		} else {	// THETA_CONTIGUOUS:
			#ifndef VKFFT_BACKEND
			res = cufftExecZ2Z(shtns->cufft_plan, x, x, CUFFT_FORWARD);
			#else
				VkFFTLaunchParams launchParams = {};
				launchParams.buffer = (void**) &x;
				VkFFTAppend(&shtns->vkfft_plan, -1, &launchParams);
			#endif
		}
		if (res != CUFFT_SUCCESS) printf("cufft error %d\n", res);
	}
}

void spat_to_fourier_host(shtns_cfg shtns, double* q, double* qf)
{
	// FFT on host
	if (shtns->fft_mode != 0) {
		if ((shtns->fft_mode & FFT_PHI_CONTIG_SPLIT) == 0) {
			fftw_execute_dft(shtns->fftc, (fftw_complex *) q, (fftw_complex *) qf);
		} else {		// split dft
			printf("ERROR fft not supported\n");
		}
	}
}

void fourier_to_spat_host(shtns_cfg shtns, double* qf, double* q)
{
	if (shtns->fft_mode != 0) {
		if ((shtns->fft_mode & FFT_PHI_CONTIG_SPLIT) == 0) {
			fftw_execute_dft(shtns->ifftc, (fftw_complex *) qf, (fftw_complex *) q);
		} else {		// split dft
			printf("ERROR fft not supported\n");
		}
	}
}

/************************
 * TRANSFORMS ON DEVICE *
 ************************/ 


/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, int NFIELDS>
void cuda_SH_to_spat(shtns_cfg shtns, cplx* d_Qlm, double *d_Vr, const long int llim, const int mmax, long spat_dist = 0)
{
	static_assert(NFIELDS==1, "only NFIELDS=1 is supported in batch mode");
	//if (spat_dist == 0) spat_dist = shtns->spat_stride;

	cplx* d_qlm = d_Qlm;
		if (S==0) {
			d_qlm = (cplx*) shtns->gpu_buf_in;
			//for (int f=0; f<NFIELDS; f++)
			//	sh2ishioka_gpu(shtns, d_Qlm + f * shtns->nlm_stride, d_qlm + f * shtns->nlm_stride, llim, mmax, S);
			sh2ishioka_gpu(shtns, d_Qlm, d_qlm, llim, mmax, S);
		} else
	if (d_Vr == (double*) d_Qlm) { printf("ERROR: cuda_SH_to_spat must have distinct in and out fields");	exit(1); }
	legendre<S,NFIELDS>(shtns, (double*) d_qlm, d_Vr, llim, mmax, shtns->nlat);
	for (int f=0; f<NFIELDS; f++)  fourier_to_spat_gpu(shtns, d_Vr + f*spat_dist, mmax);	// in-place
}

/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, int NFIELDS>
void cuda_spat_to_SH(shtns_cfg shtns, double *d_Vr, cplx* d_Qlm, const long int llim, long spat_dist = 0)
{
	static_assert(NFIELDS==1, "only NFIELDS=1 is supported in batch mode");

	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	//if (spat_dist == 0) spat_dist = shtns->spat_stride;
	if (llim < mmax*mres)	mmax = llim / mres;		// truncate mmax too !

	for (int f=0; f<NFIELDS; f++) spat_to_fourier_gpu(shtns, d_Vr + f*spat_dist, mmax);

		if (S==0) {
			cplx* d_Qlm_ish = (cplx*) shtns->gpu_buf_in;
			ilegendre<S, NFIELDS>(shtns, d_Vr, (double*) d_Qlm_ish, llim, shtns->nlat);
			ishioka2sh_gpu(shtns, d_Qlm_ish, d_Qlm, llim, mmax, S);
			//for (int f=0; f<NFIELDS; f++)
			//	ishioka2sh_gpu(shtns, d_Qlm_ish + f * shtns->nlm_stride, d_Qlm + f * shtns->nlm_stride, llim, mmax, S);
			return;
		 } else
	{
		if (d_Vr == (double*) d_Qlm) { printf("ERROR: cuda_spat_to_SH must have distinct in and out fields");	exit(1); }
		ilegendre<S, NFIELDS>(shtns, d_Vr, (double*) d_Qlm, llim, shtns->nlat);
	}
}


extern "C"
void cu_SH_to_spat(shtns_cfg shtns, cplx* d_Qlm, double *d_Vr, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !
	cuda_SH_to_spat<0,1>(shtns, d_Qlm, d_Vr, llim, mmax);
}


extern "C"
void cu_SHsphtor_to_spat(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const long nlm_stride = shtns->nlm_stride;
	double* d_vwlm = shtns->gpu_buf_in;

	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	sphtor2scal_gpu(shtns, d_Slm, d_Tlm, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,1>(shtns, (cplx*) d_vwlm, d_Vt, llim+1, mmax);
	cuda_SH_to_spat<1,1>(shtns, (cplx*) (d_vwlm + nlm_stride), d_Vp, llim+1, mmax);
//	cuda_SH_to_spat<1,2>(shtns, (cplx*) d_vwlm, d_Vt, llim+1, mmax, d_Vp-d_Vt);
}

extern "C"
void cu_SHqst_to_spat(shtns_cfg shtns, cplx* d_Qlm, cplx* d_Slm, cplx* d_Tlm, double* d_Vr, double* d_Vt, double* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !

	cuda_SH_to_spat<0,1>(shtns, d_Qlm, d_Vr, llim, mmax);
	cu_SHsphtor_to_spat(shtns, d_Slm, d_Tlm, d_Vt, d_Vp, llim);
}


extern "C"
void cu_spat_to_SH(shtns_cfg shtns, double *d_Vr, cplx* d_Qlm, int llim)
{
	cuda_spat_to_SH<0,1>(shtns, d_Vr, d_Qlm, llim);
}

extern "C"
void cu_spat_to_SHsphtor(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, int llim)
{
	const long nlm_stride = shtns->nlm_stride;
	double* d_vwlm = shtns->gpu_buf_in;

	// SHT on the GPU
	cuda_spat_to_SH<1,1>(shtns, Vt, (cplx*) d_vwlm, llim+1);
	cuda_spat_to_SH<1,1>(shtns, Vp, (cplx*) (d_vwlm + nlm_stride), llim+1);
//	cuda_spat_to_SH<1,2>(shtns, Vt, (cplx*) d_vwlm, llim+1, Vp-Vt);
	if (CUDA_ERROR_CHECK) return;
	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), Slm, Tlm, llim);
	CUDA_ERROR_CHECK;
}


extern "C"
void cu_spat_to_SHqst(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, int llim)
{
	cuda_spat_to_SH<0,1>(shtns, Vr, Qlm, llim);
	cu_spat_to_SHsphtor(shtns, Vt,Vp, Slm,Tlm, llim);
}


/*******************************************************
 * TRANSFORMS OF HOST DATA, INCLUDING TRANSFERS TO GPU *
 *******************************************************/ 

extern "C"
void SH_to_spat_gpu(shtns_cfg shtns, cplx *Qlm, double *Vr, const long int llim)
{
	cudaError_t err = cudaSuccess;
	const int mres = shtns->mres;
	long nlm = shtns->nlm;
	int mmax = shtns->mmax;

	double *d_q   = shtns->gpu_buf_out;		// outer buffer for transfer (safe)
	double *d_qlm = d_q;		// "in-place" operation possible with ishioka

	if ((llim < mmax*mres) & (shtns->howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}

	// copy spectral data to GPU
	err = cudaMemcpy(d_qlm, Qlm, 2*nlm*sizeof(double) * shtns->howmany, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	cuda_SH_to_spat<0,1>(shtns, (cplx*) d_qlm, d_q, llim, mmax);	// start with Legendre, d_qlm may be available for Fourier.
	if (CUDA_ERROR_CHECK) return;

	// copy back spatial data
	err = cudaMemcpy(Vr, d_q, shtns->nspat * sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
}

extern "C"
void SH_to_spat_gpu_hostfft(shtns_cfg shtns, cplx *Qlm, double *Vr, const long int llim)
{
	cudaError_t err = cudaSuccess;
	double *d_qlm;
	double *d_q;
	const long nlat = shtns->nlat;
	const long nphi = shtns->nphi;
	long nlm = shtns->nlm;

	// get pointers to gpu buffers.
	d_qlm = shtns->gpu_mem;
	d_q = d_qlm + shtns->nlm_stride;
	
	double* VrF = Vr;
	if (shtns->fft_mode & FFT_OOP)	VrF = shtns->xfft_cpu;

	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}

	// copy spectral data to GPU
	err = cudaMemcpy(d_qlm, Qlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// Legendre transform on gpu
	legendre<0,1>(shtns, d_qlm, d_q, llim, mmax);

	// copy back spatial data (before FFT)
	if (12*(mmax+1) < 5*nphi) {
		// copy in two parts, to avoid copying zeros:
		memzero_omp(VrF + nlat*(mmax+1), nlat*(nphi-(2*mmax+1)));
		err = cudaMemcpy(VrF, d_q, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost);
		err = cudaMemcpy(VrF + nlat*(nphi-mmax), d_q + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost);
	} else {
		err = cudaMemcpy(VrF, d_q, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost);
	}
	if (CUDA_ERROR_CHECK) return;

	fourier_to_spat_host(shtns, VrF, Vr);
}

extern "C"
void spat_to_SH_gpu_hostfft(shtns_cfg shtns, double *Vr, cplx *Qlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	long nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const long nlat = shtns->nlat;
	const long nphi = shtns->nphi;

	// get pointers to gpu buffers.
	double* d_qlm = shtns->gpu_mem;
	double* d_q = d_qlm + shtns->nlm_stride;

	double *VrF = Vr;
	if (shtns->fft_mode & FFT_OOP)	VrF = shtns->xfft_cpu;
	spat_to_fourier_host(shtns, Vr, VrF);

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	// copy FFT data to GPU
	if (12*(mmax+1) < 5*nphi) {
		// copy in two parts, to avoid copying useless data:
		err = cudaMemcpy(d_q, VrF, nlat*(mmax+1)*sizeof(double), cudaMemcpyHostToDevice);
		err = cudaMemcpy(d_q + nlat*(nphi-mmax), VrF + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyHostToDevice);
	} else {
		err = cudaMemcpy(d_q, VrF, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice);
	}
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// Legendre transform on gpu
	ilegendre<0,1>(shtns, d_q, d_qlm, llim);

	if (nlm < shtns->nlm)	memset(Qlm + nlm, 0, (shtns->nlm - nlm)*2*sizeof(double));
	// copy back spectral data from GPU
	err = cudaMemcpy(Qlm, d_qlm, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
}


extern "C"
void SHsphtor_to_spat_gpu(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm = shtns->gpu_mem;
	double* d_vtp = d_vwlm + 2*nlm_stride*howmany;

	if ((llim < mmax*mres) && (howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	// transfer and convert on gpu
	double* d_Slm = 0;
	double* d_Tlm = 0;
	if (Slm) {
		d_Slm = d_vtp;
		err = cudaMemcpy(d_Slm, Slm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	}
	if (Tlm) {
		d_Tlm = d_vtp + nlm_stride * howmany;
		err = cudaMemcpy(d_Tlm, Tlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	}

	sphtor2scal_gpu(shtns, (cplx*) d_Slm, (cplx*) d_Tlm, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride*howmany), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,1>(shtns, (cplx*) d_vwlm, d_vtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht, cudaEventDisableTiming );
	cudaEventRecord(ev_sht, shtns->comp_stream);					// record the end of scalar SH (theta).

	cuda_SH_to_spat<1,1>(shtns, (cplx*) (d_vwlm + nlm_stride*howmany), d_vtp + spat_stride, llim+1, mmax);
	if (CUDA_ERROR_CHECK) return;

	cudaStreamWaitEvent(xfer_stream, ev_sht, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(Vt, d_vtp, nspat*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaEventDestroy(ev_sht);

	// copy back spatial data (phi)
	err = cudaMemcpy(Vp, d_vtp + spat_stride, nspat*sizeof(double), cudaMemcpyDeviceToHost);
	CUDA_ERROR_CHECK;
}

extern "C"
void SHsph_to_spat_gpu(shtns_cfg shtns, cplx *Slm, double *Vt, double *Vp, const long int llim)
{
	SHsphtor_to_spat_gpu(shtns, Slm, 0, Vt,Vp, llim);
}

extern "C"
void SHtor_to_spat_gpu(shtns_cfg shtns, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	SHsphtor_to_spat_gpu(shtns, 0, Tlm, Vt,Vp, llim);
}

extern "C"
void SHsphtor_to_spat_gpu_hostfft(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	cudaEvent_t ev_sht, ev_sht2;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm = shtns->gpu_mem;
	double* d_vtp = d_vwlm + 2*nlm_stride;
	double* VtF = Vt;
	double* VpF = Vp;
	if (shtns->fft_mode & FFT_OOP) {
		VtF = Vp;
		VpF = shtns->xfft_cpu;
	}

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	// transfer and convert on gpu
	cudaMemcpy(d_vtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (CUDA_ERROR_CHECK) return;
	cudaMemcpy(d_vtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (CUDA_ERROR_CHECK) return;

	sphtor2scal_gpu(shtns, (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	legendre<1,1>(shtns, d_vwlm, d_vtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht, cudaEventDisableTiming );
	cudaEventRecord(ev_sht, shtns->comp_stream);					// record the end of scalar SH (theta).

	legendre<1,1>(shtns, (d_vwlm + nlm_stride), d_vtp + spat_stride, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht2, cudaEventDisableTiming );
	cudaEventRecord(ev_sht2, shtns->comp_stream);					// record the end of scalar SH (phi).

	if (CUDA_ERROR_CHECK) return;

	cudaStreamWaitEvent(xfer_stream, ev_sht, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(VtF, d_vtp, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaMemcpyAsync(VtF + nlat*(nphi-mmax), d_vtp + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaEventRecord(ev_sht, shtns->xfer_stream);

	cudaStreamWaitEvent(xfer_stream, ev_sht2, 0);					// xfer stream waits for end of scalar SH (phi).
	cudaMemcpyAsync(VpF, d_vtp + spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaMemcpyAsync(VpF + nlat*(nphi-mmax), d_vtp + spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaEventRecord(ev_sht2, shtns->xfer_stream);

	memzero_omp(Vt + nlat*(mmax+1), Vp + nlat*(mmax+1), nlat*(nphi-(2*mmax+1)));

	cudaEventSynchronize(ev_sht);
	fourier_to_spat_host(shtns, VtF, Vt);
	cudaEventSynchronize(ev_sht2);
	fourier_to_spat_host(shtns, VpF, Vp);

	cudaEventDestroy(ev_sht2);
	cudaEventDestroy(ev_sht);
}

/*
extern "C"
void SHsphtor_to_spat_gpu2_hostfft(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht, ev_sht2;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm = shtns->gpu_mem;
	double* d_vtp = d_vwlm + 2*nlm_stride;
	double* VtF = Vt;
	double* VpF = Vp;
	if (shtns->fft_mode & FFT_OOP) {
		VtF = Vp;
		VpF = shtns->xfft_cpu;
	}

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	// transfer and convert on gpu
	err = cudaMemcpy(d_vtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpy(d_vtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	sphtor2scal_gpu(shtns, (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	legendre<1,2>(shtns, d_vwlm, d_vtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht, cudaEventDisableTiming );
	cudaEventRecord(ev_sht, shtns->comp_stream);					// record the end of scalar SH (theta+phi).

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(VtF, d_vtp, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaMemcpyAsync(VtF + nlat*(nphi-mmax), d_vtp + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaEventRecord(ev_sht, shtns->xfer_stream);

	cudaMemcpyAsync(VpF, d_vtp + spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaMemcpyAsync(VpF + nlat*(nphi-mmax), d_vtp + spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
	cudaEventCreateWithFlags(&ev_sht2, cudaEventDisableTiming );
	cudaEventRecord(ev_sht2, shtns->xfer_stream);

	memzero_omp(Vt + nlat*(mmax+1), Vp + nlat*(mmax+1), nlat*(nphi-(2*mmax+1)));

	cudaEventSynchronize(ev_sht);
	fourier_to_spat_host(shtns, VtF, Vt);
	cudaEventSynchronize(ev_sht2);
	fourier_to_spat_host(shtns, VpF, Vp);

	cudaEventDestroy(ev_sht2);
	cudaEventDestroy(ev_sht);
	CUDA_ERROR_CHECK;
}

extern "C"
void SHsphtor_to_spat_gpu2(shtns_cfg shtns, cplx *Slm, cplx *Tlm, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;

	double* d_vwlm;
	double* d_vtp;

	d_vwlm = shtns->gpu_mem;
	d_vtp = d_vwlm + 2*nlm_stride;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	// transfer and convert on gpu
	err = cudaMemcpy(d_vtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpy(d_vtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	sphtor2scal_gpu(shtns, (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,2>(shtns, (cplx*) d_vwlm, d_vtp, llim+1, mmax);		// Vt and Vp together  (merge with sphtor2scal_gpu)
	CUDA_ERROR_CHECK;
	cudaMemcpy(Vt, d_vtp, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Vp, d_vtp + spat_stride, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost);
	CUDA_ERROR_CHECK;
}
*/

extern "C"
void SHqst_to_spat_gpu(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_sht1, ev_up;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm = shtns->gpu_mem;
	double* d_vrtp = d_qvwlm + 2*nlm_stride*howmany;

	if ((llim < mmax*mres) && (howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm, Qlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_SH_to_spat<0,1>(shtns, (cplx*) d_qvwlm, d_vrtp + 2*spat_stride, llim, mmax);

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride*howmany, Tlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride*howmany), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride*howmany), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,1>(shtns, (cplx*) d_qvwlm, d_vrtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht1, cudaEventDisableTiming );
	cudaEventRecord(ev_sht1, comp_stream);					// record the end of scalar SH (theta).

	cuda_SH_to_spat<1,1>(shtns, (cplx*) (d_qvwlm + nlm_stride*howmany), d_vrtp + spat_stride, llim+1, mmax);

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht0, 0);					// xfer stream waits for end of scalar SH (radial).
	cudaMemcpyAsync(Vr, d_vrtp + 2*spat_stride, nspat * sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventDestroy(ev_sht0);

	cudaStreamWaitEvent(xfer_stream, ev_sht1, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(Vt, d_vrtp, nspat * sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventDestroy(ev_sht1);

	// copy back the last transform (compute stream).
	err = cudaMemcpy(Vp, d_vrtp + spat_stride, nspat * sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
}

/*
extern "C"
void SHqst_to_spat_gpu_hostfft(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_sht1, ev_sht2, ev_up;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm = shtns->gpu_mem;
	double* d_vrtp = d_qvwlm + 2*nlm_stride;
	
	double* VrF = Vr;
	double* VtF = Vt;
	double* VpF = Vp;
	if (shtns->fft_mode & FFT_OOP) {
		VrF = Vt;
		VtF = Vp;
		VpF = shtns->xfft_cpu;
	}

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm, Qlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	legendre<0,1>(shtns, d_qvwlm, d_vrtp + 2*spat_stride, llim, mmax);

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	legendre<1,1>(shtns, d_qvwlm, d_vrtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht1, cudaEventDisableTiming );
	cudaEventRecord(ev_sht1, comp_stream);					// record the end of scalar SH (theta).

	legendre<1,1>(shtns, (d_qvwlm + nlm_stride), d_vrtp + spat_stride, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht2, cudaEventDisableTiming );
	cudaEventRecord(ev_sht2, comp_stream);					// record the end of scalar SH (phi).

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht0, 0);					// xfer stream waits for end of scalar SH (radial).
	cudaMemcpyAsync(VrF, d_vrtp + 2*spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VrF + nlat*(nphi-mmax), d_vrtp + 2*spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventRecord(ev_sht0, xfer_stream);

	cudaStreamWaitEvent(xfer_stream, ev_sht1, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(VtF, d_vrtp, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VtF + nlat*(nphi-mmax), d_vrtp + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventRecord(ev_sht1, xfer_stream);

	cudaStreamWaitEvent(xfer_stream, ev_sht2, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(VpF, d_vrtp + spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VpF + nlat*(nphi-mmax), d_vrtp + spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventRecord(ev_sht2, xfer_stream);

	memzero_omp(Vr + nlat*(mmax+1), Vt + nlat*(mmax+1), Vp + nlat*(mmax+1), nlat*(nphi-(2*mmax+1)));

	cudaEventSynchronize(ev_sht0);
	fourier_to_spat_host(shtns, VrF, Vr);
	cudaEventSynchronize(ev_sht1);
	fourier_to_spat_host(shtns, VtF, Vt);
	cudaEventSynchronize(ev_sht2);
	fourier_to_spat_host(shtns, VpF, Vp);

	cudaEventDestroy(ev_sht0);
	cudaEventDestroy(ev_sht1);
	cudaEventDestroy(ev_up);
}

extern "C"
void SHqst_to_spat_gpu2_hostfft(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_sht1, ev_sht2, ev_up;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm = shtns->gpu_mem;
	double* d_vrtp = d_qvwlm + 2*nlm_stride;
	
	double* VrF = Vr;
	double* VtF = Vt;
	double* VpF = Vp;
	if (shtns->fft_mode & FFT_OOP) {
		VrF = Vt;
		VtF = Vp;
		VpF = shtns->xfft_cpu;
	}

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm, Qlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	legendre<0,1>(shtns, d_qvwlm, d_vrtp + 2*spat_stride, llim, mmax);

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	legendre<1,2>(shtns, d_qvwlm, d_vrtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht1, cudaEventDisableTiming );
	cudaEventRecord(ev_sht1, comp_stream);					// record the end of scalar SH (theta+phi).

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht0, 0);					// xfer stream waits for end of scalar SH (radial).
	cudaMemcpyAsync(VrF, d_vrtp + 2*spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VrF + nlat*(nphi-mmax), d_vrtp + 2*spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventRecord(ev_sht0, xfer_stream);

	cudaStreamWaitEvent(xfer_stream, ev_sht1, 0);					// xfer stream waits for end of scalar SH (theta).
	cudaMemcpyAsync(VtF, d_vrtp, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VtF + nlat*(nphi-mmax), d_vrtp + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventRecord(ev_sht1, xfer_stream);

	cudaMemcpyAsync(VpF, d_vrtp + spat_stride, nlat*(mmax+1)*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaMemcpyAsync(VpF + nlat*(nphi-mmax), d_vrtp + spat_stride + nlat*(nphi-mmax), nlat*mmax*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventCreateWithFlags(&ev_sht2, cudaEventDisableTiming );
	cudaEventRecord(ev_sht2, xfer_stream);

	memzero_omp(Vr + nlat*(mmax+1), Vt + nlat*(mmax+1), Vp + nlat*(mmax+1), nlat*(nphi-(2*mmax+1)));

	cudaEventSynchronize(ev_sht0);
	fourier_to_spat_host(shtns, VrF, Vr);
	cudaEventSynchronize(ev_sht1);
	fourier_to_spat_host(shtns, VtF, Vt);
	cudaEventSynchronize(ev_sht2);
	fourier_to_spat_host(shtns, VpF, Vp);

	cudaEventDestroy(ev_sht2);
	cudaEventDestroy(ev_sht1);
	cudaEventDestroy(ev_sht0);
	cudaEventDestroy(ev_up);
}


extern "C"
void SHqst_to_spat_gpu2(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_up;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm = shtns->gpu_mem;
	double* d_vrtp = d_qvwlm + 2*nlm_stride;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm, Qlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_SH_to_spat<0,1>(shtns, (cplx*) d_qvwlm, d_vrtp + 2*spat_stride, llim, mmax);

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride, Tlm, 2*nlm*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,2>(shtns, (cplx*) d_qvwlm, d_vrtp, llim+1, mmax);

	CUDA_ERROR_CHECK;

	cudaStreamWaitEvent(xfer_stream, ev_sht0, 0);					// xfer stream waits for end of scalar SH (radial).
	cudaMemcpyAsync(Vr, d_vrtp + 2*spat_stride, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	cudaEventDestroy(ev_sht0);

	cudaMemcpy(Vt, d_vrtp, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost);

	// copy back the last transform (compute stream).
	err = cudaMemcpy(Vp, d_vrtp + spat_stride, nlat*nphi*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
}
*/

extern "C"
void spat_to_SH_gpu(shtns_cfg shtns, double *Vr, cplx *Qlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	double *d_q   = shtns->gpu_buf_out;
	double *d_qlm = d_q;		// "in-place" operation possible

	// copy spatial data to GPU
	err = cudaMemcpy(d_q, Vr, shtns->nspat * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	cu_spat_to_SH(shtns, d_q, (cplx*) d_qlm, llim);
	CUDA_ERROR_CHECK;

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	int nlm = shtns->nlm;
	if ((llim < mmax*mres) && (shtns->howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Qlm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}
	// copy back spectral data
	err = cudaMemcpy(Qlm, d_qlm, 2*nlm*sizeof(double) * shtns->howmany, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
}


extern "C"
void spat_to_SHsphtor_gpu(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm;
	double* d_vtp;

	//err = cudaMalloc( (void **)&d_vwlm, (4*nlm_stride + 2*spat_stride)*sizeof(double) );
	d_vtp = shtns->gpu_mem;
	d_vwlm = d_vtp + 2*spat_stride;

	// copy spatial data to gpu
	err = cudaMemcpy(d_vtp, Vt, nspat*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_spat_to_SH<1,1>(shtns, d_vtp, (cplx*) d_vwlm, llim+1);

	err = cudaMemcpyAsync(d_vtp + spat_stride, Vp, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(shtns->comp_stream, ev_up, 0);					// compute stream waits for end of data transfer (phi).
	cuda_spat_to_SH<1,1>(shtns, d_vtp + spat_stride, (cplx*) (d_vwlm + nlm_stride*howmany), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride*howmany), (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride*howmany), llim);

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	int nlm = shtns->nlm;
	if ((llim < mmax*mres) && (howmany == 1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Slm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Tlm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}

	err = cudaMemcpy(Slm, d_vtp, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost);
	err = cudaMemcpy(Tlm, d_vtp+nlm_stride*howmany, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
//    cudaFree(d_vwlm);
//    cudaFreeHost(vw);
}

/*
extern "C"
void spat_to_SHsphtor_gpu2(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	const int nlm = shtns->nlm;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;

	double* d_vwlm;
	double* d_vtp;

	//err = cudaMalloc( (void **)&d_vwlm, (4*nlm_stride + 2*spat_stride)*sizeof(double) );
	d_vtp = shtns->gpu_mem;
	d_vwlm = d_vtp + 2*spat_stride;

	// copy spatial data to gpu
	err = cudaMemcpy(d_vtp, Vt, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_vtp + spat_stride, Vp, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_spat_to_SH<1,2>(shtns, d_vtp, (cplx*) d_vwlm, llim+1);

	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), llim);

	err = cudaMemcpy(Slm, d_vtp, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost);
	err = cudaMemcpy(Tlm, d_vtp+nlm_stride, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost);
}
*/

extern "C"
void spat_to_SHqst_gpu(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up, ev_up2, ev_sh2;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm;
	double* d_vrtp;

	// Allocate the device work vectors
//	err = cudaMalloc( (void **)&d_qvwlm, (5*nlm_stride + 3*spat_stride)*sizeof(double) );
	d_qvwlm = shtns->gpu_mem;
	d_vrtp = d_qvwlm + 2*nlm_stride*howmany;

	// copy spatial data to gpu
	err = cudaMemcpy(d_vrtp, Vt, nspat*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_spat_to_SH<1,1>(shtns, d_vrtp, (cplx*) d_qvwlm, llim+1);

	err = cudaMemcpyAsync(d_vrtp + spat_stride, Vp, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(comp_stream, ev_up, 0);			// compute stream waits for end of data transfer (phi).
	cuda_spat_to_SH<1,1>(shtns, d_vrtp + spat_stride, (cplx*) (d_qvwlm + nlm_stride*howmany), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride*howmany), (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride*howmany), llim);
	cudaEventCreateWithFlags(&ev_sh2, cudaEventDisableTiming );
	cudaEventRecord(ev_sh2, comp_stream);				// record the end of vector transform.

	err = cudaMemcpyAsync(d_vrtp + 2*spat_stride, Vr, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up2, cudaEventDisableTiming );
	cudaEventRecord(ev_up2, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(comp_stream, ev_up2, 0);		// compute stream waits for end of data transfer (phi).
	// scalar SHT on the GPU
	cuda_spat_to_SH<0,1>(shtns, d_vrtp + 2*spat_stride, (cplx*) d_qvwlm, llim);

	int mmax = shtns->mmax;
	int mres = shtns->mres;
	int nlm = shtns->nlm;
	if ((llim < mmax*mres) & (shtns->howmany ==1)) {
		mmax = llim / mres;	// truncate mmax too !
		nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
		memset(Slm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Tlm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
		memset(Qlm+nlm, 0, 2*(shtns->nlm - nlm)*sizeof(double));	// zero out on cpu (during the transform on GPU).
	}

	cudaStreamWaitEvent(xfer_stream, ev_sh2, 0);					// xfer stream waits for end of vector sht.
	err = cudaMemcpyAsync(Slm, d_vrtp, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost, xfer_stream);
	err = cudaMemcpyAsync(Tlm, d_vrtp+nlm_stride*howmany, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost, xfer_stream);

	err = cudaMemcpy(Qlm, d_qvwlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);	cudaEventDestroy(ev_up2);	cudaEventDestroy(ev_sh2);
//    cudaFree(d_qvwlm);
//    cudaFreeHost(vw);
}

/*
extern "C"
void spat_to_SHqst_gpu2(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up, ev_sh2;
	const int nlm = shtns->nlm;
	const int nlat = shtns->nlat;
	const int nphi = shtns->nphi;
	const long nlm_stride = shtns->nlm_stride;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm;
	double* d_vrtp;

	d_qvwlm = shtns->gpu_mem;
	d_vrtp = d_qvwlm + 2*nlm_stride;

	// copy Vt and Vp to gpu (async)
	err = cudaMemcpy(d_vrtp, Vt, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpy(d_vrtp + spat_stride, Vp, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cuda_spat_to_SH<1,2>(shtns, d_vrtp, (cplx*) d_qvwlm, llim+1);
	scal2sphtor_gpu(shtns, (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), llim);
	cudaEventCreateWithFlags(&ev_sh2, cudaEventDisableTiming );
	cudaEventRecord(ev_sh2, comp_stream);				// record the end of vector transform.

	// copy Vr to gpu
	err = cudaMemcpyAsync(d_vrtp + 2*spat_stride, Vr, nlat*nphi*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);				// record the end of data transfer.
	cudaStreamWaitEvent(comp_stream, ev_up, 0);			// compute stream waits for end of data transfer.
	// scalar SHT on the GPU
	cuda_spat_to_SH<0,1>(shtns, d_vrtp + 2*spat_stride, (cplx*) d_qvwlm, llim);

	// copy back
	cudaStreamWaitEvent(xfer_stream, ev_sh2, 0);					// xfer stream waits for end of vector sht.
	err = cudaMemcpyAsync(Slm, d_vrtp, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);
	err = cudaMemcpyAsync(Tlm, d_vrtp+nlm_stride, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost, xfer_stream);

	err = cudaMemcpy(Qlm, d_qvwlm, 2*nlm*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);	cudaEventDestroy(ev_sh2);
}
*/

void* fgpu[4][SHT_NTYP] = {
	{ (void*) SH_to_spat_gpu, (void*) spat_to_SH_gpu, (void*) SHsphtor_to_spat_gpu, (void*) spat_to_SHsphtor_gpu, (void*) SHsph_to_spat_gpu, (void*) SHtor_to_spat_gpu, (void*) SHqst_to_spat_gpu, (void*) spat_to_SHqst_gpu },
	{0}, //{ 0, 0, (void*) SHsphtor_to_spat_gpu2, (void*) spat_to_SHsphtor_gpu2, 0, 0, (void*) SHqst_to_spat_gpu2, (void*) spat_to_SHqst_gpu2 },
	{0}, //{ (void*) SH_to_spat_gpu_hostfft, (void*) spat_to_SH_gpu_hostfft, (void*) SHsphtor_to_spat_gpu_hostfft, 0, 0, 0, (void*) SHqst_to_spat_gpu2_hostfft, 0 },
	{0}, //{ 0, 0, (void*) SHsphtor_to_spat_gpu2_hostfft, 0, 0, 0, (void*) SHqst_to_spat_gpu2_hostfft, 0}
};
