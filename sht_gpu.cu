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

#ifndef SHTNS_ISHIOKA
#error "GPU transform requires SHTNS_ISHIOKA"
#endif

#include "sht_gpu_kernels.cu"

enum cushtns_flags { CUSHT_OFF=0, CUSHT_ON=1, CUSHT_OWN_XFER_STREAM=4};

/// include a compilable version of cuda_legendre.gen.cu (zero-terminated) :
const char *src_leg =
	#include "SHT/cuda_legendre.inc"
;

/* TOOL FUNCTIONS */

extern "C"
void* shtns_malloc(size_t size) {
	void* ptr = 0;
	cudaError_t err = cudaMallocHost(&ptr, size);		// try to allocate pinned memory (for faster transfers !)
	if (err != cudaSuccess) {
		cudaGetLastError();		// clears the error status.
		#if SHT_VERBOSE > 1
		printf("!WARNING! [shtns_malloc] failed to alloc pinned memory. using regular memory instead.\n");
		#endif
		ptr = VMALLOC(size);		// return regular memory instead...
	}
	return ptr;
}

extern "C"
void shtns_free(void* p) {
	if (p) {
		cudaError_t err = cudaFreeHost(p);
		if (err != cudaSuccess) {
			cudaGetLastError();		// clears the error status.
			#if SHT_VERBOSE > 1
			printf("!WARNING! [sntns_free] not page locked memory. trying regular free...\n");
			#endif
			VFREE(p);
		}
	}
}

static void destroy_cuda_buffer_fft(shtns_cfg shtns)
{
	#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
	if (shtns->nphi > 1) cufftDestroy(shtns->cufft_plan);
	#endif
	#ifdef VKFFT_BACKEND
	deleteVkFFT(&shtns->vkfft_plan);
	#endif
	if (shtns->gpu_buf_in) cudaFree(shtns->gpu_buf_in);
}

#define CACHE_LINE_GPU 128
inline void align_ptr(void** p, size_t ofs, uintptr_t align) {
	*p = (void*) ((((uintptr_t) (*p)) + (ofs+align-1)) &~ (align-1));
}

// WARNING! streams should be set BEFORE this routine is called!!
static int init_cuda_buffer_fft(shtns_cfg shtns, int cuda_gpu_id, int sizeof_real)
{
	cudaError_t err = cudaSuccess;
	int err_count = 0;

	/* GPU FFT init */
	int nfft = shtns->nphi;
	//int nreal = 2*(nfft/2+1);
	if (nfft > 1) {
		// cufftPlanMany(cufftHandle *plan, int rank, int *n,   int *inembed, int istride, int idist,   int *onembed, int ostride, int odist,   cufftType type, int batch);
		#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
			cufftResult res = CUFFT_SUCCESS;
		#endif
		if ((shtns->fft_mode & FFT_PHI_CONTIG_CPLX) && (nfft % 16 == 0) && (shtns->nlat_2 % 16 == 0)) {	// DEPRECATED: use the fastest data-layout for large sizes in CUFFT
			printf("!!! Use phi-contiguous FFT +transpose: WARNING, the spatial data is neither phi-contiguous nor theta-contiguous !!!\n");
			#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
				res = cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, 1, shtns->nphi, &nfft, 1, shtns->nphi, (sizeof_real==4) ? CUFFT_C2C : CUFFT_Z2Z, shtns->nlat_2);
			#else
				printf("WARNING: layout not available without cuFFT/rocFFT.\n");
				err_count ++;
				return 1;
			#endif
		} else if (shtns->fft_mode & FFT_THETA_CONTIG) {
			printf("!!! Use theta-contiguous FFT on GPU !!!\n");
			int howmany = shtns->nlat_2 * shtns->howmany;		// support batched transforms
			int dist = shtns->nlat_padded / 2;
			#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
				res = cufftPlanMany(&shtns->cufft_plan, 1, &nfft, &nfft, dist, 1, &nfft, dist, 1, (sizeof_real==4) ? CUFFT_C2C : CUFFT_Z2Z, howmany);
			#endif
			#ifdef VKFFT_BACKEND
				CUdevice vkfft_device_struct;
				VkFFTConfiguration config = {};		//zero-initialize configuration
				config.FFTdim = 2; //FFT dimension: 1D, but we use a second dimension to get non-unit strides.
				config.size[0] = howmany;
				config.size[1] = nfft;
				config.bufferStride[0] = dist;
				config.bufferStride[1] = dist * nfft;
				config.omitDimension[0] = 1;		// no FFT on the first dimension.
				config.doublePrecision = sizeof_real / 8;
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
		#if defined(HAVE_LIBCUFFT) || defined(HAVE_LIBROCFFT)
			if (res != CUFFT_SUCCESS) {
				printf("cufft init FAILED with error code %d\n", res);
				err_count ++;
			}
			res = cufftSetStream(shtns->cufft_plan, shtns->comp_stream);	// select stream for cufft
			size_t worksize = 0;
			cufftGetSize(shtns->cufft_plan, &worksize);
			#if SHT_VERBOSE > 1
				printf("cufft work-area size: %ld \t nlat*nphi = %d\n", worksize/sizeof_real, shtns->nlat * shtns->nphi);
			#endif
		#endif
	}

	// Allocate working arrays for SHT on GPU:
	const int howmany = shtns->howmany;		// batch size
	const int nlm2 = shtns->nlm + (shtns->mmax+1);		// one more data per m
	const size_t nlm_stride = ((2*nlm2+WARPSZE-1)/WARPSZE) * WARPSZE;
	const size_t spat_stride = ((shtns->nlat_padded*shtns->nphi+WARPSZE-1)/WARPSZE) * WARPSZE;
	const size_t dual_stride = (spat_stride < nlm_stride*howmany) ? nlm_stride*howmany : spat_stride;		// we need two spatial buffers to also hold spectral data.

	size_t sze = nlm_stride;		// 1 spectral buffer for scalar only ...
	if (shtns->mx_stdt) sze *= 2;	// ... 2 spectral buffer for vector transforms.
	if (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) {
		if (spat_stride > sze) sze = spat_stride;		// one spatial buffer for FFT -OR- 2 spectral buffers should fit in.
	}
	err = cudaMalloc( (void **)&shtns->gpu_buf_in,  sze * sizeof_real * howmany );
	if (err != cudaSuccess)	{	err_count++;	CUDA_ERROR_CHECK;  }

	shtns->nlm_stride = nlm_stride;
	shtns->spat_stride = dual_stride;

	return err_count;
}

/// allocate buffers on the GPU for copying data to and from CPU memory 
extern "C"
int init_gpu_staging_buffer(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	if (shtns->xfer_stream == 0) {
		err = cudaStreamCreateWithFlags(&shtns->xfer_stream, cudaStreamNonBlocking);		// stream for async data transfer.
		if (err != cudaSuccess)	{	CUDA_ERROR_CHECK;  return 1; }
		shtns->cu_flags |= CUSHT_OWN_XFER_STREAM;		// mark the transfer stream as managed by shtns.
	}

	double* gpu_mem = 0;
	size_t sze = shtns->spat_stride;		// 1 spatial or spectral buffer
	if (shtns->mx_stdt) sze *= 3;		// for vector transform: 3 spatial or spectral buffers.
	err = cudaMalloc( (void **)&gpu_mem, sze * shtns->sizeof_real );		// maximum GPU memory required for SHT auto-offloading
	if (err != cudaSuccess)	{	CUDA_ERROR_CHECK;  return 1; }
	shtns->gpu_staging_mem = gpu_mem;
	return 0;
}

void read_line_int(FILE* fp, int* val)
{
	char s[32];
	char* x = fgets(s, 30, fp);		// read line
	if (x) sscanf(x, "%d", val);		// convert to int
}

/// use some apriori metric to choose a good blocksize. An optimal one would require to measure.
static int optimize_nwarp(int* nwarp, int n_target, int nw, float loss_max, const bool div_by_2=false)
{
	float loss;
	int n = (div_by_2) ? *nwarp*2 : *nwarp+1;
	int nb = 0;
	do {
		n = (div_by_2) ? n/2 : n-1;
		nb = (n_target + n*nw-1)/(n*nw);	// number of block (should be minimum)
		loss = nb*n*nw / (float) n_target;
		if (SHT_VERBOSE > 1) printf("%d %d %f\n", n, nb, loss);
	} while (n>1 && loss>loss_max);		// either we found a good value, with less than 15% overhead due to large block size, or we reach n=1
	*nwarp = n;
	return nb;
}

int init_cuda_program(shtns_cfg shtns, const char* gpu_arch_target, const int sizeof_real = 8)
{
	const int nwarp_target = (shtns->nlat_2 + WARPSZE-1)/WARPSZE;		// number of 'warps' needed for nlat_2 points
	const bool hi_llim = (shtns->mmax > 0  &&  
		shtns->lmax > ((sizeof_real == 4) ? SHT_L_RESCALE_FLY_FLOAT : SHT_L_RESCALE_FLY));	// special rescaling needed
	bool sh2ish_fuse = (SHT_ALLOW_SH2ISH_FUSE  &&  !hi_llim);		// never fuse hi_llim
	int nwarp_s=4;		// 1 to 4 warps is a good choice on V100 for vector or when sh2ish is disabled. Usually, 4 is a bit better.
	int nwarp_a=1;		// 1 WARP is by far the best choice here, at least on V100
	const int nw_a=1;	// only one point per thread possible for analysis
	int nw_s=2;		int nf_s=1;			int nf_a=1;
	int lspan_a = 16;	// V100 and MI100: 16/nf_a works best (mmax>0)
#if WARPSZE == 32
	if (nwarp_target % 3 == 0) nw_s=3;	// if we need a multiple of 3, nw_s=3 is likely a bit better
	// adjust values (heuristics)
	if (shtns->howmany % 4 == 0) 	  {	nf_s=4;	nw_s=1;		nf_a=4;	}
	else if (shtns->howmany % 2 == 0) {	nf_s=2;	nw_s=2; 	nf_a=2;	}
	else if (shtns->howmany % 3 == 0) { nf_s=3; nw_s=1; 	nf_a=1;	}
#else
	const bool gfx90a = (strcmp(gpu_arch_target,"gfx90a") >= 0);	// MI200+
	if (shtns->howmany % 2 == 0) {	nf_a=2;		nf_s=2; }
	if (gfx90a) {	// MI200+
		nw_s=4;
		lspan_a = (nf_a > 1) ? 32 : 16;		// 16 for nf_a=1
		if (hi_llim  &&  nf_s==1  &&  shtns->howmany % 3 == 0)	nf_s=3;
		if (shtns->howmany % 4 == 0) { nf_a=4;		if (hi_llim) { nf_s=4;	nw_s=2; } }
	} else {	// assume MI100
		if (nwarp_target > 2  &&  !hi_llim)	nf_s=1;
	}
	if (shtns->howmany % 4 == 0  &&  nwarp_target == 1)	nf_s=4;
	if (hi_llim)	nwarp_s=1;
#endif
	if (shtns->mmax == 0) {
		lspan_a *= 2;
		sh2ish_fuse = false;	// don't fuse mmax=0
	}
	if (hi_llim  &&  nw_s > 2) nw_s=2;	// nw_s = 1 or 2 only with hi_llim
	lspan_a /= nf_a;

	#if SHT_VERBOSE > 1
	{	// override from sht_gpu.conf file
		FILE *fp = fopen("sht_gpu.conf", "r");
		if (fp) {
			printf("WARNING! defaults override from sht_gpu.conf\n");
			read_line_int(fp, &nwarp_s);	read_line_int(fp, &nf_s);	read_line_int(fp, &nw_s);
			read_line_int(fp, &nwarp_a);	read_line_int(fp, &nf_a);	read_line_int(fp, &lspan_a);
			fclose(fp);
		}
	}
	#endif

	// for analysis, simple:
	if (SHT_VERBOSE > 1) printf("optimize analysis:\n");
	optimize_nwarp(&nwarp_a, nwarp_target, nw_a, 1.14f);
	// for regular scalar synthesis (not fused) and vector synthesis
	if (SHT_VERBOSE > 1) printf("optimize vector synthesis:\n");
	optimize_nwarp(&nwarp_s, nwarp_target, nw_s, 1.14f);
	if (nw_s > 1  &&  nwarp_s == 1)	{
		if (SHT_VERBOSE > 1) printf("optimize NW synthesis:\n");
		optimize_nwarp(&nw_s, nwarp_target, nwarp_s, 1.3f);		// maybe we should reduce nw_s ?
	}

	int nwarp_s0=0;		int nblocks_s0=0;
	if (sh2ish_fuse) {
		// for scalar synthesis we should try to fuse sh2ish and leg_m_kernel for better performance.
		// this requires a larger blocksize (nwarp_s), up to MAX_THREADS_PER_BLOCK.
		nwarp_s0 = 8;		// start with maximum number of warps per block
		if (SHT_VERBOSE > 1) printf("optimize scalar synthesis:\n");
		nblocks_s0 = optimize_nwarp(&nwarp_s0, nwarp_target, nw_s, 1.14f);
		if (nblocks_s0 > 2) sh2ish_fuse = false;	// disable sh2ish_fuse, very likely slower or only marginally faster
		if (nw_s == 4) sh2ish_fuse = false;			// MI250
	}

	// also store into plan the kernel launch parameters:
	shtns->nwarp[0] = nwarp_s;		shtns->nwarp[1] = nwarp_a;		shtns->nwarp[2] = sh2ish_fuse ? nwarp_s0 : 0;
	shtns->gridDim_x[0] = (shtns->nlat_2 + nw_s*nwarp_s*WARPSZE-1)/(nw_s*nwarp_s*WARPSZE);
	shtns->gridDim_x[1] = (shtns->nlat_2 + nw_a*nwarp_a*WARPSZE-1)/(nw_a*nwarp_a*WARPSZE);
	shtns->gridDim_x[2] = sh2ish_fuse ? nblocks_s0 : 0;
	shtns->gridDim_y[0] = shtns->howmany / nf_s;
	shtns->gridDim_y[1] = shtns->howmany / nf_a;
	#if SHT_VERBOSE > 1
		printf("launch params: nblocks=(%d, %d, %d)\n", shtns->gridDim_x[0], shtns->gridDim_x[1], shtns->gridDim_x[2]);
	#endif

	const int sze_src = 100*1024;	// 100 KB
	char* const src = (char*) malloc(sze_src);
	// define what we need
	char* s = src;
	s += sprintf(s, "#define WARPSZE %d\n", WARPSZE);
	s += sprintf(s, "#define LMAX %d\n", shtns->lmax);
	s += sprintf(s, "#define MRES %d\n", shtns->mres);
	s += sprintf(s, "#define HI_LLIM %d\n", (hi_llim) ? 1 : 0);
	s += sprintf(s, "#define M0_ONLY %d\n", (shtns->mmax == 0) ? 1 : 0);
	s += sprintf(s, "#define ROBERT_FORM %d\n", shtns->robert_form);
	s += sprintf(s, "#define BLKSZE_S %d\n", nwarp_s*WARPSZE);
	s += sprintf(s, "#define BLKSZE_A %d\n", nwarp_a*WARPSZE);
	s += sprintf(s, "#define BLKSZE_SH2ISH %d\n", shtns->nwarp[2] * WARPSZE);	// 0 in case sh2ish is disabled
	s += sprintf(s, "#define NF_S %d\n", nf_s);
	s += sprintf(s, "#define NF_A %d\n", nf_a);
	s += sprintf(s, "#define LSPAN_A %d\n", lspan_a);
	s += sprintf(s, "#define NW_S %d\n", nw_s);
	s += sprintf(s, "#define MPOS_SCALE %g\n", shtns->mpos_scale_analys);
	s += sprintf(s, "#define NLAT_2 %d\n", shtns->nlat_2);
	if (sizeof_real == 4)
		 s += sprintf(s, "#define SHT_ACCURACY 1.0e-15f\n#define SHT_SCALE_FACTOR 7.2057594037927936e16f\ntypedef float real;\n");		// for single-precision
	else s += sprintf(s, "#define SHT_ACCURACY 1.0e-33\n#define SHT_SCALE_FACTOR 2.0370359763344860863e90\ntypedef double real;\n");	// for double-precision
	#if SHT_VERBOSE > 1
		printf("%s", src);		// displays the defines for debug purposes
	#endif

	// first look for file to read (allows quick changes without recompiling), otherwise use embedded kernel source.
	FILE *fp = fopen("SHT/cuda_legendre.gen.cu", "r");
	if (fp) {
		int k = fread(s, 1, sze_src-10-(s-src), fp);
		s[k]=0;	// zero-terminated
		fclose(fp);
	} else 	snprintf(s, sze_src-10-(s-src), "%s", src_leg);		// copy embedded kernel source
	//printf(src);

	nvrtcProgram prog;
	nvrtcResult rtc_res = nvrtcCreateProgram(&prog, src, "shtns.cu", 0, NULL, NULL);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcCreateProgram failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;	// fail
	}
	const char *ker_inst[] = {"leg_m_kernel<0>", "leg_m_kernel<1>", "ileg_m_kernel<0>", "ileg_m_kernel<1>"};
	for (int k=0; k<4; k++) {
		rtc_res = nvrtcAddNameExpression(prog,  ker_inst[k]);
		if (rtc_res != NVRTC_SUCCESS) {
			printf("ERROR nvrtcAddNameExpression(\"%s\") failed with error '%s'\n", ker_inst[k], nvrtcGetErrorString(rtc_res));
			return 1;	// fail
		}
	}

	// Compile
	char arch[64];
#if WARPSZE == 32
	#if CUDA_VERSION >= 11030
		snprintf(arch, 64, "-arch=sm%s", gpu_arch_target);		// compile for the current gpu
		const char *opts[] = {"-std=c++11", "-ftz=true", "-lineinfo", "--ptxas-options","-v", arch};
	#else
		snprintf(arch, 64, "-arch=compute%s", gpu_arch_target);		// compile for the current gpu
		const char *opts[] = {"-std=c++11", "-ftz=true", "-lineinfo", arch};
	#endif
#else
	snprintf(arch, 64, "--offload-arch=%s", gpu_arch_target);             // compile for the current gpu
	const char *opts[] = {"-std=c++11", "-O3", arch};
#endif
	#if SHT_VERBOSE > 1
		printf("compiling cuda kernels (lmax=%d, nlat=%d, nbatch=%d) for %s\n", shtns->lmax, shtns->nlat, shtns->howmany, gpu_arch_target);
	#endif
	rtc_res = nvrtcCompileProgram(prog, sizeof(opts)/sizeof(const char*), opts);
	if ((rtc_res != NVRTC_SUCCESS) || (SHT_VERBOSE > 1)) {		// show compile log in case of failure, or if verbose (debug) output required
		size_t sze = 0;
		nvrtcGetProgramLogSize (prog, &sze);
		char* log = (char*) malloc(sze);
		nvrtcGetProgramLog (prog, log);
		if (sze > 0) printf("%s", log);
		free(log);
	}
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcCompileProgram failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;	// fail
	}

	// Obtain PTX of the program.
	size_t sze;
	rtc_res = nvrtcGetPTXSize(prog, &sze);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcGetPTXSize failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;
	}
	char *ptx = (char*) malloc(sze);
	rtc_res = nvrtcGetPTX(prog, ptx);
	if (rtc_res != NVRTC_SUCCESS) {
		printf("\nERROR nvrtcGetPTX failed with error '%s'\n", nvrtcGetErrorString(rtc_res));
		return 1;
	}

	// Load the generated PTX module
	CUmodule module;
	CUresult cu_res = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	if (cu_res != CUDA_SUCCESS) {
		printf("\nERROR cuModuleLoadDataEx failed with error %d\n", cu_res);
		return 1;
	}

	// get the kernel pointers
	for (int k=0; k<4; k++) {
		const char *name;
		rtc_res = nvrtcGetLoweredName(prog, ker_inst[k], &name);
		if (rtc_res != NVRTC_SUCCESS) {
			printf("\nERROR nvrtcGetLoweredName(%s) failed with error '%s'\n", ker_inst[k], nvrtcGetErrorString(rtc_res));
			return 1;
		}
		CUfunction kernel;
		cu_res = cuModuleGetFunction(&kernel, module, name);
		if (cu_res != CUDA_SUCCESS) {
			printf("\nERROR cuModuleGetFunction(%s -> %s) failed with error %d\n", ker_inst[k], name, cu_res);
			return 1;
		}
		shtns->gpu_kernels[k] = kernel;
	}
	shtns->gpu_module = module;

	nvrtcDestroyProgram(&prog);		// no longer needed.
	free(src);
	return 0;	// success
}


extern "C"
void cushtns_release_gpu(shtns_cfg shtns)
{
	if (shtns->gpu_staging_mem) cudaFree(shtns->gpu_staging_mem);
	if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
	destroy_cuda_buffer_fft(shtns);
	// TODO: arrays possibly shared between different shtns_cfg should be deallocated ONLY if not used by other shtns_cfg.
	if (shtns->d_clm) cudaFree(shtns->d_clm);
	shtns->d_clm = 0;		// disable gpu.
	shtns->cu_flags = 0;
}

int gpu_upload_convert(void* dst, double* src, size_t n_real, int dst_sizeof_real)
{
	float* tmp_f = 0;
	if (dst_sizeof_real == 4) {		// convert from double to float
		tmp_f = (float*) malloc(n_real * sizeof(float));
		for (int i=0; i<n_real; i++) tmp_f[i] = src[i];
		src = (double*) &tmp_f[0];
	}
	cudaError_t err = cudaMemcpy(dst, src, n_real * dst_sizeof_real, cudaMemcpyHostToDevice);
	if (tmp_f) free(tmp_f);
	return (err != cudaSuccess);
}

extern "C"
int cushtns_init_gpu(shtns_cfg shtns)
{
	cudaError_t err = cudaSuccess;
	const long nlm = shtns->nlm;
	const long nlat_2 = shtns->nlat_2;
	const int sizeof_real = shtns->sizeof_real;

	void *buf = 0;
	double *d_ct  = 0;
	double *d_mx_stdt = 0;
	double *d_mx_van = 0;
	double *d_xlm = 0;
	double *d_x2lm = 0;
	double *d_clm = 0;
	int err_count = 0;
	int device_id = -1;

	cudaDeviceProp prop;
	cudaGetDevice(&device_id);
	err = cudaGetDeviceProperties(&prop, device_id);
	if (err != cudaSuccess) return -1;
	#if SHT_VERBOSE > 0
	#if SHTNS_GPU == 1
	printf("  cuda GPU #%d \"%s\" found (warp size = %d, compute capabilities = %d.%d).\n", device_id, prop.name, prop.warpSize, prop.major, prop.minor);
	char gpu_arch_target[16];
	sprintf(gpu_arch_target, "_%d", prop.major*10 + prop.minor);		// the gpu_arch we will compile for!
	#elif SHTNS_GPU == 2
	printf("  hip GPU #%d \"%s\" found (warp size = %d).\n", device_id, prop.gcnArchName, prop.warpSize);
	const char* gpu_arch_target = prop.gcnArchName;
	#endif
	#endif
	if (prop.warpSize != WARPSZE) return -1;		// failure, warpsize must be known at compile time (does it?).
	if (prop.major < 3) return -1;			// failure, SHTns requires compute cap. >= 3 (warp shuffle instructions)
	if (shtns->nlat % 4) return -1;			// failure, nlat must be a multiple of 4.

	const long nlm0 = nlm_calc(LMAX+4, MMAX, MRES);
	// Allocate the coefficients vectors alm, ...
	size_t sze = nlm0 + 3*nlm0/2 + 4*nlat_2  +  (CACHE_LINE_GPU/sizeof_real-1)*2;
	if (shtns->x2lm != shtns->xlm)  sze += 3*nlm0/2 +  (CACHE_LINE_GPU/sizeof_real-1);		// reserve space for x2lm
	if (shtns->mx_stdt) sze += ( 2*nlm + (CACHE_LINE_GPU/sizeof_real-1) ) * ((shtns->mx_van == shtns->mx_stdt) ? 1 : 2);
	sze += (3*nlm0/2 +1)/2 + (CACHE_LINE_GPU/sizeof_real-1);		// float buffers, in double units
	err = cudaMalloc(&buf, (sze + MAX_THREADS_PER_BLOCK-1)*sizeof_real);	// allow some overflow.
	if (err != cudaSuccess) err_count ++;
	if (err_count == 0) {
		d_clm = (double*) buf;		align_ptr(&buf, nlm0*sizeof_real,  CACHE_LINE_GPU);
		d_xlm = (double*) buf;		align_ptr(&buf, 3*nlm0/2 * sizeof_real, CACHE_LINE_GPU);
		err_count += gpu_upload_convert(d_clm, shtns->clm, nlm0, sizeof_real);
		err_count += gpu_upload_convert(d_xlm, shtns->xlm, 3*nlm0/2, sizeof_real);
		if (shtns->x2lm != shtns->xlm) {		// different arrays for Schmidt normalization
			d_x2lm = (double*) buf;		align_ptr(&buf, 3*nlm0/2 * sizeof_real, CACHE_LINE_GPU);
			err_count += gpu_upload_convert(d_x2lm, shtns->x2lm, 3*nlm0/2, sizeof_real);
		} else d_x2lm = d_xlm;

		if (shtns->mx_stdt) {
			d_mx_van = d_mx_stdt = (double*) buf;	align_ptr(&buf, 2*nlm*sizeof_real, CACHE_LINE_GPU);	// Allocate the device matrix for d(sin(t))/dt
			err_count += gpu_upload_convert(d_mx_stdt, shtns->mx_stdt, 2*nlm, sizeof_real);
			if (shtns->mx_stdt != shtns->mx_van) {		// may be the same array
				d_mx_van = (double*) buf;	align_ptr(&buf, 2*nlm*sizeof_real, CACHE_LINE_GPU);  // Same thing for analysis
				err_count += gpu_upload_convert(d_mx_van, shtns->mx_van, 2*nlm, sizeof_real);
			}
		}
		// Allocate the device input vector cos(theta) and gauss weights, sin(theta) and 1/sin(theta)
		d_ct = (double*) buf;			align_ptr(&buf, 4*nlat_2*sizeof_real, CACHE_LINE_GPU);

		err_count += gpu_upload_convert(d_ct, shtns->ct, nlat_2, sizeof_real);
		err_count += gpu_upload_convert(((char*)d_ct) +   nlat_2*sizeof_real, shtns->wg, nlat_2, sizeof_real);
		err_count += gpu_upload_convert(((char*)d_ct) + 2*nlat_2*sizeof_real, shtns->st, nlat_2, sizeof_real);
		err_count += gpu_upload_convert(((char*)d_ct) + 3*nlat_2*sizeof_real, shtns->st_1, nlat_2, sizeof_real);
	}

	shtns->d_xlm = d_xlm;
	shtns->d_x2lm = d_x2lm;
	shtns->d_clm = d_clm;
	shtns->d_ct  = d_ct;
	shtns->d_mx_stdt = d_mx_stdt;
	shtns->d_mx_van = d_mx_van;

	err_count += init_cuda_buffer_fft(shtns, device_id, sizeof_real);
	err_count += init_cuda_program(shtns, gpu_arch_target, sizeof_real);

	if (err_count != 0) {
		cushtns_release_gpu(shtns);
		return -1;	// fail
	}

	return device_id;		// success, return device_id
}

/// WARNING: cushtns_set_streams must be called BEFORE shtns_set_grid
extern "C"
void cushtns_set_streams(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	if (shtns->gpu_buf_in) {
		printf("[cushtns_set_streams] must be called before initializing shtns on GPU");
		exit(1);
	}
	shtns->comp_stream = compute_stream;
	if (transfer_stream != 0) {
		if (shtns->cu_flags & CUSHT_OWN_XFER_STREAM) cudaStreamDestroy(shtns->xfer_stream);
		shtns->xfer_stream = transfer_stream;
		shtns->cu_flags &= ~((int)CUSHT_OWN_XFER_STREAM);		// we don't manage this stream
	}
}

extern "C"
shtns_cfg cushtns_clone(shtns_cfg shtns, cudaStream_t compute_stream, cudaStream_t transfer_stream)
{
	shtns_cfg sht_clone = shtns_create_with_grid(shtns, shtns->mmax, 0);		// copy the shtns_cfg, sharing all data.
	if (sht_clone == 0) return 0;

	int err_count = 0;
	sht_clone->cu_flags = 0;	// reset
	sht_clone->gpu_buf_in = 0;
	sht_clone->gpu_staging_mem = 0;
	sht_clone->xfer_stream = 0;
	cushtns_set_streams(sht_clone, compute_stream, transfer_stream);
	cushtns_init_gpu(sht_clone);	// for now, we should copy everything again
	//err_count += init_cuda_buffer_fft(sht_clone);
	if (shtns->gpu_staging_mem) {
		err_count += init_gpu_staging_buffer(sht_clone);
	}
	if (err_count == 0) {
		return sht_clone;
	} else {
		shtns_destroy(sht_clone);
		return 0;		// fail
	}
}

void fourier_to_spat_gpu(shtns_cfg shtns, void* q, const int mmax, const int sizeof_real = 8)
{
	const int nphi = shtns->nphi;
	if (nphi > 1) {
	#ifndef VKFFT_BACKEND
		cufftResult res = CUFFT_SUCCESS;
		void* xfft = q;
		if (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) {
			xfft = shtns->gpu_buf_in;
			transpose_cplx_zero(shtns->comp_stream, q, xfft, shtns->nlat_2, nphi, mmax, sizeof_real);		// zero out m>mmax during transpose
		} else if (2*(mmax+1) <= nphi) {
			const int nlat = shtns->nlat_padded;
			cudaMemsetAsync( ((char*)q) + sizeof_real*(mmax+1)*nlat, 0, sizeof_real*(nphi-2*mmax-1)*nlat, shtns->comp_stream );		// zero out m>mmax before fft
		}
		res = (sizeof_real==8) ? cufftExecZ2Z(shtns->cufft_plan, (cufftDoubleComplex*) xfft, (cufftDoubleComplex*) q, CUFFT_INVERSE) :
								 cufftExecC2C(shtns->cufft_plan, (cufftComplex*)       xfft, (cufftComplex*)       q, CUFFT_INVERSE);
		if (res != CUFFT_SUCCESS) printf("[fourier_to_spat_gpu] cufft error %d\n", res);
	#else
		// VkFFT: always THETA_CONTIGUOUS
		// rely on vkfft to avoid reading the unused Fourier modes above shtns->mmax
		if (mmax < shtns->mmax) {	// some zero must be added, only if more than nominal
			const int nlat = shtns->nlat_padded;
			cudaMemsetAsync( ((char*)q) + sizeof_real*(mmax+1)*nlat, 0, sizeof_real*(nphi-2*mmax-1)*nlat, shtns->comp_stream );		// zero out m>mmax before fft
		}
		VkFFTLaunchParams launchParams = {};
		launchParams.buffer = (void**) &q;
		VkFFTAppend(&shtns->vkfft_plan, 1, &launchParams);
	#endif
	}
}

void spat_to_fourier_gpu(shtns_cfg shtns, void* q, const int mmax, const int sizeof_real = 8)
{
	const int nphi = shtns->nphi;
	if (nphi > 1) {
	#ifndef VKFFT_BACKEND
		cufftResult res = CUFFT_SUCCESS;
		void* xfft = (shtns->fft_mode & FFT_PHI_CONTIG_CPLX) ? shtns->gpu_buf_in : q;
		res = (sizeof_real==8) ? cufftExecZ2Z(shtns->cufft_plan, (cufftDoubleComplex*) q, (cufftDoubleComplex*) xfft, CUFFT_FORWARD) :
								 cufftExecC2C(shtns->cufft_plan, (cufftComplex*) q, (cufftComplex*) xfft, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS) printf("[spat_to_fourier_gpu] cufft error %d\n", res);
		if (q != xfft)
			transpose_cplx_skip(shtns->comp_stream, xfft, q, nphi, shtns->nlat_2, mmax, sizeof_real);		// ignore m > mmax during transpose
	#else
		// VkFFT: always THETA_CONTIGUOUS
		VkFFTLaunchParams launchParams = {};
		launchParams.buffer = (void**) &q;
		VkFFTAppend(&shtns->vkfft_plan, -1, &launchParams);
	#endif
	}
}

/************************
 * TRANSFORMS ON DEVICE *
 ************************/

static void legendre(shtns_cfg shtns, const int S, const void *ql, void *q, const int llim, const int mmax, long spat_dist = 0)
{
	int nlat_2 = shtns->nlat_2;
	int nphi = shtns->nphi;
	void *d_alm = shtns->d_clm;
	void *d_ct  = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;
	if (spat_dist == 0) spat_dist = shtns->spat_stride;

	const bool sh2ish_fuse = (SHT_ALLOW_SH2ISH_FUSE==1 && S==0 && shtns->nwarp[2]>0);
	int nlm_stride = (sh2ish_fuse) ? shtns->spec_dist*2 : shtns->nlm_stride;
	int par_idx = (sh2ish_fuse) ? 2 : 0;

	int llim_ = llim;
	void* params[11] = {&d_alm, &d_ct, &ql, &q, &llim_, &nlat_2, &nphi, &shtns->nlat_padded, &nlm_stride, &spat_dist, &shtns->d_xlm};
	cuLaunchKernel(shtns->gpu_kernels[S], 
			shtns->gridDim_x[par_idx], shtns->gridDim_y[0], mmax+1,		// grid dim
			shtns->nwarp[par_idx]*WARPSZE, 1, 1,					// block dim
			0, stream,								 // shared memory, stream
			params, 0);		// kernel params
}


/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int NFIELDS>
static void ilegendre(shtns_cfg shtns, const int S, const void *q, void* ql, const int llim, long spat_dist = 0)
{
	int mmax = shtns->mmax;
	int mres = shtns->mres;
	int nlat_2 = shtns->nlat_2;
	int nphi = shtns->nphi;
	double *d_alm = shtns->d_clm;
	double *d_ct = shtns->d_ct;
	cudaStream_t stream = shtns->comp_stream;

	int ql_dist = shtns->nlm_stride;
	if (spat_dist == 0) spat_dist = shtns->spat_stride;
	cudaMemsetAsync(ql, 0, shtns->sizeof_real * NFIELDS * shtns->nlm_stride * shtns->howmany, stream);		// set to zero before we start.

	if (llim < mmax*mres) mmax = llim / mres;	// truncate mmax too !

	int llim_ = llim;
	void* params[10] = {&d_alm, &d_ct, &q, &ql, &llim_, &nlat_2, &nphi, &shtns->nlat_padded, &spat_dist, &ql_dist};
	cuLaunchKernel(shtns->gpu_kernels[2+S], 		// analysis kernels
			shtns->gridDim_x[1], shtns->gridDim_y[1], mmax+1,		// grid dim
			shtns->nwarp[1]*WARPSZE, 1, 1,					// block dim
			0, stream,								 // shared memory, stream
			params, 0);		// kernel params
}


/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, int NFIELDS, typename real=double>
void cuda_SH_to_spat(shtns_cfg shtns, std::complex<real>* d_Qlm, real *d_Vr, const long int llim, const int mmax, long spat_dist = 0)
{
	static_assert(NFIELDS==1, "only NFIELDS=1 is supported in batch mode");
	//if (spat_dist == 0) spat_dist = shtns->spat_stride;

	std::complex<real>* d_qlm = d_Qlm;

		if (S==0  &&  (SHT_ALLOW_SH2ISH_FUSE==0 || shtns->nwarp[2]==0)) {
			d_qlm = (std::complex<real>*) shtns->gpu_buf_in;
			//for (int f=0; f<NFIELDS; f++)
			//	sh2ishioka_gpu(shtns, d_Qlm + f * shtns->nlm_stride, d_qlm + f * shtns->nlm_stride, llim, mmax, S);
			sh2ishioka_gpu(shtns, d_Qlm, d_qlm, llim, mmax, S);
		} else
	if (d_Vr == (real*) d_Qlm) { printf("ERROR: cuda_SH_to_spat must have distinct in and out fields");	exit(1); }
	legendre(shtns, S, (real*) d_qlm, d_Vr, llim, mmax, shtns->nlat);
	for (int f=0; f<NFIELDS; f++)  fourier_to_spat_gpu(shtns, d_Vr + f*spat_dist, mmax, sizeof(real));	// in-place
}

/// Perform SH transform on data that is already on the GPU. d_Qlm and d_Vr are pointers to GPU memory (obtained by cudaMalloc() for instance)
template<int S, int NFIELDS, typename real=double>
void cuda_spat_to_SH(shtns_cfg shtns, real *d_Vr, std::complex<real>* d_Qlm, const long int llim, long spat_dist = 0)
{
	static_assert(NFIELDS==1, "only NFIELDS=1 is supported in batch mode");

	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	//if (spat_dist == 0) spat_dist = shtns->spat_stride;
	if (llim < mmax*mres)	mmax = llim / mres;		// truncate mmax too !

	for (int f=0; f<NFIELDS; f++) spat_to_fourier_gpu(shtns, d_Vr + f*spat_dist, mmax, sizeof(real));

		if (S==0) {
			std::complex<real>* d_Qlm_ish = (std::complex<real>*) shtns->gpu_buf_in;
			ilegendre<NFIELDS>(shtns, S, d_Vr, (real*) d_Qlm_ish, llim, shtns->nlat);
			ishioka2sh_gpu(shtns, d_Qlm_ish, d_Qlm, llim, mmax, S);
			//for (int f=0; f<NFIELDS; f++)
			//	ishioka2sh_gpu(shtns, d_Qlm_ish + f * shtns->nlm_stride, d_Qlm + f * shtns->nlm_stride, llim, mmax, S);
			return;
		 } else
	{
		if (d_Vr == (real*) d_Qlm) { printf("ERROR: cuda_spat_to_SH must have distinct in and out fields");	exit(1); }
		ilegendre<NFIELDS>(shtns, S, d_Vr, (real*) d_Qlm, llim, shtns->nlat);
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
void cu_SH_to_spat_float(shtns_cfg shtns, cplx_f* d_Qlm, float *d_Vr, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	if (llim < mmax*mres)	mmax = llim / mres;	// truncate mmax too !
	if (shtns->sizeof_real != 4) { printf("ERROR: SHTns plan not prepared for float");	exit(1); }
	cuda_SH_to_spat<0,1,float>(shtns, d_Qlm, d_Vr, llim, mmax);
}


extern "C"
void cu_SHsphtor_to_spat(shtns_cfg shtns, cplx* d_Slm, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
{
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
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
void cu_SHsph_to_spat(shtns_cfg shtns, cplx* d_Slm, double* d_Vt, double* d_Vp, int llim)
{
	cu_SHsphtor_to_spat(shtns, d_Slm, 0, d_Vt, d_Vp, llim);
}

extern "C"
void cu_SHtor_to_spat(shtns_cfg shtns, cplx* d_Tlm, double* d_Vt, double* d_Vp, int llim)
{
	cu_SHsphtor_to_spat(shtns, 0, d_Tlm, d_Vt, d_Vp, llim);
}

extern "C"
void cu_spat_to_SH(shtns_cfg shtns, double *d_Vr, cplx* d_Qlm, int llim)
{
	cuda_spat_to_SH<0,1>(shtns, d_Vr, d_Qlm, llim);
}

extern "C"
void cu_spat_to_SH_float(shtns_cfg shtns, float *d_Vr, cplx_f* d_Qlm, int llim)
{
	if (shtns->sizeof_real != 4) { printf("ERROR: SHTns plan not prepared for float");	exit(1); }
	cuda_spat_to_SH<0,1,float>(shtns, d_Vr, d_Qlm, llim);
}

extern "C"
void cu_spat_to_SHsphtor(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, int llim)
{
	const long nlm_stride = shtns->nlm_stride * shtns->howmany;
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

	double *d_q   = shtns->gpu_staging_mem;		// buffer for transfer (safe)
	double *d_qlm = d_q;		// "in-place" operation possible with ishioka
	if (SHT_ALLOW_SH2ISH_FUSE == 1  &&  shtns->nwarp[2]>0) d_qlm = shtns->gpu_buf_in; // include sh2ishioka into legendre kernel

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (shtns->howmany == 1) nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}

	// copy spectral data to GPU
  if (shtns->sizeof_real == 4) {
	// convert
	float* tmp_f = (float*) malloc(2*nlm*sizeof(float) * shtns->howmany);
	for (int i=0; i<2*nlm*shtns->howmany; i++) tmp_f[i] = ((double*) Qlm)[i];
	// copy spectral data to GPU
	err = cudaMemcpy(d_qlm, tmp_f, 2*nlm*sizeof(float) * shtns->howmany, cudaMemcpyHostToDevice);
	free(tmp_f);
  } else
	err = cudaMemcpy(d_qlm, Qlm, 2*nlm*sizeof(double) * shtns->howmany, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	if (shtns->sizeof_real == 4) {
		cuda_SH_to_spat<0,1,float>(shtns, (cplx_f*) d_qlm, (float*)d_q, llim, mmax);	// start with Legendre, d_qlm may be available for Fourier.
	} else
	cuda_SH_to_spat<0,1>(shtns, (cplx*) d_qlm, d_q, llim, mmax);	// start with Legendre, d_qlm may be available for Fourier.
	if (CUDA_ERROR_CHECK) return;

	// copy back spatial data
	err = cudaMemcpy(Vr, d_q, shtns->nspat * shtns->sizeof_real, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
  if (shtns->sizeof_real == 4) {	// convert float to double in-place
	for (int i=shtns->nspat-1; i>=0; i--) 	Vr[i] = ((float*)Vr)[i];
  }
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
	const long nlm_stride = shtns->nlm_stride * howmany;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm = shtns->gpu_buf_in;
	double* d_vtp = shtns->gpu_staging_mem;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (howmany == 1) nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
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
		d_Tlm = d_vtp + nlm_stride;
		err = cudaMemcpy(d_Tlm, Tlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	}

	sphtor2scal_gpu(shtns, (cplx*) d_Slm, (cplx*) d_Tlm, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	if (Vt) {
		cuda_SH_to_spat<1,1>(shtns, (cplx*) d_vwlm, d_vtp, llim+1, mmax);
		if (Vp) {
			cudaEventCreateWithFlags(&ev_sht, cudaEventDisableTiming );
			cudaEventRecord(ev_sht, shtns->comp_stream);					// record the end of scalar SH (theta).
		}
	}
	if (Vp) {
		cuda_SH_to_spat<1,1>(shtns, (cplx*) (d_vwlm + nlm_stride), d_vtp + spat_stride, llim+1, mmax);
	}
	if (CUDA_ERROR_CHECK) return;

	if (Vt) {	// copy back spatial data (theta)
		if (Vp) {
			cudaStreamWaitEvent(xfer_stream, ev_sht, 0);					// xfer stream waits for end of scalar SH (theta).
			cudaMemcpyAsync(Vt, d_vtp, nspat*sizeof(double), cudaMemcpyDeviceToHost, shtns->xfer_stream);
			cudaEventDestroy(ev_sht);
		} else {
			err = cudaMemcpy(Vt, d_vtp, nspat*sizeof(double), cudaMemcpyDeviceToHost);
		}
	}
	if (Vp) {	// copy back spatial data (phi)
		err = cudaMemcpy(Vp, d_vtp + spat_stride, nspat*sizeof(double), cudaMemcpyDeviceToHost);
	}
	if (err != cudaSuccess) CUDA_ERROR_CHECK;
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
void SHqst_to_spat_gpu(shtns_cfg shtns, cplx *Qlm, cplx *Slm, cplx *Tlm, double *Vr, double *Vt, double *Vp, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_sht0, ev_sht1, ev_up;
	int nlm = shtns->nlm;
	int mmax = shtns->mmax;
	const int mres = shtns->mres;
	const int howmany = shtns->howmany;
	const long nspat = shtns->nspat;
	const long nlm_stride = shtns->nlm_stride * howmany;
	const long spat_stride = shtns->spat_stride;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm = shtns->gpu_buf_in;
	double* d_vrtp = shtns->gpu_staging_mem;

	if (llim < mmax*mres) {
		mmax = llim / mres;	// truncate mmax too !
		if (howmany == 1) nlm = nlm_calc( shtns->lmax, mmax, mres);		// transfer less data
	}
	/// 1) start scalar SH for radial component.
	err = cudaMemcpy(d_qvwlm + nlm_stride, Qlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	// SHT on the GPU
	cuda_SH_to_spat<0,1>(shtns, (cplx*) (d_qvwlm+nlm_stride), d_vrtp + 2*spat_stride, llim, mmax);		// may use gpu_buf_in = d_qvwlm internally

	// OR transfer and convert on gpu
	err = cudaMemcpyAsync(d_vrtp, Slm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	err = cudaMemcpyAsync(d_vrtp + nlm_stride, Tlm, 2*nlm*sizeof(double) * howmany, cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	cudaEventCreateWithFlags(&ev_sht0, cudaEventDisableTiming );
	cudaEventRecord(ev_sht0, comp_stream);					// record the end of scalar SH (radial).
	cudaEventCreateWithFlags(&ev_up, cudaEventDisableTiming );
	cudaEventRecord(ev_up, xfer_stream);			// record the end of upload
	cudaStreamWaitEvent(comp_stream, ev_up, 0);				// compute stream waits for end of transfer.

	sphtor2scal_gpu(shtns, (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), llim, mmax);

	// SHT on the GPU
	cuda_SH_to_spat<1,1>(shtns, (cplx*) d_qvwlm, d_vrtp, llim+1, mmax);
	cudaEventCreateWithFlags(&ev_sht1, cudaEventDisableTiming );
	cudaEventRecord(ev_sht1, comp_stream);					// record the end of scalar SH (theta).

	cuda_SH_to_spat<1,1>(shtns, (cplx*) (d_qvwlm + nlm_stride), d_vrtp + spat_stride, llim+1, mmax);

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

extern "C"
void spat_to_SH_gpu(shtns_cfg shtns, double *Vr, cplx *Qlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	double *d_q   = shtns->gpu_staging_mem;
	double *d_qlm = d_q;		// "in-place" operation possible

	// copy spatial data to GPU
	if (shtns->sizeof_real == 4) {		// convert double to float
		float* tmp_f = (float*) malloc(sizeof(float)*shtns->nspat);
		for (int i=0; i<shtns->nspat; i++) tmp_f[i] = Vr[i];
		err = cudaMemcpy(d_q, tmp_f, shtns->nspat * sizeof(float), cudaMemcpyHostToDevice);
		free(tmp_f);
	} else
	err = cudaMemcpy(d_q, Vr, shtns->nspat * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }

	// SHT on the GPU
	if (shtns->sizeof_real == 4) {
		cu_spat_to_SH_float(shtns, (float*) d_q, (cplx_f*) d_qlm, llim);
	} else
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
	err = cudaMemcpy(Qlm, d_qlm, 2*nlm * shtns->howmany * shtns->sizeof_real, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	if (shtns->sizeof_real == 4) {	// convert float to double in-place
		for (int i=2*nlm*shtns->howmany-1; i>=0; i--)	((double*)Qlm)[i] = ((float*)Qlm)[i];
	}
}


extern "C"
void spat_to_SHsphtor_gpu(shtns_cfg shtns, double *Vt, double *Vp, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up;
	const long nspat = shtns->nspat;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	const long nlm_stride = shtns->nlm_stride * howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;

	double* d_vwlm;
	double* d_vtp;

	d_vtp = shtns->gpu_staging_mem;
	d_vwlm = shtns->gpu_buf_in;

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
	cuda_spat_to_SH<1,1>(shtns, d_vtp + spat_stride, (cplx*) (d_vwlm + nlm_stride), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_vwlm, (cplx*) (d_vwlm+nlm_stride), (cplx*) d_vtp, (cplx*) (d_vtp+nlm_stride), llim);

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
	err = cudaMemcpy(Tlm, d_vtp+nlm_stride, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);
}

extern "C"
void spat_to_SHqst_gpu(shtns_cfg shtns, double *Vr, double *Vt, double *Vp, cplx *Qlm, cplx *Slm, cplx *Tlm, const long int llim)
{
	cudaError_t err = cudaSuccess;
	cudaEvent_t ev_up, ev_up2, ev_sh2;
	const long nspat = shtns->nspat;
	const long spat_stride = shtns->spat_stride;
	const int howmany = shtns->howmany;
	const long nlm_stride = shtns->nlm_stride * howmany;
	cudaStream_t xfer_stream = shtns->xfer_stream;
	cudaStream_t comp_stream = shtns->comp_stream;

	double* d_qvwlm;
	double* d_vrtp;

	d_qvwlm = shtns->gpu_buf_in;
	d_vrtp = shtns->gpu_staging_mem;	//d_qvwlm + 2*nlm_stride;

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
	cuda_spat_to_SH<1,1>(shtns, d_vrtp + spat_stride, (cplx*) (d_qvwlm + nlm_stride), llim+1);
	CUDA_ERROR_CHECK;

	scal2sphtor_gpu(shtns, (cplx*) d_qvwlm, (cplx*) (d_qvwlm+nlm_stride), (cplx*) d_vrtp, (cplx*) (d_vrtp+nlm_stride), llim);
	cudaEventCreateWithFlags(&ev_sh2, cudaEventDisableTiming );
	cudaEventRecord(ev_sh2, comp_stream);				// record the end of vector transform.

	err = cudaMemcpyAsync(d_vrtp + 2*spat_stride, Vr, nspat*sizeof(double), cudaMemcpyHostToDevice, xfer_stream);
	if (err != cudaSuccess) { CUDA_ERROR_CHECK;	return; }
	cudaEventCreateWithFlags(&ev_up2, cudaEventDisableTiming );
	cudaEventRecord(ev_up2, xfer_stream);				// record the end of scalar SH (theta).
	cudaStreamWaitEvent(comp_stream, ev_up2, 0);		// compute stream waits for end of data transfer (phi).
	// scalar SHT on the GPU
	cuda_spat_to_SH<0,1>(shtns, d_vrtp + 2*spat_stride, (cplx*) (d_qvwlm+nlm_stride), llim);	// uses gpu_buf_in == d_qvwlm internally

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
	err = cudaMemcpyAsync(Tlm, d_vrtp+nlm_stride, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost, xfer_stream);

	err = cudaMemcpy(Qlm, d_qvwlm+nlm_stride, 2*nlm*sizeof(double) * howmany, cudaMemcpyDeviceToHost);

	cudaEventDestroy(ev_up);	cudaEventDestroy(ev_up2);	cudaEventDestroy(ev_sh2);
}

void* fgpu[4][SHT_NTYP] = {
	{ (void*) SH_to_spat_gpu, (void*) spat_to_SH_gpu, (void*) SHsphtor_to_spat_gpu, (void*) spat_to_SHsphtor_gpu, (void*) SHsph_to_spat_gpu, (void*) SHtor_to_spat_gpu, (void*) SHqst_to_spat_gpu, (void*) spat_to_SHqst_gpu },
	{0}, //{ 0, 0, (void*) SHsphtor_to_spat_gpu2, (void*) spat_to_SHsphtor_gpu2, 0, 0, (void*) SHqst_to_spat_gpu2, (void*) spat_to_SHqst_gpu2 },
	{0}, // former "hostfft" transforms
	{0}, // former "hostfft" transforms
};
