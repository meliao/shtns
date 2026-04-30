###########################################################
# Authors :
#   - Máté Ferenc Nagy-Egri - 2021
#   - Sébastien Valat (ISTerre/CNRS) - 2026
###########################################################

###########################################################
function(shtns_build_library)
	# parse arguments
	set(options OPENMP CUDA HIP)
	set(oneValueArgs LIBNAME MARCH)
	set(multiValueArgs "")
	cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")

	# select mode
	set(build_mode STATIC)
	if (USE_SHARED)
		set(build_mode SHARED)
	endif()

	# build with hip fftw
	if (arg_HIP AND TARGET hip::hipfft)
		set(arg_HIPFFTW ON)
	endif()

	# build shtns library
	add_library(${arg_LIBNAME} ${build_mode}
		sht_init.c
		sht_kernels_a.c
		sht_kernels_s.c
		sht_fly.c
		sht_odd_nlat.c
		$<$<BOOL:${arg_OPENMP}>:sht_omp.c>
		$<$<BOOL:${arg_CUDA}>:sht_gpu.cu>
	)

	add_dependencies(${arg_LIBNAME}
		codelets
		sht_version
	)

	target_include_directories(${arg_LIBNAME}
		PRIVATE
			${CMAKE_CURRENT_SOURCE_DIR}
			${CMAKE_CURRENT_BINARY_DIR}
		PUBLIC
			${FFTW_INCLUDE_DIRS}
			$<$<BOOL:${arg_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>
			$<$<BOOL:${arg_HIP}>:${HIP_HOST_INCLUDE_DIRS}>
	)

	target_link_libraries(${arg_LIBNAME}
		PUBLIC
			${FFTW_LIBRARIES}
			$<$<BOOL:${arg_OPENMP}>:${FFTW_OMP_LIBRARIES}>
			$<$<BOOL:${arg_OPENMP}>:OpenMP::OpenMP_C>
			$<$<BOOL:${arg_CUDA}>:CUDA::cufft>
			$<$<BOOL:${arg_CUDA}>:CUDA::cudart>
			$<$<BOOL:${arg_CUDA}>:CUDA::nvrtc>
			$<$<BOOL:${arg_CUDA}>:CUDA::cuda_driver>
			$<$<BOOL:${arg_HIPFFTW}>:hip::hipfft>
			$<$<BOOL:${arg_HIP}>:hip::amdhip64>
			$<$<BOOL:${arg_HIP}>:hiprtc::hiprtc>
	)

	target_compile_definitions(${arg_LIBNAME}
		PRIVATE
			# HIP
			# hip_complex.h utility doesn't function without defining the back-end explicitly, something which the legacy CMake support set.
			$<$<BOOL:${arg_HIP}>:__HIP_PLATFORM_AMD__> # Bugfix
			$<$<BOOL:${arg_HIP}>:SHTNS_GPU=2>
			$<$<BOOL:${arg_HIP}>:HAVE_HIP=1>
			$<$<BOOL:${arg_HIPFFTW}>:HAVE_LIBHIPFFT=1>
			$<$<BOOL:${arg_HIP}>:VKFFT_BACKEND=2>
			# CUDA
			$<$<BOOL:${arg_CUDA}>:SHTNS_GPU=1>
			$<$<BOOL:${arg_CUDA}>:HAVE_CUDA=1>
			$<$<BOOL:${arg_CUDA}>:HAVE_LIBCUFFT=1>
			$<$<BOOL:${arg_CUDA}>:HAVE_LIBCUDART=1>
			$<$<BOOL:${arg_CUDA}>:VKFFT_BACKEND=1>
			_GNU_SOURCE
	)

	if(HAVE_LIBM)
		target_link_libraries(${arg_LIBNAME}
			PUBLIC
				m
		)
	endif()

	if (BUILD_FOR_PYTHON)
		set_target_properties(${arg_LIBNAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
	endif()

	if (arg_MARCH)
		target_compile_options(${arg_LIBNAME} PRIVATE ${arg_MARCH})
	endif()

	# Note: this should be done after all set target properties
	# because it imports them to build the new .o targs (-D & -I).
	if (arg_HIP)
		if (USE_HIP_CMAKE)
			target_sources(${arg_LIBNAME} PRIVATE sht_gpu.cu)
			set_source_files_properties(sht_gpu.cu PROPERTIES LANGUAGE HIP)
		else()
			shtns_hip_add_library(TARGET ${arg_LIBNAME} SOURCES sht_gpu.cu)
		endif()		
	endif()

	# install it
	install(
		TARGETS ${arg_LIBNAME}
		LIBRARY	DESTINATION ${CMAKE_INSTALL_LIBDIR}
	)
endfunction()
