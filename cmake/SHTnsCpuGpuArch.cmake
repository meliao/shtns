###########################################################
function(shtns_config_cpu_gpu_march)
	# set default
	set(SHTNS_CPU_MARCH OFF)
	set(SHTNS_GPU_MARCH OFF)

	# handle AUTO mode
	if (USE_MARCH STREQUAL "AUTO")
		if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
			message(STATUS "Building for a 64-bit Intel/AMD system.")
			set(SHTNS_CPU_MARCH "-march=native")
			set(SHTNS_GPU_MARCH "-march=core-avx2")
		elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86")
			message(STATUS "Building for a 32-bit Intel/AMD system.")
			set(SHTNS_CPU_MARCH "-march=native")
			set(SHTNS_GPU_MARCH "-march=core-avx2")
		elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^armv7|aarch64")
			message(STATUS "Building for an ARM system.")
			set(SHTNS_CPU_MARCH "-march=native")
			set(SHTNS_GPU_MARCH "-march=native")
		else()
			message(FATAL_ERROR "Unknown or unsupported processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")
		endif()
	elseif(NOT USE_MARCH STREQUAL OFF)
		set(SHTNS_CPU_MARCH "-march=${USE_MARCH}")
		set(SHTNS_GPU_MARCH "-march=${USE_MARCH}")
	endif()

	# Handle USE_GPU_MARCH
	if (NOT (USE_GPU_MARCH STREQUAL OFF OR USE_GPU_MARCH STREQUAL "AUTO"))
		set(SHTNS_GPU_MARCH "-march=${USE_GPU_MARCH}")
	endif()

	# export
	set(SHTNS_CPU_MARCH "${SHTNS_CPU_MARCH}" PARENT_SCOPE)
	set(SHTNS_GPU_MARCH "${SHTNS_GPU_MARCH}" PARENT_SCOPE)
endfunction()

###########################################################
function(shtns_config_gpu_cuda_arch)
	# default cuda arch
	# Note: this trick can also be removed if we consider "all" as the default value !
	if (USE_CUDA)
		include(cmake/SHTnsCheckCudaArch.cmake)
		if (USE_CUDA_ARCH STREQUAL "AUTO")
			shtns_check_cuda_arch(CUDA_ARCH_AVAIL)
			if (CUDA_ARCH_AVAIL)
				set(USE_CUDA_ARCH ${CUDA_ARCH_AVAIL})
			else()
				set(USE_CUDA_ARCH all)
			endif()
		endif()
	endif()

	# build some derivates
	if (USE_CUDA AND NOT CMAKE_CUDA_ARCHITECTURES)
		if (USE_CUDA_ARCH STREQUAL "kepler")
			set(CMAKE_CUDA_ARCHITECTURES 30)
		elseif (USE_CUDA_ARCH STREQUAL "pascal")
			set(CMAKE_CUDA_ARCHITECTURES 60)
		elseif (USE_CUDA_ARCH STREQUAL "volta")
			set(CMAKE_CUDA_ARCHITECTURES 70)
		elseif (USE_CUDA_ARCH STREQUAL "ampere")
			set(CMAKE_CUDA_ARCHITECTURES 80)
		elseif (USE_CUDA_ARCH STREQUAL "ada")
			set(CMAKE_CUDA_ARCHITECTURES 89)
		elseif (USE_CUDA_ARCH STREQUAL "hopper")
			set(CMAKE_CUDA_ARCHITECTURES 90)
		elseif (USE_CUDA_ARCH STREQUAL "blackwell")
			set(CMAKE_CUDA_ARCHITECTURES 100)
		elseif (NOT CMAKE_CUDA_ARCHITECTURES)
			set(CMAKE_CUDA_ARCHITECTURES ${USE_CUDA_ARCH})
		endif()
	endif()

	# export to parent
	set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
endfunction()

###########################################################
function(shtns_config_gpu_hip_arch)
	# default cuda arch
	# Note: this trick can also be removed if we consider "all" as the default value !
	if (USE_HIP)
		include(cmake/SHTnsCheckHipArch.cmake)
		if (USE_HIP_ARCH STREQUAL "AUTO")
			shtns_check_hip_arch(HIP_ARCH_AVAIL)
			if (HIP_ARCH_AVAIL)
				set(USE_HIP_ARCH ${HIP_ARCH_AVAIL})
			else()
				set(USE_HIP_ARCH all)
			endif()
		endif()
	endif()

	# build some derivates
	if (USE_HIP AND NOT CMAKE_HIP_ARCHITECTURES)
		if (USE_HIP_ARCH STREQUAL "mi100")
			set(CMAKE_HIP_ARCHITECTURES gfx908)
		elseif (USE_HIP_ARCH STREQUAL "mi200")
			set(CMAKE_HIP_ARCHITECTURES gfx90a)
		elseif (USE_HIP_ARCH STREQUAL "mi250")
			set(CMAKE_HIP_ARCHITECTURES gfx90a)
		elseif (USE_HIP_ARCH STREQUAL "mi300")
			set(CMAKE_HIP_ARCHITECTURES gfx942)
		elseif (NOT CMAKE_HIP_ARCHITECTURES)
			set(CMAKE_HIP_ARCHITECTURES ${USE_HIP_ARCH})
		endif()
	endif()

	# export to parent
	set(CMAKE_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES} PARENT_SCOPE)
endfunction()
