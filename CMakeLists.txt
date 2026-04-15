###########################################################
# Authors :
#   - Máté Ferenc Nagy-Egri - 2021
#   - Sébastien Valat (ISTerre/CNRS) - 2026
###########################################################

###########################################################
# min:
#  - 3.8  : CUDA LANGUAGE SUPPORT
#  - 3.12 : CMP0074
cmake_minimum_required(VERSION 3.12)

# handle policies
cmake_policy(SET CMP0074 NEW)

###########################################################
project(SHTns VERSION 3.7 LANGUAGES C)

###########################################################
# set default build type to release
if (CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release")
endif()

###########################################################
# configure options
option(BUILD_TESTING "Build unit tests" OFF)
option(BUILD_DOCUMENTATION "Build docs" OFF)
option(BUILD_FORTRAN "Build the Fortran77 API" OFF)
option(BUILD_FOR_PYTHON "Build the library as python module." OFF)
option(USE_HIP "Enable HIP support" OFF)
option(USE_HIP_CMAKE "Enable HIP support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_MAGIC "I need the transforms compatible with the MagIC code, to speed it up!" OFF)
option(USE_ISHIOKA "Enable the new recurrence proposed by Ishioka (2018) see https://doi.org/10.2151/jmsj.2018-019 (faster)" ON)
option(USE_SHARED "Enable building shared libraries" OFF)
# options with values
set(USE_HIP_ARCH "AUTO" CACHE STRING "Select the architecture to build for (mi100, mi200, mi250, gfx908, gfx90a...).")
set(USE_CUDA "AUTO" CACHE STRING "Enable CUDA support (ON/OFF/AUTO)")
set(USE_MARCH "AUTO" CACHE STRING "Select the -march value to apply, can use AUTO to autoselect depending on ARM or x86 familly.")
set(USE_GPU_MARCH "AUTO" CACHE STRING "Select the -march value to apply on GPU binary, can use AUTO to autoselect depending on ARM or x86 familly.")
set(USE_CUDA_ARCH "AUTO" CACHE STRING "Select the architecture to build for (nativ, all, all-major, kepler, pascal, volta, ampere, 70, 80, ...).")
set(SHT_VERBOSE 1 CACHE STRING "0:no output, 1:output info to stdout, 2:more output (debug info), 3:also print fftw plans.]")

###########################################################
# TODO
#add_definitions(-DSHTNS_GIT="v${CMAKE_PROJECT_VERSION}")
add_definitions(-DSHTNS_VER="${CMAKE_PROJECT_VERSION}")
set(SHT_C_FLAGS "-std=gnu99 -fno-tree-loop-distribute-patterns -mno-vzeroupper ${CMAKE_C_FLAGS}")
set(SHT_C_FLAGS "${SHT_C_FLAGS} -Wno-declared_but_not_referenced -Wno-mixed_enum_type")
set(CMAKE_C_FLAGS "${SHT_C_FLAGS} ${CMAKE_C_FLAGS}")

###########################################################
# dependency
if (USE_HIP_CMAKE)
	set(USE_HIP ON)
endif()

###########################################################
# Dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(FFTW REQUIRED)

###########################################################
# include some custom scripts
include(cmake/SHTnsConfigHeader.cmake)
include(cmake/SHTnsBuildLib.cmake)
include(cmake/SHTnsBuildTimeSht.cmake)
include(cmake/SHTnsBuildInfos.cmake)
include(cmake/SHTnsCpuGpuArch.cmake)
# Initializes ${CMAKE_INSTALL_*DIR} variables
include(GNUInstallDirs) 

###########################################################
# configure march for cpu & gpu
shtns_config_cpu_gpu_march()
shtns_config_gpu_cuda_arch()
shtns_config_gpu_hip_arch()

###########################################################
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sht_config.h)
	message(FATAL_ERROR "Caution, you have previously run SHTNS 'configure' script in the sources, this is not compatible with cmake. Please run 'make clean' in the sources.")
endif()

###########################################################
# standards to use
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

###########################################################
# search deps
if(USE_OPENMP)
	find_package(OpenMP REQUIRED COMPONENTS C)
endif()

if(USE_CUDA STREQUAL "AUTO")
	find_package(CUDAToolkit QUIET)
	if (CUDAToolkit_FOUND)
		enable_language(CUDA)
	endif()
elseif(USE_CUDA STREQUAL "ON")
	enable_language(CUDA)
	find_package(CUDAToolkit REQUIRED)
endif()

if(USE_HIP)
	if (USE_HIP_CMAKE)
		cmake_minimum_required(
			VERSION 3.21 # HIP LANGUAGE SUPPORT
		)
		# Enabling CXX (and searching for a C++ compiler) is a workaround for a bug in ROCm 4.3, where hipfft pulls in
		# /opt/rocm/lib/cmake/hip/hip-config.cmake:273 which still expects CMAKE_CXX_COMPILER_ID to exist (as previous) HIP support
		# in CMake required overriding the C++ compiler with hipcc. Once this is changed to CMAKE_HIP_COMPILER_ID, this can be removed.
		enable_language(CXX) # Workaround
		enable_language(HIP)

		# search hip libs
		find_package(hip REQUIRED)
		find_package(hipfft REQUIRED)
		find_package(hiprtc REQUIRED)
	else()
		# use SHTns manual HIP binding
		find_package(SHTnsHIP REQUIRED)
		include(cmake/SHTnsHIP.cmake)
	endif()
endif()

# Find the Python development files
if (BUILD_FOR_PYTHON)
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
	find_package(Numpy REQUIRED)
endif()

###########################################################
# tricks to get the version file generated all the time
# but the file being touch only if it changes by using the cmake
# internals (configure_file)
# from : https://www.marcusfolkesson.se/blog/git-version-in-cmake/
add_custom_target(sht_version ALL
	${CMAKE_COMMAND}
		-D SRC=${CMAKE_CURRENT_SOURCE_DIR}/sht_version.h.in
		-D DST=${CMAKE_CURRENT_BINARY_DIR}/sht_version.h
		-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateVersion.cmake
)
add_definitions(-DSHTNS_CMAKE=1)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

###########################################################
# build libraries variants
shtns_build_library(LIBNAME shtns)
if (USE_OPENMP)
	shtns_build_library(LIBNAME shtns_omp OPENMP MARCH ${SHTNS_CPU_MARCH})
endif()
if (USE_CUDA AND CUDAToolkit_FOUND)
	shtns_build_library(LIBNAME shtns_cuda CUDA MARCH ${SHTNS_GPU_MARCH})
endif()
if (USE_CUDA AND CUDAToolkit_FOUND AND USE_OPENMP)
	shtns_build_library(LIBNAME shtns_cuda_omp CUDA OPENMP MARCH ${SHTNS_GPU_MARCH})
endif()
if (USE_HIP)
	shtns_build_library(LIBNAME shtns_cuda HIP MARCH ${SHTNS_GPU_MARCH})
endif()
if (USE_HIP AND USE_OPENMP)
	shtns_build_library(LIBNAME shtns_cuda_omp HIP OPENMP MARCH ${SHTNS_GPU_MARCH})
endif()

###########################################################
# enable testing
if(BUILD_TESTING)
	enable_testing()
endif()

###########################################################
# build and run time_SHT test for all variants
if(BUILD_TESTING)
	shtns_build_time_shtns(EXENAME time_SHT LIBNAME shtns)
	if (USE_OPENMP)
		shtns_build_time_shtns(EXENAME time_SHT_omp LIBNAME shtns_omp)
	endif()
	if (USE_CUDA AND CUDAToolkit_FOUND)
		shtns_build_time_shtns(EXENAME time_SHT_cuda LIBNAME shtns_cuda)
	endif()
	if (USE_CUDA AND CUDAToolkit_FOUND AND USE_OPENMP)
		shtns_build_time_shtns(EXENAME time_SHT_cuda_omp LIBNAME shtns_cuda_omp)
	endif()
	if (USE_HIP)
		shtns_build_time_shtns(EXENAME time_SHT_cuda LIBNAME shtns_cuda)
	endif()
	if (USE_HIP AND USE_OPENMP)
		shtns_build_time_shtns(EXENAME time_SHT_cuda_omp LIBNAME shtns_cuda_omp)
	endif()
endif()

###########################################################
# install
if(USE_CUDA OR USE_HIP)
	install(FILES shtns_cuda.h TYPE INCLUDE)
endif()
install(FILES ${CMAKE_BINARY_DIR}/sht_config.h shtns.h TYPE INCLUDE)

###########################################################
if (BUILD_FOR_PYTHON)
	# Python wrapper for shtns cuda
	Python3_add_library(_shtns MODULE "shtns_numpy_wrap.c" WITH_SOABI)
	target_link_libraries(_shtns PRIVATE shtns_omp)
	target_include_directories(_shtns PRIVATE ${NUMPY_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
	install(TARGETS _shtns COMPONENT SHTnsPy DESTINATION .)

	# Python wrapper for shtns cuda
	if ((USE_CUDA AND CUDAToolkit_FOUND) OR USE_HIP)
		Python3_add_library(_shtns_cuda MODULE "shtns_cuda_wrap.c" WITH_SOABI)
		target_link_libraries(_shtns_cuda PRIVATE shtns_cuda_omp)
		target_include_directories(_shtns_cuda PRIVATE ${NUMPY_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
		install(TARGETS _shtns_cuda COMPONENT SHTnsPy DESTINATION .)
	endif()
endif()

###########################################################
# subdirs
add_subdirectory(SHT)

###########################################################
shtns_print_status()
