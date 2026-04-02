# Locate the HIP (https://rocm.docs.amd.com/projects/HIP/en/latest/index.html) Framework.
#
# Defines the following variables:
#
#   HIP_FOUND - Found the HIP framework
#   HIP_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   HIP_LIBRARIES - libfftw
#
# Accepts the following variables as input:
#
#   HIP_ROOT - (as a CMake or environment variable)
#                The root directory of the fftw install prefix
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether 
#               findHIP should search for 64bit or 32bit libs
#-----------------------------------------------
# Example Usage:
#
#    find_package(HIP REQUIRED)
#    include_directories(${HIP_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${HIP_LIBRARIES})
#
#-----------------------------------------------

###########################################################
# How to configure
set(CACHE HIP_ROOT TYPE PATH HELP "Prefix installation path for the HIP framework." $ENV{HIP_ROOT})

# append to prefixes to search in
if (HIP_ROOT)
	list(APPEND CMAKE_PREFIX_PATH ${HIP_ROOT})
endif()

###########################################################
# Search for hipcc
find_program(
	HIP_COMPILER_BINARY hipcc
	DOC "The path to hipcc compiler."
)

###########################################################
# Get prefix
# down to parent dir
get_filename_component(HIP_PREFIX_PATH ${HIP_COMPILER_BINARY} DIRECTORY)
get_filename_component(HIP_PREFIX_PATH ${HIP_COMPILER_BINARY} DIRECTORY)

###########################################################
# Search for hip include dir
find_path(HIP_HOST_INCLUDE_DIRS
	NAMES hip/hip_runtime.h
)

###########################################################
# Search for hip fft include dir
find_path(HIP_FFT_INCLUDE_DIRS
	NAMES hipfft.h
	PATH_SUFFIXES hipfft
)

# search for hip fft lib
find_library(HIP_FFT_LIBRARIES
	NAMES hipfft
)

# set found
if (HIP_FFT_INCLUDE_DIRS AND HIP_FFT_LIBRARIES)
	set(HIP_FFT_FOUND ON)
endif()

###########################################################
# Search for hip rtc include dir
find_path(HIP_RTC_INCLUDE_DIRS
	NAMES hiprtc.h
	PATH_SUFFIXES hip
)

# search for hip fft lib
find_library(HIP_RTC_LIBRARIES
	NAMES hiprtc
)

###########################################################
# search for hip fft lib
find_library(HIP_AMDHIP_LIBRARIES
	NAMES amdhip64
)

###########################################################
# Search for hip rtc include dir
find_path(HIP_ROCBLAS_INCLUDE_DIRS
	NAMES rocblas.h
	PATH_SUFFIXES rocblas
)

# search for hip fft lib
find_library(HIP_ROCBLAS_LIBRARIES
	NAMES rocblas
)

# set found
if (HIP_ROCBLAS_INCLUDE_DIRS AND HIP_ROCBLAS_LIBRARIES)
	set(HIP_ROCBLAS_FOUND ON)
endif()

###########################################################
# export vars
mark_as_advanced(
	HIP_COMPILER_BINARY
	HIP_PREFIX_PATH
	HIP_HOST_INCLUDE_DIRS
	HIP_FFT_FOUND
	HIP_FFT_INCLUDE_DIRS
	HIP_FFT_LIBRARIES
	HIP_RTC_INCLUDE_DIRS
	HIP_RTC_LIBRARIES
	HIP_AMDHIP_LIBRARIES
	HIP_ROCBLAS_FOUND
	HIP_ROCBLAS_INCLUDE_DIRS
	HIP_ROCBLAS_LIBRARIES
)

###########################################################
# handle
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SHTnsHIP DEFAULT_MSG
	HIP_COMPILER_BINARY
	HIP_PREFIX_PATH
	HIP_HOST_INCLUDE_DIRS
	HIP_RTC_INCLUDE_DIRS
	HIP_RTC_LIBRARIES
	HIP_AMDHIP_LIBRARIES
)
