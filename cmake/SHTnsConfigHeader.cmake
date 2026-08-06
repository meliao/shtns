###########################################################
# Authors :
#   - Máté Ferenc Nagy-Egri - 2021
#   - Sébastien Valat (ISTerre/CNRS) - 2026
###########################################################

include(CheckSymbolExists)
include(CheckIncludeFile)
include(CheckLibraryExists)

check_symbol_exists(clock_gettime "time.h" HAVE_CLOCK_GETTIME)
check_include_file("complex.h" HAVE_COMPLEX_H)
check_include_file("c_asm.h" HAVE_C_ASM_H)

set(CMAKE_REQUIRED_INCLUDES ${FFTW_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${FFTW_LIBRARIES})
check_symbol_exists(fftw_cost "fftw3.h" HAVE_FFTW_COST)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_LIBRARIES)

check_symbol_exists(gethrtime "sys/time.h" HAVE_GETHRTIME)
check_symbol_exists(hrtime_t "sys/time.h" HAVE_HRTIME_T)

check_include_file("intrinsics.h" HAVE_INTRINSICS_H)
check_include_file("inttypes.h" HAVE_INTTYPES_H)

if(CUDAToolkit_FOUND)
	set(HAVE_LIBCUDART 1 CACHE BOOL "Define to 1 if `cudart' has been found by find_package(CUDAToolkit)")
	set(HAVE_LIBCUFFT 1 CACHE BOOL "Define to 1 if `cufft' has been found by find_package(CUDAToolkit)")
endif()

get_property(LANGS GLOBAL PROPERTY ENABLED_LANGUAGES)
if("HIP" IN_LIST LANGS)
	set(HAVE_HIP 1 CACHE BOOL "Define to 1 if HIP language has been enabled")
	set(HAVE_LIBHIPFFT 1 CACHE BOOL "Define to 1 if `hipfft' has been found by find_package(hipfft)")
endif()
unset(LANGS)

if(FFTW_FOUND)
	set(HAVE_LIBFFTW3 1 CACHE BOOL "Define to 1 if `FFTW3' has been found by find_package(FFTW)")
endif()

# TODO: Test for HAVE_LIBFFTW3_OMP

check_library_exists(m sin "" HAVE_LIBM)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test_ldbl.c"
	[[
		#include <float.h>
		int main()
		{
			return DBL_DECIMAL_DIG == LDBL_DECIMAL_DIG;
		}
	]]
)

try_run(LDBL_RUN_RESULT
	LDBL_COMPILE_RESULT
	"${CMAKE_CURRENT_BINARY_DIR}"
	"${CMAKE_CURRENT_BINARY_DIR}/test_ldbl.c"
)

file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/test_ldbl.c")

if(LDBL_COMPILE_RESULT EQUAL 0 AND LDBL_RUN_RESULT EQUAL 0)
	set(HAVE_LONG_DOUBLE_WIDER 1 CACHE BOOL "Whether type `long double' works and has more range or precision
	than `double'.")
endif()

check_symbol_exists(mach_absolute_time "mach/mach_time.h" HAVE_MACH_ABSOLUTE_TIME)
check_include_file("mach/mach_time.h" HAVE_MACH_MACH_TIME_H)
check_include_file("math.h" HAVE_MATH_H)
check_include_file("memory.h" HAVE_MEMORY_H)
check_symbol_exists(read_real_time "sys/time.h" HAVE_READ_REAL_TIME)
check_include_file("stdatomic.h" HAVE_STDATOMIC_H)
check_include_file("stdint.h" HAVE_STDINT_H)
check_include_file("stdio.h" HAVE_STDIO_H)
check_include_file("stdlib.h" HAVE_STDLIB_H)
check_include_file("strings.h" HAVE_STRINGS_H)
check_include_file("string.h" HAVE_STRING_H)
check_include_file("sys/stat.h" HAVE_SYS_STAT_H)
check_include_file("sys/time.h" HAVE_SYS_TIME_H)
check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
check_symbol_exists(time_base_to_time "sys/time.h" HAVE_TIME_BASE_TO_TIME)
check_include_file("unistd.h" HAVE_UNISTD_H)

check_symbol_exists(_rtc "linux/rtc.h" HAVE__RTC)

set(PACKAGE_BUGREPORT [[""]])
set(PACKAGE_NAME "${PROJECT_NAME}")
set(PACKAGE_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
set(PACKAGE_TARNAME shtns)
set(PACKAGE_URL "https://bitbucket.org/nschaeff/shtns")
set(PACKAGE_VERSION "${PROJECT_VERSION}")

if(USE_MAGIC)
	set(SHTNS4MAGIC 1 CACHE BOOL "")
endif()

if(USE_ISHIOKA)
	set(SHTNS_ISHIOKA 1 CACHE BOOL "")
endif()

# https://gist.github.com/likema/97841d77e5f384fbdb46629c5b429013
include(cmake/CheckStdCHeaders.cmake)
check_stdc_headers()

configure_file(
	sht_config.cmake.h.in
	sht_config.h
)
