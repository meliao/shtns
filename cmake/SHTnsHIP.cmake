###########################################################
# Create imported target hip::host
if (NOT HIP_TARGET_DEFINED)
	add_library(hip::host INTERFACE IMPORTED)
	target_include_directories(hip::host INTERFACE ${HIP_HOST_INCLUDE_DIRS})
	target_link_libraries(hip::host INTERFACE hip::amdhip64)
endif()

###########################################################
# Create imported target hip::fftw
if (NOT HIP_TARGET_DEFINED AND HIP_FFT_FOUND)
	add_library(hip::hipfft INTERFACE IMPORTED)
	target_include_directories(hip::hipfft INTERFACE ${HIP_FFT_INCLUDE_DIRS})
	target_link_libraries(hip::hipfft INTERFACE ${HIP_FFT_LIBRARIES})
endif()

###########################################################
# Create imported target hip::host
if (NOT HIP_TARGET_DEFINED)
	add_library(hiprtc::hiprtc INTERFACE IMPORTED)
	target_include_directories(hiprtc::hiprtc INTERFACE ${HIP_RTC_INCLUDE_DIRS})
	target_link_libraries(hiprtc::hiprtc INTERFACE ${HIP_RTC_LIBRARIES})
endif()

###########################################################
# Create imported target hip::host
if (NOT HIP_TARGET_DEFINED)
	add_library(hip::amdhip64 INTERFACE IMPORTED)
	target_link_libraries(hip::amdhip64 INTERFACE ${HIP_AMDHIP_LIBRARIES})
endif()

###########################################################
# Create imported target hip::host
if (NOT HIP_TARGET_DEFINED AND HIP_ROCBLAS_FOUND)
	add_library(roc::rocblas INTERFACE IMPORTED)
	target_include_directories(roc::rocblas INTERFACE ${HIP_ROCBLAS_INCLUDE_DIRS})
	target_link_libraries(roc::rocblas INTERFACE ${HIP_ROCBLAS_LIBRARIES})
endif()

###########################################################
# not to redo if done in parent
set(HIP_TARGET_DEFINED ON)

###########################################################
# function to compile a CU file to object
function(shtns_hip_add_library)
	# parse arguments
	set(options "")
	set(oneValueArgs TARGET INCLUDED_IN)
	set(multiValueArgs SOURCES)
	cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")

	# import if related
	get_target_property(cxx_standard ${arg_TARGET} CXX_STANDARD)

	# build offloat
	set(offload "")
	foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES)
		list(APPEND offload --offload-arch=${arch}:xnack+)
		list(APPEND offload --offload-arch=${arch}:xnack-)
	endforeach()

	# create custom target
	foreach(src IN LISTS arg_SOURCES)
		# get filename
		get_filename_component(src_fname ${src} NAME_WLE)

		# cal ofile
		set(o_file ${arg_TARGET}_${src_fname}.o)

		add_custom_target(${o_file}
			COMMAND hipcc
				-fPIE
				-O3
				-std=c++${cxx_standard}
				#--rocm-path=/opt/rocm-7.2.0
				${offload}
				# get the -D & -I from the target and reproduce them here
				"$<LIST:TRANSFORM,$<TARGET_PROPERTY:${arg_TARGET},COMPILE_DEFINITIONS>,PREPEND,-D>"
				"$<LIST:TRANSFORM,$<TARGET_PROPERTY:${arg_TARGET},INCLUDE_DIRECTORIES>,PREPEND,-I>"
				-c ${CMAKE_CURRENT_SOURCE_DIR}/${src} -o ${o_file}
			BYPRODUCTS
				${o_file}
			SOURCES
				${arg_SOURCES}
			COMMENT
				"Building HIP target ${o_file}"
			COMMAND_EXPAND_LISTS
			VERBATIM
		)

		# mark as object
		set_target_properties(${o_file} PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${o_file})

		# link
		target_sources(${arg_TARGET} PRIVATE ${o_file})
	endforeach()
endfunction()
