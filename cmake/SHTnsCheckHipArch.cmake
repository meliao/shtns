function (shtns_check_hip_arch OUTVAR)
	set(archs "")

	# test all
	foreach(arch IN ITEMS gfx908 gfx90a)
		message(STATUS "Checking HIP_ARCH=${arch}....")
		execute_process(
			COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/shtu_gpu_empty.cu
			COMMAND hipcc ${CMAKE_CURRENT_BINARY_DIR}/shtu_gpu_empty.cu -c -std=c++11 --offload-arch=${arch}:xnack+ --offload-arch=${arch}:xnack- -o /dev/null
			OUTPUT_QUIET
			ERROR_QUIET
			RESULT_VARIABLE hipcc_status
		)
		if (hipcc_status EQUAL 0)
			list(APPEND archs ${arch})
			message(STATUS "Found valid HIP_ARCH=${arch}")
		endif()
	endforeach()

	# join
	list(JOIN archs ";" archs)

	# export
	set(${OUTVAR} ${archs} PARENT_SCOPE)
endfunction()
