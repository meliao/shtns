function (shtns_check_cuda_arch OUTVAR)
	set(archs "")

	# test all
	foreach(arch IN ITEMS 30 60 70 80 90 100)
		message(STATUS "Checking CUDA_ARCH=${arch}....")
		execute_process(
			COMMAND nvcc ${CMAKE_PROJECT_SOURCE_DIR}/sht_gpu.cu -c -arch=compute_${arch} --dryrun
			OUTPUT_QUIET
			ERROR_QUIET
			RESULT_VARIABLE nvcc_status
		)
		if (nvcc_status EQUAL 0)
			list(APPEND archs ${arch})
			message(STATUS "Found valid CUDA_ARCH=${arch}")
		endif()
	endforeach()

	# join
	list(JOIN archs ";" archs)

	# export
	set(${OUTVAR} ${archs} PARENT_SCOPE)
endfunction()
