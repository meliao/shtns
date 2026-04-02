###########################################################
# Authors :
#   - Máté Ferenc Nagy-Egri - 2021
#   - Sébastien Valat (ISTerre/CNRS) - 2026
###########################################################

###########################################################
function(shtns_build_time_shtns)
	# parse arguments
	set(options "")
	set(oneValueArgs EXENAME LIBNAME)
	set(multiValueArgs "")
	cmake_parse_arguments(PARSE_ARGV 0 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")

	# build exe
	add_executable(${arg_EXENAME} time_SHT.c)
	target_link_libraries(${arg_EXENAME} ${arg_LIBNAME})
	target_include_directories(${arg_EXENAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

	# append tests
	foreach(SIZE IN ITEMS 127 1023)
		add_test(
			NAME ${arg_EXENAME}-${SIZE}
			COMMAND ${arg_EXENAME} ${SIZE} -quickinit
		)
	endforeach()
endfunction()
