# from https://www.marcusfolkesson.se/blog/git-version-in-cmake/

# search git
find_package(Git)

# call describe to get version
if(GIT_EXECUTABLE)
	get_filename_component(WORKING_DIR ${SRC} DIRECTORY)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty=*
		WORKING_DIRECTORY ${WORKING_DIR}
		OUTPUT_VARIABLE SHTNS_VERSION
		RESULT_VARIABLE ERROR_CODE
		OUTPUT_STRIP_TRAILING_WHITESPACE
		)
endif()

# fallback
if(SHTNS_VERSION STREQUAL "")
	set(SHTNS_VERSION ${CMAKE_PROJECT_VERSION})
	message(WARNING "Failed to determine version from Git tags. Using default version \"${SHTNS_VERSION}\".")
endif()

# generate config file
configure_file(${SRC} ${DST} @ONLY)
