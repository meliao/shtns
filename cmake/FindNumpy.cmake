# search for python
find_package(Python REQUIRED COMPONENTS Interpreter)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
	COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
	OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

# export variable
mark_as_advanced( NUMPY_INCLUDE_DIR )

# handle
include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( Numpy DEFAULT_MSG NUMPY_INCLUDE_DIR )
