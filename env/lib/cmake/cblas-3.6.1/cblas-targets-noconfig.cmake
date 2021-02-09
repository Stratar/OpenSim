#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cblas" for configuration ""
set_property(TARGET cblas APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cblas PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "blas"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcblas.so"
  IMPORTED_SONAME_NOCONFIG "libcblas.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS cblas )
list(APPEND _IMPORT_CHECK_FILES_FOR_cblas "${_IMPORT_PREFIX}/lib/libcblas.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
