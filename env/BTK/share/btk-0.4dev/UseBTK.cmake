#
# This file sets up include directories, link directories, and
# compiler settings for a project to use BTK.  It should not be
# included directly, but rather through the BTK_USE_FILE setting
# obtained from BTKConfig.cmake.
#

IF(BTK_BUILD_SETTINGS_FILE AND NOT SKIP_BTK_BUILD_SETTINGS_FILE)
  INCLUDE(${CMAKE_ROOT}/Modules/CMakeImportBuildSettings.cmake)
  CMAKE_IMPORT_BUILD_SETTINGS(${BTK_BUILD_SETTINGS_FILE})
ENDIF(BTK_BUILD_SETTINGS_FILE AND NOT SKIP_BTK_BUILD_SETTINGS_FILE)

# Add compiler flags needed to use BTK.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BTK_REQUIRED_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BTK_REQUIRED_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BTK_REQUIRED_LINK_FLAGS}")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BTK_REQUIRED_LINK_FLAGS}")
SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${BTK_REQUIRED_LINK_FLAGS}")

# Add include directories needed to use BTK.
INCLUDE_DIRECTORIES(BEFORE ${BTK_INCLUDE_DIRS})

# Add link directories needed to use BTK.
LINK_DIRECTORIES(${BTK_LIBRARY_DIRS})
