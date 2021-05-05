############################################################
#
#
# Copyright (c) simglib 2020
#
# AUTHOR:
# Sylvain Prigent
# 

## #################################################################
## Header
## #################################################################

cmake_minimum_required(VERSION 3.8)
cmake_policy(VERSION 3.8)

project(simglib VERSION 0.0.1)

set(${PROJECT_NAME}_VERSION_MAJOR 0 CACHE STRING "simglib major version number." FORCE)
set(${PROJECT_NAME}_VERSION_MINOR 0 CACHE STRING "simglib minor version number." FORCE)
set(${PROJECT_NAME}_VERSION_BUILD 1 CACHE STRING "simglib build version number." FORCE)
set(${PROJECT_NAME}_VERSION
  "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_BUILD}")

mark_as_advanced(${PROJECT_NAME}_VERSION_MAJOR)
mark_as_advanced(${PROJECT_NAME}_VERSION_MINOR)
mark_as_advanced(${PROJECT_NAME}_VERSION_BUILD)

## #################################################################
## Config file
## #################################################################

# Cmake config file directory path
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(simglibBuild)

set(simglibProject_USE_DOXYGEN 0)



## #################################################################
## OpemMP option
## #################################################################
option(SL_USE_OPENMP "use OpenMP" OFF)
if (SL_USE_OPENMP)
    add_compile_definitions(SL_USE_OPENMP)
endif(SL_USE_OPENMP)

include(simglibDependencies)

## #################################################################
## Define Internal Architecture (for includes)
## #################################################################
option(simglib_BUILD_TESTS "Build Tests" ON)
option(simglib_BUILD_TOOLS "Build Tools" OFF)
option(simglib_BUILD_CUDA "Build Cuda" OFF)


add_subdirectory( src )
if(simglib_BUILD_TOOLS)
  add_subdirectory( tools )
endif(simglib_BUILD_TOOLS)
if(simglib_BUILD_TESTS)
  add_subdirectory( tests )
endif(simglib_BUILD_TESTS)

## #################################################################
## Build Documentation
## #################################################################
IF ( ${SL_USE_DOXYGEN} )
  message(STATUS "Using Doxygen")
  SET( EXCLUDE_FROM_DOC_PATTERNS "*/external/*" )
  SET( DOC_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc)
  SET( DOC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src )
  
  configure_file( cmake/Doxyfile.in 
    ${DOC_OUTPUT_DIR}/Doxyfile
    @ONLY
    )
  
  add_custom_target( doc
    ${DOXYGEN_EXECUTABLE} ${DOC_OUTPUT_DIR}/Doxyfile
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
ELSE( ${SL_USE_DOXYGEN} )
  message(STATUS "Not Using Doxygen")
ENDIF( ${SL_USE_DOXYGEN} )
