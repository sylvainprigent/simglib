############################################################
#
# $Id$
#
# Copyright (c) simglib 2020
#
# AUTHOR:
# Sylvain Prigent
# 

## #################################################################
## Doxygen
## #################################################################

find_package(Doxygen QUIET)
if(${DOXYGEN_FOUND})
  set(simglib_USE_DOXYGEN 1)
endif(${DOXYGEN_FOUND})

if(${SL_USE_OPENMP})
  find_package(OpenMP REQUIRED)
endif(${SL_USE_OPENMP})  

## #################################################################
## libfftw3
## #################################################################

find_package (FFTW REQUIRED)
if (FFTW_FOUND)
  set (SL_INCLUDE_DIRS ${SL_INCLUDE_DIRS} ${FFTW_INCLUDE_DIR})
  set (SL_LIBRARIES ${SL_LIBRARIES} ${FFTW_LIBRARIES})
  if(${SL_USE_OPENMP})
    set (SL_LIBRARIES ${SL_LIBRARIES} ${FFTW3F_THREAD_LIBRARIES})
  endif(${SL_USE_OPENMP})  
  message(STATUS "FFTW3F found")
else (FFTW_FOUND)
  message(STATUS "FFTW3F NOT found.")
  message (FATAL_ERROR "You need fftw3f to compile this program. Please install libs and developpement headers")
endif (FFTW_FOUND)

if(simglib_BUILD_TOOLS)
## #################################################################
## libtiff
## #################################################################
find_package (TIFF REQUIRED)
if (TIFF_FOUND)
  set (SL_INCLUDE_DIRS ${SL_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR})
  set (SL_LIBRARIES ${SL_LIBRARIES} ${TIFF_LIBRARIES}) 
  message(STATUS "TIFF found")
else (TIFF_FOUND)
  message(STATUS "TIFF NOT found.")
  message (FATAL_ERROR "You need libtiff to compile this program. Please install libs and developpement headers")
endif (TIFF_FOUND)
endif(simglib_BUILD_TOOLS)
## #################################################################
## definitions
## #################################################################
add_definitions (${SL_DEFINITIONS})
include_directories (${SL_INCLUDE_DIRS})
link_directories(${SL_LIBRARY_DIRS})
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SL_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SL_CXX_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(${SL_USE_OPENMP})
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fopenmp")
endif(${SL_USE_OPENMP})  

#set (SL_LIBRARIES ${SL_LIBRARIES} m) 
set (SL_INCLUDE_DIRS ${SL_INCLUDE_DIRS} CACHE STRING "include directories for spartion dependancies")
set (SL_LIBRARIES ${SL_LIBRARIES} CACHE STRING "spartion required and optional 3rd party libraries")
set (SL_DEFINITIONS ${SL_DEFINITIONS} CACHE STRING "SL_USE_XXX defines")
set (SL_C_FLAGS ${SL_C_FLAGS}  CACHE STRING "c flags for cimg")
set (SL_CXX_FLAGS ${SL_CXX_FLAGS} CACHE STRING "c++ flags for cimg")