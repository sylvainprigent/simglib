############################################################
#
# $Id$
#
# Copyright (c) SimgLib 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(cuDenoiseTools LANGUAGES CUDA CXX)

## #################################################################
## Define files
## #################################################################
include_directories(${sl_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME simgcuspitfire2ddenoise)
SET(FILE ${SRC_FILES} cuspitfire2ddenoise.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cudenoise smanipulate score simage simageio scli)
target_compile_features(cuspitfire2ddenoise PUBLIC cxx_std_11)
set_target_properties( cuspitfire2ddenoise
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET(NAME simgcuspitfire3ddenoise)
SET(FILE ${SRC_FILES} cuspitfire3ddenoise.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cudenoise smanipulate score simage simageio scli)
target_compile_features(cuspitfire3ddenoise PUBLIC cxx_std_11)
set_target_properties( cuspitfire3ddenoise
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET(NAME simgcuspitfire4ddenoise)
SET(FILE ${SRC_FILES} cuspitfire4ddenoise.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cudenoise smanipulate score simage simageio scli)
target_compile_features(cuspitfire3ddenoise PUBLIC cxx_std_11)
set_target_properties( cuspitfire3ddenoise
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)                       
                     