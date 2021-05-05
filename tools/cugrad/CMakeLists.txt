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
project(cuGradTools LANGUAGES CUDA CXX)

## #################################################################
## Define files
## #################################################################
include_directories(${sl_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME cuconvbin)
SET(FILE ${SRC_FILES} cuconv.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cufft cugrad score smanipulate simage simageio scli)
target_compile_features(cuconvbin PUBLIC cxx_std_11)
set_target_properties( cuconvbin
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET(NAME cugradbin)
SET(FILE ${SRC_FILES} cugrad.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cufft cugrad smanipulate score simage simageio scli)
target_compile_features(cugradbin PUBLIC cxx_std_11)
set_target_properties( cugradbin
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET(NAME cufft2dbin)
SET(FILE ${SRC_FILES} cufft2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cufft cugrad smanipulate score simage simageio scli)
target_compile_features(cufft2dbin PUBLIC cxx_std_11)
set_target_properties( cufft2dbin
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)                       