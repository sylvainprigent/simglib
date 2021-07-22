############################################################
#
# $Id$
#
# Copyright (c) sdeconv 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(cuDeconvTools LANGUAGES CUDA CXX)

## #################################################################
## Define files
## #################################################################
include_directories(${sl_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME simgcuspitfire2ddeconv)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cudeconv sdeconv smanipulate score simage simageio scli)
#target_compile_features(cuspitfire2ddeconv PUBLIC cxx_std_11)
#set_target_properties( cuspitfire2ddeconv
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)       
                       
SET(NAME simgcuspitfire3ddeconv)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} cudeconv sdeconv smanipulate score simage simageio scli)
#target_compile_features(cuspitfire2ddeconv PUBLIC cxx_std_11)
#set_target_properties( cuspitfire2ddeconv
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)                         
