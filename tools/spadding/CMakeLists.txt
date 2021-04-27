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
project(SPaddingTools)

## #################################################################
## Define files
## #################################################################
include_directories(${sl_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME zeropadding2d)
SET(FILE ${SRC_FILES} zeropadding2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME zeropadding3d)
SET(FILE ${SRC_FILES} zeropadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME mirrorpadding3d)
SET(FILE ${SRC_FILES} mirrorpadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME normalizez)
SET(FILE ${SRC_FILES} normalizez.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME hannzeropadding2d)
SET(FILE ${SRC_FILES} hannzeropadding2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME hannzeropadding3d)
SET(FILE ${SRC_FILES} hannzeropadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME removepadding3d)
SET(FILE ${SRC_FILES} removepadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)

SET(NAME resize3d)
SET(FILE ${SRC_FILES} resize3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
