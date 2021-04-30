############################################################
#
# $Id$
#
# Copyright (c) SImgLib 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(sdenoisecli)

## #################################################################
## Define files
## #################################################################
include_directories(${simglib_SRC_INCLUDE_DIR})


## #################################################################
## Exe generation
## #################################################################
SET(NAME spitfiredenoise2d)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score sdenoise scli simage simageio ${SL_LIBRARIES})

SET(NAME spitfiredenoise3d)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score sdenoise scli simage simageio ${SL_LIBRARIES})