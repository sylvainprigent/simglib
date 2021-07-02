############################################################
#
# $Id$
#
# Copyright (c) SCli 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(SCliTools)

## #################################################################
## Define files
## #################################################################
include_directories(${scli_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME simgclidemo)
SET(FILE ${SRC_FILES} demo.cpp)
add_executable( ${NAME} ${FILE} )
target_link_libraries(${NAME} score scli ${SL_LIBRARIES})
