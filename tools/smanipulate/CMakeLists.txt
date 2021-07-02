############################################################
#
# $Id$
#
# Copyright (c) SManipulate 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(smanipulatecli)

## #################################################################
## Define files
## #################################################################
include_directories(${smanipulate_SRC_INCLUDE_DIR})


## #################################################################
## Exe generation
## #################################################################
SET(NAME simgcrop)
SET(FILE ${SRC_FILES} crop.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} smanipulate scli simage simageio ${SL_LIBRARIES})
INSTALL(TARGETS simgcrop
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)