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
SET(NAME simgspitfiredenoise2d)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score sdenoise scli simage simageio smanipulate ${SL_LIBRARIES})

INSTALL(TARGETS simgspitfiredenoise2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredenoise3d)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score sdenoise scli simage simageio smanipulate ${SL_LIBRARIES})

INSTALL(TARGETS simgspitfiredenoise3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredenoise4d)
SET(FILE ${SRC_FILES} spitfire4d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score sdenoise scli simage simageio smanipulate ${SL_LIBRARIES})

INSTALL(TARGETS simgspitfiredenoise4d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
