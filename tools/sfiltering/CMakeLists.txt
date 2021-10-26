############################################################
#
# $Id$
#
# Copyright (c) SImgLib 2021
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(sfilteringcli)

## #################################################################
## Define files
## #################################################################
include_directories(${simglib_SRC_INCLUDE_DIR})


## #################################################################
## Exe generation
## #################################################################
SET(NAME simgmedian2d)
SET(FILE ${SRC_FILES} median2d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score scli simage simageio sfiltering spadding ${SL_LIBRARIES})

INSTALL(TARGETS simgmedian2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgmedian3d)
SET(FILE ${SRC_FILES} median3d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score scli simage simageio sfiltering spadding ${SL_LIBRARIES})

INSTALL(TARGETS simgmedian3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgmedian4d)
SET(FILE ${SRC_FILES} median4d.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} score scli simage simageio sfiltering spadding ${SL_LIBRARIES})

INSTALL(TARGETS simgmedian4d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)