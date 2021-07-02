############################################################
#
# $Id$
#
# Copyright (c) SMpp 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(sroiTools)

## #################################################################
## Define files
## #################################################################
include_directories(${smpp_SRC_INCLUDE_DIR})


## #################################################################
## Exe generation
## #################################################################
SET(NAME simgroiinfo)
SET(FILE ${SRC_FILES} roiinfo.cpp SRoiInfo.cpp SRoiInfo.h)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} sroi scli simage simageio sdata sdataio sfiltering smanipulate ${SL_LIBRARIES})
INSTALL(TARGETS simgroiinfo
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgdrawrois)
SET(FILE ${SRC_FILES} drawrois.cpp)
add_executable(${NAME} ${FILE} )
target_link_libraries(${NAME} sroi scli simage simageio ${SL_LIBRARIES})
INSTALL(TARGETS simgdrawrois
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
