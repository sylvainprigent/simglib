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
SET(NAME simgzeropadding2d)
SET(FILE ${SRC_FILES} zeropadding2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgzeropadding2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgzeropadding3d)
SET(FILE ${SRC_FILES} zeropadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgzeropadding3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgmirrorpadding3d)
SET(FILE ${SRC_FILES} mirrorpadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgmirrorpadding3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgnormalizez)
SET(FILE ${SRC_FILES} normalizez.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgnormalizez
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simghannzeropadding2d)
SET(FILE ${SRC_FILES} hannzeropadding2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simghannzeropadding2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simghannzeropadding3d)
SET(FILE ${SRC_FILES} hannzeropadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simghannzeropadding3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgremovepadding3d)
SET(FILE ${SRC_FILES} removepadding3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgremovepadding3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgresize3d)
SET(FILE ${SRC_FILES} resize3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} spadding scli simage simageio)
INSTALL(TARGETS simgresize3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
