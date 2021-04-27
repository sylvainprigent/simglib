############################################################
#
# $Id$
#
# Copyright (c) SRoi 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(sroi)

## #################################################################
## Define files
## #################################################################

SET(SRC_FILES
    SPoint.cpp
    SRoi.cpp
    SRoiDrawer.cpp
    SRoiPatch.cpp
    SRoiReader.cpp
    SRoiShortcuts.cpp
    SRoiTypes.cpp
)

SET(HEADERS_FILES
    SPoint.h
    SRoi.h
    SRoiDrawer.h
    SRoiPatch.h
    SRoiReader.h
    SRoiShortcuts.h
    SRoiTypes.h
)
  
## #################################################################
## Library target
## #################################################################

add_library(${PROJECT_NAME} ${SRC_FILES} ${HEADERS_FILES})
target_link_libraries(${PROJECT_NAME} sdata sdataio ${SL_LIBRARIES})

## #################################################################
## Export header file
## #################################################################

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

generate_export_header(${PROJECT_NAME}
  EXPORT_FILE_NAME "${PROJECT_NAME}Export.h")

add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}Export.h" "${CMAKE_BINARY_DIR}"
  COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different "${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}Export.h" "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Export")

