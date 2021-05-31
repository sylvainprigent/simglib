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
project(scli)

## #################################################################
## Define files
## #################################################################

SET(SRC_FILES
  SCliParameter.cpp
  SCliParameterList.cpp
  SCliParser.cpp
  SCliUtils.cpp
  SCliSelect.cpp
  SCliStringOp.cpp
  SCliTimer.cpp
  STimer.cpp
)

SET(HEADERS_FILES
  SCliException.h
  SCliParameter.h
  SCliParameterList.h
  SCliParser.h
  SCliUtils.h
  SCliSelect.h
  SCliStringOp.h
  SCliTimer.h
  STimer.h
)
  
## #################################################################
## Library target
## #################################################################

add_library(${PROJECT_NAME} ${SRC_FILES} ${HEADERS_FILES})

target_link_libraries(${PROJECT_NAME} score ${SL_LIBRARIES})

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
