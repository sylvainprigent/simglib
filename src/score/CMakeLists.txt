############################################################
#
# $Id$
#
# Copyright (c) score 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(score)

## #################################################################
## Define files
## #################################################################

SET(SRC_FILES
  SColor.cpp
  SCoreShortcuts.cpp
  SFile.cpp
  SMath.cpp
  SObserver.cpp
  SObservable.cpp
  SPath.cpp
  SProcess.cpp
  SRandomGeneratorPM.cpp
  SRandomGeneratorPMS.cpp
  SStringOp.cpp
)

SET(HEADERS_FILES
  SColor.h
  SCoreShortcuts.h
  SException.h
  SFile.h
  SMath.h
  SObserver.h
  SObservable.h
  SPath.h
  SProcess.h
  SRandomGeneratorPM.h
  SRandomGeneratorPMS.h
  SSingleton.h
  SStringOp.h
)
  
## #################################################################
## Library target
## #################################################################

add_library(${PROJECT_NAME} ${SRC_FILES} ${HEADERS_FILES})

target_link_libraries(${PROJECT_NAME} ${SL_LIBRARIES})


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
