############################################################
#
# $Id$
#
# Copyright (c) SMeasuring 2020
#
# AUTHOR:
# Sylvain Prigent
# 

# Project name
project(smeasuring)

## #################################################################
## Define files
## #################################################################

SET(SRC_FILES
  SRoiInfo.cpp
)

SET(HEADERS_FILES
  SRoiInfo.h
)
  
## #################################################################
## Library target
## #################################################################

add_library(${PROJECT_NAME} ${SRC_FILES} ${HEADERS_FILES})
target_link_libraries(${PROJECT_NAME} score simage sroi smanipulate sfiltering ${SL_LIBRARIES})

INSTALL(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
install(FILES ${HEADERS_FILES} DESTINATION include/simglib/${PROJECT_NAME})
install(FILES ../../build/${PROJECT_NAME}Export.h DESTINATION include/simglib/${PROJECT_NAME})

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
