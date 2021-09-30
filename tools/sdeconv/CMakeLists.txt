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
project(SDeconvTools)

## #################################################################
## Define files
## #################################################################
include_directories(${sl_SRC_INCLUDE_DIR})

SET(SRC_FILES 
	)

## #################################################################
## Exe generation
## #################################################################
SET(NAME simggaussian3dpsf)
SET(FILE ${SRC_FILES} gaussian3dpsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} score simage sdeconv spadding sfft smanipulate simageio scli)

INSTALL(TARGETS simggaussian3dpsf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simggaussian2dpsf)
SET(FILE ${SRC_FILES} gaussian2dpsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simggaussian2dpsf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simggibsonlannipsf)
SET(FILE ${SRC_FILES} sgibsonlannipsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simggaussian2dpsf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgrichardsonlucy2d)
SET(FILE ${SRC_FILES} richardsonlucy2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgrichardsonlucy2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgrichardsonlucy3d)
SET(FILE ${SRC_FILES} richardsonlucy3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgrichardsonlucy3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgwiener2d)
SET(FILE ${SRC_FILES} wiener2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgwiener2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgwiener3d)
SET(FILE ${SRC_FILES} wiener3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgwiener3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgwienerairyscan2d)
SET(FILE ${SRC_FILES} wienerairyscan2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli )

INSTALL(TARGETS simgwienerairyscan2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredeconv2d)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgspitfiredeconv2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredeconv3d)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgspitfiredeconv3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredeconv2dplan)
SET(FILE ${SRC_FILES} spitfire2dplan.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS simgspitfiredeconv2dplan
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME simgspitfiredeconv2destimate)
SET(FILE ${SRC_FILES} spitfire2d_estimate.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate sfiltering simage simageio scli sdata sdataio)
