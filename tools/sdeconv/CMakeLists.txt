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
SET(NAME gaussian3dpsf)
SET(FILE ${SRC_FILES} gaussian3dpsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

SET(NAME gaussian2dpsf)
SET(FILE ${SRC_FILES} gaussian2dpsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

SET(NAME richardsonlucy2d)
SET(FILE ${SRC_FILES} richardsonlucy2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)


SET(NAME richardsonlucy3d)
SET(FILE ${SRC_FILES} richardsonlucy3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

SET(NAME wiener2d)
SET(FILE ${SRC_FILES} wiener2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

SET(NAME wiener3d)
SET(FILE ${SRC_FILES} wiener3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

SET(NAME wienerairyscan2d)
SET(FILE ${SRC_FILES} wienerairyscan2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli )

SET(NAME spitfiredeconv2d)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

SET(NAME spitfiredeconv3d)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

SET(NAME spitfiredeconv2dplan)
SET(FILE ${SRC_FILES} spitfire2dplan.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)