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
target_link_libraries(${NAME} ${SL_LIBRARIES} score simage sdeconv spadding sfft smanipulate simageio scli)

INSTALL(TARGETS gaussian3dpsf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME gaussian2dpsf)
SET(FILE ${SRC_FILES} gaussian2dpsf.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS gaussian2dpsf
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME richardsonlucy2d)
SET(FILE ${SRC_FILES} richardsonlucy2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS richardsonlucy2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME richardsonlucy3d)
SET(FILE ${SRC_FILES} richardsonlucy3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

INSTALL(TARGETS richardsonlucy3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME wiener2d)
SET(FILE ${SRC_FILES} wiener2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS wiener2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME wiener3d)
SET(FILE ${SRC_FILES} wiener3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli)

INSTALL(TARGETS wiener3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME wienerairyscan2d)
SET(FILE ${SRC_FILES} wienerairyscan2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv sfft smanipulate simage simageio scli )

INSTALL(TARGETS wienerairyscan2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME spitfiredeconv2d)
SET(FILE ${SRC_FILES} spitfire2d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS spitfiredeconv2d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME spitfiredeconv3d)
SET(FILE ${SRC_FILES} spitfire3d.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS spitfiredeconv3d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

SET(NAME spitfiredeconv2dplan)
SET(FILE ${SRC_FILES} spitfire2dplan.cpp)
add_executable(${NAME} ${FILE})
target_link_libraries(${NAME} ${SL_LIBRARIES} sdeconv spadding sfft smanipulate simage simageio scli)

INSTALL(TARGETS spitfiredeconv2dplan
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)