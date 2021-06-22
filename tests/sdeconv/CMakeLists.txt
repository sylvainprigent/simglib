### CMakeLists.txt --- 
##
######################################################################
## 
### Commentary: 
## 
######################################################################
## 
### Change log:
## 
######################################################################
enable_testing()

project(sdenoiseTest)

enable_testing()

## #################################################################
## Input
## #################################################################

set(${PROJECT_NAME}_MAIN_SOURCES
  TestRichardsonLucy2d.cpp
  TestRichardsonLucy3d.cpp
  TestSpitfireDeconv2dHV.cpp
  TestSpitfireDeconv2dSV.cpp
  TestSpitfireDeconv3dHV.cpp
  TestSpitfireDeconv3dSV.cpp
  TestWiener2d.cpp
)

## ###################################################################
## Configure file
## ###################################################################

if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/celegans.tif)
  set(SAMPLE2D ${PROJECT_SOURCE_DIR}/../dataset/celegans.tif)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/pollen.tif)
  set(SAMPLE3D ${PROJECT_SOURCE_DIR}/../dataset/pollen.tif)
endif()
if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/pollen_psf.tif)
  set(PSF3D ${PROJECT_SOURCE_DIR}/../dataset/pollen_psf.tif)
endif()

set(SAMPLE2DDECONV ${PROJECT_SOURCE_DIR}/../dataset/TestSpitfire2dHV_ref.tif)
set(SAMPLE2DDECONVSV ${PROJECT_SOURCE_DIR}/results/TestSpitfire2dSV_ref.tif)
set(SAMPLE2DDECONVHV ${PROJECT_SOURCE_DIR}/results/TestSpitfire2dHV_ref.tif)
set(SAMPLE2DDECONVRL ${PROJECT_SOURCE_DIR}/results/TestRichardsonLucy2d_ref.tif)
set(SAMPLE2DDECONVW ${PROJECT_SOURCE_DIR}/results/TestWiener2d_ref.tif)
set(SAMPLE3DDECONV ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dHV_ref.tif)
set(SAMPLE3DDECONVSV ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dSV_ref.tif)
set(SAMPLE3DDECONVHV ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dHV_ref.tif)
set(SAMPLE3DDECONVRL ${PROJECT_SOURCE_DIR}/results/TestRichardsonLucy3d_ref.tif)


if(EXISTS ${PROJECT_SOURCE_DIR}/sdeconvTestConfig.h.in)
  configure_file(sdeconvTestConfig.h.in ${CMAKE_BINARY_DIR}/sdeconvTestConfig.h)
endif()

## #################################################################
## Build rules
## #################################################################

SET(TESTS "")

foreach (test ${${PROJECT_NAME}_MAIN_SOURCES})
  get_filename_component(TName ${test} NAME_WE)

  add_executable(${TName} ${test})
  target_link_libraries(${TName} score scli sdeconv simage simageio smanipulate ${SL_LIBRARIES})

  add_test(${TName} ${EXECUTABLE_OUTPUT_PATH}/${TName})

  set(TEST ${TESTS} ${EXECUTABLE_OUTPUT_PATH}/${TName})
endforeach()

add_custom_target(testsdeconv COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${TESTS})

## #################################################################
## Source file layout in development environments like Visual Studio
## #################################################################
SOURCE_GROUP("Header Files" REGULAR_EXPRESSION .*\\.h\$)
SOURCE_GROUP("Source Files" REGULAR_EXPRESSION .*\\.cpp\$)
