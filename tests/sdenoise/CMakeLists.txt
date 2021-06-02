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
  TestSpitfireDenoise2dHV.cpp
  TestSpitfireDenoise2dSV.cpp
  TestSpitfireDenoise3dHV.cpp
  TestSpitfireDenoise4dHV.cpp
)

## ###################################################################
## Configure file
## ###################################################################

if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/celegans.tif)
  set(SAMPLE2D ${PROJECT_SOURCE_DIR}/../dataset/celegans.tif)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/Cell02_HELA_SD_Tubtracker_CY5_15p_M6PR_15p_GFP_15p_20ms_50tp_w14-Spinning1-CY5_t06.tif)
  set(SAMPLE3D ${PROJECT_SOURCE_DIR}/../dataset/Cell02_HELA_SD_Tubtracker_CY5_15p_M6PR_15p_GFP_15p_20ms_50tp_w14-Spinning1-CY5_t06.tif)
endif()

if(EXISTS ${PROJECT_SOURCE_DIR}/../dataset/Cell02_HELA_SD_Tubtracker_CY5_15p_M6PR_15p_GFP_15p_20ms_50tp_w14-Spinning1-CY5.txt)
  set(SAMPLE4D ${PROJECT_SOURCE_DIR}/../dataset/Cell02_HELA_SD_Tubtracker_CY5_15p_M6PR_15p_GFP_15p_20ms_50tp_w14-Spinning1-CY5.txt)
endif()

set(SAMPLE2DDENOISED ${PROJECT_SOURCE_DIR}/../dataset/celegans_denoised.tif)
set(SAMPLE2DDENOISEDSV ${PROJECT_SOURCE_DIR}/results/TestSpitfire2dSV_ref.tif)
set(SAMPLE3DDENOISED ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dHV_ref.tif)
set(SAMPLE4DDENOISED ${PROJECT_SOURCE_DIR}/results/denoised_4d.txt)

if(EXISTS ${PROJECT_SOURCE_DIR}/sdenoiseTestConfig.h.in)
  configure_file(sdenoiseTestConfig.h.in ${CMAKE_BINARY_DIR}/sdenoiseTestConfig.h)
endif()

## #################################################################
## Build rules
## #################################################################

SET(TESTS "")

foreach (test ${${PROJECT_NAME}_MAIN_SOURCES})
  get_filename_component(TName ${test} NAME_WE)

  add_executable(${TName} ${test})
  target_link_libraries(${TName} score scli sdenoise simage simageio smanipulate ${SL_LIBRARIES})

  add_test(${TName} ${EXECUTABLE_OUTPUT_PATH}/${TName})

  set(TEST ${TESTS} ${EXECUTABLE_OUTPUT_PATH}/${TName})
endforeach()

add_custom_target(testsdenoise COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${TESTS})

## #################################################################
## Source file layout in development environments like Visual Studio
## #################################################################
SOURCE_GROUP("Header Files" REGULAR_EXPRESSION .*\\.h\$)
SOURCE_GROUP("Source Files" REGULAR_EXPRESSION .*\\.cpp\$)
