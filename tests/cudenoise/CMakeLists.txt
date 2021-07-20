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

project(cudenoiseTest)

enable_testing()

## #################################################################
## Input
## #################################################################

set(${PROJECT_NAME}_MAIN_SOURCES
  TestSpitfireCuDenoise2dHV.cpp
  TestSpitfireCuDenoise2dSV.cpp
  TestSpitfireCuDenoise3dHV.cpp
  TestSpitfireCuDenoise3dSV.cpp
  TestSpitfireCuDenoise4dHV.cpp
  TestSpitfireCuDenoise4dSV.cpp
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

set(SAMPLE2DDENOISED ${PROJECT_SOURCE_DIR}/results/TestSpitfire2dHV_ref.tif)
set(SAMPLE2DDENOISEDSV ${PROJECT_SOURCE_DIR}/results/TestSpitfire2dSV_ref.tif)
set(SAMPLE3DDENOISED ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dHV_ref.tif)
set(SAMPLE3DDENOISEDMEAN ${PROJECT_SOURCE_DIR}/results/TestMeanFilter_ref.tif)
set(SAMPLE3DDENOISEDSV ${PROJECT_SOURCE_DIR}/results/TestSpitfire3dSV_ref.tif)
set(SAMPLE4DDENOISED ${PROJECT_SOURCE_DIR}/results/denoised_4d.txt)
set(SAMPLE4DDENOISEDSV ${PROJECT_SOURCE_DIR}/results/denoised_4d_sv.txt)


if(EXISTS ${PROJECT_SOURCE_DIR}/cudenoiseTestConfig.h.in)
  configure_file(cudenoiseTestConfig.h.in ${CMAKE_BINARY_DIR}/cudenoiseTestConfig.h)
endif()

## #################################################################
## Build rules
## #################################################################

SET(TESTS "")

foreach (test ${${PROJECT_NAME}_MAIN_SOURCES})
  get_filename_component(TName ${test} NAME_WE)

  add_executable(${TName} ${test})
  target_link_libraries(${TName} score scli cudenoise simage simageio smanipulate ${SL_LIBRARIES})

  add_test(${TName} ${EXECUTABLE_OUTPUT_PATH}/${TName})

  set(TEST ${TESTS} ${EXECUTABLE_OUTPUT_PATH}/${TName})
endforeach()

add_custom_target(testcudenoise COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${TESTS})

## #################################################################
## Source file layout in development environments like Visual Studio
## #################################################################
SOURCE_GROUP("Header Files" REGULAR_EXPRESSION .*\\.h\$)
SOURCE_GROUP("Source Files" REGULAR_EXPRESSION .*\\.cpp\$)
