############################################################
#
#
# Copyright (c) SImg 2020
#
# AUTHOR:
# Sylvain Prigent (sylvain.prigent@inria.fr)
# 
## ###############################################################

enable_testing()

if (${simglib_BUILD_CUDA})
  add_subdirectory( cudeconv )
  add_subdirectory( cudenoise )
endif(${simglib_BUILD_CUDA})  

add_subdirectory(scli)
add_subdirectory(score)
#add_subdirectory(sdata)
add_subdirectory(sdeconv)
add_subdirectory(sdenoise)
add_subdirectory(simage)