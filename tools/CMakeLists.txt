############################################################
#
# $Id$
#
# Copyright (c) simglib 2020
#
# AUTHOR:
# Sylvain Prigent
# 

## #################################################################
## Add subdirectory
## #################################################################
add_subdirectory( scli )
add_subdirectory( sdeconv )
add_subdirectory( sdenoise )
add_subdirectory( smanipulate )
add_subdirectory( spadding )
add_subdirectory( sroi )
add_subdirectory( sshape )

if (${simglib_BUILD_CUDA})
  add_subdirectory( cugrad )
  add_subdirectory( cudeconv )
  add_subdirectory( cudenoise )
endif(${simglib_BUILD_CUDA})  