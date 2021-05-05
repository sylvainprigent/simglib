############################################################
#
# $Id$
#
# Copyright (c) SImg 2010
#
# AUTHOR:
# Sylvain Prigent
# 

## #################################################################
## Options
## #################################################################

## #################################################################
## Add subdirectory
## #################################################################
add_subdirectory( scli )
add_subdirectory( score )
add_subdirectory( sdata )
add_subdirectory( sdataio )
add_subdirectory( sdeconv )
add_subdirectory( sdenoise )
add_subdirectory( sfft )
add_subdirectory( sfiltering )
add_subdirectory( simage )
add_subdirectory( simageio )
add_subdirectory( smanipulate )
add_subdirectory( smeasuring )
add_subdirectory( soperate )
add_subdirectory( spadding )
add_subdirectory( sroi )
add_subdirectory( sshape )

if (${simglib_BUILD_CUDA})
  add_subdirectory( cugrad )
  add_subdirectory( cudeconv )
  add_subdirectory( cudenoise )
endif(${simglib_BUILD_CUDA})  