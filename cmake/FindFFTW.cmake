find_path(FFTW_INCLUDE_DIR "fftw3.h" 
  HINTS ${FFTW_ROOT}/include
  /usr/include
  /usr/local/include
  /opt/local/include 
  /Users/sprigent/opt/anaconda3/include
)

find_library(FFTW_LIBRARIES NAMES fftw3f 
  HINTS ${FFTW_ROOT}/lib
  PATHS /usr/lib /usr/local/lib /opt/local/lib /Users/sprigent/opt/anaconda3/lib
)

#find_library(FFTW3F_THREAD_LIBRARIES NAMES fftw3f_threads 
#  HINTS ${FFTW_ROOT}/lib
#  PATHS /usr/lib /usr/local/lib /opt/local/lib
#)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDE_DIR)