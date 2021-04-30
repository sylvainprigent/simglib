find_path(FFTW_INCLUDE_DIR "fftw3.h" 
  HINTS ${FFTW_ROOT}/include
  /usr/include
  /usr/local/include
  /opt/local/include 
  /home/sprigent/anaconda3/include
)

find_library(FFTW_LIBRARIES NAMES fftw3f 
  HINTS ${FFTW_ROOT}/lib
  PATHS /usr/lib /usr/local/lib /opt/local/lib /home/sprigent/anaconda3/lib
)

find_library(FFTW3F_THREAD_LIBRARIES NAMES fftw3f_threads 
  HINTS ${FFTW_ROOT}/lib
  PATHS /usr/lib /usr/local/lib /opt/local/lib /home/sprigent/anaconda3/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG FFTW_LIBRARIES FFTW_INCLUDE_DIR)