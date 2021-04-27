/// \file SConvolve.cpp
/// \brief SConvolve functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SConvolve.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

    float* naive_convolution(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, 
                             float* kernel, unsigned int ksx, unsigned int ksy, unsigned int ksz, unsigned int kst)
                             {

        float* outputBuffer = new float[sx*sy*sz*st*sc];

        int halfX = ksx/2;
        int halfY = ksy/2;
        int halfZ = ksz/2;
        int halfT = kst/2;

        float val;
        for (int c = 0 ; c < sc ; c++){

    #pragma omp parallel for
            for (int x = halfX ; x < sx-halfX ; x++){
                for (int y = halfY ; y < sy-halfY ; y++){
                    for (int z = halfZ ; z < sz-halfZ ; z++){
                        for (int t = halfT ; t < st-halfT ; t++){

                            val = 0.0;
                            for (int fx = -halfX ; fx <= halfX ; fx++){
                                for (int fy = -halfY ; fy <= halfY ; fy++){
                                    for (int fz = -halfZ ; fz <= halfZ ; fz++){
                                        for (int ft = -halfT ; ft <= halfT ; ft++){
                                            val += image[c + sc*(t+ft + st*(z+fz + sz*((y+fy)) + sy*(x+fx)))] * kernel[t+halfT + kst*(z+halfZ + ksz*((y+halfY)) + ksy*(x+halfX))];
                                        }
                                    }
                                }
                            }
                            outputBuffer[c + sc*(t + st*(z + sz*(y + sy*x)))] = val;

                        }
                    }
                }
            }
        }
        return outputBuffer;
}

}