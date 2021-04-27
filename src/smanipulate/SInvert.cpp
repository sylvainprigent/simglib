/// \file SInvert.cpp
/// \brief SInvert function
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SInvert.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void invert(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output){

    // calculate max
    unsigned long bs = sx*sy*sz*st*sc;
    float max = image[0];
    for (unsigned int i = 1 ; i < bs ; i++){
        if (image[i] > max){
            max = image[i];
        }
    }

    // invert
    float* bufferOut = new float[bs];
#pragma omp parallel for
    for (unsigned int i = 1 ; i < bs ; i++){
        output[i] = max - image[i];
    }
}

}
