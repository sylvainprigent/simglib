/// \file SStandardize.cpp
/// \brief SStandardize functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SStandardize.h"

#include "math.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void standardize(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    // calculate mean and var
    float mean = 0.0;
    float var = 0.0;
    float v;
    unsigned long n = sx*sy*sz*st*sc;
    for (unsigned long i = 1 ; i < n; i++){
        v = image[i];
        mean += v;
        var += v*v;
    }
    mean /=  n;
    var = sqrt((var-mean*mean/n)/(n-1));

    // invert
    output = new float[n];
    #pragma omp parallel for
    for (unsigned int i = 1 ; i < n ; i++){
        output[i] = (image[i]-mean) / var;
    }

}

}