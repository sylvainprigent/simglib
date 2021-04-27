/// \file SResizeCanvas.cpp
/// \brief SResizeCanvas function
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SResizeCanvas.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void resizeCanvas(float* image, unsigned int sx, unsigned int sy, unsigned int sz, float* output, const unsigned int& csx, const unsigned int& csy, const unsigned int& csz)
{
    unsigned int offsetX = (csx-sx)/2;
    unsigned int offsetY = (csy-sy)/2;

    if ( sz == 1 ){
        unsigned int bs = csx*csy;
        output = new float[bs];
        for (unsigned int i = 0 ; i < bs ; ++i)
        {
            output[i] = 0.0;
        }

        for (unsigned int x = 0 ; x < sx ; x++){
            for (unsigned int y = 0 ; y < sy ; y++){
                output[csy*(x+offsetX)+(y+offsetY)] = image[sy*x+y];
            }
        }
    }
    else{
        unsigned int bs = csx*csy*csz;
        output = new float[bs];
        for (unsigned int i = 0 ; i < bs ; ++i)
        {
            output[i] = 0.0;
        }

        unsigned int offsetZ = (csz-sz)/2;
        for (unsigned int x = 0 ; x < sx ; x++){
            for (unsigned int y = 0 ; y < sy ; y++){
                for (unsigned int z = 0 ; z < sz ; z++){
                    output[ (z+offsetZ) + csz*((y+offsetY) + csy*(x+offsetX))] = image[ z + sz*(y + sy*x)];
                }
            }
        }
    }
}

}