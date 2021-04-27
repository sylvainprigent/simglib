/// \file SVarianceFilter.cpp
/// \brief SVarianceFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SVarianceFilter.h"
#include "score/SException.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void varianceFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, int rx, int ry, int rz, int rt, float* output){

    output = new float[sx*sy*sz*st*sc];
    float filterSize = (2*rx+1)*(2*ry+1)*(2*rz+1)*(2*rt+1);
    float* neighbors = new float[unsigned(filterSize)];
    //unsigned int pos;
    //float mean, var;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = rx ; x < sx-rx ; x++){
            for (int y = ry ; y < sy-ry ; y++){
                for (int z = rz ; z < sz-rz ; z++){
                    for (int t = rt ; t < st-rt ; t++){
                        unsigned int pos = 0;
                        float mean = 0.0;
                        float var = 0.0;
                        for (int fx = -rx ; fx <= rx ; fx++){
                            for (int fy = -ry ; fy <= ry ; fy++){
                                for (int fz = -rz ; fz <= rz ; fz++){
                                    for (int ft = -rt ; ft <= rt ; ft++){
                                        neighbors[pos] = image[c + sc*((t-ft) + st*((z-fz) + sz*((y-fy) + sy*(x-fx))))];
                                        mean += neighbors[pos];
                                        var += neighbors[pos]*neighbors[pos];
                                        pos++;
                                    }
                                }
                            }
                        }
                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = (var-mean*mean/filterSize)/(filterSize-1);
                    }
                }
            }
        }
    }
}

}