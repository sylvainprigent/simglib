/// \file SMedianFilter.cpp
/// \brief SMedianFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SMedianFilter.h"
#include "score/SException.h"
#include "score/SMath.h"

#include <vector>
#include <algorithm>

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void medianFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, int rx, int ry, int rz, int rt, float* output)
{
    //output = new float[sx*sy*sz*st*sc];
    float filterSize = (2*rx+1)*(2*ry+1)*(2*rz+1)*(2*rt+1);
    std::vector<float> neighbors;
    neighbors.resize(unsigned(filterSize));
    for (int c = 0 ; c < sc ; c++){
        for (int x = rx ; x < sx-rx ; x++){
            for (int y = ry ; y < sy-ry ; y++){
                for (int z = rz ; z < sz-rz ; z++){
                    for (int t = rt ; t < st-rt ; t++){
                        unsigned int pos = 0;
                        for (int fx = -rx ; fx <= rx ; fx++){
                            for (int fy = -ry ; fy <= ry ; fy++){
                                for (int fz = -rz ; fz <= rz ; fz++){
                                    for (int ft = -rt ; ft <= rt ; ft++){
                                        neighbors[pos] = image[c + sc*((t-ft) + st*((z-fz) + sz*((y-fy) + sy*(x-fx))))];
                                        pos++;
                                    }
                                }
                            }
                        }
                        std::sort(neighbors.begin(), neighbors.end());
                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = neighbors[ unsigned(filterSize/2) ];
                    }
                }
            }
        }
    }
}

}