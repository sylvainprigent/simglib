/// \file SMinMaxFilter.cpp
/// \brief SMinMaxFilter functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SMinMaxFilter.h"
#include "score/SException.h"
#include "score/SMath.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void minMaxFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, int rx, int ry, int rz, int rt, float* output, std::string direction){

    output = new float[sx*sy*sz*st*sc];
    //float value, v;

    if (direction == "min"){
        float min = SMath::FMAX;

        for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
            for (int x = rx ; x < sx-rx ; x++){
                for (int y = ry ; y < sy-ry ; y++){
                    for (int z = rz ; z < sz-rz ; z++){
                        for (int t = rt ; t < st-rt ; t++){

                            float value = min;
                            for (int fx = -rx ; fx <= rx ; fx++){
                                for (int fy = -ry ; fy <= ry ; fy++){
                                    for (int fz = -rz ; fz <= rz ; fz++){
                                        for (int ft = -rt ; ft <= rt ; ft++){
                                            float v = image[c + sc*((t-ft) + st*((z-fz) + sz*((y-fy) + sy*(x-fx))))];
                                            if (v < value){
                                                value = v;
                                            }
                                        }
                                    }
                                }
                            }
                            output[c + sc*(t + st*(z + sz*(y + sy*x)))] = value;
                        }
                    }
                }
            }

        }
    }
    else if (direction == "max"){
        float max = SMath::FMIN;

        for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
            for (int x = rx ; x < sx-rx ; x++){
                for (int y = ry ; y < sy-ry ; y++){
                    for (int z = rz ; z < sz-rz ; z++){
                        for (int t = rt ; t < st-rt ; t++){

                            float value = max;
                            for (int fx = -rx ; fx <= rx ; fx++){
                                for (int fy = -ry ; fy <= ry ; fy++){
                                    for (int fz = -rz ; fz <= rz ; fz++){
                                        for (int ft = -rt ; ft <= rt ; ft++){
                                            float v = image[c + sc*((t-ft) + st*((z-fz) + sz*((y-fy) + sy*(x-fx))))];
                                            if (v > value){
                                                value = v;
                                            }
                                        }
                                    }
                                }
                            }
                            output[c + sc*(t + st*(z + sz*(y + sy*x)))] = value;
                        }
                    }
                }
            }

        }
    }
    else{
        throw SException("SMinMaxFilter direction must be 'min' or 'max'");
    }
}

}