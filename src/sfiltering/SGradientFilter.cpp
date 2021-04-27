/// \file SGradientFilter.cpp
/// \brief SGradientFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SGradientFilter.h"
#include "score/SException.h"
#include "math.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void gradientXY(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    output = new float[sx*sy*sz*st*sc];
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                for (int z = 0 ; z < sz ; z++){
                    for (int t = 0 ; t < st ; t++){
                        float x_m1 = image[c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        float x_p1 = image[c + sc*(t + st*(z + sz*(y + sy*(x+1))))];
                        float y_m1 = image[c + sc*(t + st*(z + sz*((y-1) + sy*x)))];
                        float y_p1 = image[c + sc*(t + st*(z + sz*((y+1) + sy*x)))];

                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = float(sqrt( pow(x_m1 - x_p1,2.0) + pow(y_m1 - y_p1, 2.0) ));
                    }
                }
            }
        }
    }
}

void gradientXYZ(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{    
    output = new float[sx*sy*sz*st*sc];
    for (int c = 0 ; c < sc ; c++){
        #pragma omp parallel for
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                for (int z = 1 ; z < sz-1 ; z++){
                    for (int t = 0 ; t < st ; t++){
                        float x_m1 = image[c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        float x_p1 = image[c + sc*(t + st*(z + sz*(y + sy*(x+1))))];
                        float y_m1 = image[c + sc*(t + st*(z + sz*((y-1) + sy*x)))];
                        float y_p1 = image[c + sc*(t + st*(z + sz*((y+1) + sy*x)))];
                        float z_m1 = image[c + sc*(t + st*(z-1 + sz*(y + sy*x)))];
                        float z_p1 = image[c + sc*(t + st*(z+1 + sz*(y + sy*x)))];

                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = float(sqrt( pow(x_m1 - x_p1,2.0) + pow(y_m1 - y_p1, 2.0) + pow(z_m1 - z_p1, 2.0)));
                    }
                }
            }
        }
    }
}

void gradientXYT(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{    
    output = new float[sx*sy*sz*st*sc];
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                for (int z = 0 ; z < sz ; z++){
                    for (int t = 1 ; t < st-1 ; t++){
                        float x_m1 = image[c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        float x_p1 = image[c + sc*(t + st*(z + sz*(y + sy*(x+1))))];
                        float y_m1 = image[c + sc*(t + st*(z + sz*((y-1) + sy*x)))];
                        float y_p1 = image[c + sc*(t + st*(z + sz*((y+1) + sy*x)))];
                        float t_m1 = image[c + sc*(t-1 + st*(z + sz*(y + sy*x)))];
                        float t_p1 = image[c + sc*(t+1 + st*(z + sz*(y + sy*x)))];

                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = float(sqrt( pow(x_m1 - x_p1,2.0) + pow(y_m1 - y_p1, 2.0) + pow(t_m1 - t_p1, 2.0)));
                    }
                }
            }
        }
    }
}

void gradientXYZT(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output)
{
    output = new float[sx*sy*sz*st*sc];
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                for (int z = 1 ; z < sz-1 ; z++){
                    for (int t = 1 ; t < st-1 ; t++){
                        float x_m1 = image[c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        float x_p1 = image[c + sc*(t + st*(z + sz*(y + sy*(x+1))))];
                        float y_m1 = image[c + sc*(t + st*(z + sz*((y-1) + sy*x)))];
                        float y_p1 = image[c + sc*(t + st*(z + sz*((y+1) + sy*x)))];
                        float z_m1 = image[c + sc*(t + st*(z-1 + sz*(y + sy*x)))];
                        float z_p1 = image[c + sc*(t + st*(z+1 + sz*(y + sy*x)))];
                        float t_m1 = image[c + sc*(t-1 + st*(z + sz*(y + sy*x)))];
                        float t_p1 = image[c + sc*(t+1 + st*(z + sz*(y + sy*x)))];
                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = float(sqrt( pow(x_m1 - x_p1,2.0) + pow(y_m1 - y_p1, 2.0) + pow(z_m1 - z_p1, 2.0) + pow(t_m1 - t_p1, 2.0)));
                    }
                }
            }
        }
    }
}

}