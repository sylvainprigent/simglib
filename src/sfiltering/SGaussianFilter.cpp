/// \file SGaussianFilter.cpp
/// \brief SGaussianFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SGaussianFilter.h"
#include "score/SException.h"
#include "score/SMath.h"
#include "math.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void SGaussian2dFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const float& sigma)
{
    // compute the kernel
    float sigma2 = sigma*sigma;
    int halfSizeGF = int(0.5+sqrt( double(2*double(sigma2)*log( double(300) ) )  ) );
    int sizeGF = 2*halfSizeGF+1;
    float norm = float( sqrt( double(1/(2*SMath::PI*sigma2) ) ) );

    float *kernel = new float[sizeGF*sizeGF];
    for (int i = - halfSizeGF; i <= halfSizeGF ; i++ ){
        for (int j = - halfSizeGF; j <= halfSizeGF ; j++ ){
            kernel[sizeGF*(i+halfSizeGF)+(j+halfSizeGF)] = norm*exp( - i*i/(2*sigma2) - j*j/(2*sigma2)  );
        }
    }
    output = new float[sx*sy*sz*st*sc];
    for (int c = 0 ; c < sc ; c++){
        for (int t = 0 ; t < st ; t++){
            for (int z = 0 ; z < sz ; z++){

#pragma omp parallel for
                for (int x = halfSizeGF ; x <= sx-halfSizeGF ; x++){
                    for (int y = halfSizeGF ; y <= sy-halfSizeGF ; y++){

                        float val = 0;
                        for (int fx = -halfSizeGF ; fx < halfSizeGF ; fx++ ){
                            for (int fy = -halfSizeGF ; fy < halfSizeGF ; fy++ ){
                                val += image[c + sc*(t + st*(z + sz*((y+fy) + sy*(x+fx))))]*kernel[sizeGF*(fx+halfSizeGF)+(fy+halfSizeGF)];
                            }
                        }
                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = val;
                    }
                }
            }
        }
    }
}

void SGaussian3dFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const float& sigmaX, const float& sigmaY, const float& sigmaZ)
{
    // compute the kernel
    float sigmaX2 = sigmaX*sigmaX;
    float sigmaY2 = sigmaY*sigmaY;
    float sigmaZ2 = sigmaZ*sigmaZ;
    int halfX = int(0.5+sqrt( double(2*double(sigmaX2)*log( double(300) ) )  ) );
    int halfY = int(0.5+sqrt( double(2*double(sigmaY2)*log( double(300) ) )  ) );
    int halfZ = int(0.5+sqrt( double(2*double(sigmaZ2)*log( double(300) ) )  ) );
    int sizeX = 2*halfX+1;
    int sizeY = 2*halfY+1;
    int sizeZ = 2*halfZ+1;

    float *kernel = new float[sizeX*sizeY*sizeZ];
    float norm = 0;
    float kv;
    for (int i = - halfX; i <= halfX ; i++ ){
        for (int j = - halfY; j <= halfY ; j++ ){
            for (int k = - halfZ; k <= halfZ ; k++ ){

                kv = exp( - i*i/(2*sigmaX2) - j*j/(2*sigmaY2) - k*k/(2*sigmaZ2)  );
                kernel[k* sizeZ*(sizeY*i*j)] = kv;
                norm += kv;
            }
        }
    }
    for (int i = 0 ; i < sizeX*sizeY*sizeZ ; i++){
        kernel[i] /= norm;
    }

    // do convolution
    output = new float[sx*sy*sz*st];
    float val;
    for (int c = 0 ; c < sc ; c++){
        for (int t = 0 ; t < st ; t++){

#pragma omp parallel for
                for (int x = halfX ; x <= sx-halfX ; x++){
                    for (int y = halfY ; y <= sy-halfY ; y++){
                        for (int z = halfZ ; z < sz-halfZ ; z++){

                        val = 0;
                        for (int fx = -halfX ; fx < halfX ; fx++ ){
                            for (int fy = -halfY ; fy < halfY ; fy++ ){
                                for (int fz = -halfZ ; fz < halfZ ; fz++ ){
                                    val += image[c + sc*(t + st*(z+fz + sz*((y+fy) + sy*(x+fx))))]*kernel[fz+halfZ + sizeZ*(sizeY*(fx+halfX)+(fy+halfY))];

                                }
                            }
                        }
                        output[c + sc*(t + st*(z + sz*(y + sy*x)))] = val;

                    }

                }

            }
        }
    }
}

}