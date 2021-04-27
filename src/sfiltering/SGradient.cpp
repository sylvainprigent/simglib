/// \file SGradient.cpp
/// \brief SGradient functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "math.h"

#include "SGradient.h"

namespace SImg{

void gradient2d(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* bufferGx, float* bufferGy)
{
    bufferGx = new float[sx*sy*sz*st*sc];
    bufferGy = new float[sx*sy*sz*st*sc];
    for (unsigned int c = 0 ; c < sc ; c++){
        for (unsigned int t = 0 ; t < st ; t++){
            for (unsigned int z = 0 ; z < sz ; z++){
                // gradient x, y
                for (unsigned int x = 1 ; x < sx-1 ; x++){
                    for (unsigned int y = 1 ; y < sy-1 ; y++){
                        //buffer[ c + sc*(t + st*(z + sz*(y + sy*x)))];
                        bufferGx[c + sc*(t + st*(z + sz*(y + sy*x)))] = image[ c + sc*(t + st*(z + sz*(y + sy*(x+1))))] - image[ c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        bufferGy[c + sc*(t + st*(z + sz*(y + sy*x)))] = image[ c + sc*(t + st*(z + sz*(y+1 + sy*x)))] - image[ c + sc*(t + st*(z + sz*(y-1 + sy*x)))];
                    }
                }

                // set borders to 0.0
                for (unsigned int y = 0 ; y <  sy ; y++){
                    bufferGx[c + sc*(t + st*(z + sz*(y + sy*0)))] = 0.0;
                    bufferGx[c + sc*(t + st*(z + sz*(y + sy*(sx-1))))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(y + sy*0)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(y + sy*(sx-1))))] = 0.0;
                }
                for (unsigned int x = 0 ; x <  sx ; x++){
                    bufferGx[c + sc*(t + st*(z + sz*(0 + sy*x)))] = 0.0;
                    bufferGx[c + sc*(t + st*(z + sz*(sy-1 + sy*x)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(0 + sy*x)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(sy-1 + sy*x)))] = 0.0;
                }
            }
        }
    }
}

void gradient3d(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* bufferGx, float* bufferGy, float* bufferGz){

    bufferGx = new float[sx*sy*sz*st*sc];
    bufferGy = new float[sx*sy*sz*st*sc];
    bufferGz = new float[sx*sy*sz*st*sc];
    for (unsigned int c = 0 ; c < sc ; c++){
        for (unsigned int t = 0 ; t < st ; t++){
            // gradient x, y, z
            for (unsigned int z = 1 ; z < sz-1 ; z++){
                for (unsigned int x = 1 ; x < sx-1 ; x++){
                    for (unsigned int y = 1 ; y < sy-1 ; y++){
                        bufferGx[c + sc*(t + st*(z + sz*(y + sy*x)))] = image[ c + sc*(t + st*(z + sz*(y + sy*(x+1))))] - image[ c + sc*(t + st*(z + sz*(y + sy*(x-1))))];
                        bufferGy[c + sc*(t + st*(z + sz*(y + sy*x)))] = image[ c + sc*(t + st*(z + sz*(y+1 + sy*x)))] - image[ c + sc*(t + st*(z + sz*(y-1 + sy*x)))];
                        bufferGz[c + sc*(t + st*(z + sz*(y + sy*x)))] = image[ c + sc*(t + st*(z+1 + sz*(y + sy*x)))] - image[ c + sc*(t + st*(z-1 + sz*(y + sy*x)))];
                    }
                }
            }
            // set borders to 0.0
            for (unsigned int z = 0 ; z < sz ; z++){
                for (unsigned int y = 0 ; y <  sy ; y++){
                    bufferGx[c + sc*(t + st*(z + sz*(y + sy*0)))] = 0.0;
                    bufferGx[c + sc*(t + st*(z + sz*(y + sy*(sx-1))))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(y + sy*0)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(y + sy*(sx-1))))] = 0.0;
                    bufferGz[c + sc*(t + st*(z + sz*(y + sy*0)))] = 0.0;
                    bufferGz[c + sc*(t + st*(z + sz*(y + sy*(sx-1))))] = 0.0;
                }
                for (unsigned int x = 0 ; x <  sx ; x++){
                    bufferGx[c + sc*(t + st*(z + sz*(0 + sy*x)))] = 0.0;
                    bufferGx[c + sc*(t + st*(z + sz*(sy-1 + sy*x)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(0 + sy*x)))] = 0.0;
                    bufferGy[c + sc*(t + st*(z + sz*(sy-1 + sy*x)))] = 0.0;
                    bufferGz[c + sc*(t + st*(z + sz*(0 + sy*x)))] = 0.0;
                    bufferGz[c + sc*(t + st*(z + sz*(sy-1 + sy*x)))] = 0.0;
                }
            }

        }
    }
}

float gradient2dL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc)
{
    float* bufferGx;
    float* bufferGy;
    gradient2d(image, sx, sy, sz, st, sc, bufferGx, bufferGy);
    unsigned long bs = sx*sy*sz*st*sc;
    float normL2 = 0.0;
    float val;
    for (unsigned long i = 0 ; i < bs ; i++){
        val = sqrt(bufferGx[i]*bufferGx[i] + bufferGy[i]*bufferGy[i]);
        normL2 +=val;
    }
    return normL2;
}

float gradient3dL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc)
{
    float* bufferGx; float* bufferGy; float* bufferGz;
    gradient3d(image, sx, sy, sz, st, sc, bufferGx, bufferGy, bufferGz);
    unsigned long bs = sx*sy*sz*st*sc;
    float normL2 = 0.0;
    for (unsigned long i = 0 ; i < bs ; i++){
        normL2 = sqrt(bufferGx[i]*bufferGx[i] + bufferGy[i]*bufferGy[i] + bufferGz[i]*bufferGz[i]);
    }
    return normL2;
}

float gradient2dL1(float* image,  unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc)
{
     float* bufferGx; float* bufferGy;
    gradient2d(image, sx, sy, sz, st, sc, bufferGx, bufferGy);
    unsigned long bs = sx*sy*sz*st*sc;
    float normL1 = 0.0;
    float val;
    for (unsigned long i = 0 ; i < bs ; i++){
        val = fabs(bufferGx[i]) + fabs(bufferGy[i]);
        normL1 +=val;
    }
    return normL1;
}

float gradient3dL1(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc)
{
    float* bufferGx; float* bufferGy; float* bufferGz;
    gradient3d(image, sx, sy, sz, st, sc, bufferGx, bufferGy, bufferGz);
    unsigned long bs = sx*sy*sz*st*sc;
    float normL1 = 0.0;
    for (unsigned long i = 0 ; i < bs ; i++){
        normL1 += fabs(bufferGx[i]) + fabs(bufferGy[i]) + fabs(bufferGz[i]);
    }
    return normL1;
}

float gradient2dMagnitude(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* magnitude)
{
    float* bufferGx; float* bufferGy;
    gradient2d( image, sx, sy, sz, st, sc, bufferGx, bufferGy);
    unsigned long bs = sx*sy*sz*st*sc;
    magnitude = new float[bs];
    for (unsigned long i = 0 ; i < bs ; i++){
        magnitude[i]  = sqrt(bufferGx[i]*bufferGx[i] + bufferGy[i]*bufferGy[i]);
    }
}

float gradient3dMagnitude(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* magnitude)
{
    float* bufferGx; float* bufferGy; float* bufferGz;
    gradient3d(image, sx, sy, sz, st, sc, bufferGx, bufferGy, bufferGz);
    unsigned long bs = sx*sy*sz*st*sc;
    magnitude = new float[bs];
    for (unsigned long i = 0 ; i < bs ; i++){
        magnitude[i]  = sqrt(bufferGx[i]*bufferGx[i] + bufferGy[i]*bufferGy[i] + bufferGz[i]*bufferGz[i]);
    }
}

}
