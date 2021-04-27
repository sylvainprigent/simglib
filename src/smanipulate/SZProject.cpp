/// \file SZProject.cpp
/// \brief SZProject class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SZProject.h"
#include "score/SException.h"
#include <vector>
#include <algorithm>

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

int zprojectCheckIputsStart(const int& startZ, unsigned int sz)
{
    int start = startZ;
    if (startZ == -1){
        start = 0;
    }
    if ( startZ > int(sz)){
        throw SException("STProject: Bounds are higher than the image z size");
    }
    return start;
}

int zprojectCheckIputsEnd(const int& endZ, unsigned int sz)
{
    int end = endZ;
    if (endZ == -1){
        end = int(sz)-1;
    }
    if ( endZ > int(sz)){
        throw SException("STProject: Bounds are higher than the image z size");
    }
    return end;
}


void sproject(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ, std::string methodName){

    if (methodName == "MAX"){
        zprojectMax(image, sx, sy, sz, st, sc, output, startZ, endZ);
    }
    if (methodName == "MIN"){
        zprojectMin(image, sx, sy, sz, st, sc, output, startZ, endZ);
    }
    if (methodName == "MEAN"){
        zprojectMean(image, sx, sy, sz, st, sc, output, startZ, endZ);
    }
    if (methodName == "VAR"){
        zprojectVar(image, sx, sy, sz, st, sc, output, startZ, endZ);
    }
    if (methodName == "MEDIAN"){
        zprojectMedian(image, sx, sy, sz, st, sc, output, startZ, endZ);
    }
}

void zprojectMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ)
{
    int m_startZ = zprojectCheckIputsStart(startZ, sz);
    int m_endZ = zprojectCheckIputsEnd(endZ, sz);

    output = new float[sx*sy*st*sc];
    //int filterSize = m_endZ - m_startZ;
    //float val, max;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int t = 0 ; t < st ; t++){

                    float max = float(-999999999.9);
                    for (int z = m_startZ ; z <= m_endZ ; z++){
                        float val = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        if ( val > max ){
                            max = val;
                        }
                    }
                    output[c + sc*(t + st*(0 + sz*(y + sy*x)))] = max;
                }
            }
        }

    }
}

void zprojectMin(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ)
{
    int m_startZ = zprojectCheckIputsStart(startZ, sz);
    int m_endZ = zprojectCheckIputsEnd(endZ, sz);

    output = new float[sx*sy*st*sc];
    //int filterSize = m_endZ - m_startZ;
    //float val, min;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int t = 0 ; t < st ; t++){

                    float min = float(999999999.9);
                    for (int z = m_startZ ; z <= m_endZ ; z++){
                        float val = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        if ( val > min ){
                            min = val;
                        }
                    }
                    output[c + sc*(t + st*(0 + sz*(y + sy*x)))] = min;
                }
            }
        }

    }
}

void zprojectMean(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ)
{
    int m_startZ = zprojectCheckIputsStart(startZ, sz);
    int m_endZ = zprojectCheckIputsEnd(endZ, sz);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endZ - m_startZ;
    //float mean;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int t = 0 ; t < st ; t++){

                    float mean = 0.0;
                    for (int z = m_startZ ; z <= m_endZ ; z++){
                        mean += image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                    output[c + sc*(t + st*(0 + sz*(y + sy*x)))] = mean/filterSize;

                }
            }
        }

    }
}

void zprojectVar(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ)
{
    int m_startZ = zprojectCheckIputsStart(startZ, sz);
    int m_endZ = zprojectCheckIputsEnd(endZ, sz);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endZ - m_startZ;
    //float mean, var, v;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int t = 0 ; t < st ; t++){

                    float mean = 0.0;
                    float var = 0.0;
                    for (int z = m_startZ ; z <= m_endZ ; z++){
                        float v = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        mean += v;
                        var += v*v;
                    }
                    output[c + sc*(t + st*(0 + sz*(y + sy*x)))] = (var - mean*mean/filterSize)/(filterSize-1);
                }
            }
        }
    }
}

void zprojectMedian(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ)
{
    int m_startZ = zprojectCheckIputsStart(startZ, sz);
    int m_endZ = zprojectCheckIputsEnd(endZ, sz);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endZ - m_startZ;

    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            std::vector<float> vals;
            vals.resize(unsigned(filterSize));
            for (int y = 0 ; y < sy ; y++){
                for (int t = 0 ; t < st ; t++){

                    for (int z = m_startZ ; z <= m_endZ ; z++){
                        vals[unsigned(z-m_startZ)] = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                    std::sort(vals.begin(), vals.end());
                    output[c + sc*(t + st*(0 + sz*(y + sy*x)))] = vals[unsigned(filterSize/2)];
                }
            }
        }
    }
}

}