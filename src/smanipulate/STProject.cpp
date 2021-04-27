/// \file STProject.cpp
/// \brief STProject functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "STProject.h"
#include "score/SException.h"
#include <vector>
#include <algorithm>

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

int tprojectCheckIputsStart(const int& startT, unsigned int st)
{
    int start = startT;
    if (startT == -1){
        start = 0;
    }
    if ( startT > int(st)){
        throw SException("STProject: Bounds are higher than the image t size");
    }
    return start;
}

int tprojectCheckIputsEnd(const int& endT, unsigned int st)
{
    int end = endT;
    if (endT == -1){
        end = int(st)-1;
    }
    if ( endT > int(st)){
        throw SException("STProject: Bounds are higher than the image t size");
    }
    return end;
}

void tproject(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT, const std::string& operatorName)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    if (operatorName == "MAX"){
        tprojectMax(image, sx, sy, sz, st, sc, output, startT, endT);
    }
    if (operatorName == "MIN"){
        tprojectMin(image, sx, sy, sz, st, sc, output, startT, endT);
    }
    if (operatorName == "MEAN"){
        tprojectMean(image, sx, sy, sz, st, sc, output, startT, endT);
    }
    if (operatorName == "VAR"){
        tprojectVar(image, sx, sy, sz, st, sc, output, startT, endT);
    }
    if (operatorName == "MEDIAN"){
        tprojectMedian(image, sx, sy, sz, st, sc, output, startT, endT);
    }
}

void tprojectMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    output = new float[sx*sy*st*sc];
    //float val, max;
    for (int c = 0 ; c < sc ; c++){

        #pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int z = 0 ; z < sz ; z++){

                    float max = float(-999999999.9);
                    for (int t = m_startT ; t <= m_endT ; t++){
                        float val = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        if ( val > max ){
                            max = val;
                        }
                    }
                    output[c + sc*(0 + st*(z + sz*(y + sy*x)))] = max;
                }
            }
        }
    }
}

void tprojectMin(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    output = new float[sx*sy*st*sc];
    //int filterSize = m_endZ - m_startZ;
    //float val, min;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int z = 0 ; z < sz ; z++){

                    float min = float(999999999.9);
                    for (int t = m_startT ; t <= m_endT ; t++){
                        float val = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        if ( val > min ){
                            min = val;
                        }
                    }
                    output[c + sc*(0 + st*(z + sz*(y + sy*x)))] = min;
                }
            }
        }
    }
}

void tprojectMean(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endT - m_startT;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int z = 0 ; z < sz ; z++){

                    float mean = 0.0;
                    for (int t = m_startT ; t <= m_endT ; t++){
                        mean += image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                    output[c + sc*(0 + st*(z + sz*(y + sy*x)))] = mean/filterSize;

                }
            }
        }
    }
}

void tprojectVar(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endT - m_startT;
    float mean, var, v;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            for (int y = 0 ; y < sy ; y++){
                for (int z = 0 ; z < sz ; z++){

                    mean = 0.0;
                    var = 0.0;
                    for (int t = m_startT ; t <= m_endT ; t++){
                        v = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                        mean += v;
                        var += v*v;
                    }
                    output[c + sc*(0 + st*(z + sz*(y + sy*x)))] = (var - mean*mean/filterSize)/(filterSize-1);
                }
            }
        }
    }
}

void tprojectMedian(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT)
{
    int m_startT = tprojectCheckIputsStart(startT, st);
    int m_endT = tprojectCheckIputsEnd(endT, st);

    output = new float[sx*sy*st*sc];
    float filterSize = m_endT - m_startT;
    for (int c = 0 ; c < sc ; c++){
#pragma omp parallel for
        for (int x = 0 ; x < sx ; x++){
            std::vector<float> vals;
            vals.resize(unsigned(filterSize));
            for (int y = 0 ; y < sy ; y++){
                for (int z = 0 ; z < sz ; z++){

                    for (int t = m_startT ; z <= m_endT ; z++){
                        vals[unsigned(t-m_startT)] = image[c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                    std::sort(vals.begin(), vals.end());
                    output[c + sc*(0 + st*(z + sz*(y + sy*x)))] = vals[unsigned(filterSize/2)];
                }
            }
        }

    }
}

}