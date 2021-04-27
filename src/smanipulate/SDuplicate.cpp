/// \file SDuplicate.cpp
/// \brief SDuplicate functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SDuplicate.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void duplicate(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, 
                     float* output, int minZ, int maxZ, int minT, int maxT, int minC, int maxC){

    unsigned int _minZ, _maxZ, _minT, _maxT, _minC, _maxC;
    if (minZ < 0 ){
        _minZ = 0;
    }
    else{
        _minZ = unsigned(minZ);
    }
    if (maxZ < 0 ){
        _maxZ = sz;
    }
    else{
        _maxZ = unsigned(maxZ);
    }
    if (minT < 0 ){
        _minT = 0;
    }
    else{
        _minT = unsigned(minT);
    }
    if (maxT < 0 ){
        _maxT = st;
    }
    else{
        _maxT = unsigned(maxT);
    }
    if (minC < 0 ){
        _minC = 0;
    }
    else{
        _minC = unsigned(minC);
    }
    if (maxC < 0 ){
        _maxC = sc;
    }
    else{
        _maxC = unsigned(maxC);
    }

    unsigned int szo = _maxZ - _minZ;
    unsigned int sto = _maxT - _minT;
    unsigned int sco = _maxC - _minC;

    output = new float[sx*sy*szo*sto*sco];

#pragma omp parallel for
    for (unsigned int c = minC ; c < maxC ; c++){
        for (unsigned int t = minT ; t < maxT ; t++){
            for (unsigned int z = minZ ; z < maxZ ; z++){
                for (unsigned int y = 0 ; y < sy ; y++){
                    for (unsigned int x = 0 ; x < sx ; x++){
                        output[(c-minC) + sco*((t-minT) + sto*((z-minZ) + szo*(y + sy*x)))] = image[ c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                }
            }
        }
    }
}

}