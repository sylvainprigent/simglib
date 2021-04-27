/// \file SCrop.cpp
/// \brief SCrop functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SCrop.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void crop(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output,
          int minX, int maxX, int minY, int maxY, int minZ, int maxZ, int minT, int maxT, int minC, int maxC){

    unsigned int _minX, _maxX, _minY, _maxY, _minZ, _maxZ, _minT, _maxT, _minC, _maxC;
    if (minX < 0 ){
        _minX = 0;
    }
    else{
        _minX = unsigned(minX);
    }
    if (maxX < 0 ){
        _maxX = sx;
    }
    else{
        _maxX = unsigned(_maxX);
    }
    if (minY < 0 ){
        _minY = 0;
    }
    else{
        _minY = unsigned(minY);
    }
    if (maxY < 0 ){
        _maxY = sy;
    }
    else{
        _maxY = unsigned(maxY);
    }
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

    unsigned int sxo = _maxX - _minX;
    unsigned int syo = _maxY - _minY;
    unsigned int szo = _maxZ - _minZ;
    unsigned int sto = _maxT - _minT;
    unsigned int sco = _maxC - _minC;

    output = new float[sx*sy*szo*sto*sco];

    for (unsigned int c = _minC ; c < _maxC ; c++){
        for (unsigned int t = _minT ; t < _maxT ; t++){
            for (unsigned int z = _minZ ; z < _maxZ ; z++){
#pragma omp parallel for
                for (unsigned int y = _minY ; y < _maxY ; y++){
                    for (unsigned int x = _minX ; x < _maxX ; x++){
                        output[(c-_minC) + sco*((t-_minT) + sto*((z-_minZ) + szo*((y-_minY) + syo*(x-_minX))))] = image[ c + sc*(t + st*(z + sz*(y + sy*x)))];
                    }
                }
            }
        }
    }
}

}