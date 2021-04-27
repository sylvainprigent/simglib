/// \file SZProject.h
/// \brief SZProject class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"


namespace SImg{

    int zprojectCheckIputsStart(const int& startT, unsigned int st);
    int zprojectCheckIputsEnd(const int& endT, unsigned int st);

    void zproject(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ, std::string methodName);
    void zprojectMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ);
    void zprojectMin(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ);
    void zprojectMean(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ);
    void zprojectVar(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ);
    void zprojectMedian(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startZ, const int& endZ);

}
