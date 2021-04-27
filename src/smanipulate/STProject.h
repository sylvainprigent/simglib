/// \file STProject.h
/// \brief STProject functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"


namespace SImg{

    int tprojectCheckIputsStart(const int& startT, unsigned int st);
    int tprojectCheckIputsEnd(const int& endT, unsigned int st);

    void tproject(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT, const std::string& operatorName);
    void tprojectMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT);
    void tprojectMin(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT);
    void tprojectMean(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT);
    void tprojectVar(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT);
    void tprojectMedian(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const int& startT, const int& endT);

}
