/// \file SNormalize.h
/// \brief SNormalize class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"


namespace SImg{

/// \class SNormalize
/// \brief Change image intensity range to [min, max]
class SMANIPULATE_EXPORT SNormalize{

public:
    static std::string Max;
    static std::string Sum;
    static std::string L2;
    static std::string RC;
    static std::string Bits8;
    static std::string Bits12;
    static std::string Bits16;

};

void normMinMax(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
void normSum(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
void normL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
void normRC(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
void normValue(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, float normValue);

}
