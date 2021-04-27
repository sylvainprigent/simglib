/// \file slInvert.h
/// \brief slInvert class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "soperateExport.h"

namespace SImg{

/// \class SImageCalculator
/// \brief Apply pixel wiseoperation between two images (ADD, SUBTRACT, MULTIPLY, DIVIDE, MIN, MAX)
class SOPERATE_EXPORT SImageCalculator{

public:
    static const std::string ADD;
    static const std::string SUBTRACT;
    static const std::string MULTIPLY;
    static const std::string DIVIDE;
    static const std::string MIN;
    static const std::string MAX;

};

void imageCalculator(float* image1, float* image2, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const std::string& operatorName);

}