/// \file SMaskCalculator.h
/// \brief SMaskCalculator class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "soperateExport.h"

namespace SImg{

/// \class SMaskCalculator
/// \brief Perform pixelwise operation between two masks (AND, OR, XOR)
class SOPERATE_EXPORT SMaskCalculator{

public:
    static const std::string AND;
    static const std::string OR;
    static const std::string XOR;
};


void maskCalculator(unsigned int* image1, unsigned int* image2, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, unsigned int* output, const std::string& operatorName);

}