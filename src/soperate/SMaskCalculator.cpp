/// \file SMaskCalculator.cpp
/// \brief SMaskCalculator class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "SMaskCalculator.h"
#include "score/SException.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

const std::string SMaskCalculator::AND = "AND";
const std::string SMaskCalculator::OR = "OR";
const std::string SMaskCalculator::XOR = "XOR";

void maskCalculator(unsigned int* image1, unsigned int* image2, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, unsigned int* output, const std::string& operatorName)
{
    unsigned long bs = sx*sy*sz*st*sc;
    output = new unsigned int[bs];

    if (operatorName == SMaskCalculator::AND){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = bool(image1[i]) * bool(image2[i]);
        }
    }
    else if (operatorName == SMaskCalculator::OR){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = bool(image1[i]) + bool(image2[i]);
        }
    }
    else if (operatorName == SMaskCalculator::XOR){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = bool(image1[i]) ^ bool(image2[i]);
        }
    }
    else{
        throw SException(("biMaskCalculator: method '" + operatorName + "' not known").c_str());
    }
}

}
