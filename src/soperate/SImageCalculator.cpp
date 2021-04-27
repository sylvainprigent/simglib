/// \file SImageCalculator.cpp
/// \brief SImageCalculator class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SImageCalculator.h"
#include "score/SException.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

const std::string SImageCalculator::ADD = "ADD";
const std::string SImageCalculator::SUBTRACT = "SUBTRACT";
const std::string SImageCalculator::MULTIPLY = "MULTIPLY";
const std::string SImageCalculator::DIVIDE = "DIVIDE";
const std::string SImageCalculator::MIN = "MIN";
const std::string SImageCalculator::MAX = "MAX";

void imageCalculator(float* image1, float* image2, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const std::string& operatorName)
{
    unsigned long bs = sx*sy*sz*st*sc;
    output = new float[bs];

    if (operatorName == SImageCalculator::ADD){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = image1[i] + image2[i];
        }
    }
    else if (operatorName == SImageCalculator::SUBTRACT){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = image1[i] - image2[i];
        }
    }
    else if (operatorName == SImageCalculator::MULTIPLY){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = image1[i] * image2[i];
        }
    }
    else if (operatorName == SImageCalculator::DIVIDE){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            output[i] = image1[i] / image2[i];
        }
    }
    else if (operatorName == SImageCalculator::MIN){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            if ( image1[i] < image2[i] ){
                output[i] = image1[i];
            }
            else{
                output[i] = image2[i];
            }
        }
    }
    else if (operatorName == SImageCalculator::MAX){
#pragma omp parallel for
        for (unsigned i = 0 ; i < bs ; i++){
            if ( image1[i] > image2[i] ){
                output[i] = image1[i];
            }
            else{
                output[i] = image2[i];
            }
        }
    }
    else{
        throw SException(("biImageCalculator: method '" + operatorName + "' not known").c_str());
    }
}

}
