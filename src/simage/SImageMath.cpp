/// \file SImageMath.cpp
/// \brief SImageMath class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SImageMath.h"
#include "score/SException.h"

void SImageMath::add(SImageFloat* image, float value){
    float* buffer = image->getBuffer();
    for (unsigned int i = 0 ; i < image->getBufferSize() ; i++){
        buffer[i] += value;
    }

}

void SImageMath::subtract(SImageFloat* image, float value){
    float* buffer = image->getBuffer();
    for (unsigned int i = 0 ; i < image->getBufferSize() ; i++){
        buffer[i] -= value;
    }
}

void SImageMath::multiply(SImageFloat* image, float value){
    float* buffer = image->getBuffer();
    for (unsigned int i = 0 ; i < image->getBufferSize() ; i++){
        buffer[i] *= value;
    }
}

void SImageMath::divide(SImageFloat* image, float value){
    float* buffer = image->getBuffer();
    for (unsigned int i = 0 ; i < image->getBufferSize() ; i++){
        buffer[i] /= value;
    }
}
