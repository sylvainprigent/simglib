/// \file SImageFilter.cpp
/// \brief SImageFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SImageFilter.h"
#include "score/SException.h"

SImageFilter::SImageFilter() : SImageProcess(){

}

SImageFilter::~SImageFilter(){

}

SImage* SImageFilter::getOutput(){
    return m_output;
}
