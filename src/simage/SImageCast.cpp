/// \file SImageCast.cpp
/// \brief SImageCast class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SImageCast.h"

SImageUInt* SImageCast::toChar(SImage* image){
    return dynamic_cast<SImageUInt*>(image);
}

SImageInt* SImageCast::toInt(SImage* image){
    return dynamic_cast<SImageInt*>(image);
}

SImageFloat* SImageCast::toFloat(SImage* image){
    return dynamic_cast<SImageFloat*>(image);
}
