/// \file SImageColor.cpp
/// \brief SImageColor class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "score/SException.h"

#include "SImageColor.h"

SImageUInt* SImageColor::toRGB(SImageFloat* image)
{
    unsigned int sx = image->getSizeX();
    unsigned int sy = image->getSizeY();
    unsigned int sz = image->getSizeZ();
    unsigned int st = image->getSizeT();

    if (image->getSizeC() > 1){
        throw SException("SImageColor::toRGB can only convert one channel image to RGB");
    }

    float maxi = image->getMax();
    float mini = image->getMin();
    SImageUInt* rgbImage = new SImageUInt(sx, sy, sz, st, 3);
    unsigned int* rgbBuffer = rgbImage->getBuffer();
    float* grayBuffer = image->getBuffer();
    for (unsigned int x = 0 ; x < sx ; x++){
        for (unsigned int y = 0 ; y < sy ; y++){
            for (unsigned int z = 0 ; z < sz ; z++){
                for (unsigned int t = 0 ; t < st ; t++){
                    for (unsigned int c = 0 ; c < 3 ; c++){
                        rgbBuffer[c + 3*(t + st*(z + sz*(y + sy*x)))] = unsigned( 255*(grayBuffer[t + st*(z + sz*(y + sy*x))]-mini)/(maxi-mini));
                    }
                }        
            }
        }
    }
    return rgbImage;
}
