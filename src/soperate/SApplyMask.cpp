/// \file SApplyMask.cpp
/// \brief SApplyMask class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "SApplyMask.h"

namespace SImg{

void applyMask(float* image, unsigned int sx, unsigned int sy, unsigned int sz, float* mask, float* output, const float& maskValue)
{
    output = new float[sx*sy*sz];
    float m_epsilon = float(0.0000001);

    for (unsigned int x = 0 ; x < sx ; x++){
        for (unsigned int y = 0 ; y < sy ; y++){
            for (unsigned int z = 0 ; z < sz ; z++){

                if (mask[ z + sz*(y + sy*x)] > maskValue - m_epsilon && mask[ z + sz*(y + sy*x)] < maskValue + m_epsilon){
                    output[ z + sz*(y + sy*x)] = 0.0;
                }
                else{
                    output[ z + sz*(y + sy*x)] = image[ z + sz*(y + sy*x)];
                }
            }
        }
    }
}

}