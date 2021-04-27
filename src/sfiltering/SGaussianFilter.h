/// \file SGaussianFilter.h
/// \brief SGaussianFilter functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "sfilteringExport.h"

namespace SImg{

    void gaussian2dFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const float& sigma);
    void gaussian3dFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output, const float& sigmaX, const float& sigmaY, const float& sigmaZ);

}