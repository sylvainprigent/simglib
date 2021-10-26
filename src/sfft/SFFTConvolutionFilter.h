/// \file SFFT.h
/// \brief SFFT class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <fftw3.h>

#include "sfftExport.h"

namespace SImg
{
    float* convolution_2d(float* image1, float* image2, unsigned int sx, unsigned int sy);
    float* convolution_3d(float* image1, float* image2, unsigned int sx, unsigned int sy, unsigned int sz);
}
