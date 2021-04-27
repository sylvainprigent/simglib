/// \file SGradientFilter.h
/// \brief SGradientFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "sfilteringExport.h"

namespace Simg
{
    void gradientXY(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
    void gradientXYZ(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
    void gradientXYT(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
    void gradientXYZT(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);
}