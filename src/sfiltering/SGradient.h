/// \file gradient.h
/// \brief gradient functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <vector>

#include "sfilteringExport.h"

namespace SImg{

void gradient2d(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* bufferGx, float* bufferGy);
void gradient3d(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* bufferGx, float* bufferGy, float* bufferGz);
float gradient2dL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc);
float gradient3dL2(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc);
float gradient2dL1(float* image,  unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc);
float gradient3dL1(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc);
float gradient2dMagnitude(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* magnitude);
float gradient3dMagnitude(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* magnitude);

}
