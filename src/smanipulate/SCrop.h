/// \file SCrop.h
/// \brief SCrop functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "smanipulateExport.h"

namespace SImg{

void crop(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output,
          int minX = -1, int maxX = -1, int minY = -1, int maxY = -1, int minZ = -1, int maxZ = -1, int minT = -1, int maxT = -1, int minC = -1, int maxC = -1);
}

