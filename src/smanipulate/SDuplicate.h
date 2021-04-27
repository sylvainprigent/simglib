/// \file SDuplicate.h
/// \brief SDuplicate functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"

namespace SImg{

void duplicate(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, 
                     float* output, int minZ, int maxZ, int minT, int maxT, int minC, int maxC);

}
