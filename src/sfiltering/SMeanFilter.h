/// \file SMeanFilter.h
/// \brief SMeanFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "sfilteringExport.h"

namespace SImg{

    void meanFilter(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, int rx, int ry, int rz, int rt, float* output);

}
