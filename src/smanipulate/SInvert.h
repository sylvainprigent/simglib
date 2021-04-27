/// \file SInvert.h
/// \brief SInvert function
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"

namespace SImg{

void invert(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, float* output);

}
