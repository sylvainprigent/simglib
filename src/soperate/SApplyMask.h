/// \file applyMask.h
/// \brief applyMask class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "soperateExport.h"

namespace SImg{

void applyMask(float* image, unsigned int sx, unsigned int sy, unsigned int sz, float* mask, float* output, const float& maskValue);

}
