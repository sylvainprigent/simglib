/// \file SResizeCanvas.h
/// \brief SResizeCanvas function
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "smanipulateExport.h"

namespace SImg{

void resizeCanvas(float* image, unsigned int sx, unsigned int sy, unsigned int sz, float* output, const unsigned int& csx, const unsigned int& csy, const unsigned int& csz);
    
}
