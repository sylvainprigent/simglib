/// \file SConvolve.h
/// \brief SConvolve functions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "sfilteringExport.h"


namespace SImg{

    float* convolutionNaive(float* image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, unsigned int sc, 
                             float* kernel, unsigned int ksx, unsigned int ksy, unsigned int ksz, unsigned int kst);

}
                             
