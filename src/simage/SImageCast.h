/// \file SImageCast.h
/// \brief SImageCast class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"
#include "SImageUInt.h"
#include "SImageInt.h"
#include "SImageFloat.h"

/// \class SImageCast
/// \brief SImageCast cast image
class SIMAGE_EXPORT SImageCast{

public:
    static SImageUInt* toChar(SImage* image);
    static SImageInt* toInt(SImage* image);
    static SImageFloat* toFloat(SImage* image);

};

