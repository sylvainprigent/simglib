/// \file SImageColor.h
/// \brief SImageColor class
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
class SIMAGE_EXPORT SImageColor{

public:
    /// \fn static SImageUInt* toRGB(SImageFloat* image);
    /// \brief create a new RGB image by copying a gray image to each color channels
    static SImageUInt* toRGB(SImageFloat* image);

};