/// \file SImageDimension.h
/// \brief SImageDimension class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"

/// \class SImageDimension
/// \brief Define dimensions for images.
class SIMAGE_EXPORT SImageDimension{

public:
    static const std::string XY;
    static const std::string XYZ;
    static const std::string XYT;
    static const std::string XYZT;
};
