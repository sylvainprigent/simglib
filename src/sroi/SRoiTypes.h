/// \file SRoiTypes.h
/// \brief SRoiTypes class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "sroiExport.h"

/// \class SRoiTypes
/// \brief Define all the types of ROIs
class SROI_EXPORT SRoiTypes{

public:
    static std::string Patch;
    static std::string Rectangle;
    static std::string Cuboid;
    static std::string Circle;
    static std::string Sphere;
    static std::string Path;
    static std::string Polygon;
};
