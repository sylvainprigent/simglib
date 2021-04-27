/// \file SCoreShortcuts.h
/// \brief SCoreShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <chrono>

#include "SSingleton.h"
#include "scoreExport.h"

/// \namespace SImg
/// \brief Shortut function to call modules functionalities
namespace SImg{
    std::string int2string(int value);
    std::string uint2string(unsigned int value);
    std::string float2string(float value);
    float string2float(std::string str);
}
