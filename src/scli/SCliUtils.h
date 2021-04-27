/// \file SCliUtils.h
/// \brief SCliUtils class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#pragma once

#include <string>
#include "scliExport.h"

/// \class SCliUtils
/// \brief Static functions to manipulate files and dir paths
class SCLI_EXPORT SCliUtils{


public:
    SCliUtils();

public:
    static std::string getCurentPath();
    static std::string getFileSeparator();
};
