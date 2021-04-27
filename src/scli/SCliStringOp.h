/// \file SStringOp.h
/// \brief SStringOp class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "scliExport.h"

#include <vector>

/// \class SCliStringOp
/// \brief Static functions for string operations
class SCLI_EXPORT SCliStringOp{

public:
    static std::vector<std::string> split(const std::string& str, const std::string& delim);
    static bool endsWith(std::string const & value, std::string const & ending);
    static bool contains(const std::string &str, const std::string &substr );
    static std::string remove(const std::string &str, char value);
    static std::string int2string(int value);
    static std::string uint2string(unsigned int value);
    static std::string float2string(float value);
    static float string2float(std::string str);
};
