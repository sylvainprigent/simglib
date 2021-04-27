/// \file SCliStringOp.cpp
/// \brief SCliStringOp class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SCliStringOp.h"

#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::string> SCliStringOp::split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

bool SCliStringOp::endsWith(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


bool SCliStringOp::contains(const std::string &str, const std::string &substr ){
    std::size_t found = str.find(substr);
    if (found!=std::string::npos){
        return true;
    }
    return false;
}

std::string SCliStringOp::remove(const std::string &str, char value){
    std::string outStr = "";
    for (unsigned int i = 0 ; i < str.size() ; i++){
        if (str[i] != value){
            outStr.push_back(str[i]);
        }
    }
    return outStr;
}

std::string SCliStringOp::int2string(int value){
    std::ostringstream convert;
    convert << value;
    return convert.str();
}

std::string SCliStringOp::uint2string(unsigned int value){
    std::ostringstream convert;
    convert << value;
    return convert.str();
}

std::string SCliStringOp::float2string(float value){
    std::ostringstream convert;
    convert << value;
    return convert.str();
}

float SCliStringOp::string2float(std::string str){
    double temp = ::atof(str.c_str());
    return float(temp);
}
