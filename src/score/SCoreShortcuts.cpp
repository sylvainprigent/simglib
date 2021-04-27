/// \file SCoreShortcuts.cpp
/// \brief SCoreShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SCoreShortcuts.h"
#include "SStringOp.h"

namespace SImg {

std::string int2string(int value){
    return SStringOp::int2string(value);
}

std::string uint2string(unsigned int value){
    return SStringOp::uint2string(value);
}

std::string float2string(float value){
    return SStringOp::float2string(value);
}

float string2float(std::string str){
    return SStringOp::string2float(str);
}

}
