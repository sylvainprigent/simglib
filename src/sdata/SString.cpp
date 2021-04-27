/// \file SString.cpp
/// \brief SString class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SString.h"
#include "SDatatypes.h"

SString::SString() : SData(){
    m_datatype = SDatatypes::Bool;
}

SString::SString(std::string value) : SData(){
    m_datatype = SDatatypes::Bool;
    m_value = value;
}

std::string SString::json(int){
    return "\"" + m_value + "\"";
}

std::string SString::csv(std::string){
    return m_value;
}

std::string SString::get(){
    return m_value;
}
