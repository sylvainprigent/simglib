/// \file SFloat.cpp
/// \brief SFloat class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <sstream>

#include "SFloat.h"
#include "SDatatypes.h"

SFloat::SFloat() : SData(){
    m_datatype = SDatatypes::Float;
}

SFloat::SFloat(float value) : SData(){
    m_datatype = SDatatypes::Float;
    m_value = value;
}

SFloat::SFloat(std::string value) : SData(){
    m_datatype = SDatatypes::Float;
    m_value = std::stof(value);
}

std::string SFloat::json(int){
    std::ostringstream ss;
    ss << m_value;
    std::string s(ss.str());
    return s;
}

std::string SFloat::csv(std::string){
    return this->json();
}

float SFloat::get(){
    return m_value;
}
