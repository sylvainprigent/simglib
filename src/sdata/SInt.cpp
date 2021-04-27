/// \file SInt.cpp
/// \brief SInt class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <sstream>

#include "SInt.h"
#include "SDatatypes.h"

SInt::SInt() : SData(){
    m_datatype = SDatatypes::Int;
}

SInt::SInt(int value) : SData(){
    m_datatype = SDatatypes::Int;
    m_value = value;
}

SInt::SInt(std::string value){
    m_datatype = SDatatypes::Int;
    m_value = std::stoi(value);
}

std::string SInt::json(int){
    std::ostringstream ss;
    ss << m_value;
    std::string s(ss.str());
    return s;
}

std::string SInt::csv(std::string){
    return this->json();
}

int SInt::get(){
    return m_value;
}
