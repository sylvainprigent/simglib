/// \file SBool.cpp
/// \brief SBool class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SBool.h"
#include "SDatatypes.h"

SBool::SBool() : SData(){
    m_datatype = SDatatypes::Bool;
}

SBool::SBool(bool value) : SData(){
    m_datatype = SDatatypes::Bool;
    m_value = value;
}

std::string SBool::json(int){
    if (m_value){
        return "true";
    }
    return "false";
}

std::string SBool::csv(std::string){
    return this->json();
}

bool SBool::get(){
    return m_value;
}

void SBool::set(bool value){
    m_value = value;
}
