/// \file SData.cpp
/// \brief SData class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SData.h"

SData::SData(){
    m_datatype = "data";
}

SData::~SData(){

}

std::string SData::getDatatype(){
    return m_datatype;
}
