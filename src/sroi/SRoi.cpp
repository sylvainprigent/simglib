/// \file SRoi.cpp
/// \brief SRoi class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SRoi.h"


SRoi::SRoi(){
    m_type = "";
}

SRoi::~SRoi(){

}

std::string SRoi::getType(){
    return m_type;
}

SObject* SRoi::getProperties(){
    return m_properties;
}

void SRoi::setProperties(SObject* properties){
    m_properties = properties;
}
