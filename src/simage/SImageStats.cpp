/// \file SImageStats.cpp
/// \brief SImageStats class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "math.h"

#include "SImageStats.h"
#include "SImageFloat.h"

SImageStats::SImageStats(){
    m_processPrecision = 32;
    m_processZ = true;
    m_processT = true;
    m_processC = true;
}

void SImageStats::setInput(SImage* image){
    m_input = image;
}

float SImageStats::positiveMin(){

    float* buffer = dynamic_cast<SImageFloat*>(m_input)->getBuffer();
    float min = 999999;
    float val;

    for (unsigned long i = 0 ; i < m_input->getBufferSize() ; i++){
        val = buffer[i];
        if (val > 0 && val < min){
            min = val;
        }
    }
    return min;
}

float SImageStats::normL2(){

    double norm = 0.0;
    float* buffer = dynamic_cast<SImageFloat*>(m_input)->getBuffer();
    for (unsigned long i = 0 ; i < m_input->getBufferSize() ; i++){
        norm += buffer[i]*buffer[i];
    }
    return sqrt(norm);
}

