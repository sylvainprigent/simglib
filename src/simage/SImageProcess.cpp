/// \file SImageProcess.cpp
/// \brief SImageProcess class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SImageProcess.h"
#include "score/SException.h"

SImageProcess::SImageProcess() : SProcess(){
    m_processName = "SImageProcess";
    m_processPrecision = 32;
    m_processZ = false;
    m_processT = false;
    m_processC = false;
    m_input = nullptr;
}

SImageProcess::~SImageProcess(){

}

void SImageProcess::setInput(SImage* image){
    m_input = image;
    this->checkInput();
}

void SImageProcess::checkInputs(){
    if (!m_input){
        throw SException(std::string(m_processName + " Input image not set ").c_str());
    }
}

void SImageProcess::checkInput(){

    if ( m_input->getPrecision() != m_processPrecision ){
        throw SException(std::string(m_processName +" only process datatype " + std::to_string(m_processPrecision)  + ", input image is " + m_input->getPrecision()).c_str()  );
    }

    if ( m_input->getSizeZ() > 1 && !m_processZ){
        throw SException(std::string( m_processName + " can only process 2D images" ).c_str());
    }

    if ( m_input->getSizeT() > 1 && !m_processT){
        throw SException(std::string( m_processName + " cannot process time series" ).c_str());
    }

    if ( m_input->getSizeC() > 1 && !m_processC){
        throw SException(std::string( m_processName + " cannot process multichannel images" ).c_str());
    }
}

SImageUInt* SImageProcess::castInputToUInt(){
    return dynamic_cast<SImageUInt*>(m_input);
}

SImageInt* SImageProcess::castInputToInt(){
    return dynamic_cast<SImageInt*>(m_input);
}

SImageFloat* SImageProcess::castInputToFloat(){
    return dynamic_cast<SImageFloat*>(m_input);
}
