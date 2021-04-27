/// \file SProcess.cpp
/// \brief SProcess class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SProcess.h"

SProcess::SProcess(){
    m_processName = "SProcess";
}

SProcess::~SProcess(){

}

void SProcess::notify(std::string message, int type){
    for (unsigned int i = 0 ; i < m_observers.size() ; i++){
        m_observers[i]->message(m_processName + ": " + message, type);
    }
}

std::string SProcess::getProcessName(){
    return m_processName;
}
