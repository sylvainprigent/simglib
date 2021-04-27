/// \file SProcess.h
/// \brief SProcess class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "scoreExport.h"

#include "SObservable.h"

/// \class SProcess
/// \brief Define a common interface for all processes
class SCORE_EXPORT SProcess : public SObservable{

public:
    SProcess();
    virtual ~SProcess();

public:
    virtual void checkInputs() = 0;
    virtual void run() = 0;


public:
    std::string getProcessName();

protected:
    void notify(std::string message, int type =  SObserver::MessageTypeDefault);

protected:
    std::string m_processName;

};
