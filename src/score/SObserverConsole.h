/// \file SObserver.h
/// \brief SObserver class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include "scoreExport.h"

#include "score/SObserver.h"

/// \class SProcess
/// \brief Define a common interface for all processes
class SCORE_EXPORT SObserverConsole : public SObserver{

public:
    SObserverConsole();
    virtual ~SObserverConsole();

public:
    virtual void progress(int value);
    virtual void message(std::string message, int type =  SObserver::MessageTypeDefault);

protected:
    bool m_inProgress;
};
