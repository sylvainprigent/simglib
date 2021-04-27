/// \file SProcessObserver.h
/// \brief SProcessObserver class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include <string>
#include <vector>
#include "scoreExport.h"

#include "SObserver.h"

/// \class SObservable
/// \brief Define a common interface for observable object
class SCORE_EXPORT SObservable{

public:
    SObservable();
    virtual ~SObservable();

public:
    /// \brief Add an observer to the object
    /// \param[in] observer Pointer to the observer
    void addObserver(SObserver* observer);

protected:
    void notifyProgress(int value);
    void notify(std::string message, int type =  SObserver::MessageTypeDefault);

protected:
    std::vector<SObserver*> m_observers;

};
