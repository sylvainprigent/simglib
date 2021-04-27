/// \file STimer.h
/// \brief STimer class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <chrono>

#include "score/SSingleton.h"
#include "score/SObserver.h"

#include "scliExport.h"


/// \class SLog
/// \brief Log messages.
class SCLI_EXPORT STimer {

public:
    STimer();
    ~STimer();

public:
    void tic();
    void toc();
    void printTime();

public:
    void setObserver(SObserver* observer);
    SObserver* getObserver();

protected:
    std::chrono::steady_clock::time_point m_begin;
    std::chrono::steady_clock::time_point m_end;
    SObserver *m_observer;
    bool m_localObserver;

};


/// \class STimerAccess
/// \brief Singleton that allow to access STimer
class SCLI_EXPORT STimerAccess : public SSingleton<STimer>{

    friend class SSingleton<STimer>;

private:
    /// \fn STimerAccess();
    /// \brief Constructor
    STimerAccess();
    /// \fn ~SLogAccess();
    /// \brief Desctructor
    ~STimerAccess();

public:
    // getters
    /// \fn STimer *timer();
    /// \return a pointer to the STimer class
    STimer *timer();

    /// \fn void tic();
    /// \brief Start the timer
    void tic();

    /// \fn void toc();
    /// \brief Stop the timer
    void toc();

private:
    STimer *m_timer; ///< settings containers
};
