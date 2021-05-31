/// \file STimer.cpp
/// \brief STimer class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SCliStringOp.h"

#include "STimer.h"
#include "score/SObserverConsole.h"

STimer::STimer(){
    m_observer = nullptr;
}

STimer::~STimer(){

}

void STimer::tic(){
    m_begin = std::chrono::steady_clock::now();
}

void STimer::toc(){
    m_end = std::chrono::steady_clock::now();
    this->printTime();
}

void STimer::printTime(){

    unsigned int dt = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_begin).count() /1000.0;

    unsigned int ndays = unsigned(dt/86400000.);
    unsigned int nhours = unsigned((dt - ndays*86400000.)/3600000.);
    unsigned int nmin = unsigned((dt - ndays*86400000. - nhours*3600000.)/60000.);
    unsigned int nsec = unsigned((dt - ndays*86400000. - nhours*3600000. - nmin*60000.)/1000.);
    unsigned int nmil = unsigned((dt - ndays*86400000. - nhours*3600000. - nmin*60000. - nsec*1000.));

    std::string message = "Elapsed time: ";
    if (ndays > 0){
        message = SCliStringOp::uint2string(ndays) + " days "
                + SCliStringOp::uint2string(nhours) + " hours "
                + SCliStringOp::uint2string(nmin) + " minutes ";
    }
    else if (nhours > 0){
        message = SCliStringOp::uint2string(nhours) + " hours "
                + SCliStringOp::uint2string(nmin) + " minutes ";
    }
    else if (nmin > 0){
        message = SCliStringOp::uint2string(nmin) + " minutes ";
    }
    message += SCliStringOp::uint2string(nsec) + " seconds "
            + SCliStringOp::uint2string(nmil) + " ms ";

    if (m_observer){
        m_observer->message("\e[1m\e[92m" + message + "\e[0m");
    }
}

void STimer::setObserver(SObserver* observer){
    m_observer = observer;
}

SObserver* STimer::getObserver(){
    return m_observer;
}

STimerAccess::STimerAccess(){
    m_timer = new STimer();
}

STimerAccess::~STimerAccess(){

}

STimer *STimerAccess::timer(){
    return m_timer;
}

void STimerAccess::tic(){
    m_timer->tic();
}

void STimerAccess::toc(){
    m_timer->toc();
}
