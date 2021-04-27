/// \file SCoreShortcuts.cpp
/// \brief SCoreShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SCliTimer.h"
#include "SObserverConsole.h"
#include "STimer.h"
#include "score/SStringOp.h"

namespace SImg {

void tic(){
    STimerAccess::instance()->setObserver(new SObserverConsole());
    STimerAccess::instance()->tic();
}

void toc(){
    STimerAccess::instance()->toc();
    delete STimerAccess::instance()->getObserver();
}

}
