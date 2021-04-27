/// \file SCoreShortcuts.h
/// \brief SCoreShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <chrono>

#include "score/SSingleton.h"
#include "scliExport.h"

/// \namespace SImg
/// \brief Shortut function to call modules functionalities
namespace SImg{
    void tic();
    void toc();
}
