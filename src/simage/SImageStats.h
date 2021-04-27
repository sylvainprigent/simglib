/// \file SImageStats.h
/// \brief SImageStats class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include "simageExport.h"
#include "SImage.h"
#include "SImageProcess.h"

/// \class SImageStats
/// \brief Calculate stats on images.
class SIMAGE_EXPORT SImageStats : public SImageProcess{

public:
    SImageStats();
    ~SImageStats();

public:
    float positiveMin();
    float normL2();

};
