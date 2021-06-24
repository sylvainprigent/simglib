/// \file SImageStats.h
/// \brief SImageStats class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include "simageExport.h"
#include "SImage.h"

/// \class SImageStats
/// \brief Calculate stats on images.
class SIMAGE_EXPORT SImageStats{

public:
    SImageStats();
    ~SImageStats();

    void setInput(SImage* image);

public:
    float positiveMin();
    float normL2();

protected:
    SImage* m_input;
    char m_processPrecision;
    bool m_processZ;
    bool m_processT;
    bool m_processC;
};
