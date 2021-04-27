/// \file SHistogram.h
/// \brief SHistogram class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "SImageFloat.h"
#include "sdata/STable.h"
#include "SImageProcess.h"

#include "simageExport.h"

/// \class SHistogram
/// \brief Class to compute and store image histogram
class SIMAGE_EXPORT SHistogram : public SImageProcess{

public:
    SHistogram();

public:
    void setNumberOfBins(unsigned numberOfBins);
    unsigned NumberOfBins();

public:
    int* getCount();
    float* getValues();
    int countAt(int idx);
    float valueAt(int idx);
    STable* toTable();

public:
    void run();

protected:
    // parameters
    unsigned m_numberOfBins;
    bool m_verbose;

protected:
    // output
    int* m_count;
    float* m_values;

};
