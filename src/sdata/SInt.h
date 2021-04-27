/// \file SInt.h
/// \brief SInt class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"

#include "SData.h"

/// \class SInt
/// \brief define a integer container
class SDATA_EXPORT SInt : public SData{

public:
    SInt();
    SInt(int value);
    SInt(std::string value);

public:
    std::string json(int level = 0);
    std::string csv(std::string separator);

public:
    int get();

protected:
    int m_value;
};
