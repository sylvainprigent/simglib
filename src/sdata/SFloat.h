/// \file SFloat.h
/// \brief SFloat class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"

#include "SData.h"

/// \class SFloat
/// \brief define a float container
class SDATA_EXPORT SFloat : public SData{

public:
    SFloat();
    SFloat(float value);
    SFloat(std::string value);

public:
    std::string json(int level = 0);
    std::string csv(std::string separator);

public:
    float get();

protected:
    float m_value;
};
