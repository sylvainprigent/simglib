/// \file SBool.h
/// \brief SBool class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"

#include "SData.h"

/// \class SBool
/// \brief define a boolean container
class SDATA_EXPORT SBool : public SData{

public:
    SBool();
    SBool(bool value);

public:
    std::string json(int level = 0);
    std::string csv(std::string separator = ",");

public:
    bool get();
    void set(bool value);

protected:
    bool m_value;
};
