/// \file SString.h
/// \brief SString class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"

#include "SData.h"

/// \class SString
/// \brief define a string container
class SDATA_EXPORT SString : public SData{

public:
    SString();
    SString(std::string value);

public:
    std::string json(int level = 0);
    std::string csv(std::string separator = "");

public:
    std::string get();

protected:
    std::string m_value;
};
