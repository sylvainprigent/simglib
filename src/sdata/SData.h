/// \file SData.h
/// \brief SData class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"


/// \class SData
/// \brief Abstract object to define interface for any serpicolib data
class SDATA_EXPORT SData{

public:
    SData();
    virtual ~SData();

public:
    virtual std::string json(int level = 0) = 0;
    virtual std::string csv(std::string = ",") = 0;

public:
    std::string getDatatype();

protected:
    std::string m_datatype;
};
