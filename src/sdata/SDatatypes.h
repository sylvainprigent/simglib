/// \file slData.h
/// \brief slData class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataExport.h"


/// \class SDatatypes
/// \brief List of available data types
class SDATA_EXPORT SDatatypes{

public: static std::string Bool;
public: static std::string Int;
public: static std::string Float;
public: static std::string Array;
public: static std::string Object;
public: static std::string Table;
};
