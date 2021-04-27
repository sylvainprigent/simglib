/// \file SObject.h
/// \brief SObject class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include <map>
#include "sdataExport.h"

#include "SData.h"
#include "SInt.h"
#include "SFloat.h"
#include "SString.h"

/// \class SObject
/// \brief define an object container
class SDATA_EXPORT SObject : public SData{

public:
    SObject();

public:
    std::string json(int level = 0);
    std::string csv(std::string separator = ",");

public:
    SData* get(std::string key);

public:
    void set(std::string key, SData* value);

public:
    bool hasKey(std::string key);

protected:
    std::string indent(int level);

protected:
    std::map<std::string, SData*> m_data;
};
