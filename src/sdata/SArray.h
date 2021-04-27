/// \file SArray.h
/// \brief SArray class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include <map>
#include "sdataExport.h"

#include "SData.h"

/// \class SArray
/// \brief define an array container
class SDATA_EXPORT SArray : public SData{

public:
    SArray();
    SArray(int size);

public:
    std::string json(int level = 0);
    std::string csv(std::string separator = ",");

public:
    void resize(unsigned int size);
    unsigned int size();
    void set(int idx, SData* data);
    SData* get(int idx);
    void append(SData* data);

protected:
    std::string indent(int level);

protected:
    std::vector<SData*> m_data;
};
