/// \file SCliSelect.h
/// \brief SCliSelect class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "scliExport.h"

/// \class SCliSelect
/// \brief class container for a parameter of type select
class SCLI_EXPORT SCliSelect{

public:
    SCliSelect();

public:
    int count();
    void add(std::string name, std::string value);
    std::string name(int i);
    std::string value(int i);

protected:
    std::vector<std::string> m_names;
    std::vector<std::string> m_values;
};
