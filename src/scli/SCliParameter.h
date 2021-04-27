/// \file SCliParameter.h
/// \brief SCliParameter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "scliExport.h"
#include "SCliSelect.h"

/// \class SCliParameter
/// \brief class that define a container for unix shell single input
class SCLI_EXPORT SCliParameter{

public:
    static const std::string TypeData; 
    static const std::string TypeFloat;
    static const std::string TypeInt;
    static const std::string TypeString;
    static const std::string TypeSelect;
    static const std::string TypeBoolean;

    static const std::string IOInput;
    static const std::string IOOutput;

public:
    SCliParameter(std::string name, std::string description="");

public:
    std::string getName();
    std::string getValueSuffix();
    std::string getDescription();
    std::string getValue();
    std::string getType();
    std::string getIO();
    std::string getDefault();
    bool getIsAdvanced();
    SCliSelect getSelectInfo();

public:
    void setDescription(std::string description);
    void setType(std::string type);
    void setIO(std::string io);
    void setValue(std::string value);
    void setDefault(std::string value);
    void setIsAdvanced(bool value);
    void setValueSuffix(std::string suffix);
    void setSelectInfo(SCliSelect select);

protected:
    std::string m_name;
    std::string m_description;
    std::string m_value;
    std::string m_type; // image, number, string, choice
    std::string m_io; // input | output
    std::string m_defaultValue;
    SCliSelect m_selectInfo; // for select
    std::string m_valueSuffix;
    bool m_isAdvanced;
};
