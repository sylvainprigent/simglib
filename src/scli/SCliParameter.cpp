/// \file SCliParameter.cpp
/// \brief SCliParameter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#include "SCliParameter.h"

const std::string SCliParameter::TypeData = "data";
const std::string SCliParameter::TypeFloat = "float";
const std::string SCliParameter::TypeInt = "int";
const std::string SCliParameter::TypeString = "string";
const std::string SCliParameter::TypeSelect = "select";
const std::string SCliParameter::TypeBoolean = "boolean";

const std::string SCliParameter::IOInput = "input";
const std::string SCliParameter::IOOutput = "output";

SCliParameter::SCliParameter(std::string name, std::string description)
{
    m_name = name;
    m_description = description;
    m_isAdvanced = false;
}

SCliParameter::~SCliParameter()
{

}

std::string SCliParameter::getName()
{
    return m_name;
}

std::string SCliParameter::getValueSuffix(){
    return m_valueSuffix;
}

std::string SCliParameter::getDescription()
{
    return m_description;
}

std::string SCliParameter::getValue()
{
    return m_value;
}

std::string SCliParameter::getType()
{
    return m_type;
}

std::string SCliParameter::getIO()
{
    return m_io;
}

std::string SCliParameter::getDefault()
{
    return m_defaultValue;
}

bool SCliParameter::getIsAdvanced()
{
    return m_isAdvanced;
}

SCliSelect SCliParameter::getSelectInfo(){
    return m_selectInfo;
}

void SCliParameter::setDescription(std::string description)
{
    m_description = description;
}

void SCliParameter::setType(std::string type)
{
    m_type = type;
}

void SCliParameter::setIO(std::string io)
{
    m_io = io;
}

void SCliParameter::setValue(std::string value)
{
    m_value = value;
}

void SCliParameter::setDefault(std::string value)
{
    m_defaultValue = value;
}

void SCliParameter::setIsAdvanced(bool value)
{
    m_isAdvanced = value;
}

void SCliParameter::setValueSuffix(std::string suffix)
{
    m_valueSuffix = suffix;
}

void SCliParameter::setSelectInfo(SCliSelect select){
    m_selectInfo = select;
}
