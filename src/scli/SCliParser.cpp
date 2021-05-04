/// \file SCliParser.cpp
/// \brief SCliParser class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SCliParser.h"
#include "SCliException.h"
#include "SCliStringOp.h"
#include <iostream>

SCliParser::SCliParser(int argc, char *argv[])
{
    m_argc = argc;
    m_argv = argv;
    m_args.setProgramName(argv[0]);
}

SCliParser::~SCliParser()
{

}

void SCliParser::addParameter(std::string name, std::string description, std::string defaultValue, std::string type,  bool isAdvanced)
{
    SCliParameter *arginput = new SCliParameter(name, description);
    arginput->setIO("input");
    arginput->setType(type);
    arginput->setDefault(defaultValue);
    arginput->setValue(defaultValue);
    arginput->setIsAdvanced(isAdvanced);
    m_args.add(arginput);
}

void SCliParser::addParameterFloat(std::string name, std::string description, float defaultValue, bool isAdvanced){
    this->addParameter(name, description, SCliStringOp::float2string(defaultValue), SCliParameter::TypeFloat, isAdvanced);
}

void SCliParser::addParameterInt(std::string name, std::string description, int defaultValue, bool isAdvanced){
    this->addParameter(name, description, SCliStringOp::int2string(defaultValue), SCliParameter::TypeInt, isAdvanced);
}

void SCliParser::addParameterString(std::string name, std::string description, std::string defaultValue, bool isAdvanced){
    this->addParameter(name, description, defaultValue, SCliParameter::TypeString, isAdvanced);
}

void SCliParser::addParameterSelect(std::string name, std::string description, std::string defaultValue, bool isAdvanced){
    this->addParameter(name, description, defaultValue, SCliParameter::TypeSelect, isAdvanced);
}

void SCliParser::addParameterBoolean(std::string name, std::string description, bool defaultValue, bool isAdvanced){
    std::string val = "False";
    if (defaultValue){
        val = "True";
    }
    this->addParameter(name, description, val, SCliParameter::TypeBoolean, isAdvanced);
}

void SCliParser::addInputData(std::string name, std::string description)
{
    SCliParameter *arginput = new SCliParameter(name, description);
    arginput->setIO("input");
    arginput->setType(SCliParameter::TypeData);
    m_args.add(arginput);
}

void SCliParser::addOutputData(std::string name, std::string description)
{
    SCliParameter *arginput = new SCliParameter(name, description);
    arginput->setIO("output");
    arginput->setType(SCliParameter::TypeData);
    m_args.add(arginput);
}

void SCliParser::setMan(std::string man)
{
    m_man = man;
}

// parse
void SCliParser::parse(int minArgc)
{

    if (m_argc == 2 && ( std::string(m_argv[1]) == "-h" || std::string(m_argv[1]) == "--help"))
    {

        std::string man = "OVERVIEW: " +  m_man + "\n\n";
        man += "USAGE: "+std::string(m_argv[0])+" [parameters] \n\n";
        man += "PARAMETERS:\n";
        man += m_args.man();
        throw SCliException(man.c_str());
    }

    if (m_argc <= minArgc)
    {
        throw SCliException(("ERROR: "+std::string(m_argv[0])+"  needs at least " + std::to_string(minArgc) + " arguments. \n Call it with -h or --help to read the man page").c_str());
    }
    else
    {
        for (int i = 1; i < m_argc; ++i)
        {
            std::string argName = std::string(m_argv[i]);
            if (i + 1 < m_argc && !argName.compare(0, 1, "-"))
            {
                i++;
                this->setArgValue(argName, std::string(m_argv[i]));
            }
            else
            {
                throw SCliException((argName + " option require one argument").c_str());
            }
        }
    }
}

void SCliParser::print()
{
    for (int i = 0; i < m_args.size(); i++)
    {
        std::cout << m_args.at(i)->getName() << " " << m_args.at(i)->getValue() << std::endl;
    }
}

void SCliParser::setArgValue(std::string name, std::string value)
{
    bool found = false;
    for (int j = 0; j < m_args.size(); ++j)
    {
        if (m_args.at(j)->getName() == name)
        {
            m_args.at(j)->setValue(value);
            found = true;
            break;
        }
    }
    if (!found)
    {
        throw SCliException((name + " option is unknown").c_str());
    }
}

bool SCliParser::getParameterBool(std::string name){
    std::string value = this->getParameterString(name);
    if ( value == "true" || value == "True" || value == "TRUE" ){
        return true;
    }
    return false;
}

int SCliParser::getParameterInt(std::string name)
{
    return std::stoi(this->getParameterString(name));
}

float SCliParser::getParameterFloat(std::string name)
{
    return std::stof(this->getParameterString(name));
}

std::string SCliParser::getParameterString(std::string name)
{
    SCliParameter* param = m_args.get(name);
    if (param){
        return param->getValue();
    }
    return "";
}

const char* SCliParser::getParameterChar(std::string name)
{
    return this->getParameterString(name).c_str();
}

std::string SCliParser::getDataURI(std::string name){
    return this->getParameterString(name);
}
