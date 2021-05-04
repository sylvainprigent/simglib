/// \file SCliParser.h
/// \brief SCliParser class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2018

#pragma once

#include "SCliParameterList.h"
#include "scliExport.h"

/// \class SCliParser
/// \brief class that define function to parse a unix shell commande
class SCLI_EXPORT SCliParser{

public:
    SCliParser(int argc, char *argv[]);
    ~SCliParser();

public:
    // inputs
    void addParameter(std::string name, std::string description, std::string defaultValue, std::string type = SCliParameter::TypeString , bool isAdvanced=false);
    void addParameterFloat(std::string name, std::string description, float defaultValue, bool isAdvanced=false);
    void addParameterInt(std::string name, std::string description, int defaultValue, bool isAdvanced=false);
    void addParameterString(std::string name, std::string description, std::string defaultValue, bool isAdvanced=false);
    void addParameterSelect(std::string name, std::string description, std::string defaultValue, bool isAdvanced=false);
    void addParameterBoolean(std::string name, std::string description, bool defaultValue, bool isAdvanced=false);

    static const std::string TypeFloat;
    static const std::string TypeInt;
    static const std::string TypeString;
    static const std::string TypeSelect;
    static const std::string TypeBoolean;
    void addInputData(std::string name, std::string description);
    void addOutputData(std::string name, std::string Description);
    void setMan(std::string man);

    // parse
    void parse(int minArgc = 0);

    // outputs
    bool getParameterBool(std::string name);
    int getParameterInt(std::string name);
    float getParameterFloat(std::string name);
    std::string getParameterString(std::string name);
    const char* getParameterChar(std::string name);
    std::string getDataURI(std::string name);

public:
    void print();

private:
    void setArgValue(std::string argName, std::string value);

private:
    int m_argc;
    char **m_argv;

private:
    SCliParameterList m_args;
    std::string m_man;
};
