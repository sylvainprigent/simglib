/// \file SCliParameterList.h
/// \brief SCliParameterList class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include "SCliParameter.h"
#include <vector>
#include "scliExport.h"

/// \class SCliParameterList
/// \brief class that define a container for unix shell single parameter
class SCLI_EXPORT SCliParameterList{

public:
    SCliParameterList();
    ~SCliParameterList();

public:
    void add(SCliParameter* input);
    void setProgramName(std::string name);
    void setDescription(std::string description);
    void setWrapperName(std::string name);
    void setToolboxName(std::string name);
    void setHelp(std::string help);
    void setWrapperPath(std::string path);
    std::string getProgramName();
    std::string getDescription();
    std::string getWrapperName();
    std::string getToolboxName();
    std::string getHelp();
    std::string getWrapperPath();
    bool exists(std::string parameterName);

public:
    SCliParameter* get(std::string name);
    std::vector<SCliParameter*> getInputImages();
    std::vector<SCliParameter*> getOutputImages();


    SCliParameter* at(int i);
    int size();
    std::string man();

protected:
    std::vector< SCliParameter* > m_parameters;
    std::string m_programName;
    std::string m_wrapperName;
    std::string m_wrapperPath;
    std::string m_description;
    std::string m_toolboxName;
    std::string m_help;
};
