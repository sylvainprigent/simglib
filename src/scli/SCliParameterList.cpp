/// \file SCliParameterList.cpp
/// \brief SCliParameterList class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SCliParameterList.h"

SCliParameterList::SCliParameterList()
{
}

SCliParameterList::~SCliParameterList()
{
    for (int i = 0; i < int(m_parameters.size()) ; i++)
    {
        //delete m_parameters[i];
    }
}

void SCliParameterList::add(SCliParameter *paramater)
{
    m_parameters.push_back(paramater);
}

void SCliParameterList::setProgramName(std::string name)
{
    m_programName = name;
}

void SCliParameterList::setDescription(std::string description){
    m_description = description;
}

void SCliParameterList::setWrapperName(std::string name){
    m_wrapperName = name;
}

void SCliParameterList::setWrapperPath(std::string path){
    m_wrapperPath = path;
}

void SCliParameterList::setToolboxName(std::string name){
    m_toolboxName = name;
}

std::string SCliParameterList::getDescription(){
    return m_description;
}

std::string SCliParameterList::getProgramName(){
    return m_programName;
}

std::string SCliParameterList::getWrapperName(){
    return m_wrapperName;
}

std::string SCliParameterList::getWrapperPath(){
    return m_wrapperPath;
}

std::string SCliParameterList::getToolboxName(){
    return m_toolboxName;
}

void SCliParameterList::setHelp(std::string help){
    m_help = help;
}

std::string SCliParameterList::getHelp(){
    return m_help;
}

bool SCliParameterList::exists(std::string parameterName){

    for (std::vector<SCliParameter>::size_type i = 0; i < m_parameters.size(); i++)
    {
        if ( m_parameters[i]->getName() == parameterName){
            return true;
        }
    }
    return false;
}

std::string SCliParameterList::man()
{
    std::string txt = this->getProgramName() + ": " + this->getDescription() + "\n";
    for (std::vector<SCliParameter>::size_type i = 0; i < m_parameters.size(); i++)
    {
        txt += "\t" + m_parameters[i]->getName() + "\t" + m_parameters[i]->getDefault() + "\t" + m_parameters[i]->getDescription() + "\n";
    }
    return txt;
}

int SCliParameterList::size()
{
    return int(m_parameters.size());
}

SCliParameter *SCliParameterList::at(int i)
{
    return m_parameters[std::vector<SCliParameter>::size_type(i)];
}

SCliParameter *SCliParameterList::get(std::string name)
{
    for (std::vector<SCliParameter>::size_type i = 0; i < m_parameters.size(); i++)
    {
        if (m_parameters[i]->getName() == name)
        {
            return m_parameters[i];
        }
    }
    return NULL;
}

std::vector<SCliParameter*> SCliParameterList::getInputImages(){
    std::vector<SCliParameter*> inputImages;
    for(std::vector<SCliParameter>::size_type i = 0 ; i < m_parameters.size() ; i++){
        if ( m_parameters[i]->getType() == "image" && m_parameters[i]->getIO() == "input"){
            inputImages.push_back(m_parameters[i]);        
        } 
    }
    return inputImages;
}

std::vector<SCliParameter*> SCliParameterList::getOutputImages(){
    std::vector<SCliParameter*> outputImages;
    for(std::vector<SCliParameter>::size_type i = 0 ; i < m_parameters.size() ; i++){
        if ( m_parameters[i]->getType() == "image" && m_parameters[i]->getIO() == "output"){
            outputImages.push_back(m_parameters[i]);        
        } 
    }
    return outputImages;
}
