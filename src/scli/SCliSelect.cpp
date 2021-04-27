/// \file SCliSelect.cpp
/// \brief SCliSelect class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SCliSelect.h"

SCliSelect::SCliSelect(){

}

int SCliSelect::count(){
    return int(m_names.size());
}

void SCliSelect::add(std::string name, std::string value){
    m_names.push_back(name);
    m_values.push_back(value);
}

std::string SCliSelect::name(int i){
    return m_names[std::vector<std::string>::size_type(i)];
}

std::string SCliSelect::value(int i){
    return m_values[std::vector<std::string>::size_type(i)];
}
