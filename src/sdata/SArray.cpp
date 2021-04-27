/// \file SArray.cpp
/// \brief SArray class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SArray.h"
#include "SDatatypes.h"

SArray::SArray() : SData(){
    m_datatype = SDatatypes::Array;
}

SArray::SArray(int size) : SData(){
    m_datatype = SDatatypes::Object;
    m_data.resize(size);
}

std::string SArray::json(int level){
    // check is there is an object in the array
    bool hasObject = false;
    for (int i = 0 ; i < int(m_data.size()) ; i++){
        if (m_data[i]->getDatatype() == SDatatypes::Object){
            hasObject = true;
            break;
        }
    }
    std::string strData = "";
    if (hasObject){
        strData += "[\n";
        for (int i = 0 ; i < int(m_data.size()) ; i++){
           strData +=  m_data[i]->json(level+1) + "\n"; // add a parameter in json for tabs
        }
        if (strData[strData.size()-2] == ','){
            strData.pop_back();
            strData.pop_back();
            strData.append("\n");
        }
        strData += this->indent(level) + "]";
    }
    else{
        strData += "[";
        for (int i = 0 ; i < int(m_data.size()) ; i++){
           strData += m_data[i]->json(level) + ", "; // add a parameter in json for tabs
        }
        strData.pop_back();
        strData.pop_back();
        strData += "]";
    }
    return strData;
}

std::string SArray::indent(int level){

    std::string ind = "";
    for (int i = 0 ; i < 4*level ; i++){
        ind += " ";
    }
    return ind;

}

std::string SArray::csv(std::string separator) {
    std::string strData = "";
    for (unsigned int i = 0 ; i < m_data.size() ; i++){
        strData += m_data[i]->csv(separator) + '\n';
    }
    strData.pop_back();
    return strData;
}

void SArray::resize(unsigned int size){
    m_data.resize(size);
}

unsigned int SArray::size(){
    return m_data.size();
}

void SArray::set(int idx, SData* data){
    m_data[idx] = data;
}

SData* SArray::get(int idx){
    return m_data[idx];
}

void SArray::append(SData* data){
    m_data.push_back(data);
}

