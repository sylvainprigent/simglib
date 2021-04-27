/// \file SObject.cpp
/// \brief SObject class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SObject.h"
#include "SDatatypes.h"
#include "SDataException.h"

SObject::SObject() : SData(){
    m_datatype = SDatatypes::Object;
}

std::string SObject::json(int level){
    std::string strData = this->indent(level) + "{\n";
    std::map<std::string, SData*>::iterator itr;
    for (itr = m_data.begin(); itr != m_data.end(); ++itr) {
        strData += this->indent(level+1) + "\"" + itr->first + "\": ";
        strData += itr->second->json(level+1);
        if ( itr->second->getDatatype() != SDatatypes::Object ){
            strData += ",\n";
        }
        else{
            strData += "\n";
        }
    }
    strData = strData.substr(0, strData.size()-2) + "\n"+this->indent(level)+"}";
    if (level > 0){
        strData += ",";
    }
    return strData;
}

std::string SObject::indent(int level){

    std::string ind = "";
    for (int i = 0 ; i < 4*level ; i++){
        ind += " ";
    }
    return ind;

}

std::string SObject::csv(std::string){
    return "{SObject: \"cannot be exported in csv\"}";
}

SData* SObject::get(std::string key){
    return m_data[key];
}

void SObject::set(std::string key, SData* value){
    m_data[key] = value;
}

bool SObject::hasKey(std::string key){
    std::map<std::string, SData*>::iterator itr;
    for (itr = m_data.begin(); itr != m_data.end(); ++itr) {
        if (itr->first == key){
            return true;
        }
    }
    return false;
}
