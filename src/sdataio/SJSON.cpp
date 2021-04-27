/// \file SJSON.cpp
/// \brief SJSON class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>
#include <fstream>

#include "score/SStringOp.h"

#include "SJSON.h"
#include "sdata/SDatatypes.h"
#include "sdata/SDataException.h"
#include "sdata/SObject.h"
#include "sdata/SString.h"
#include "sdata/SArray.h"
#include "sdata/SFloat.h"
#include "sdata/SInt.h"
#include "sdata/SBool.h"

SJSON::SJSON(){
    m_data = nullptr;
}

void SJSON::read(std::string file){
    std::string content = this->readFileContent(file);

    if (content[0] == '{'){
        int* interval = this->findObjectLimits(content, 0, '{', '}');

        //std::string c = content.substr(interval[0]+1, (interval[1]-1) -interval[0] );
        //std::cout << c << std::endl;
        //return;
        m_data = this->parseObject(content, interval[0]+1, interval[1]);
    }
    else{
        int* interval = this->findObjectLimits(content, 0, '[', ']');
        m_data = this->parseArray(content, interval[0]+1, interval[1]);
    }
}

void SJSON::write(std::string file){
    std::ofstream ofile (file);
    if (ofile.is_open())
    {
        ofile << m_data->json();
        ofile.close();
    }
}

void SJSON::set(SData* data){
    m_data = data;
}

SData* SJSON::get(){
    return m_data;
}

std::string SJSON::readFileContent(std::string file){
    std::string content, line;
    std::ifstream ofile (file);
    if (ofile.is_open())
    {
        while ( getline (ofile,line) )
        {
            content += line;
        }
        ofile.close();
    }
    return content;
}

SData* SJSON::parseObject(std::string fileContent, int start, int end){

    std::string c = fileContent.substr(start, end-start );

    SObject* object = new SObject();

    int pos = start;
    int posColon;
    std::string currentKey;

    while ( pos < end){

        posColon = this->findNextChar(fileContent, pos, end, ':' );
        currentKey = this->getKeyIn(fileContent, pos, posColon);
        char objDelim = this->getObjectDelimiter(fileContent, posColon);
        if (objDelim == '"'){
            int* valueIdx = this->getValueIndex(fileContent, posColon, end);
            SString* strObject = new SString( fileContent.substr(valueIdx[0]+1, (valueIdx[1]-1)-valueIdx[0]) );
            object->set(currentKey, strObject);
            pos = valueIdx[1];
        }
        else if (objDelim == 'n'){
            int* valueIdx = this->getNumberIndex(fileContent, posColon, end);
            std::string number = fileContent.substr(valueIdx[0]+1, (valueIdx[1]-1)-valueIdx[0]);
            number = SStringOp::remove(number, ' ');
            //number.erase(std::remove(number.begin(),number.end(), " "),number.end()); // remove spaces
            if (this->contains(number, '.')){
                SFloat *floatObj = new SFloat(number);
                object->set(currentKey, floatObj);
            }
            else if(number == "true" || number == "false" ){
                SBool *boolObj = new SBool();
                if (number == "true"){
                    boolObj->set(true);
                }
                else{
                    boolObj->set(false);
                }
                object->set(currentKey, boolObj);
            }
            else{
                SInt *intObj = new SInt(number);
                object->set(currentKey, intObj);
            }
            pos = valueIdx[1];
        }
        else if (objDelim == '{'){
            int* interval = this->findObjectLimits(fileContent, pos, '{', '}');
            SData* data = this->parseObject(fileContent, interval[0]+1, interval[1]);
            object->set(currentKey, data);
            pos = interval[1];
        }
        else if (objDelim == '['){
            int* interval = this->findObjectLimits(fileContent, pos, '[', ']');
            SData* data = this->parseArray(fileContent, interval[0]+1, interval[1]);
            object->set(currentKey, data);
            pos = interval[1];
        }

        // go the the next key
        pos = this->findNextChar(fileContent, pos, end, ',');

    }
    return object;
}

bool SJSON::contains(std::string str, char c){
    for (int i = 0 ; i < int(str.size()) ; i++){
        if (str[i] == c){
            return true;
        }
    }
    return false;
}

SData* SJSON::parseArray(std::string fileContent, int start, int end){

    std::string c = fileContent.substr(start, end-start );

    SArray* array = new SArray();

    int pos = start;
    while (pos < end){

        char objDelim = this->getObjectDelimiter(fileContent, pos);
        if (objDelim == '"'){
            int* valueIdx = this->getValueIndex(fileContent, pos, end);
            SString* strObject = new SString( fileContent.substr(valueIdx[0]+1, (valueIdx[1]-1)-valueIdx[0]) );
            array->append(strObject);
            pos = valueIdx[1];
        }
        else if (objDelim == 'n'){
            int* valueIdx = this->getNumberIndex(fileContent, pos, end);
            std::string number = fileContent.substr(valueIdx[0]+1, (valueIdx[1]-1)-valueIdx[0]);
            number = SStringOp::remove(number, ' ');
            //number.erase(std::remove(number.begin(),number.end()," "),number.end()); // remove spaces
            if (this->contains(number, '.')){
                SFloat *floatObj = new SFloat(number);
                array->append(floatObj);
            }
            else if(number == "true" || number == "false" ){
                SBool *boolObj = new SBool();
                if (number == "true"){
                    boolObj->set(true);
                }
                else{
                    boolObj->set(false);
                }
                array->append(boolObj);
            }
            else{
                SInt *floatObj = new SInt(number);
                array->append(floatObj);
            }
            pos = valueIdx[1];
        }
        else if (objDelim == '['){
            int* interval = this->findObjectLimits(fileContent, pos, '[', ']');

            SData* data = this->parseArray(fileContent, interval[0]+1, interval[1]);
            array->append(data);
            pos = interval[1];
        }
        else if (objDelim == '{'){
            int* interval = this->findObjectLimits(fileContent, pos, '{', '}');
            SData* data = this->parseObject(fileContent, interval[0]+1, interval[1]);
            array->append(data);
            pos = interval[1];
        }

        // go the the next item
        pos = this->findNextChar(fileContent, pos, end, ',');
        pos++;
    }

    return array;
}

char SJSON::getObjectDelimiter(const std::string &content, int start){

    char c;
    for (int i = start ; i < int(content.size()) ; i++){
        c = content[i];
        if ( c == '"' || c == '[' || c == '{'){
            return c;
        }
        if ( c == '0' || c == '1' || c == '2' || c == '3' || c == '4' || c == '5' || c == '6' || c == '7' || c == '8' || c == '9'){
            return 'n';
        }
        if ( c == 't' || c == 'f'){
            return 'n';
        }
    }
    throw SDataException("ERROR: SJSON cannot parse object value");
}

int* SJSON::getValueIndex(const std::string &content, int start, int stop){

    int* interval = new int[2];
    interval[0] = -1;
    for (int i = start ; i <  stop ; i++){
        if (content[i] == '"'){
            if (interval[0] == -1){
                interval[0] = i;
            }
            else{
                interval[1] = i;
                return interval;
            }
        }
    }
    throw SDataException("ERROR: SJSON Parse error in value" );
}

int* SJSON::getNumberIndex(const std::string &content, int start, int stop){

    int* interval = new int[2];
    interval[0] = start;
    interval[1] = stop;
    for (int i = start ; i <  stop ; i++){
        if (content[i] == ',' || content[i] == '\n' || content[i] == '\r' || content[i] == ']'){
            interval[1] = i;
            return interval;
        }
    }
    return interval;
}

std::string SJSON::getKeyIn(const std::string &content, int start, int stop){

    int firstQuote = -1;
    for (int i = start ; i <= stop ; i++){
        if (content[i] == '"'){
            if (firstQuote == -1){
                firstQuote = i;
            }
            else{
                return content.substr(firstQuote+1, i-1-firstQuote);
            }
        }
    }
    throw SDataException("ERROR parse error in key" );
}

int SJSON::findNextChar(const std::string &content, int start, int end, char s ){

    for (int i = start ; i < end ; i++){
        if (content[i] == s){
            return i;
        }
    }
    return end;
}

int* SJSON::findObjectLimits(std::string content, int start, char open, char close){

    int pos0 = start;
    int pos1 = start;

    // position of the first open
    for (int i = start ; i < int(content.size()) ; i++){
        if (content[i] == open){
            pos0 = i;
            break;
        }
    }

    // get position of the close
    int countDiff = -1;
    for (int i = start ; i < int(content.size()) ; i++){
        if (content[i] == open){
            countDiff++;
        }
        if (content[i] == close){
            countDiff--;

            if (countDiff == -1){
                pos1 = i;
                break;
            }
        }
    }

    int* interval = new int[2];
    interval[0] = pos0;
    interval[1] = pos1;
    return interval;
}
