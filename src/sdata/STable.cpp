/// \file STable.cpp
/// \brief STable class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "STable.h"
#include "SDatatypes.h"
#include "SDataException.h"


STable::STable(){
    m_datatype = SDatatypes::Table;
    m_row = 0;
    m_col = 0;
}

STable::STable(unsigned int row, unsigned int col){
    m_row = row;
    m_col = col;

    m_headers.resize(col);
    m_content.resize(row);
    for (unsigned int i = 0 ; i < row ; i++){
        m_content[i].resize(col);
    }
}

STable::~STable(){

}

std::string STable::json(int level){

    std::string content = this->indent(level) + "{\ntable: [\n";
    for (unsigned int r = 0 ; r < m_row ; r++){
        content += this->indent(level+1) + "{ ";
        for (unsigned int c = 0 ; c < m_col ; c++){
            content += m_headers[c] + ": " + m_content[r][c] + ",";
        }
        content.pop_back();
        content += "},\n";
    }
    content.pop_back();
    content.pop_back();
    content += this->indent(level+1) + "\n]\n";
    content += this->indent(level) +  "}";
    if (level > 0){
        content += ",";
    }
    return content;
}

std::string STable::indent(int level){

    std::string ind = "";
    for (int i = 0 ; i < 4*level ; i++){
        ind += " ";
    }
    return ind;

}

std::string STable::csv(std::string separator){

    std::string content;
    // headers
    for (unsigned int c = 0 ; c < m_headers.size() ; c++){
        content += m_headers[c] + ",";
    }
    content.pop_back();
    content += '\n';

    // content
    for (unsigned int r = 0 ; r < m_row ; r++){
        for (unsigned int c = 0 ; c < m_col ; c++){
            content += m_content[r][c] + separator;
        }
        content.pop_back();
        content += "\n";
    }
    content.pop_back();
    return content;

}

void setHeader(unsigned int col, std::string value);
std::string getHeader(unsigned int col);

unsigned int STable::getHeight(){
    return m_row;
}

unsigned int STable::getWidth(){
    return m_col;
}

void STable::setHeader(unsigned int col, std::string value){
    if (col < m_row){
        m_headers[col] = value;
    }
    else{
        std::string message = "STable::setHeader: index out of range value [col: "+std::to_string(col)+", value: "+value+"]";
        throw SDataException(message.c_str());
    }
}

void STable::setHeaders(std::vector<std::string>& value){
    m_headers = value;
    m_col = m_headers.size();
}

std::string STable::getHeader(unsigned int col){
    if (col < m_row){
        return m_headers[col];
    }
    else{
        std::string message = "STable::getHeader: index out of range value [col: "+std::to_string(col)+"]";
        throw SDataException(message.c_str());
    }
}

std::string STable::get(unsigned int row, unsigned int col){
    if (row < m_row && col < m_col){
        return m_content[row][col];
    }
    else{
        std::string message = "STable::getHeader: index out of range value [row: "+std::to_string(row)+", col: "+std::to_string(col)+"]";
        throw SDataException(message.c_str());
    }
}

float STable::getFloat(unsigned int row, unsigned int col){
    return std::stof(this->get(row,col));
}

int STable::getInt(unsigned int row, unsigned int col){
    return std::atoi(this->get(row,col).c_str());
}

unsigned int STable::getUnsigned(unsigned int row, unsigned int col){
    return unsigned(std::atoi(this->get(row,col).c_str()));
}

void STable::set(unsigned int row, unsigned int col, std::string value){
    if (row < m_row && col < m_col){
        m_content[row][col] = value;
    }
    else{
        std::string message = "STable::set: index out of range value [row: "+std::to_string(row)+", col: "+std::to_string(col)+"] = "+value;
        throw SDataException(message.c_str());
    }
}

void STable::set(unsigned int row, unsigned int col, float value){
    this->set(row,col,std::to_string(value));
}

void STable::set(unsigned int row, unsigned int col, int value){
    this->set(row,col,std::to_string(value));
}

void STable::set(unsigned int row, unsigned int col, unsigned int value){
    this->set(row,col,std::to_string(value));
}

void STable::addRow(std::vector<std::string>& data){
    m_content.push_back(data);
    m_row++;
}

std::vector< std::vector<std::string> > STable::buffer(){
    return m_content;
}

std::vector< std::string > STable::headers(){
    return m_headers;
}
