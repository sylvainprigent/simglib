/// \file SCSV.cpp
/// \brief SCSV class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>
#include <fstream>

#include "SCSV.h"
#include "sdata/SArray.h"
#include "sdata/SBool.h"
#include "sdata/SDatatypes.h"
#include "sdata/SDataException.h"
#include "sdata/SFloat.h"
#include "sdata/SInt.h"
#include "sdata/SObject.h"
#include "sdata/SString.h"
#include "sdata/STable.h"

SCSV::SCSV(){
    m_data = nullptr;
}

void SCSV::read(std::string file, std::string separator){

    int number_of_lines = 0;
    std::string line;
    std::ifstream myfile(file);

    while (std::getline(myfile, line))
        ++number_of_lines;
    myfile.seekg(0);

    if ( number_of_lines == 1 ){
        std::getline(myfile, line);
        std::istringstream s(line);
        int xi;
        float xf;
        if (s >> xi) {
            m_data = new SInt(line);
        }
        else if( s >> xf ){
            m_data = new SFloat(line);
        }
        else if(line == "true"){
            m_data = new SBool(true);
        }
        else if(line == "false"){
            m_data = new SBool(false);
        }
        else{
            m_data = new SString(line);
        }
    }
    else{
        std::getline(myfile, line);
        std::vector<std::string> lineArray = this->parseLine(line, separator.c_str()[0]);
        if ( lineArray.size() == 0 ){
            throw SDataException(std::string("Unable to parse the table or array line: " + line).c_str());
        }
        else if ( lineArray.size() == 1 ){
            // read array
            SArray* dataArray = new SArray();
            dataArray->resize(number_of_lines);
            dataArray->set(0, new SString(line));
            int idx = 0;
            while(std::getline(myfile, line)){
                idx++;
                dataArray->set(idx, new SString(line));
            }
            m_data = dataArray;
        }
        else{
            STable* dataTable = new STable();
            // set header
            dataTable->setHeaders(lineArray);

            // set data
            while(std::getline(myfile, line)){
                std::vector<std::string> dataArray = this->parseLine(line, separator.c_str()[0]);
                dataTable->addRow(dataArray);
            }
            m_data = dataTable;
        }
    }

    std::cout << "Number of lines in text file: " << number_of_lines;
}

void SCSV::write(std::string file, std::string separator){
    std::ofstream ofile (file);
    if (ofile.is_open())
    {
        ofile << m_data->csv(separator);
        ofile.close();
    }
}

void SCSV::set(SData* data){
    m_data = data;
}

SData* SCSV::get(){
    return m_data;
}

std::vector<std::string> SCSV::parseLine(const std::string& line, char separator){

    std::vector<std::string> data;
    unsigned int previousSeparatorIdx = -1;
    unsigned int countQuote = 0;
    unsigned int countDoubleQuote = 0;
    unsigned int countSingleQuote = 0;
    for (int i = 0 ; i < int(line.size()) ; i++){
        if ( line[i] == '`' ){
            countQuote++;
        }
        else if ( line[i] == '"'){
            countDoubleQuote++;
        }
        else if ( line[i] == '\''){
            countSingleQuote++;
        }
        else if ( line[i] == separator){
            if ( countQuote % 2 == 0 && countDoubleQuote % 2 == 0 && countSingleQuote % 2 == 0){
                data.push_back( line.substr(previousSeparatorIdx+1, i-previousSeparatorIdx-1 ) );
                previousSeparatorIdx = i;
                countQuote = 0;
                countDoubleQuote = 0;
                countSingleQuote = 0;
            }

        }
    }
    return data;
}
