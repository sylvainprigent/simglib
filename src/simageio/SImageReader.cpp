/// \file SImageReader.cpp
/// \brief SImageReader class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>
#include <fstream>

#include "SImageReader.h"

#include "SImageIOException.h"
#include "STiffIO.h"
#include "SImIO.h"
#include "STxtIO.h"


bool endsWith(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

SImage* SImageReader::read(std::string file, char precision){

    std::ifstream infile(file);
    if (!infile.good()){
        throw SImageIOException(std::string("File does not exists: " + file).c_str());
    }

    if (endsWith(file, ".im")){
        SImIO reader;
        return reader.read(file, precision);
    }
    else if (endsWith(file, ".tif") || endsWith(file, ".tiff") || endsWith(file, ".TIFF") || endsWith(file, ".TIF")){
        STiffIO reader;
        return reader.read(file, precision);
    }
    else if (endsWith(file, ".txt") || endsWith(file, ".TXT") ){
        STxtIO reader;    
        return reader.read(file, precision);    
    }
    else{
        //SCImgIO reader;
        //return reader.read(file, precision);
        return nullptr;
    }
    return nullptr;
}

void SImageReader::write(SImage* image, std::string file){

    //std::cout << "SImageReader::write: " << file << std::endl;     
    if ( endsWith(file, ".im") ){
        SImIO reader;
        return reader.write(image, file);
    }
    else if (endsWith(file, ".tif") || endsWith(file, ".tiff") || endsWith(file, ".TIFF") || endsWith(file, ".TIF")){
        STiffIO reader;
        return reader.write(image, file);
    }
    else if ( endsWith(file, ".txt") || endsWith(file, ".TXT") ){
        STxtIO reader;
        return reader.write(image, file);
    }
    else{
        //SCImgIO reader;
        //return reader.write(image, file);
    }
}
