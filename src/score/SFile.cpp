/// \file SFile.cpp
/// \brief SFile class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#include "SFile.h"

#include <iostream>
#include <fstream>

bool SFile::exists(std::string file){
    std::ifstream infile(file);
    return infile.good();
}
