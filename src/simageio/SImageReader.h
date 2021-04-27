/// \file SImageReader.h
/// \brief SImageReader class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "simageioExport.h"

#include "simage/SImage.h"

/// \class SImageReader
/// \brief Read an image from file
class SIMAGEIO_EXPORT SImageReader{

public:
    static SImage* read(std::string file, char precision  = 0);
    static void write(SImage* image, std::string file);

};
