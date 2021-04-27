/// \file slImageIO.h
/// \brief slImageIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <tiffio.h>

#include "simageioExport.h"

#include "SImageIO.h"

#include "simage/SImageFloat.h"
#include "simage/SImageUInt.h"
#include "simage/SImageInt.h"

/// \class slTiffIO
/// \brief Read and write images using libtiff
class SIMAGEIO_EXPORT STiffIO : public SImageIO{

public:
    STiffIO();

public:
    SImage* read(std::string file, char precision);
    void write(SImage* image, std::string file);

public:
    void write(SImageFloat* image, std::string file);
    void write(SImageUInt* image, std::string file);

public:
    SImageUInt* read_uint(std::string file); 
    SImageInt* read_int(std::string file); 
    SImageFloat* read_float(std::string file); 
        

protected:
    void write_rgb(SImageUInt* image, std::string file);    

protected:
    void parseDescription(SImage* image, std::string description);
    void printTIFFTags(TIFF *tif);
    std::string createDescription(SImage* image);
};
