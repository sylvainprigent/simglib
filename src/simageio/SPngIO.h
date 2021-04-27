/// \file SPngIO.h
/// \brief SPngIO class
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

/// \class SPngIO
/// \brief Read and write images using libpng
class SIMAGEIO_EXPORT SPngIO : public SImageIO{

public:
    SPngIO();

public:
    SImage* read(std::string file, char precision);
    void write(SImage* image, std::string file);

public:
    void write(SImageFloat* image, std::string file);
    void write(SImageUInt* image, std::string file);

};
