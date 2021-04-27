/// \file SImIO.h
/// \brief SImIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageioExport.h"

#include "SImageIO.h"

/// \class SCimgIO
/// \brief Read and write images using CImg library
class SIMAGEIO_EXPORT SImIO : public SImageIO {

public:
    SImIO();

public:
    SImage* read(std::string file, char precision);
    void write(SImage* image, std::string file);

protected:
    SImage* readDefault(std::string file);
    SImage* readUInt(std::string file);
    SImage* readInt(std::string file);
    SImage* readFloat(std::string file);
    void writeUInt(SImage* image, std::string file);
    void writeInt(SImage* image, std::string file);
    void writeFloat(SImage* image, std::string file);

};
