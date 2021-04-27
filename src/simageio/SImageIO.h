/// \file SCimgIO.h
/// \brief SCimgIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageioExport.h"

#include "simage/SImage.h"
#include "score/SObservable.h"

/// \class SCimgIO
/// \brief Read and write images using CImg library
class SIMAGEIO_EXPORT SImageIO : public SObservable {

public:
    SImageIO();

public:
    virtual SImage* read(std::string file, char precision) = 0;
    virtual void write(SImage* image, std::string file) = 0;

protected:
    std::string m_readerName;

};
