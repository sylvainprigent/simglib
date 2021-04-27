/// \file SImage.h
/// \brief SImage class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"

#include "SImage.h"
#include "SImageProcess.h"

/// \class SImageFilter
/// \brief Define a common interface for all image filters
class SIMAGE_EXPORT SImageFilter : public SImageProcess{

public:
    SImageFilter();
    virtual ~SImageFilter();

public:
    virtual void run() = 0;

public:
    SImage* getOutput();

protected:
    SImage* m_output;
};
