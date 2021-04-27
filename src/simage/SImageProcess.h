/// \file SImageProcess.h
/// \brief SImageProcess class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2019

#pragma once

#include "simageExport.h"

#include "score/SProcess.h"
#include "SImage.h"
#include "SImageUInt.h"
#include "SImageInt.h"
#include "SImageFloat.h"

/// \class SImageProcess
/// \brief Define a common interface for an process on an image
class SIMAGE_EXPORT SImageProcess : public SProcess{

public:
    SImageProcess();
    virtual ~SImageProcess();

public:
    void setInput(SImage* image);

public:
    virtual void checkInputs();
    virtual void run() = 0;

protected:
    void checkInput();
    SImageUInt* castInputToUInt();
    SImageInt* castInputToInt();
    SImageFloat* castInputToFloat();

protected:
    SImage* m_input;
    char m_processPrecision;
    bool m_processZ;
    bool m_processT;
    bool m_processC;

};
