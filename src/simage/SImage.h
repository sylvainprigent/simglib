/// \file SImage.h
/// \brief SImage class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"

/// \class SImage
/// \brief SImage an image
class SIMAGE_EXPORT SImage{

public:
    SImage();
    virtual ~SImage();

public:
    void setSizeX(unsigned int value);
    void setSizeY(unsigned int value);
    void setSizeZ(unsigned int value);
    void setSizeT(unsigned int value);
    void setSizeC(unsigned int value);

    unsigned int getSizeX();
    unsigned int getSizeY();
    unsigned int getSizeZ();
    unsigned int getSizeT();
    unsigned int getSizeC();
    unsigned long getBufferSize();

public:
    virtual void allocate() = 0;

public:
    std::string info();

public:
    // metadata
    float getResX();
    float getResY();
    float getResZ();
    float getResT();
    std::string getUnit();
    void setRes(float rx, float ry, float rz, float rt);
    void setResX(float value);
    void setResY(float value);
    void setResZ(float value);
    void setResT(float value);
    void setUnit(std::string value);
    char getPrecision();

public:
    virtual void initAtZero() = 0;

protected:
    unsigned int m_sx; // sizeX = width
    unsigned int m_sy; // sizeY = height
    unsigned int m_sz; // sizeZ = slice
    unsigned int m_st; // sizeT = frame
    unsigned int m_sc; // sizeC = channel
    float m_rx; // resolutionX
    float m_ry; // resolutionY
    float m_rz; // resolutionZ
    float m_rt; // resolutionT
    std::string m_unit; // resolution unit
    char m_precision; // precision

};

