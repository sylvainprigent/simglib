/// \file SImageFloat.h
/// \brief SImageFloat class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"

#include "SImage.h"

/// \class SImage
/// \brief SImage an image
class SIMAGE_EXPORT SImageFloat : public SImage{

public:
    SImageFloat();
    SImageFloat(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);
    SImageFloat(float* buffer, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);

    virtual ~SImageFloat();

public:
    float getMin();
    float getMax();
    float getMean();
    float getSum();
    SImageFloat* getSlice(unsigned int z);
    SImageFloat* getFrame(unsigned int t);
    SImageFloat* getChannel(unsigned int c);
    float getPixel(unsigned int x, unsigned int y);
    float getPixel(unsigned int x, unsigned int y, unsigned int z);
    float getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t);
    float getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c);
    float getPixelXYC(unsigned int x, unsigned int y, unsigned int c);
    float getPixelXYZC(unsigned int x, unsigned int y, unsigned int z, unsigned int c);

    void setPixel(float value, unsigned int x, unsigned int y, unsigned int z = 0, unsigned int t = 0, unsigned int c = 0);

public:
    virtual void allocate();
    float* getBuffer();

public:
    virtual void initAtZero();

protected:
    float* m_buffer;

};

