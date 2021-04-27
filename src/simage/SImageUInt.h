/// \file SImageUInt.h
/// \brief SImageUInt class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "simageExport.h"

#include "SImage.h"

/// \class SImage
/// \brief SImage an image
class SIMAGE_EXPORT SImageUInt : public SImage{

public:
    SImageUInt();
    SImageUInt(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);
    SImageUInt(unsigned int* buffer, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);

    virtual ~SImageUInt();

public:
    unsigned int getMin();
    unsigned int getMax();
    float getMean();
    unsigned int getSum();
    SImageUInt* getSlice(unsigned int z);
    SImageUInt* getFrame(unsigned int t);
    SImageUInt* getChannel(unsigned int c);
    unsigned int getPixel(unsigned int x, unsigned int y);
    unsigned int getPixel(unsigned int x, unsigned int y, unsigned int z);
    unsigned int getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t);
    unsigned int getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c);
    unsigned int getPixelXYC(unsigned int x, unsigned int y, unsigned int c);
    unsigned int getPixelXYZC(unsigned int x, unsigned int y, unsigned int z, unsigned int c);

    void setPixel(unsigned int value, unsigned int x, unsigned int y, unsigned int z = 0, unsigned int t = 0, unsigned int c = 0);

public:
    virtual void allocate();
    unsigned int* getBuffer();

public:
    virtual void initAtZero();

protected:
    unsigned int* m_buffer;

};

