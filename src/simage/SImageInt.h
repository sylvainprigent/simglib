/// \file SImageInt.h
/// \brief SImageInt class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"

#include "SImage.h"

/// \class SImage
/// \brief SImage an image
class SIMAGE_EXPORT SImageInt : public SImage{

public:
    SImageInt();
    SImageInt(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);
    SImageInt(int* buffer, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1, unsigned int sizeT = 1, unsigned int sizeC = 1);

    virtual ~SImageInt();

public:
    int getMin();
    int getMax();
    float getMean();
    int getSum();
    SImageInt* getSlice(unsigned int z);
    SImageInt* getFrame(unsigned int t);
    SImageInt* getChannel(unsigned int c);
    int getPixel(unsigned int x, unsigned int y);
    int getPixel(unsigned int x, unsigned int y, unsigned int z);
    int getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t);
    int getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c);
    int getPixelXYC(unsigned int x, unsigned int y, unsigned int c);
    int getPixelXYZC(unsigned int x, unsigned int y, unsigned int z, unsigned int c);

    void setPixel(int value, unsigned int x, unsigned int y, unsigned int z = 0, unsigned int t = 0, unsigned int c = 0);

public:
    virtual void allocate();
    int* getBuffer();

public:
    virtual void initAtZero();

protected:
    int* m_buffer;

};

