/// \file SPoint.h
/// \brief SPoint class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sroiExport.h"

/// \class SPoint
/// \brief Container for point coordinates
class SROI_EXPORT SPoint{

public:
    SPoint();
    SPoint(unsigned int x, unsigned int y);
    SPoint(unsigned int x, unsigned int y, unsigned int z);
    SPoint(unsigned int x, unsigned int y, unsigned int z, unsigned int t);
    SPoint(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c);

public:
    bool dimension();
    unsigned int getX();
    unsigned int getY();
    unsigned int getZ();
    unsigned int getT();
    unsigned int getC();

public:
    void setX(unsigned int value);
    void setY(unsigned int value);
    void setZ(unsigned int value);
    void setT(unsigned int value);
    void setC(unsigned int value);

protected:
    unsigned int m_dimension;
    unsigned int m_x;
    unsigned int m_y;
    unsigned int m_z;
    unsigned int m_t;
    unsigned int m_c;
};
