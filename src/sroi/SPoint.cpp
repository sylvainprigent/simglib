/// \file SPoint.cpp
/// \brief SPoint class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SPoint.h"


SPoint::SPoint(){
    m_dimension = 0;
}

SPoint::SPoint(unsigned int x, unsigned int y){
    m_dimension = 2;
    m_x = x;
    m_y = y;
}

SPoint::SPoint(unsigned int x, unsigned int y, unsigned int z){
    m_dimension = 3;
    m_x = x;
    m_y = y;
    m_z = z;
}

SPoint::SPoint(unsigned int x, unsigned int y, unsigned int z, unsigned int t){
    m_dimension = 4;
    m_x = x;
    m_y = y;
    m_z = z;
    m_t = t;
}

SPoint::SPoint(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c){
    m_dimension = 5;
    m_x = x;
    m_y = y;
    m_z = z;
    m_t = t;
    m_c = c;
}

bool SPoint::dimension(){
    return m_dimension;
}

unsigned int SPoint::getX(){
    return m_x;
}

unsigned int SPoint::getY(){
    return m_y;
}

unsigned int SPoint::getZ(){
    return m_z;
}

unsigned int SPoint::getT(){
    return m_t;
}

unsigned int SPoint::getC(){
    return m_c;
}

void SPoint::setX(unsigned int value){
    m_x = value;
}

void SPoint::setY(unsigned int value){
    m_y = value;
}

void SPoint::setZ(unsigned int value){
    m_z = value;
}

void SPoint::setT(unsigned int value){
    m_t = value;
}
void SPoint::setC(unsigned int value){
    m_c = value;
}
