/// \file SImageInt.cpp
/// \brief SImageInt class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "score/SException.h"
#include "SImageInt.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

SImageInt::SImageInt() : SImage(){
    m_sx = 0;
    m_sy = 0;
    m_sz = 0;
    m_st = 0;
    m_sc = 0;
    m_precision = 16;
}

SImageInt::SImageInt(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, unsigned int sizeT, unsigned int sizeC) : SImage(){

    m_sx = sizeX;
    m_sy = sizeY;
    m_sz = sizeZ;
    m_st = sizeT;
    m_sc = sizeC;
    m_precision = 16;

    m_buffer = new int[m_sx*m_sy*m_sz*m_st*m_sc];
}

SImageInt::SImageInt(int *buffer, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, unsigned int sizeT, unsigned int sizeC){
    m_sx = sizeX;
    m_sy = sizeY;
    m_sz = sizeZ;
    m_st = sizeT;
    m_sc = sizeC;
    m_precision = 16;

    m_buffer = buffer;
}

SImageInt::~SImageInt(){
    delete m_buffer;
}

int SImageInt::getMin(){
    int mVal = m_buffer[0];
    int val;
    for (unsigned int i = 1 ; i < this->getBufferSize() ; i++){
        val = m_buffer[i];
        if (val < mVal){
            mVal = val;
        }
    }
    return mVal;
}

int SImageInt::getMax(){
    int mVal = m_buffer[0];
    int val;
    for (unsigned int i = 1 ; i < this->getBufferSize() ; i++){
        val = m_buffer[i];
        if (val > mVal){
            mVal = val;
        }
    }
    return mVal;
}

float SImageInt::getMean(){
    float val = 0.0;
    unsigned long bs = this->getBufferSize();
    for (unsigned int i = 0 ; i < bs ; i++){
        val += m_buffer[i];
    }
    return val/float(bs);
}

int SImageInt::getSum(){
    float val = 0.0;
    unsigned long bs = this->getBufferSize();
    for (unsigned int i = 0 ; i < bs ; i++){
        val += m_buffer[i];
    }
    return val;
}


SImageInt* SImageInt::getSlice(unsigned int z){

    int *slice = new int[m_sx*m_sy*m_st*m_sc];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int t = 0 ; t < m_st ; t++){
                for (unsigned int c = 0 ; c < m_sc ; c++){
                    slice[ c + m_sc*(t + m_st*(0 + m_sz*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageInt(slice, m_sx, m_sy, 1, m_st, m_sc);
}

SImageInt* SImageInt::getFrame(unsigned int t){

    int *frame = new int[m_sx*m_sy*m_sz*m_sc];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int z = 0 ; z < m_sz ; z++){
                for (unsigned int c = 0 ; c < m_sc ; c++){
                    frame[ c + m_sc*(0 + m_st*(z + m_sz*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageInt(frame, m_sx, m_sy, m_sz, 1, m_sc);
}

SImageInt* SImageInt::getChannel(unsigned int c){

    int *channel = new int[m_sx*m_sy*m_sz*m_st];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int z = 0 ; z < m_sz ; z++){
                for (unsigned int t = 0 ; t < m_st ; t++){
                    channel[ 0 + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageInt(channel, m_sx, m_sy, m_sz, m_st, 1);
}

int SImageInt::getPixel(unsigned int x, unsigned int y){
    return m_buffer[y + m_sy*x];
}

int SImageInt::getPixel(unsigned int x, unsigned int y, unsigned int z){
    return m_buffer[ z + m_sz*(y + m_sy*x)];
}

int SImageInt::getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t){
    return m_buffer[ t + m_st*(z + m_sz*(y + m_sy*x))];
}

int SImageInt::getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c){
    return m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
}

int SImageInt::getPixelXYC(unsigned int x, unsigned int y, unsigned int c){
    return m_buffer[ c + m_sc*(m_st*(m_sz*(y + m_sy*x)))];
}

int SImageInt::getPixelXYZC(unsigned int x, unsigned int y, unsigned int z, unsigned int c){
    return m_buffer[ c + m_sc*(m_st*(z + m_sz*(y + m_sy*x)))];
}

void SImageInt::setPixel(int value, unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c){
    m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))] = value;
}

void SImageInt::allocate(){
    m_buffer = new int[m_sx*m_sy*m_sz*m_st*m_sc];
}

int* SImageInt::getBuffer(){
    return m_buffer;
}

void SImageInt::initAtZero(){
#pragma omp parallel for
    for (unsigned int i = 0 ; i < unsigned(m_sx*m_sy*m_sz*m_st*m_sc) ; i++){
        m_buffer[i] = 0;
    }
}
