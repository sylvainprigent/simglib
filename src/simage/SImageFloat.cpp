/// \file SImageFloat.cpp
/// \brief SImageFloat class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "score/SException.h"
#include "SImageFloat.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

SImageFloat::SImageFloat() : SImage(){
    m_sx = 0;
    m_sy = 0;
    m_sz = 0;
    m_st = 0;
    m_sc = 0;
    m_precision = 32;
    m_buffer = nullptr;
}

SImageFloat::SImageFloat(unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, unsigned int sizeT, unsigned int sizeC) : SImage(){

    m_sx = sizeX;
    m_sy = sizeY;
    m_sz = sizeZ;
    m_st = sizeT;
    m_sc = sizeC;
    m_precision = 32;

    m_buffer = new float[m_sx*m_sy*m_sz*m_st*m_sc];
}

SImageFloat::SImageFloat(float* buffer, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ, unsigned int sizeT, unsigned int sizeC){
    m_sx = sizeX;
    m_sy = sizeY;
    m_sz = sizeZ;
    m_st = sizeT;
    m_sc = sizeC;
    m_precision = 32;
    //if (m_buffer){
    //    delete[] m_buffer;
    //}
    m_buffer = buffer;
}

SImageFloat::~SImageFloat(){

    delete[] m_buffer;
}

float SImageFloat::getMin(){
    float mVal = m_buffer[0];
    float val;
    for (unsigned int i = 1 ; i < this->getBufferSize() ; i++){
        val = m_buffer[i];
        if (val < mVal){
            mVal = val;
        }
    }
    return mVal;
}

float SImageFloat::getMax(){
    float mVal = m_buffer[0];
    float val;
    for (unsigned int i = 1 ; i < this->getBufferSize() ; i++){
        val = m_buffer[i];
        if (val > mVal){
            mVal = val;
        }
    }
    return mVal;
}

float SImageFloat::getMean(){
    float val = 0.0;
    unsigned long bs = this->getBufferSize();
    for (unsigned int i = 0 ; i < bs ; i++){
        val += m_buffer[i];
    }
    return val/float(bs);
}

float SImageFloat::getSum(){
    float val = 0.0;
    unsigned long bs = this->getBufferSize();
    for (unsigned int i = 0 ; i < bs ; i++){
        val += m_buffer[i];
    }
    return val;
}


SImageFloat* SImageFloat::getSlice(unsigned int z){

    float *slice = new float[m_sx*m_sy*m_st*m_sc];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int t = 0 ; t < m_st ; t++){
                for (unsigned int c = 0 ; c < m_sc ; c++){
                    slice[ c + m_sc*(t + m_st*(0 + 1*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageFloat(slice, m_sx, m_sy, 1, m_st, m_sc);
}

SImageFloat* SImageFloat::getFrame(unsigned int t){

    float *frame = new float[m_sx*m_sy*m_sz*m_sc];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int z = 0 ; z < m_sz ; z++){
                for (unsigned int c = 0 ; c < m_sc ; c++){
                    frame[ c + m_sc*(0 + 1*(z + m_sz*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageFloat(frame, m_sx, m_sy, m_sz, 1, m_sc);
}

SImageFloat* SImageFloat::getChannel(unsigned int c){

    float *channel = new float[m_sx*m_sy*m_sz*m_st];
#pragma omp parallel for
    for (unsigned int x = 0 ; x < m_sx ; x++){
        for (unsigned int y = 0 ; y < m_sy ; y++){
            for (unsigned int z = 0 ; z < m_sz ; z++){
                for (unsigned int t = 0 ; t < m_st ; t++){
                    channel[ 0 + 1*(t + m_st*(z + m_sz*(y + m_sy*x)))] = m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
                }
            }
        }
    }
    return new SImageFloat(channel, m_sx, m_sy, m_sz, m_st, 1);
}

float SImageFloat::getPixel(unsigned int x, unsigned int y){
    return m_buffer[y + m_sy*x];
}

float SImageFloat::getPixel(unsigned int x, unsigned int y, unsigned int z){
    return m_buffer[ z + m_sz*(y + m_sy*x)];
}

float SImageFloat::getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t){
    return m_buffer[ t + m_st*(z + m_sz*(y + m_sy*x))];
}

float SImageFloat::getPixel(unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c){
    return m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))];
}

float SImageFloat::getPixelXYC(unsigned int x, unsigned int y, unsigned int c){
    return m_buffer[ c + m_sc*(m_st*(m_sz*(y + m_sy*x)))];
}

float SImageFloat::getPixelXYZC(unsigned int x, unsigned int y, unsigned int z, unsigned int c){
    return m_buffer[ c + m_sc*(m_st*(z + m_sz*(y + m_sy*x)))];
}

void SImageFloat::setPixel(float value, unsigned int x, unsigned int y, unsigned int z, unsigned int t, unsigned int c){
    m_buffer[ c + m_sc*(t + m_st*(z + m_sz*(y + m_sy*x)))] = value;
}

void SImageFloat::allocate(){
    m_buffer = new float[m_sx*m_sy*m_sz*m_st*m_sc];
}

float* SImageFloat::getBuffer(){
    return m_buffer;
}

void SImageFloat::initAtZero(){
#pragma omp parallel for
    for (int i = 0 ; i < int(m_sx*m_sy*m_sz*m_st*m_sc) ; i++){
        m_buffer[i] = 0;
    }
}
