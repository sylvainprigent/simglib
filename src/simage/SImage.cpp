/// \file SImage.cpp
/// \brief SImage class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SImage.h"

SImage::SImage(){
    m_sx = 0;
    m_sy = 0;
    m_sz = 0;
    m_st = 0;
    m_sc = 0;
    m_rx = 0;
    m_ry = 0;
    m_rz = 0;
    m_rt = 0;
    m_unit = new char[8];
    m_precision = 0;
}

SImage::~SImage(){
}

void SImage::setSizeX(unsigned int value){
    m_sx = value;
}

void SImage::setSizeY(unsigned int value){
    m_sy = value;
}

void SImage::setSizeZ(unsigned int value){
    m_sz = value;
}

void SImage::setSizeT(unsigned int value){
    m_st = value;
}

void SImage::setSizeC(unsigned int value){
    m_sc = value;
}

unsigned int SImage::getSizeX(){
    return m_sx;
}

unsigned int SImage::getSizeY(){
    return m_sy;
}

unsigned int SImage::getSizeZ(){
    return m_sz;
}

unsigned int SImage::getSizeT(){
    return m_st;
}

unsigned int SImage::getSizeC(){
    return m_sc;
}

unsigned long SImage::getBufferSize(){
    return m_sx*m_sy*m_sz*m_st*m_sc;
}

std::string SImage::info(){
    std::string meta = "{ \n";
    meta += "    size: {\n";
    meta += "        \"x\": " + std::to_string(m_sx) + ",\n";
    meta += "        \"y\": " + std::to_string(m_sy)+ ",\n";
    meta += "        \"z\": " + std::to_string(m_sz)+ ",\n";
    meta += "        \"t\": " + std::to_string(m_st)+ ",\n";
    meta += "        \"c\": " + std::to_string(m_sc)+ ",\n";
    meta += "    }, \n";
    meta += "    resolution: {\n";
    meta += "        \"x\": " + std::to_string(m_rx) + ",\n";
    meta += "        \"y\": " + std::to_string(m_ry) + ",\n";
    meta += "        \"z\": " + std::to_string(m_rz) + ",\n";
    meta += "    } ";
    meta += "    \"unit\": " + std::string(m_unit) + ",\n";
    meta += "    \"precision\": " + std::to_string(m_precision) + "\n" ;
    meta += "}\n";

    return meta;
}

float SImage::getResX(){
    return m_rx;
}

float SImage::getResY(){
    return m_ry;
}

float SImage::getResZ(){
    return m_rz;
}

float SImage::getResT(){
    return m_rt;
}

std::string SImage::getUnit(){
    return m_unit;
}

void SImage::setRes(float rx, float ry, float rz, float rt){
    m_rx = rx;
    m_ry = ry;
    m_rz = rz;
    m_rt = rt;
}

void SImage::setResX(float value){
    m_rx = value;
}

void SImage::setResY(float value){
    m_ry = value;
}

void SImage::setResZ(float value){
    m_rz = value;
}

void SImage::setResT(float value){
    m_rt = value;
}

void SImage::setUnit(std::string value){
    m_unit = value;
}

char SImage::getPrecision(){
    return m_precision;
}
