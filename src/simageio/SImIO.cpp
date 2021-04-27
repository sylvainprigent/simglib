/// \file SImIO.cpp
/// \brief SImIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <fstream>

#include "SImIO.h"

#include "SImageIOException.h"
#include "simage/SImage.h"
#include "simage/SImageUInt.h"
#include "simage/SImageInt.h"
#include "simage/SImageFloat.h"

SImIO::SImIO(){
    m_readerName = "SImIO";
}

SImage* SImIO::read(std::string file, char precision){

    if ( precision == 0){
        return this->readDefault(file);
    }
    if ( precision == 8 ){
        return this->readUInt(file);
    }
    else if ( precision == 16 ){
        return this->readInt(file);
    }
    else if (precision == 32){
        return this->readFloat(file);
    }
    else{
        throw SImageIOException(std::string("SImageReader: Cannot read image with precision " + std::to_string(precision)).c_str());
    }
}

SImage* SImIO::readUInt(std::string file){
    // open file
    std::ifstream infile;
    infile.open(file.c_str(),std::ios::binary|std::ios::in);

    unsigned int sx, sy, sz, st, sc;
    float rx, ry, rz, rt;
    std::string unit;
    char precision;

    // read header
    infile.read(reinterpret_cast<char *>(&sx),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sy),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sz),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&st),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sc),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&rx),sizeof(float));
    infile.read(reinterpret_cast<char *>(&ry),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rz),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rt),sizeof(float));
    unit.resize(16);
    for (unsigned int i = 0 ; i < 16 ; i++){
        infile.read(reinterpret_cast<char *>(&unit[i]),sizeof(char));
    }
    infile.read(reinterpret_cast<char *>(&precision),sizeof(char));

    unsigned int *buffer = new unsigned int[sx*sy*sz*st*sc];

    // load values to buffer
    if ( precision == 8){
        unsigned int value;
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(unsigned int));
            buffer[i] = value;
        }
    }
    else if (precision == 16){
        infile.close();
        throw SImageIOException("slImage::read cannot read precision 16 to 8");
    }
    else if (precision == 32){
        infile.close();
        throw SImageIOException("slImage::read cannot read precision 32 to 8");
    }
    else{
        infile.close();
        throw SImageIOException("slImage::read precision unknown");
    }
    infile.close();

    SImageUInt* imageUInt = new SImageUInt(buffer, sx, sy, sz, st, sc);
    imageUInt->setRes(rx, ry, rz, rt);
    imageUInt->setUnit(unit);
    return imageUInt;
}

SImage* SImIO::readInt(std::string file){
    // open file
    std::ifstream infile;
    infile.open(file.c_str(),std::ios::binary|std::ios::in);

    unsigned int sx, sy, sz, st, sc;
    float rx, ry, rz, rt;
    std::string unit;
    char precision;

    // read header
    infile.read(reinterpret_cast<char *>(&sx),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sy),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sz),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&st),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sc),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&rx),sizeof(float));
    infile.read(reinterpret_cast<char *>(&ry),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rz),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rt),sizeof(float));
    unit.resize(16);
    for (unsigned int i = 0 ; i < 16 ; i++){
        infile.read(reinterpret_cast<char *>(&unit[i]),sizeof(char));
    }
    infile.read(reinterpret_cast<char *>(&precision),sizeof(char));

    int *buffer = new int[sx*sy*sz*st*sc];

    // load values to buffer
    if ( precision == 8){
        unsigned int value;
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(unsigned int));
            buffer[i] = int(value);
        }
    }
    else if (precision == 16){
        unsigned int value;
        for (unsigned long i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),2*sizeof(unsigned int));
            buffer[i] = value;
        }
    }
    else if (precision == 32){
        infile.close();
        throw SImageIOException("slImage::read cannot read precision 32 to 16");
    }
    else{
        infile.close();
        throw SImageIOException("slImage::read precision unknown");
    }
    infile.close();

    SImageInt* imageInt = new SImageInt(buffer, sx, sy, sz, st, sc);
    imageInt->setRes(rx, ry, rz, rt);
    imageInt->setUnit(unit);
    return imageInt;
}

SImage* SImIO::readFloat(std::string file){

    // open file
    std::ifstream infile;
    infile.open(file.c_str(),std::ios::binary|std::ios::in);

    unsigned int sx, sy, sz, st, sc;
    float rx, ry, rz, rt;
    std::string unit;
    char precision;

    // read header
    infile.read(reinterpret_cast<char *>(&sx),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sy),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sz),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&st),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sc),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&rx),sizeof(float));
    infile.read(reinterpret_cast<char *>(&ry),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rz),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rt),sizeof(float));
    unit.resize(16);
    for (unsigned int i = 0 ; i < 16 ; i++){
        infile.read(reinterpret_cast<char *>(&unit[i]),sizeof(char));
    }
    infile.read(reinterpret_cast<char *>(&precision),sizeof(char));

    float *buffer = new float[sx*sy*sz*st*sc];

    // load values to buffer
    if ( precision == 8){
        char value;
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(char));
            buffer[i] = float(value);
        }
    }
    else if (precision == 16){
        unsigned int value;
        for (unsigned long i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),2*sizeof(unsigned int));
            buffer[i] = float(value);
        }
    }
    else if (precision == 32){
        float value = float(0.0);
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(float));
            buffer[i] = value;
        }
    }
    else{
        infile.close();
        throw SImageIOException("slImage::read precision unknown");
    }
    infile.close();

    SImageFloat* imageFloat = new SImageFloat(buffer, sx, sy, sz, st, sc);
    imageFloat->setRes(rx, ry, rz, rt);
    imageFloat->setUnit(unit);
    return imageFloat;
}

SImage* SImIO::readDefault(std::string file){

    // open file
    std::ifstream infile;
    infile.open(file.c_str(),std::ios::binary|std::ios::in);

    unsigned int sx, sy, sz, st, sc;
    float rx, ry, rz, rt;
    std::string unit;
    char precision;

    // read header
    infile.read(reinterpret_cast<char *>(&sx),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sy),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sz),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&st),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&sc),sizeof(unsigned int));
    infile.read(reinterpret_cast<char *>(&rx),sizeof(float));
    infile.read(reinterpret_cast<char *>(&ry),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rz),sizeof(float));
    infile.read(reinterpret_cast<char *>(&rt),sizeof(float));
    unit.resize(16);
    for (unsigned int i = 0 ; i < 16 ; i++){
        infile.read(reinterpret_cast<char *>(&unit[i]),sizeof(char));
    }
    infile.read(reinterpret_cast<char *>(&precision),sizeof(char));


    // load values to buffer
    if ( precision == 8){
        unsigned int *buffer = new unsigned int[sx*sy*sz*st*sc];
        unsigned int value;
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(unsigned int));
            buffer[i] = (unsigned int)(value);
        }
        infile.close();

        SImageUInt* image = new SImageUInt(buffer, sx, sy, sz, st, sc);
        image->setRes(rx, ry, rz, rt);
        image->setUnit(unit);
        return image;
    }
    else if (precision == 16){
        int *buffer = new int[sx*sy*sz*st*sc];
        int value;
        for (unsigned long i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),2*sizeof(unsigned int));
            buffer[i] = int(value);
        }
        infile.close();

        SImageInt* image = new SImageInt(buffer, sx, sy, sz, st, sc);
        image->setRes(rx, ry, rz, rt);
        image->setUnit(unit);
        return image;
    }
    else if (precision == 32){
        float *buffer = new float[sx*sy*sz*st*sc];
        float value = float(0.0);
        for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
            infile.read(reinterpret_cast<char *>(&value),sizeof(float));
            buffer[i] = float(value);
        }
        infile.close();

        SImageFloat* image = new SImageFloat(buffer, sx, sy, sz, st, sc);
        image->setRes(rx, ry, rz, rt);
        image->setUnit(unit);
        return image;
    }
    else{
        infile.close();
        throw SImageIOException("slImage::read precision unknown");
    }

}


void SImIO::write(SImage* image, std::string file){

    if ( image->getPrecision() == 8 ){
        this->writeUInt(image, file);
    }
    else if ( image->getPrecision() == 16 ){
        this->writeInt(image, file);
    }
    else if (image->getPrecision() == 32){
        this->writeFloat(image, file);
    }
    else{
        throw SImageIOException(std::string("SImageWriter: Cannot write image with precision " + std::to_string(image->getPrecision())).c_str());
    }

}

void SImIO::writeUInt(SImage* image, std::string file){

    SImageUInt* img = dynamic_cast<SImageUInt*>(image);
    unsigned int sx = img->getSizeX();
    unsigned int sy = img->getSizeY();
    unsigned int sz = img->getSizeZ();
    unsigned int st = img->getSizeT();
    unsigned int sc = img->getSizeC();
    float rx = img->getResX();
    float ry = img->getResY();
    float rz = img->getResZ();
    float rt = img->getResT();
    std::string unit = img->getUnit();
    char precision = img->getPrecision();
    unsigned int* buffer = img->getBuffer();

    // write metadata
    std::ofstream outStream(file, std::ios::binary);
    outStream.write(reinterpret_cast<const char*>(&sx), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sy), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sz), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&st), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sc), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&rx), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&ry), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rz), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rt), sizeof(float));
    for (unsigned int i = 0 ; i < 16 ; i++){
        outStream.write(reinterpret_cast<const char*>(&unit[i]), sizeof(char));
    }
    outStream.write(reinterpret_cast<const char*>(&precision), sizeof(char));

    for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
        outStream.write(reinterpret_cast<const char*>(&buffer[i]), sizeof(char));
    }
    outStream.close();

}

void SImIO::writeInt(SImage* image, std::string file){

    SImageInt* img = dynamic_cast<SImageInt*>(image);
    unsigned int sx = img->getSizeX();
    unsigned int sy = img->getSizeY();
    unsigned int sz = img->getSizeZ();
    unsigned int st = img->getSizeT();
    unsigned int sc = img->getSizeC();
    float rx = img->getResX();
    float ry = img->getResY();
    float rz = img->getResZ();
    float rt = img->getResT();
    std::string unit = img->getUnit();
    char precision = img->getPrecision();
    int* buffer = img->getBuffer();

    // write metadata
    std::ofstream outStream(file, std::ios::binary);
    outStream.write(reinterpret_cast<const char*>(&sx), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sy), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sz), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&st), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sc), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&rx), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&ry), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rz), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rt), sizeof(float));
    for (unsigned int i = 0 ; i < 16 ; i++){
        outStream.write(reinterpret_cast<const char*>(&unit[i]), sizeof(char));
    }
    outStream.write(reinterpret_cast<const char*>(&precision), sizeof(char));

    for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
        outStream.write(reinterpret_cast<const char*>(&buffer[i]), sizeof(int));
    }
    outStream.close();
}

void SImIO::writeFloat(SImage* image, std::string file){

    SImageFloat* img = dynamic_cast<SImageFloat*>(image);
    unsigned int sx = img->getSizeX();
    unsigned int sy = img->getSizeY();
    unsigned int sz = img->getSizeZ();
    unsigned int st = img->getSizeT();
    unsigned int sc = img->getSizeC();
    float rx = img->getResX();
    float ry = img->getResY();
    float rz = img->getResZ();
    float rt = img->getResT();
    std::string unit = img->getUnit();
    char precision = img->getPrecision();
    float* buffer = img->getBuffer();

    // write metadata
    std::ofstream outStream(file, std::ios::binary);
    outStream.write(reinterpret_cast<const char*>(&sx), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sy), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sz), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&st), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&sc), sizeof(unsigned int));
    outStream.write(reinterpret_cast<const char*>(&rx), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&ry), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rz), sizeof(float));
    outStream.write(reinterpret_cast<const char*>(&rt), sizeof(float));
    for (unsigned int i = 0 ; i < 16 ; i++){
        outStream.write(reinterpret_cast<const char*>(&unit[i]), sizeof(char));
    }
    outStream.write(reinterpret_cast<const char*>(&precision), sizeof(char));

    for (unsigned int i = 0 ; i < sx*sy*sz*st*sc ; i++){
        outStream.write(reinterpret_cast<const char*>(&buffer[i]), sizeof(float));
    }
    outStream.close();
}
