/// \file SImageIO.cpp
/// \brief SImageIO class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "STiffIO.h"
#include "SImageIOException.h"
#include "simage/SImage.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

bool str_contains(const std::string &str, const std::string &substr ){
    std::size_t found = str.find(substr);
    if (found!=std::string::npos){
        return true;
    }
    return false;
}

std::vector<std::string> str_split(const std::string& str, const std::string& delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();
        std::string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

float string2float(std::string str){
    double temp = ::atof(str.c_str());
    return float(temp);
}

std::string float2string(float value){
    std::ostringstream convert;
    convert << value;
    return convert.str();
}

std::string int2string(int value){
    std::ostringstream convert;
    convert << value;
    return convert.str();
}

STiffIO::STiffIO(){
    m_readerName = "STiffIO";
}

SImage* STiffIO::read(std::string file, char precision){
    if (precision == 8){
        return STiffIO::read_uint(file);
    } 
    if (precision == 16){
        return STiffIO::read_int(file);
    }
    else{
        return STiffIO::read_float(file);
    }
} 

SImageUInt* STiffIO::read_uint(std::string file){

    std::ifstream infile(file);
    if (!infile.good()){
        throw SImageIOException("ERROR: (STiffIO::read) File does not exists");
    }

    SImageUInt *image = new SImageUInt();

    TIFF *tif = TIFFOpen(file.c_str(), "r");

    // calculate the number of 2D plan
    unsigned int nb_images = 0;
    do ++nb_images; while (TIFFReadDirectory(tif));
    image->setSizeZ(nb_images);
    image->setSizeT(1);
    image->setSizeC(1);

    TIFFSetDirectory(tif, 0);
    uint32 nx,ny;
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);

    // allocate
    image->setSizeX(nx);
    image->setSizeY(ny);
    image->allocate();

    // get metadata from description
    char* s_description = 0;
    TIFFGetField(tif,TIFFTAG_IMAGEDESCRIPTION,&s_description);
    if (s_description){
        this->parseDescription(image, s_description);
    }

    // parse resolution
    float x_resolution = 0;
    TIFFGetField(tif,TIFFTAG_XRESOLUTION,&x_resolution);
    if (x_resolution > 0){
        image->setResX(1/x_resolution);
    }
    float y_resolution = 0;
    TIFFGetField(tif,TIFFTAG_YRESOLUTION,&y_resolution);
    if (y_resolution > 0){
        image->setResY(1/y_resolution);
    }

    // read each 2D image plan
    for (unsigned int i = 0 ; i < nb_images ; i++){

        if (!TIFFSetDirectory(tif,uint16(i))){
            throw SImageIOException("ERROR: (STiffIO::read) bad directory index ");
        }

        // read plan dimensions
        uint16 samplesperpixel = 1; // spectrum
        uint16 bitspersample; // encoding
        uint16 photo; // number of plan ?
        uint16 sampleformat = SAMPLEFORMAT_UINT;
        uint32 nx,ny;
        TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
        TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);
        TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&samplesperpixel);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
        TIFFGetFieldDefaulted(tif,TIFFTAG_BITSPERSAMPLE,&bitspersample);
        TIFFGetField(tif,TIFFTAG_PHOTOMETRIC,&photo);

        // read the data
        unsigned int* imageBuffer = image->getBuffer();
        if (TIFFIsTiled(tif)){
            throw SImageIOException("ERROR: (STiffIO::read) cannot read Tiled image yet");
        }
        else{
            tmsize_t nxScan = TIFFScanlineSize(tif);

            if (sampleformat == SAMPLEFORMAT_UINT){
                //std::cout << "sample format uint = " << sampleformat << std::endl;
                tdata_t buf = _TIFFmalloc(nxScan);
                uint8 s;
                uint8* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<uint8*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = data[row];
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else if (sampleformat == SAMPLEFORMAT_INT){
                throw SImageIOException("ERROR: (STiffIO::read) image is INT cannot be read as CHAR");
            }
            else if (sampleformat == SAMPLEFORMAT_IEEEFP){
                throw SImageIOException("ERROR: (STiffIO::read) image is FLOAT cannot be read as CHAR");
            }
            else{
                throw SImageIOException("ERROR: (STiffIO::read) unsupported bit depth");
            }
        }
    }
    TIFFClose(tif);
    return image;

}

SImageInt* STiffIO::read_int(std::string file){

    std::ifstream infile(file);
    if (!infile.good()){
        throw SImageIOException("ERROR: (STiffIO::read) File does not exists");
    }

    SImageInt *image = new SImageInt();

    TIFF *tif = TIFFOpen(file.c_str(), "r");

    // calculate the number of 2D plan
    unsigned int nb_images = 0;
    do ++nb_images; while (TIFFReadDirectory(tif));
    image->setSizeZ(nb_images);
    image->setSizeT(1);
    image->setSizeC(1);

    TIFFSetDirectory(tif, 0);
    uint32 nx,ny;
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);

    // allocate
    image->setSizeX(nx);
    image->setSizeY(ny);
    image->allocate();

    // get metadata from description
    char* s_description = 0;
    TIFFGetField(tif,TIFFTAG_IMAGEDESCRIPTION,&s_description);
    if (s_description){
        this->parseDescription(image, s_description);
    }

    // parse resolution
    float x_resolution = 0;
    TIFFGetField(tif,TIFFTAG_XRESOLUTION,&x_resolution);
    if (x_resolution > 0){
        image->setResX(1/x_resolution);
    }
    float y_resolution = 0;
    TIFFGetField(tif,TIFFTAG_YRESOLUTION,&y_resolution);
    if (y_resolution > 0){
        image->setResY(1/y_resolution);
    }

    // read each 2D image plan
    for (unsigned int i = 0 ; i < nb_images ; i++){

        if (!TIFFSetDirectory(tif,uint16(i))){
            throw SImageIOException("ERROR: (STiffIO::read) bad directory index ");
        }

        // read plan dimensions
        uint16 samplesperpixel = 1; // spectrum
        uint16 bitspersample; // encoding
        uint16 photo; // number of plan ?
        uint16 sampleformat = SAMPLEFORMAT_UINT;
        uint32 nx,ny;
        TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
        TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);
        TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&samplesperpixel);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
        TIFFGetFieldDefaulted(tif,TIFFTAG_BITSPERSAMPLE,&bitspersample);
        TIFFGetField(tif,TIFFTAG_PHOTOMETRIC,&photo);

        // read the data
        int* imageBuffer = image->getBuffer();
        if (TIFFIsTiled(tif)){
            throw SImageIOException("ERROR: (STiffIO::read) cannot read Tiled image yet");
        }
        else{
            tmsize_t nxScan = TIFFScanlineSize(tif);

            if (sampleformat == SAMPLEFORMAT_UINT){
                tdata_t buf = _TIFFmalloc(nxScan);
                uint8 s;
                uint8* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<uint8*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = int(data[row]);
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else if (sampleformat == SAMPLEFORMAT_INT){
                tdata_t buf = _TIFFmalloc(nxScan);
                uint8 s;
                uint8* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<uint8*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = int(data[row]);
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else if (sampleformat == SAMPLEFORMAT_IEEEFP){
                throw SImageIOException("ERROR: (STiffIO::read) image is FLOAT cannot be read as INT");
            }
            else{
                throw SImageIOException("ERROR: (STiffIO::read) unsupported bit depth");
            }
        }
    }
    TIFFClose(tif);
    return image;

}


SImageFloat* STiffIO::read_float(std::string file){

    std::ifstream infile(file);
    if (!infile.good()){
        throw SImageIOException("ERROR: (STiffIO::read) File does not exists");
    }

    SImageFloat *image = new SImageFloat();

    TIFF *tif = TIFFOpen(file.c_str(), "r");

    // calculate the number of 2D plan
    unsigned int nb_images = 0;
    do ++nb_images; while (TIFFReadDirectory(tif));
    image->setSizeZ(nb_images);
    image->setSizeT(1);
    image->setSizeC(1);


    TIFFSetDirectory(tif, 0);
    uint32 nx,ny;
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);

    // allocate
    image->setSizeX(nx);
    image->setSizeY(ny);
    image->allocate();

    // get metadata from description
    char* s_description = 0;
    TIFFGetField(tif,TIFFTAG_IMAGEDESCRIPTION,&s_description);
    if (s_description){
        this->parseDescription(image, s_description);
    }

    // parse resolution
    float x_resolution = 0;
    TIFFGetField(tif,TIFFTAG_XRESOLUTION,&x_resolution);
    if (x_resolution > 0){
        image->setResX(1/x_resolution);
    }
    float y_resolution = 0;
    TIFFGetField(tif,TIFFTAG_YRESOLUTION,&y_resolution);
    if (y_resolution > 0){
        image->setResY(1/y_resolution);
    }

    // read each 2D image plan
    for (unsigned int i = 0 ; i < nb_images ; i++){

        if (!TIFFSetDirectory(tif,uint16(i))){
            throw SImageIOException("ERROR: (STiffIO::read) bad directory index ");
        }

        // read plan dimensions
        uint16 samplesperpixel = 1; // spectrum
        uint16 bitspersample; // encoding
        uint16 photo; // number of plan ?
        uint16 sampleformat = 0;// = SAMPLEFORMAT_UINT;
        uint32 nx,ny;
        TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
        TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);
        TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&samplesperpixel);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
        TIFFGetFieldDefaulted(tif,TIFFTAG_BITSPERSAMPLE,&bitspersample);
        TIFFGetField(tif,TIFFTAG_PHOTOMETRIC,&photo);

        if (bitspersample == 8)
        {
            sampleformat = SAMPLEFORMAT_UINT;
        }
        else if(bitspersample == 16)
        {
            sampleformat = SAMPLEFORMAT_INT;
        }
        else if(bitspersample == 32)
        {
            sampleformat = SAMPLEFORMAT_IEEEFP;
        }

        //std::cout << "SImageIO read to float tiff image with sampleformat=" << sampleformat << std::endl;
        //std::cout << "SImageIO read to float tiff image with bitspersample=" << bitspersample << std::endl;



        // read the data
        float* imageBuffer = image->getBuffer();
        if (TIFFIsTiled(tif)){
            throw SImageIOException("ERROR: (STiffIO::read) cannot read Tiled image yet");
        }
        else{
            tmsize_t nxScan = TIFFScanlineSize(tif);

            if (sampleformat == SAMPLEFORMAT_UINT){
                //std::cout << "Read format " << SAMPLEFORMAT_UINT << std::endl;
                tdata_t buf = _TIFFmalloc(nxScan);
                uint8 s;
                uint8* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<uint8*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            //image->setPixel(data[col], row, col, i, 0, 0);
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = data[row];
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else if (sampleformat == SAMPLEFORMAT_INT){
                //std::cout << "Read format " << SAMPLEFORMAT_INT << std::endl;
                tdata_t buf = _TIFFmalloc(nxScan);
                uint16 s;
                uint16* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<uint16*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            //image->setPixel(data[col], row, col, i, 0, 0);
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = data[row];
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else if (sampleformat == SAMPLEFORMAT_IEEEFP){
                //std::cout << "Read format " << SAMPLEFORMAT_IEEEFP << std::endl;
                tdata_t buf = _TIFFmalloc(nxScan);
                uint16 s;
                float* data;
                for (s = 0; s < samplesperpixel; s++)
                {
                    for (uint32 col = 0; col < ny; col++)
                    {
                        TIFFReadScanline(tif, buf, col, s);
                        data=static_cast<float*>(buf);
                        for (uint32 row = 0 ; row < nx ; row++){
                            //image->setPixel(data[col], row, col, i, 0, 0);
                            imageBuffer[s + 1*(0 + 1*(i + nb_images*(col + ny*row)))] = data[row];
                        }
                    }
                }
                _TIFFfree(buf);
            }
            else{
                throw SImageIOException("ERROR: (STiffIO::read) unsupported bit depth");
            }
            //std::cout << "nxScan = " << nxScan << std::endl;
        }

    }
    TIFFClose(tif);
    return image;

}

void STiffIO::write(SImage* image, std::string file){

    if (dynamic_cast<SImageFloat*>(image)){
        SImageFloat *imageFloat = dynamic_cast<SImageFloat*>(image);
        STiffIO::write(imageFloat, file);
    } 
    else if(dynamic_cast<SImageUInt*>(image)){
        SImageUInt *imageUInt = dynamic_cast<SImageUInt*>(image);
        STiffIO::write(imageUInt, file);
    }
    else{
        std::cout << "STiffIO::write image type not recognized" << std::endl;
    }
}

void STiffIO::write(SImageFloat* image, std::string file){
   
    SImageFloat *imageFloat = dynamic_cast<SImageFloat*>(image);

    std::cout << "save float image " << imageFloat->getSizeX() << ", " << imageFloat->getSizeY() << ", " << imageFloat->getSizeZ() << std::endl;

    TIFF *tif = TIFFOpen(file.c_str(), "w");
    if (tif) {

        std::string desciption = createDescription(image);

        for (unsigned int z = 0 ; z < image->getSizeZ() ; z++ ){
            TIFFSetDirectory(tif,uint16(z));

            //if (z == 0){
            TIFFSetField(tif,TIFFTAG_IMAGEWIDTH, image->getSizeX());
            TIFFSetField(tif,TIFFTAG_IMAGELENGTH, image->getSizeY());
            TIFFSetField(tif,TIFFTAG_RESOLUTIONUNIT,RESUNIT_NONE);
            TIFFSetField(tif,TIFFTAG_XRESOLUTION, 1.0/double(image->getResX()));
            TIFFSetField(tif,TIFFTAG_YRESOLUTION, 1.0/double(image->getResY()));
            TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL, 1); // gray scale
            TIFFSetField(tif,TIFFTAG_BITSPERSAMPLE, 32) ; // float
            TIFFSetField(tif,TIFFTAG_SAMPLEFORMAT,3); // float
            TIFFSetField(tif,TIFFTAG_SOFTWARE,"SIMG");
            TIFFSetField(tif,TIFFTAG_COMPRESSION,COMPRESSION_NONE);
            TIFFSetField(tif,TIFFTAG_PHOTOMETRIC,PHOTOMETRIC_MINISBLACK);
            TIFFSetField(tif,TIFFTAG_PLANARCONFIG,PLANARCONFIG_CONTIG);
            TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, desciption.c_str());
            //}

            float* buf = new float[image->getSizeX()]; // (float *)_TIFFmalloc(int(image->getSizeX())*sizeof(float));

            float* img_buffer = imageFloat->getBuffer();
            unsigned int sy = imageFloat->getSizeY();
            unsigned int sz = imageFloat->getSizeZ();
            for ( int y = 0 ; y < int(image->getSizeY()) ; y++){
                for( int x = 0 ; x < int(image->getSizeX()) ; x++){
                    buf[x] = img_buffer[sz*(sy*x+y)+z];
                }
                if (TIFFWriteScanline(tif, buf, y) != 1 )
                {
                    throw SImageIOException("ERROR: (STiffIO::write) Unable to write a row.");
                }
            }
            TIFFWriteDirectory(tif);
            delete[] buf;
            //_TIFFfree(buf);
        }

        TIFFClose(tif);
    }
    else {
        throw SImageIOException( ("save tiff : Failed to open file " + file + " for writing.").c_str());
    }
}

void STiffIO::write_rgb(SImageUInt* image, std::string file){

    std::cout << "save as RGB" << std::endl;
    TIFF *tif = TIFFOpen(file.c_str(), "w");
    if (tif) {

        // copy data to raster
        int sx = image->getSizeX();
        int sy = image->getSizeY();
        unsigned int* buffer = image->getBuffer();

        // transpose the image buffer
        int sx_inv = sy;
        int sy_inv = sx;
        unsigned char* buffer_inv = new unsigned char[sx*sy*3];
        for (int x = 0 ; x < sx_inv ; x++){
            for (int y = 0 ; y < sy_inv ; y++){
                buffer_inv[3*(sy_inv*x+y)+0] = (unsigned char)buffer[3*(sy*y+x)+0];
                buffer_inv[3*(sy_inv*x+y)+1] = (unsigned char)buffer[3*(sy*y+x)+1];
                buffer_inv[3*(sy_inv*x+y)+2] = (unsigned char)buffer[3*(sy*y+x)+2];
            }
        }
        
        // tags
        TIFFSetField(tif,TIFFTAG_IMAGEWIDTH, image->getSizeX());
        TIFFSetField(tif,TIFFTAG_IMAGELENGTH, image->getSizeY());
        TIFFSetField(tif,TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(tif,TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tif,TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(tif,TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL, 3);

        if (TIFFWriteEncodedStrip(tif, 0, buffer_inv, image->getSizeX()*image->getSizeY()*3) == 0){
            throw SImageIOException("Could not write the image");
        }
        delete[] buffer_inv;
        TIFFClose(tif);
    }

}

void STiffIO::write(SImageUInt* image, std::string file){
   
    SImageUInt *imageChar = dynamic_cast<SImageUInt*>(image);

    std::cout << "save char image " << imageChar->getSizeX() << ", " << imageChar->getSizeY() << ", " << imageChar->getSizeZ() << ", " << imageChar->getSizeT() << ", " << imageChar->getSizeC() << std::endl;

    if (imageChar->getSizeZ() == 1 && imageChar->getSizeT() == 1 && imageChar->getSizeC() == 3){
        STiffIO::write_rgb(image, file);
        return;
    }

    TIFF *tif = TIFFOpen(file.c_str(), "w");
    if (tif) {

        std::string desciption = createDescription(image);
        unsigned int sy = imageChar->getSizeY();
        unsigned int sz = imageChar->getSizeZ();
        unsigned int st = imageChar->getSizeT();
        unsigned int sc = imageChar->getSizeC();
        for (unsigned int c = 0 ; c < image->getSizeC() ; c++){
            for (unsigned int t = 0 ; t < image->getSizeT() ; t++){
                for (unsigned int z = 0 ; z < image->getSizeZ() ; z++ ){
                    TIFFSetDirectory(tif,uint16(c+sc*(t*st+z)));

                    //std::cout << "write directory: " << c << ", " << t << ", " << z << ", idx = " << c+sc*(t*st+z) << std::endl;
                    //if (z == 0){
                    TIFFSetField(tif,TIFFTAG_IMAGEWIDTH, image->getSizeX());
                    TIFFSetField(tif,TIFFTAG_IMAGELENGTH, image->getSizeY());
                    TIFFSetField(tif,TIFFTAG_RESOLUTIONUNIT,RESUNIT_NONE);
                    TIFFSetField(tif,TIFFTAG_XRESOLUTION, 1.0/double(image->getResX()));
                    TIFFSetField(tif,TIFFTAG_YRESOLUTION, 1.0/double(image->getResY()));
                    TIFFSetField(tif,TIFFTAG_SAMPLESPERPIXEL, 1); // gray scale
                    TIFFSetField(tif,TIFFTAG_BITSPERSAMPLE, 8) ; // float
                    TIFFSetField(tif,TIFFTAG_SAMPLEFORMAT,1); // float
                    TIFFSetField(tif,TIFFTAG_SOFTWARE,"SERPICO");
                    TIFFSetField(tif,TIFFTAG_COMPRESSION,COMPRESSION_NONE);
                    TIFFSetField(tif,TIFFTAG_PHOTOMETRIC,PHOTOMETRIC_MINISBLACK);
                    TIFFSetField(tif,TIFFTAG_PLANARCONFIG,PLANARCONFIG_CONTIG);
                    TIFFSetField(tif, TIFFTAG_IMAGEDESCRIPTION, desciption.c_str());
                    //}

                    unsigned int* buf = new unsigned int[image->getSizeX()]; // (float *)_TIFFmalloc(int(image->getSizeX())*sizeof(float));

                    unsigned int* img_buffer = imageChar->getBuffer();
                    for ( int y = 0 ; y < int(image->getSizeY()) ; y++){
                        for( int x = 0 ; x < int(image->getSizeX()) ; x++){
                            buf[x] = img_buffer[  sc*(st*(sz*(sy*x+y)+z) + t) + c];
                        }
                        if (TIFFWriteScanline(tif, buf, y) != 1 )
                        {
                            throw SImageIOException("ERROR: (STiffIO::write) Unable to write a row.");
                        }
                    }
                    TIFFWriteDirectory(tif);
                    _TIFFfree(buf);
                }
            }
        }
        TIFFClose(tif);
    }
    else {
        throw SImageIOException( ("save tiff : Failed to open file " + file + " for writing.").c_str());
    }
}

void STiffIO::parseDescription(SImage* image, std::string description){

    /// \todo implement parse metadata
    if ( str_contains(description, "ImageJ" ) ){

        // parse ImageJ metadata
        std::vector<std::string> lines = str_split(description, "\n");
        for (std::vector<std::string>::size_type l = 0 ; l < lines.size() ; l++){
            std::string line = lines[l];
            if ( str_contains(line, "unit=") ){
                std::size_t pos = line.find("=");
                std::string unit = line.substr (pos+1);
                image->setUnit(unit);
            }
        }

        image->setResZ(0);
        image->setResT(0);
    }
    else if ( str_contains(description, "SERPICO" ) ){

        // parse SERPICO metadata

        std::vector<std::string> lines = str_split(description, "\n");
        for (std::vector<std::string>::size_type l = 0 ; l < lines.size() ; l++){
            std::string line = lines[l];
            if ( str_contains(line, "unit=") ){
                std::size_t pos = line.find("=");
                std::string unit = line.substr (pos+1);
                image->setUnit(unit);
            }
            else if ( str_contains(line, "Zresolution=") ){
                std::size_t pos = line.find("=");
                float value = string2float(line.substr (pos+1));
                if (value > 0){
                    image->setResZ(value);
                }
            }
            else if ( str_contains(line, "Tresolution=") ){
                std::size_t pos = line.find("=");
                float value = string2float(line.substr (pos+1));
                if (value > 0){
                    image->setResT(value);
                }
            }
        }
    }

}

std::string STiffIO::createDescription(SImage* image){

    std::string description = "SIMG\n";
    description += "unit=" + image->getUnit() + "\n";
    description += "Zresolution=" + float2string( image->getResZ() ) + "\n";
    description += "Tresolution=" + float2string( image->getResT() ) + "\n";
    return description;

}

void STiffIO::printTIFFTags(TIFF *tif){

    // general info
    uint16 samplesperpixel = 1; // spectrum
    uint16 bitspersample; // encoding
    uint16 photo; // number of plan ?
    uint16 sampleformat = SAMPLEFORMAT_UINT;
    uint32 nx,ny;
    const char *const filename = TIFFFileName(tif);
    TIFFGetField(tif,TIFFTAG_IMAGEWIDTH,&nx);
    TIFFGetField(tif,TIFFTAG_IMAGELENGTH,&ny);
    TIFFGetField(tif,TIFFTAG_SAMPLESPERPIXEL,&samplesperpixel);
    TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);
    TIFFGetFieldDefaulted(tif,TIFFTAG_BITSPERSAMPLE,&bitspersample);
    TIFFGetField(tif,TIFFTAG_PHOTOMETRIC,&photo);

    this->notify("filename =  " + std::string(filename));
    this->notify("nx =  " + float2string(nx));
    this->notify("ny =  " + float2string(ny));
    this->notify("samplesperpixel =  " + int2string(samplesperpixel));
    this->notify("sampleformat =  " + int2string(sampleformat) );
    this->notify("bitspersample =  " + int2string(bitspersample));
    this->notify("photo =  " + int2string(photo));


    // Software
    char* s_software = 0;
    TIFFGetField(tif,TIFFTAG_SOFTWARE,&s_software);
    if ( s_software ){
        this->notify("software =  " + std::string(s_software));
    }

    // Description
    char *s_description = 0;
    TIFFGetField(tif,TIFFTAG_IMAGEDESCRIPTION,&s_description);
    if (s_description){
        this->notify("description =  " + std::string(s_description));
    }

    // resolution
    float x_resolution;
    TIFFGetField(tif,TIFFTAG_XRESOLUTION,&x_resolution);
    this->notify("x_resolution =  " + float2string(float(1.0/double(x_resolution))));
    float y_resolution;
    TIFFGetField(tif,TIFFTAG_YRESOLUTION,&y_resolution);
    this->notify("y_resolution =  " + float2string(float(1.0/double(y_resolution))));

}
