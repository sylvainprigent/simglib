/// \file SRoiDrawer.cpp
/// \brief SRoiDrawer class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SRoiDrawer.h"
#include "sdata/SArray.h"
#include "sdata/SInt.h"

#include <iostream>

SRoiDrawer::SRoiDrawer(){
    m_processName = "SRoiDrawer";
    m_processPrecision = 32;
    m_processZ = true;
    m_processT = false;
    m_processC = false;
}

void SRoiDrawer::setRois(std::vector<SRoi*> rois){
    m_rois = rois;
}

void SRoiDrawer::run(){
    this->copyInputImage2RGB();

    for (unsigned int r = 0 ; r < m_rois.size() ; r++){ 
        SRoi *roi = m_rois[r];
        int* color = this->exctractRoiColor(roi);  
        std::vector<SPoint*> points = roi->getContour(1);
        for (unsigned int p = 0 ; p < points.size() ; p++){
                 
            SPoint* point = points[p];
            m_output->setPixel(color[0], point->getX(),point->getY(), 0, 0, 0);
            m_output->setPixel(color[1], point->getX(),point->getY(), 0, 0, 1);
            m_output->setPixel(color[2], point->getX(),point->getY(), 0, 0, 2);
        }
    }
}

SImage *SRoiDrawer::getOutput(){
    return m_output;
}

void SRoiDrawer::copyInputImage2RGB(){

    SImageFloat* input = this->castInputToFloat();

    m_output = new SImageFloat(input->getSizeX(),input->getSizeY(),3);
    float* outBuffer = m_output->getBuffer(); //new unsigned int[input->getSizeX()*input->getSizeY()*3];
    float *inputBuffer = input->getBuffer();
    unsigned int sz = input->getSizeZ();
    unsigned int sy = input->getSizeY();
    unsigned int sx = input->getSizeX();
    float maxImage = input->getMax();
    float val;
    if (input->getSizeZ() > 1){
        float maxZ = 0;
        for (unsigned int x = 0 ; x < sx ; x++){
            for (unsigned int y = 0 ; y < sy ; y++){
                maxZ = inputBuffer[0 + sz*(y + sy*x)];

                for (unsigned int z = 1 ; z < sz ; z++){
                    val = inputBuffer[z + sz*(y + sy*x)];
                    if ( val > maxZ ){
                        maxZ = val;
                    }
                }
                outBuffer[0 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*(maxZ/maxImage);
                outBuffer[1 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*(maxZ/maxImage);
                outBuffer[2 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*(maxZ/maxImage);
            }
        }
    }
    else{
        for (unsigned int x = 0 ; x < sx ; x++){
            for (unsigned int y = 0 ; y < sy ; y++){
                outBuffer[0 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*( inputBuffer[y + sy*x]/maxImage);
                outBuffer[1 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*( inputBuffer[y + sy*x]/maxImage);
                outBuffer[2 + 3*(0 + 1*(0 + 1*(y + sy*x)))] = 255*( inputBuffer[y + sy*x]/maxImage);
            }
        }
    }
}

int* SRoiDrawer::exctractRoiColor(SRoi* roi){

    int*color = new int[3];
    color[0] = 255;
    color[1] = 0;
    color[2] = 0;
    SObject * properties = roi->getProperties();
    if (properties->hasKey("color")){
        SArray* cArray= dynamic_cast<SArray*>(properties->get("color"));
        if (cArray && cArray->size() == 3){
            color[0] = dynamic_cast<SInt*>(cArray->get(0))->get();
            color[1] = dynamic_cast<SInt*>(cArray->get(1))->get();
            color[2] = dynamic_cast<SInt*>(cArray->get(2))->get();
        }
    }
    return color;
}
