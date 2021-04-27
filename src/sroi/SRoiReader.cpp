/// \file SRoiReader.cpp
/// \brief SRoiReader class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SRoiReader.h"

#include <sdata>
#include <sdataio>
#include "SRoiPatch.h"


SRoiReader::SRoiReader(){

}

void SRoiReader::setFile(std::string filePath){
    m_filePath = filePath;
}

void SRoiReader::run(){

    m_rois.clear();

    SJSON parser;
    parser.read(m_filePath);
    SObject* roisObject = SDataCast::toSObject(parser.get());

    if (roisObject->hasKey("rois")){
        SArray *roisArray = SDataCast::toSArray(roisObject->get("rois"));

        for (unsigned int r = 0 ; r < roisArray->size() ; r++){
            SObject* roiObj = SDataCast::toSObject(roisArray->get(r));
            SRoi* roi = this->parseRoi(roiObj);
            if (roi){
                m_rois.push_back(roi);
            }
        }
    }
}

SRoi* SRoiReader::parseRoi(SObject* object){
    // type
    if (object->hasKey("type")){
        std::string roiType = SDataCast::toString(object->get("type"));
        if (roiType == "patch"){

            // coordinates
            SObject* coordObj = SDataCast::toSObject(object->get("coordinates"));
            float x = SDataCast::toFloat(coordObj->get("x"));
            float y = SDataCast::toFloat(coordObj->get("y"));
            float z = SDataCast::toFloat(coordObj->get("z"));
            int rx = SDataCast::toInt(coordObj->get("rx"));
            int ry = SDataCast::toInt(coordObj->get("ry"));
            int rz = SDataCast::toInt(coordObj->get("rz"));

            SRoiPatch *roi = new SRoiPatch(int(x), int(y), int(z), rx, ry, rz);

            // properties
            SObject* propObj = SDataCast::toSObject(object->get("properties"));
            roi->setProperties(propObj);

            return roi;
        }
    }
    return nullptr;

}

std::vector<SRoi*> SRoiReader::getRois(){
    return m_rois;
}
