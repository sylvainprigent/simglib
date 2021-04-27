/// \file SRoiShortcuts.cpp
/// \brief SRoiShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SRoiShortcuts.h"

namespace SImg {

std::vector<SRoi*> readRois(std::string file){

    SRoiReader reader;
    reader.setFile(file);
    reader.run();
    return reader.getRois();
}

SImage* drawRois(SImage* image, std::vector<SRoi*> rois){
    SRoiDrawer drawer;
    drawer.setInput(image);
    drawer.setRois(rois);
    drawer.run();
    return drawer.getOutput();

}

}
