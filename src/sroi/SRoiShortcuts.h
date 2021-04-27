/// \file SRoiShortcuts.h
/// \brief SRoiShortcuts class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>

#include "SRoiReader.h"
#include "SRoiDrawer.h"

/// \namespace SImg
/// \brief Shortut function to call modules functionalities
namespace SImg{

    std::vector<SRoi*> readRois(std::string file);
    SImage* drawRois(SImage* image, std::vector<SRoi*> rois);
}
