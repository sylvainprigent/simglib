/// \file SRoiDrawer.h
/// \brief SRoiDrawer class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>

#include "sroiExport.h"

#include "simage/SImageProcess.h"
#include "simage/SImageFloat.h"
#include "SRoi.h"


/// \class SPoint
/// \brief Container for point coordinates
class SROI_EXPORT SRoiDrawer : public SImageProcess{

public:
    SRoiDrawer();

public:
    // inputs
    void setRois(std::vector<SRoi*> rois);

public:
    void run();

public:
    SImage* getOutput();

protected:
    void copyInputImage2RGB();
    int *exctractRoiColor(SRoi* roi);

protected:
    std::vector<SRoi*> m_rois;
    SImageFloat* m_output;
};
