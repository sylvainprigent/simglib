/// \file roiInfo.h
/// \brief roiInfo class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once


#include "sroiExport.h"

#include "SRoi.h"
#include "sdata/SObject.h"

/// \class SRoiPatch
/// \brief Patch ROI: this ROI is a cuboid defined by it center and size in x, y and z directions
class SROI_EXPORT SRoiPatch : public SRoi{

public:
    SRoiPatch();
    SRoiPatch(float center_x, float center_y, float center_z, float radius_x, float radius_y, float radius_z);

public:
    std::vector<SPoint*> getContour(int thikness = 0);
    unsigned int x0();
    unsigned int y0();
    unsigned int z0();
    unsigned int x1();
    unsigned int y1();
    unsigned int z1();

protected:
    unsigned int m_x0;
    unsigned int m_y0;
    unsigned int m_z0;
    unsigned int m_x1;
    unsigned int m_y1;
    unsigned int m_z1;

    SObject* properties;

};
