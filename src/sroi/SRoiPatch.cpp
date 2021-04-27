/// \file SRoiPatch.cpp
/// \brief SRoiPatch class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "SRoiPatch.h"
#include "SRoiTypes.h"

SRoiPatch::SRoiPatch() : SRoi(){
    m_type = SRoiTypes::Patch;
}

SRoiPatch::SRoiPatch(float center_x, float center_y, float center_z, float radius_x, float radius_y, float radius_z) : SRoi(){
    m_type = SRoiTypes::Patch;
    m_x0 = center_x - radius_x;
    m_y0 = center_y - radius_y;
    m_z0 = center_z - radius_z;
    m_x1 = center_x + radius_x;
    m_y1 = center_y + radius_y;
    m_z1 = center_z + radius_z;
}

std::vector<SPoint*> SRoiPatch::getContour(int thikness){
    std::vector<SPoint*> contour;
    for (unsigned int x = m_x0 ; x <= m_x1 ; x++ ){
        for (int t = -thikness ; t <= thikness ; t++){
            contour.push_back(new SPoint(x, m_y0+t, 0)); // top horizontal line
            contour.push_back(new SPoint(x, m_y1+t, 0)); // bottom horizontal line
        }
    }
    for (unsigned int y = m_y0 ; y <= m_y1 ; y++ ){
        for (int t = -thikness ; t <= thikness ; t++){
            contour.push_back(new SPoint(m_x0+t, y, 0)); // left vertical line
            contour.push_back(new SPoint(m_x1+t, y, 0)); // right vertical line
        }
    }
    /// \todo add z points
    return contour;
}

unsigned int SRoiPatch::x0(){
    return m_x0;
}

unsigned int SRoiPatch::y0(){
    return m_y0;
}

unsigned int SRoiPatch::z0(){
    return m_z0;
}

unsigned int SRoiPatch::x1(){
    return m_x1;
}

unsigned int SRoiPatch::y1(){
    return m_y1;
}

unsigned int SRoiPatch::z1(){
    return m_z1;
}
