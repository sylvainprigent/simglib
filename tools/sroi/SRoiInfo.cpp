/// \file SRoiInfo.cpp
/// \brief SRoiInfo class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "sfiltering/SGradient.h"
#include "smanipulate/SCrop.h"
#include "SRoiInfo.h"
#include "sroi/SRoiTypes.h"
#include "sroi/SRoiPatch.h"

SRoiInfo::SRoiInfo(){
    m_table = new STable();
    std::vector<std::string> header;
    header.resize(4);
    header[0] = "Min";
    header[1] = "Max";
    header[2] = "Mean";
    header[3] = "TV";
    m_table->setHeaders(header);
}

void SRoiInfo::setRois(const std::vector<SRoi*>& rois){
    m_rois = rois;
}

STable* SRoiInfo::getOutput(){
    return m_table;
}

void SRoiInfo::run(){

    float tvNorm;
    for (unsigned int r = 0 ; r < m_rois.size() ; r++){
        if ( m_rois[r]->getType() == SRoiTypes::Patch ){
            SRoiPatch* patch = dynamic_cast<SRoiPatch*>(m_rois[r]);
            unsigned int x0 = patch->x0();
            unsigned int y0 = patch->y0();
            unsigned int z0 = patch->z0();
            unsigned int x1 = patch->x1();
            unsigned int y1 = patch->y1();
            unsigned int z1 = patch->z1();
            if (z1 == 0){
                z1 = 1;
            }

            if ( x0 < 0
                 || y0 < 0
                 || z0 < 0
                 || x1 >= unsigned(m_input->getSizeX())
                 || y1 >= unsigned(m_input->getSizeY())
                 || z1 > unsigned(m_input->getSizeZ())
                 ){
                std::cout << "SRoiInfo ERROR: The ROI is out of the image" << std::endl;
            }
            else{
                float* image_buffer =  dynamic_cast<SImageFloat*>(m_input)->getBuffer();
                unsigned int sx = m_input->getSizeX();
                unsigned int sy = m_input->getSizeY();
                unsigned int sz = m_input->getSizeZ();
                unsigned int st = m_input->getSizeT();
                unsigned int sc = m_input->getSizeC();  
                float* crop_buffer;
                SImg::crop(image_buffer, sx, sy, sz, st, sc, crop_buffer, x0, x1, y0, y1, z0, z1, -1, -1, -1, -1);    

                SImageFloat* cropImg = new SImageFloat(crop_buffer, x1-x0, y1-y0, z1-z0, st, sc);
                //std::cout << "crop img size = " << cropImg->getSizeX() << ", " << cropImg->getSizeY() << ", " << cropImg->getSizeZ() << std::endl;

                float tvNorm = 0.0;
                if (sz > 1){
                    tvNorm = SImg::gradient3dL1(crop_buffer, x1-x0, y1-y0, z1-z0, st, sc);
                }
                else{
                    tvNorm = SImg::gradient2dL1(crop_buffer, x1-x0, y1-y0, sz, st, sc);
                }

                std::vector<std::string> dataRow;
                dataRow.resize(4);
                dataRow[0] = std::to_string(static_cast<SImageFloat*>(cropImg)->getMin());
                dataRow[1] = std::to_string(static_cast<SImageFloat*>(cropImg)->getMax());
                dataRow[2] = std::to_string(static_cast<SImageFloat*>(cropImg)->getMean());
                dataRow[3] = std::to_string(tvNorm);
                m_table->addRow(dataRow);
            }

        }
    }
}
