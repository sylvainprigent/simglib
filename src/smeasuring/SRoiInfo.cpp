/// \file SRoiInfo.cpp
/// \brief SRoiInfo class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

#include "SRoiInfo.h"
/*
#include "sfiltering/SGradient.h"
#include "smanipulate/SCrop.h"
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

            if ( x0 < 0
                 || y0 < 0
                 || z0 < 0
                 || x1 >= unsigned(m_input->getSizeX())
                 || y1 >= unsigned(m_input->getSizeY())
                 || z1 >= unsigned(m_input->getSizeZ())
                 ){
                std::cout << "SRoiInfo ERROR: The ROI is out of the image" << std::endl;
            }
            else{

                SCrop crop;
                crop.setInput(m_input);
                crop.setRangeX(x0, x1);
                crop.setRangeY(y0, y1);
                crop.setRangeZ(z0, z1);
                crop.run();
                SImage* cropImg = crop.getOutput();

                SGradient gradient;
                gradient.setInput(cropImg);
                gradient.run();
                tvNorm = gradient.getNormL1();

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
*/