/// \file SRoiReader.h
/// \brief SRoiReader class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>

#include "sroiExport.h"

#include "SRoi.h"

/// \class SPoint
/// \brief Container for point coordinates
class SROI_EXPORT SRoiReader{

public:
    SRoiReader();

public:
    void setFile(std::string filePath);

public:
    void run();

public:
    std::vector<SRoi*> getRois();

protected:
    SRoi* parseRoi(SObject* object);

protected:
    std::string m_filePath;
    std::vector<SRoi*> m_rois;

};
