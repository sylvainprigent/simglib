/// \file SRoi.h
/// \brief SRoi class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sroiExport.h"

#include "sdata/SObject.h"
#include "SPoint.h"

/// \class SRoi
/// \brief Interface for ROIs
class SROI_EXPORT SRoi{

public:
    SRoi();
    virtual ~SRoi();

public:
    std::string getType();
    virtual std::vector<SPoint*> getContour(int thikness = 1) = 0;

public:
    void setProperties(SObject* properties);
    SObject* getProperties();

protected:
    std::string m_type;
    SObject* m_properties;
};
