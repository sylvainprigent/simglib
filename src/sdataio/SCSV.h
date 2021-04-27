/// \file SCSV.h
/// \brief SCSV class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "sdataioExport.h"

#include "sdata/SData.h"

/// \class SCSV
/// \brief Read and write SData to SCSV
class SDATAIO_EXPORT SCSV{

public:
    SCSV();

public:
    void read(std::string file, std::string separator = ",");
    void write(std::string file, std::string separator = ",");

public:
    void set(SData* data);
    SData* get();

protected:
    std::string readFileContent(std::string file);
    std::vector<std::string> parseLine(const std::string& line, char separator);

protected:
    SData* m_data;
};
