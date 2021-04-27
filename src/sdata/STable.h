/// \file STable.h
/// \brief STable class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>

#include "sdataExport.h"

#include "SData.h"

/// \class STable
/// \brief define an table container
class SDATA_EXPORT STable : public SData{

public:
    STable();
    STable(unsigned int row, unsigned int col);

    virtual ~STable();

public:
    std::string json(int level = 0);
    std::string csv(std::string separator = "");

public:
    // header
    void setHeader(unsigned int col, std::string value);
    void setHeaders(std::vector<std::string>& value);
    std::string getHeader(unsigned int col);

    // content
    std::string get(unsigned int row, unsigned int col);
    float getFloat(unsigned int row, unsigned int col);
    int getInt(unsigned int row, unsigned int col);
    unsigned int getUnsigned(unsigned int row, unsigned int col);
    void set(unsigned int row, unsigned int col, std::string value);
    void set(unsigned int row, unsigned int col, float value);
    void set(unsigned int row, unsigned int col, int value);
    void set(unsigned int row, unsigned int col, unsigned int value);
    void addRow(std::vector<std::string>& data);

public:
    std::vector< std::vector<std::string> > buffer();
    std::vector< std::string > headers();

public:
    unsigned int getWidth();
    unsigned int getHeight();

protected:
    std::string indent(int level);

protected:
    std::vector< std::string > m_headers;
    std::vector< std::vector<std::string> > m_content;
    unsigned int m_row;
    unsigned int m_col;
};
