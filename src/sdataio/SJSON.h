/// \file SJSON.h
/// \brief SJSON class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include <vector>
#include "sdataioExport.h"

#include "sdata/SData.h"

/// \class SJSON
/// \brief Read and write SData to JSON
class SDATAIO_EXPORT SJSON{

public:
    SJSON();

public:
    void read(std::string file);
    void write(std::string file);

public:
    void set(SData* data);
    SData* get();

protected:
    std::string readFileContent(std::string file);
    SData* parseObject(std::string fileContent, int start, int end);
    SData* parseArray(std::string fileContent, int start, int end);
    int *findObjectLimits(std::string content, int start, char open, char close);
    int findNextChar(const std::string &content, int start, int end, char s );
    std::string getKeyIn(const std::string &content, int start, int stop);
    char getObjectDelimiter(const std::string &content, int start);
    int* getValueIndex(const std::string &content, int start, int stop);
    int* getNumberIndex(const std::string &content, int start, int stop);
    bool contains(std::string str, char c);

protected:
    SData* m_data;
};
