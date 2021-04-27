/// \file slFile.h
/// \brief slFile class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "scoreExport.h"

/// \class SFile
/// \brief Static functions for files operations
class SCORE_EXPORT SFile{

public:
    static bool exists(std::string file);

};
