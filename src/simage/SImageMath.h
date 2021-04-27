/// \file SImageMath.h
/// \brief SImageMath class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <string>
#include "simageExport.h"
#include "SImageUInt.h"
#include "SImageInt.h"
#include "SImageFloat.h"

/// \class SImageMath
/// \brief basic math between image and numbers
class SIMAGE_EXPORT SImageMath{

public:
    static void add(SImageFloat* image, float value);
    static void subtract(SImageFloat* image, float value);
    static void multiply(SImageFloat* image, float value);
    static void divide(SImageFloat* image, float value);

};

