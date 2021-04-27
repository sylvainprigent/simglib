/// \file SColor.h
/// \brief SColor class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2014

#pragma once

#include <vector>
#include "scoreExport.h"

/// \class SColor
/// \brief class allowing to get a random color
class SCORE_EXPORT SColor{

public:
    SColor();
    SColor(const float& red, const float& green, const float& blue);

public:
    // getters
    float red();
    float green();
    float blue();
    float r();
    float g();
    float b();

public:
    /// \fn SColor* getRandRGB();
    /// \return a random RGB color
    static SColor* getRandRGB();
    /// \fn SColor* hsvToRGB(float h, float s, float v);
    /// \brief Convert an HSV color into an RGB color
    /// \param[in] h H value
    /// \param[in] s S value
    /// \param[in] v V value
    static SColor* hsvToRGB(float h, float s, float v);

private:
    float m_red;
    float m_green;
    float m_blue;    
};