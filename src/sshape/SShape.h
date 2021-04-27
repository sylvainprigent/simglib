/// \file SShape.h
/// \brief SShape class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#pragma once

#include <string>
#include <vector>

#include "score/SColor.h"
#include "simage/SImage.h"

#include "sshapeExport.h"

/// \class SShape
/// \brief Interface a shape
/// A shape is a geometric shape (or object) that can be drawn on an image (SImage)
class SSHAPE_EXPORT SShape{

public:
    /// \fn SShape();
    /// \brief Constructor
    SShape();

    /// \fn ~SShape();
    /// \brief Destructor
    virtual ~SShape();

public:
    void setColor(SColor* color);

public:
    // --------------------- Virtuals function -----------------
    /// \fn void draw(SImage* image);
    /// \brief Draw the shape on an image
    /// \param[in] image Pointer to the image where to draw the shape
    virtual void draw(SImage* image) = 0;

protected:
    SColor* m_color;     

};
