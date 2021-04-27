/// \file SLine.h
/// \brief SLine class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#pragma once

#include <string>
#include <vector>

#include "SShape.h"

#include "sshapeExport.h"

/// \class SLine
/// \brief Line shape
/// A line can be 2D or 3D
class SSHAPE_EXPORT SLine : public SShape{

public:
    /// \fn SLine();
    /// \brief Constructor
    SLine();

    /// \fn SLine(float x_start, float y_start, float z_start, x_end, float y_end, float z_end);
    /// \brief Constructor for 2D line
    /// \param[in] x_start X position of the line starting point
    /// \param[in] y_start Y position of the line starting point
    /// \param[in] x_end X position of the line ending point
    /// \param[in] y_end Y position of the line ending point
    SLine(float x_start, float y_start, float x_end, float y_end);

    /// \fn SLine(float x_start, float y_start, float z_start, x_end, float y_end, float z_end);
    /// \brief Constructor for 3D line
    /// \param[in] x_start X position of the line starting point
    /// \param[in] y_start Y position of the line starting point
    /// \param[in] z_start Z position of the line starting point
    /// \param[in] x_end X position of the line ending point
    /// \param[in] y_end Y position of the line ending point
    /// \param[in] z_end Z position of the line ending point
    SLine(float x_start, float y_start, float z_start, float x_end, float y_end, float z_end);
    
    /// \fn ~SLine();
    /// \brief Destructor
    virtual ~SLine();

public:
    // --------------------- Virtuals function -----------------
    /// \fn void draw(SImage* image);
    /// \brief Draw the shape on an image
    /// \param[in] image Pointer to the image where to draw the shape
    void draw(SImage* image);
      
private:
    float m_x_start;
    float m_y_start;
    float m_z_start;
    float m_x_end;
    float m_y_end;
    float m_z_end;
};

void calculate2DLineCoordinates( int x1, int y1, int x2, int y2, std::vector<int> &px, std::vector<int> &py);
std::vector<float> linSpace(float d1, float d2, int n);
std::vector<int> linSpaceRound(float d1, float d2, int n);