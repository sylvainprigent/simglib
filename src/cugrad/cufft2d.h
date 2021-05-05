/// \file cufft2d.h
/// \brief cufft2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

namespace SImg
{
    /// \brief Calculate the fft2D and back to original space for cudat testing
    /// \param[in] image Buffer of the input image
    /// \param[in] sx Number of rows in the input image
    /// \param[in] sy Number of columns in the input image
    /// \param[in] output Buffer of the output image. 
    void cufft2d(float *image, unsigned int sx, unsigned int sy, float *output);
}