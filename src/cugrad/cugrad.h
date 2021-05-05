/// \file cugrad.h
/// \brief cugrad definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

namespace SImg
{
    /// \brief Calculate the left gradient of an image using cuda
    /// \param[in] image Buffer of the input image
    /// \param[in] sx Number of rows in the input image
    /// \param[in] sy Number of columns in the input image
    /// \param[in] output Buffer of the output image. 
    void cugrad(float *image, unsigned int sx, unsigned int sy, float *output);
}
