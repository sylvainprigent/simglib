/// \file wiener.h
/// \brief wiener definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#ifndef sl_wiener_cuda_H
#define sl_wiener_cuda_H

namespace SImg{

void cuda_wiener_deconv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, const float& lambda, const int& connectivity = 4);
void cuda_wiener_deconv_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const float& lambda, const int& connectivity = 4);

}

#endif /* !sl_wiener_cuda_H */