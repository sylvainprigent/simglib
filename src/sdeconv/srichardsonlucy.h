/// \file sl_richardson_lucy.h
/// \brief sl_richardson_lucy definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sl_richardson_lucy_H
#define sl_richardson_lucy_H

namespace SImg{

void richardsonlucy_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter);
void richardsonlucy_tv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter, float lambda = 0);

void richardson_lucy_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int niter);

}
#endif /* !sl_richardson_lucy_H */