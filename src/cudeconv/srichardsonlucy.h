/// \file srichardsonlucy.h
/// \brief srichardsonlucy definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef srichardsonlucy_cuda_H
#define srichardsonlucy_cuda_H

namespace SImg{

void cuda_richardsonlucy_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter);
void cuda_richardson_lucy_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int niter);

}
#endif /* !srichardsonlucy_cuda_H */