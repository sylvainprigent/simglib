/// \file spitfire4d.h
/// \brief spitfire4d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <score/SObservable.h>

namespace SImg{

void cuda_spitfire4d_denoise_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const float& deltaz, const float& deltat, const unsigned int& niter, bool verbose, SObservable* observable);
void cuda_spitfire4d_denoise_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const float& deltaz, const float& deltat, const unsigned int& niter, bool verbose, SObservable* observable);
void cuda_spitfire4d_denoise(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const float& deltaz, const float& deltat, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable);

}