/// \file spitfire3d.h
/// \brief spitfire3d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <score/SObservable.h>

namespace SImg{

void cuda_spitfire3d_denoise_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& delta, bool verbose, SObservable* observable);
void cuda_spitfire3d_denoise_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& delta, bool verbose, SObservable* observable);
void cuda_spitfire3d_denoise(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float& delta, const std::string &method, bool verbose, SObservable *observable);

}