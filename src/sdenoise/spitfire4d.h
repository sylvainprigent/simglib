/// \file spitfire4.h
/// \brief spitfire4d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <score/SObservable.h>

namespace SImg{

void spitfire4d_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat);
void spitfire4d_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat);

void spitfire4d_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat, bool verbose, SObservable* observable);
void spitfire4d_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat, bool verbose, SObservable* observable);

}