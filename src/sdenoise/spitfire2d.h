/// \file spitfire2d.h
/// \brief spitfire2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#pragma once

#include <score/SObservable.h>

namespace SImg{

void spitfire2d_sv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable);
void spitfire2d_hv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable);

}
