/// \file SFFTConvolutionFilter.cpp
/// \brief SFFTConvolutionFilter class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <fftw3.h>

#include "SFFTConvolutionFilter.h"
#include "SFFT.h"

#include <iostream>

float* SImg::convolution_2d(float* image1, float* image2, unsigned int sx, unsigned int sy)
{
    std::cout << "convolution_2d start" << std::endl;
    int nfft = sx * (sy / 2 + 1);
    std::cout << "convolution_2d calculate fft2D" << std::endl;
    fftwf_complex* image1FFT = SImg::fft2D(image1, sx, sy);
    fftwf_complex* image2FFT = SImg::fft2D(image2, sx, sy);
    fftwf_complex* outputFFT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * nfft);
    float scale = 1.0 / float(nfft);
    std::cout << "convolution_2d loop" << std::endl;
    for (unsigned int i = 0 ; i < nfft ; i++){
        outputFFT[i][0] = (image1FFT[i][0] * image2FFT[i][0] - image1FFT[i][1] * image2FFT[i][1])*scale;
        outputFFT[i][1] = (image1FFT[i][1] * image2FFT[i][0] + image1FFT[i][0] * image2FFT[i][1])*scale;
    }
    std::cout << "convolution_2d calculate ifft" << std::endl;

    float* out = new float[sx*sy];
    SImg::ifft2D(outputFFT, out, sx, sy);
    return out;
}
