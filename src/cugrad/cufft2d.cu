/// \file cufft2d.cpp
/// \brief cufft2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>
#include <cufft.h>

#include <smanipulate>

__global__
void convolve_fft(unsigned int Nfft, float scale, cufftComplex* image1FFT, cufftComplex* image2FFT, cufftComplex* outputFFT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nfft)
    {
        outputFFT[i].x = (image1FFT[i].x * image2FFT[i].x - image1FFT[i].y * image2FFT[i].y)*scale;
        outputFFT[i].y = (image1FFT[i].y * image2FFT[i].x + image1FFT[i].x * image2FFT[i].y)*scale;
    }
}

namespace SImg
{

    void convolve2d(float *image1, float* image2, unsigned int sx, unsigned int sy, float*output)
    {

        // shift image 2
        float *image2Shift = new float[sx * sy];
        shift2D(image2, image2Shift, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));

        int N = sx * sy;
        int Nfft = sx*(sy/2+1);
        float scale =  1.0 / float(sx*sy);
        float* d_image1;
        float* d_image2;
        float* d_output;
        cufftComplex *d_image1FFT;
        cufftComplex *d_image2FFT;
        cufftComplex *d_outputFFT;

        // alloc
        cudaMalloc(&d_image1, N * sizeof(float)); 
        cudaMalloc(&d_image2, N * sizeof(float)); 
        cudaMalloc(&d_output, N * sizeof(float)); 
        cudaMalloc((void**)&d_image1FFT, sizeof(cufftComplex)*Nfft); 
        cudaMalloc((void**)&d_image2FFT, sizeof(cufftComplex)*Nfft); 
        cudaMalloc((void**)&d_outputFFT, sizeof(cufftComplex)*Nfft); 

        // copy inputs
        cudaMemcpy(d_image1, image1, N*sizeof(float), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_image2, image2Shift, N*sizeof(float), cudaMemcpyHostToDevice); 

        // convolution
        cufftHandle planR2C;
        cufftPlan2d(&planR2C, sx, sy, CUFFT_R2C);

        cufftExecR2C(planR2C, (cufftReal*)d_image1, (cufftComplex*)d_image1FFT);
        cufftExecR2C(planR2C, (cufftReal*)d_image2, (cufftComplex*)d_image2FFT);

        int blockSize1d = 256;
        int numBlocks1d = (Nfft + blockSize1d - 1) / blockSize1d;
        convolve_fft<<<numBlocks1d,blockSize1d>>>(Nfft, scale, d_image1FFT, d_image2FFT, d_outputFFT);

        cufftHandle planC2R;
        cufftPlan2d(&planC2R, sx, sy, CUFFT_C2R);
        cufftExecC2R(planC2R, (cufftComplex*)d_outputFFT, (cufftReal*)d_output);
        cudaMemcpy(output, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_image1);
        cudaFree(d_image2);
        cudaFree(d_output);
        cudaFree(d_image1FFT);
        cudaFree(d_image2FFT);
        cudaFree(d_outputFFT);

    }

    void cufft2d(float *image, unsigned int sx, unsigned int sy, float *output)
    {
        int bs = sx * sy;
        float *cuimage;
        float *cuoutput;
        cufftComplex *cuimageFFT;

        // allocate input to GPU
        std::cout << "copy image to cuda" << std::endl;
        cudaMalloc(&cuimage, bs * sizeof(float));   
        cudaMalloc(&cuoutput, bs * sizeof(float)); 
        cudaMemcpy(cuimage, image, bs*sizeof(float), cudaMemcpyHostToDevice); 
        cudaMalloc((void**)&cuimageFFT, sizeof(cufftComplex)*sx*(sy/2+1));
        
        cufftHandle pF, pI;
        cufftPlan2d(&pF, sx, sy, CUFFT_R2C);
        cufftPlan2d(&pI, sx, sy, CUFFT_C2R);

        cufftExecR2C(pF, (cufftReal*)cuimage, (cufftComplex*)cuimageFFT);
        cufftExecC2R(pI, (cufftComplex*)cuimageFFT, (cufftReal*)cuoutput);

        cufftDestroy(pF);
        cufftDestroy(pI);

        // copy output to cpu
        std::cout << "copy memory to output" << std::endl;
        cudaMemcpy(output, cuoutput, bs*sizeof(float), cudaMemcpyDeviceToHost);
    }
}