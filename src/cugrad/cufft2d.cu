/// \file cufft2d.cpp
/// \brief cufft2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>
#include <cufft.h>

namespace SImg
{
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

        // copy output to cpu
        std::cout << "copy memory to output" << std::endl;
        cudaMemcpy(output, cuoutput, bs*sizeof(float), cudaMemcpyDeviceToHost);
    }
}