/// \file cugrad.cpp
/// \brief cugrad definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include <iostream>

__global__
void grad_module(unsigned int sx, unsigned int sy, float *image, float *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    int i = sy*x+y;
    float dx = image[i] - image[i - 1];
    float dy = image[i] - image[i - sy];
    output[i] = sqrt(dx*dx+dy*dy);
}

__global__ 
void x_grad_left(unsigned int sx, unsigned int sy, float *image, float *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = sx * sy;
    for (int i = index; i < size; i += stride)
    {
        float dx = 0;
        float dy = 0;
        //output[i] = image[i];
        if (i > sy)
        {
            dy = image[i] - image[i - sy];
        }
        if (i > 0)
        {
            dx = image[i] - image[i - 1];
        }
        output[i] = sqrt(dx*dx+dy*dy);
    }
}

namespace SImg
{
    void cugrad(float *image, unsigned int sx, unsigned int sy, float *output)
    {
        int bs = sx * sy;
        float *cuimage;
        float *cuoutput;

        // allocate input to GPU
        std::cout << "copy image to cuda" << std::endl;
        cudaMalloc(&cuimage, bs * sizeof(float));   
        cudaMalloc(&cuoutput, bs * sizeof(float)); 
        cudaMemcpy(cuimage, image, bs*sizeof(float), cudaMemcpyHostToDevice);    

        // calculate 
        std::cout << "run kernel" << std::endl;
        //int blockSize = 256;
        //int numBlocks = (bs + blockSize - 1) / blockSize;
        //x_grad_left<<<numBlocks, blockSize>>>(sx, sy, cuimage, cuoutput);
        dim3 blockSize(32, 32);
        dim3 gridSize = dim3((sx + 32 - 1) / 32, (sy + 32 - 1) / 32);
        grad_module<<<gridSize, blockSize>>>(sx, sy, cuimage, cuoutput);

        // copy output to cpu
        std::cout << "copy memory to output" << std::endl;
        cudaMemcpy(output, cuoutput, bs*sizeof(float), cudaMemcpyDeviceToHost);
    }
}