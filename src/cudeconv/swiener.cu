/// \file wiener.cu
/// \brief wiener cuda implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#include "swiener.h"
#include <smanipulate>
#include <cufft.h>

#include <iostream>

__global__
void wiener_fourier(unsigned int n_fft, cufftComplex *fft_psf, cufftComplex *fft_laplacian, cufftComplex *fft_in, cufftComplex *fft_out, float lambda, float scale)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_fft){

        float den = (pow(fft_psf[p].x, 2) + pow(fft_psf[p].y, 2)) + lambda * (pow(fft_laplacian[p].x, 2) + pow(fft_laplacian[p].y, 2));
        fft_psf[p].x = fft_psf[p].x / den;
        fft_psf[p].y = -fft_psf[p].y / den;

        fft_out[p].x = (fft_psf[p].x * fft_in[p].x - fft_psf[p].y * fft_in[p].y) * scale;
        fft_out[p].y = (fft_psf[p].y * fft_in[p].x + fft_psf[p].x * fft_in[p].y) * scale;
    }
}

__global__
void wiener_normalize(unsigned int n, float *buffer_out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n){
        if (buffer_out[p] < 0)
        {
            buffer_out[p] = 0;
        }
    }
}

namespace SImg{

    void cuda_wiener_deconv_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, const float& lambda, const int& connectivity)
    {
        // memory initialization
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);


        cufftComplex *fft_in;
        cufftComplex *fft_out;
        cufftComplex *fft_psf;
        cufftComplex *fft_laplacian;

        float* cu_buffer_in;
        float* cu_psf;
        float* cu_buffer_out;

        cudaMalloc((void**)&fft_in, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_laplacian, sizeof(cufftComplex)*n_fft);

        cudaMalloc ( &cu_buffer_in, n*sizeof(float));
        cudaMalloc ( &cu_psf, n*sizeof(float));
        cudaMalloc ( &cu_buffer_out, n*sizeof(float));

        cudaMemcpy(cu_buffer_in, buffer_in, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_psf, psf, n*sizeof(float), cudaMemcpyHostToDevice);

        int blockSize1d = 256;
        int numBlocks1d = (n + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (n_fft + blockSize1d - 1) / blockSize1d;

        cufftHandle Planfft;
        cufftPlan2d(&Planfft, sx, sy, CUFFT_R2C);
        cufftHandle Planifft;
        cufftPlan2d(&Planifft, sx, sy, CUFFT_C2R);

        // calculate the filter: G
        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_in, (cufftComplex*)fft_in);
        
        // H: fft_psf
        float *buffer_psf_shift = (float *)malloc(sizeof(float) * n);
        shift2D(psf, buffer_psf_shift, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        float* cu_buffer_psf_shift;
        cudaMalloc ( &cu_buffer_psf_shift, n*sizeof(float));
        cudaMemcpy(cu_buffer_psf_shift, buffer_psf_shift, n*sizeof(float), cudaMemcpyHostToDevice);

        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_psf_shift, (cufftComplex*)fft_psf);    
        
        delete buffer_psf_shift;
        cudaFree(cu_buffer_psf_shift);

        // laplacian regularization
        unsigned int xc = sx / 2;
        unsigned int yc = sy / 2;
        float *buffer_laplacian = (float *)malloc(sizeof(float) * n);
#pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            buffer_laplacian[p] = 0;
        }
        if (connectivity == 4)
        {
            buffer_laplacian[sy * xc + yc] = 4;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;
        }
        else if (connectivity == 8)
        {
            buffer_laplacian[sy * xc + yc] = 8;
            buffer_laplacian[sy * (xc - 1) + yc - 1] = -1;
            buffer_laplacian[sy * xc + yc - 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc - 1] = -1;

            buffer_laplacian[sy * (xc - 1) + yc] = -1;
            buffer_laplacian[sy * (xc + 1) + yc] = -1;

            buffer_laplacian[sy * (xc - 1) + yc + 1] = -1;
            buffer_laplacian[sy * xc + yc + 1] = -1;
            buffer_laplacian[sy * (xc + 1) + yc + 1] = -1;
        }

        float* cu_buffer_laplacian;
        cudaMalloc ( &cu_buffer_laplacian, n*sizeof(float));
        cudaMemcpy(cu_buffer_laplacian, buffer_laplacian, n*sizeof(float), cudaMemcpyHostToDevice);
        delete buffer_laplacian;

        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_laplacian, (cufftComplex*)fft_laplacian);   
        cudaFree(cu_buffer_laplacian);

        wiener_fourier<<<numBlocks1dfft,blockSize1d>>>(n_fft, fft_psf, fft_laplacian, fft_in, fft_out, lambda, scale);
        cufftExecC2R(Planifft, (cufftComplex*)fft_out, (cufftReal*)cu_buffer_out); 

        wiener_normalize<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out);

        cudaMemcpy(buffer_out, cu_buffer_out, n*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(cu_buffer_out);
        cudaFree(cu_buffer_in);

        cudaFree(fft_in);
        cudaFree(fft_out);
        cudaFree(fft_psf);
        cudaFree(fft_laplacian);
    }

    void cuda_wiener_deconv_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, const float& lambda, const int& connectivity)
    {
        // memory initialization
        unsigned int n = sx * sy * sz;
        unsigned int n_fft = sx * sy * (sz / 2 + 1);
        float scale = 1.0 / float(n_fft);

        cufftComplex *fft_in;
        cufftComplex *fft_out;
        cufftComplex *fft_psf;
        cufftComplex *fft_laplacian;

        float* cu_buffer_in;
        float* cu_psf;
        float* cu_buffer_out;

        cudaMalloc((void**)&fft_in, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_laplacian, sizeof(cufftComplex)*n_fft);

        cudaMalloc ( &cu_buffer_in, n*sizeof(float));
        cudaMalloc ( &cu_psf, n*sizeof(float));
        cudaMalloc ( &cu_buffer_out, n*sizeof(float));

        cudaMemcpy(cu_buffer_in, buffer_in, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_psf, psf, n*sizeof(float), cudaMemcpyHostToDevice);

        int blockSize1d = 256;
        int numBlocks1d = (n + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (n_fft + blockSize1d - 1) / blockSize1d;

        cufftHandle Planfft;
        cufftPlan3d(&Planfft, sx, sy, sz, CUFFT_R2C);
        cufftHandle Planifft;
        cufftPlan3d(&Planifft, sx, sy, sz, CUFFT_C2R);

        // calculate the filter: G
        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_in, (cufftComplex*)fft_in);
        
        // H: fft_psf
        float *buffer_psf_shift = (float *)malloc(sizeof(float) * n);
        shift3D(psf, buffer_psf_shift, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        float* cu_buffer_psf_shift;
        cudaMalloc ( &cu_buffer_psf_shift, n*sizeof(float));
        cudaMemcpy(cu_buffer_psf_shift, buffer_psf_shift, n*sizeof(float), cudaMemcpyHostToDevice);

        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_psf_shift, (cufftComplex*)fft_psf);    
        
        delete buffer_psf_shift;
        cudaFree(cu_buffer_psf_shift);

        // laplacian regularization
        float *buffer_laplacian = (float *)malloc(sizeof(float) * n);
        #pragma omp parallel for
        for (int p = 0; p < n; p++)
        {
            buffer_laplacian[p] = 0;
        }
        unsigned int xc = sx / 2;
        unsigned int yc = sy / 2;
        unsigned int zc = sz / 2;

        if (connectivity == 4)
        {
            buffer_laplacian[zc + sz*(sy * xc + yc)] = 6;
            buffer_laplacian[zc-1 + sz*(sy * xc + yc)] = -1;
            buffer_laplacian[zc+1 + sz*(sy * xc + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc)] = -1;
        }
        else if (connectivity == 8)
        {
            buffer_laplacian[zc + sz*(sy * xc + yc)] = 26;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc - 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc - 1)] = -1;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc)] = -1;

            buffer_laplacian[zc + sz*(sy * (xc - 1) + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * xc + yc + 1)] = -1;
            buffer_laplacian[zc + sz*(sy * (xc + 1) + yc + 1)] = -1;

            for (int i = -1 ; i <= 1 ; ++i){
                for (int j = -1 ; j <= 1 ; ++j){
                    buffer_laplacian[zc-1 + sz*(sy * (xc + i) + yc + j)] = -1; 
                    buffer_laplacian[zc+1 + sz*(sy * (xc + i) + yc + j)] = -1;            
                }
            }
        }

        float* cu_buffer_laplacian;
        cudaMalloc ( &cu_buffer_laplacian, n*sizeof(float));
        cudaMemcpy(cu_buffer_laplacian, buffer_laplacian, n*sizeof(float), cudaMemcpyHostToDevice);
        delete buffer_laplacian;

        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_laplacian, (cufftComplex*)fft_laplacian);   
        cudaFree(cu_buffer_laplacian);

        wiener_fourier<<<numBlocks1dfft,blockSize1d>>>(n_fft, fft_psf, fft_laplacian, fft_in, fft_out, lambda, scale);
        cufftExecC2R(Planifft, (cufftComplex*)fft_out, (cufftReal*)cu_buffer_out); 

        wiener_normalize<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out);

        cudaMemcpy(buffer_out, cu_buffer_out, n*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(cu_buffer_out);
        cudaFree(cu_buffer_in);

        cudaFree(fft_in);
        cudaFree(fft_out);
        cudaFree(fft_psf);
        cudaFree(fft_laplacian);
    }

}