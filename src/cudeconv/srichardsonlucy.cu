/// \file srichardsonlucy.cu
/// \brief srichardsonlucy definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020


#include "srichardsonlucy.h"
#include <smanipulate>
#include <cufft.h>

#include <iostream>

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                 \
    if(e!=cudaSuccess) {                                              \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
      exit(0); \
    }                                                                 \
   }

__global__
void rl_init_buffer_out(unsigned int N, float* buffer_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        buffer_out[i] = 0.5;
    }
}

__global__
void rl_mirror_psf(unsigned int sx, unsigned int sy, float* psf_mirror, float* psf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sx || y >= sy)
    {
        return;
    }
    psf_mirror[sy * x + y] = psf[sy * x + (sy - 1 - y)];
}

__global__
void rl_mirror_psf_3d(unsigned int sx, unsigned int sy, unsigned int sz, float* psf_mirror, float* psf)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    int z = p % sz;    
    int y = ((p-z)/sz) % sy;
    int x = (p-y)/sy;

    if (x >= sx || y >= sy || z >= sz)
    {
        return;
    }
    psf_mirror[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
}

__global__
void rl_convolve(unsigned int n_fft, float scale, cufftComplex *image1, cufftComplex *image2, cufftComplex *image_out)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_fft)
    {
        image_out[p].x = (image1[p].x * image2[p].x - image1[p].y * image2[p].y) * scale;
        image_out[p].y = (image1[p].y * image2[p].x + image1[p].x * image2[p].y) * scale;
    }
}

__global__
void rl_convolve_tmp_psf(unsigned int n_fft, float scale, cufftComplex *fft_tmp, cufftComplex *fft_psf)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_fft)
    {
        fft_tmp[p].x = (fft_tmp[p].x * fft_psf[p].x - fft_tmp[p].y * fft_psf[p].y) * scale;
        fft_tmp[p].y = (fft_tmp[p].y * fft_psf[p].x + fft_tmp[p].x * fft_psf[p].y) * scale;
    }
}

__global__
void rl_normalize_tmp(unsigned int n, float* tmp, float* buffer_in)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n)
    {
        if (tmp[p] > 1e-9)
        {
            tmp[p] = buffer_in[p] / tmp[p];
        }
        else
        {
            tmp[p] = 0.0;
        }
    }
}

__global__
void rl_update_iter(unsigned int n, float* buffer_out, float* tmp)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n){
        buffer_out[p] *= tmp[p];
    }
}


namespace SImg{

    void cuda_richardsonlucy_2d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int niter)
    {
        // memory
        unsigned int n = sx * sy;
        unsigned int n_fft = sx * (sy / 2 + 1);
        float scale = 1.0 / float(n_fft);

        cufftComplex *fft_in;
        cufftComplex *fft_out;
        cufftComplex *fft_psf;
        cufftComplex *fft_psf_mirror;
        cufftComplex *fft_tmp;
        float *psf_mirror;
        float *tmp;
        float* cu_buffer_in;
        float* cu_psf;
        float* cu_buffer_out;

        cudaMalloc((void**)&fft_in, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf_mirror, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_tmp, sizeof(cufftComplex)*n_fft);
        cudaMalloc ( &psf_mirror, n*sizeof(float));
        cudaMalloc ( &tmp, n*sizeof(float));
        cudaMalloc ( &cu_buffer_in, n*sizeof(float));
        cudaMalloc ( &cu_psf, n*sizeof(float));
        cudaMalloc ( &cu_buffer_out, n*sizeof(float));

        cudaMemcpy(cu_buffer_in, buffer_in, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_psf, psf, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        cufftHandle Planfft;
        cufftPlan2d(&Planfft, sx, sy, CUFFT_R2C);
        cufftHandle Planifft;
        cufftPlan2d(&Planifft, sx, sy, CUFFT_C2R);

        int blockSize1d = 256;
        int numBlocks1d = (n + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (n_fft + blockSize1d - 1) / blockSize1d;
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16);

        // initialization
        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_in, (cufftComplex*)fft_in);

        rl_init_buffer_out<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out);

        float *psf_shifted = new float[sx * sy];
        shift2D(psf, psf_shifted, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        float* cu_psf_shifted;
        cudaMalloc ( &cu_psf_shifted, n*sizeof(float));
        cudaMemcpy(cu_psf_shifted, psf_shifted, n*sizeof(float), cudaMemcpyHostToDevice);
        
        cufftExecR2C(Planfft, (cufftReal*)cu_psf_shifted, (cufftComplex*)fft_psf);
        cudaFree(cu_psf_shifted);
        delete[] psf_shifted;

        // flip psf
        rl_mirror_psf<<<gridSize2d,blockSize2d>>>(sx, sy, psf_mirror, cu_psf);
        cudaDeviceSynchronize();
        cufftExecR2C(Planfft, (cufftReal*)psf_mirror, (cufftComplex*)fft_psf_mirror);

        cudaFree(psf_mirror);

        unsigned int iter = 0;
        cudaDeviceSynchronize();
        
        while (iter < niter)
        {
            iter++;
            // tmp = convolve(buffer_out, psf)
            cufftExecR2C(Planfft, (cufftReal*)cu_buffer_out, (cufftComplex*)fft_out);

            rl_convolve<<<numBlocks1dfft,blockSize1d>>>(n_fft, scale, fft_out, fft_psf, fft_tmp);
            cufftExecC2R(Planifft, (cufftComplex*)fft_tmp, (cufftReal*)tmp);    
            
            // tmp = buffer_in / tmp
            rl_normalize_tmp<<<numBlocks1d, blockSize1d>>>(n, tmp, cu_buffer_in);

            // im_deconv *= convolve(tmp, psf_mirror)
            cufftExecR2C(Planfft, (cufftReal*)tmp, (cufftComplex*)fft_tmp);
            rl_convolve_tmp_psf<<<numBlocks1dfft,blockSize1d>>>(n_fft, scale, fft_tmp, fft_psf);
            cufftExecC2R(Planifft, (cufftComplex*)fft_tmp, (cufftReal*)tmp); 

            rl_update_iter<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out, tmp);
            cudaDeviceSynchronize();

        }
        cudaDeviceSynchronize();

        // free output
        cufftDestroy(Planfft);
        cufftDestroy(Planifft);
        
        cudaMemcpy(buffer_out, cu_buffer_out, n*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(cu_buffer_out);
        cudaFree(cu_buffer_in);

        cudaFree(fft_in);
        cudaFree(fft_out);
        cudaFree(fft_psf);
        cudaFree(fft_psf_mirror);
        cudaFree(fft_tmp);
        cudaFree(tmp);
    }

    void cuda_richardsonlucy_3d(float* buffer_in, float* psf, float* buffer_out, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int niter)
    {
        // memory
        unsigned int n = sx * sy * sz;
        unsigned int n_fft = sx * sy * (sz / 2 + 1);
        float scale = 1.0 / float(n_fft);

        cufftComplex *fft_in;
        cufftComplex *fft_out;
        cufftComplex *fft_psf;
        cufftComplex *fft_psf_mirror;
        cufftComplex *fft_tmp;
        float *psf_mirror;
        float *tmp;
        float* cu_buffer_in;
        float* cu_psf;
        float* cu_buffer_out;

        cudaMalloc((void**)&fft_in, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_out, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_psf_mirror, sizeof(cufftComplex)*n_fft);
        cudaMalloc((void**)&fft_tmp, sizeof(cufftComplex)*n_fft);
        cudaMalloc ( &psf_mirror, n*sizeof(float));
        cudaMalloc ( &tmp, n*sizeof(float));
        cudaMalloc ( &cu_buffer_in, n*sizeof(float));
        cudaMalloc ( &cu_psf, n*sizeof(float));
        cudaMalloc ( &cu_buffer_out, n*sizeof(float));

        cudaMemcpy(cu_buffer_in, buffer_in, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_psf, psf, n*sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        cufftHandle Planfft;
        cufftPlan3d(&Planfft, sx, sy, sz, CUFFT_R2C);
        cufftHandle Planifft;
        cufftPlan3d(&Planifft, sx, sy, sz, CUFFT_C2R);

        int blockSize1d = 256;
        int numBlocks1d = (n + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (n_fft + blockSize1d - 1) / blockSize1d;

        // initialization
        cufftExecR2C(Planfft, (cufftReal*)cu_buffer_in, (cufftComplex*)fft_in);

        rl_init_buffer_out<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out);

        float *psf_shifted = new float[sx * sy * sz];
        shift3D(psf, psf_shifted, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        float* cu_psf_shifted;
        cudaMalloc ( &cu_psf_shifted, n*sizeof(float));
        cudaMemcpy(cu_psf_shifted, psf_shifted, n*sizeof(float), cudaMemcpyHostToDevice);
        
        cufftExecR2C(Planfft, (cufftReal*)cu_psf_shifted, (cufftComplex*)fft_psf);
        cudaFree(cu_psf_shifted);
        delete[] psf_shifted;

        // flip psf
        rl_mirror_psf_3d<<<numBlocks1d, blockSize1d>>>(sx, sy, sz, psf_mirror, cu_psf);
        cudaDeviceSynchronize();
        cufftExecR2C(Planfft, (cufftReal*)psf_mirror, (cufftComplex*)fft_psf_mirror);

        cudaFree(psf_mirror);

        unsigned int iter = 0;
        cudaDeviceSynchronize();
        
        while (iter < niter)
        {
            iter++;
            // tmp = convolve(buffer_out, psf)
            cufftExecR2C(Planfft, (cufftReal*)cu_buffer_out, (cufftComplex*)fft_out);

            rl_convolve<<<numBlocks1dfft,blockSize1d>>>(n_fft, scale, fft_out, fft_psf, fft_tmp);
            cufftExecC2R(Planifft, (cufftComplex*)fft_tmp, (cufftReal*)tmp);    
            
            // tmp = buffer_in / tmp
            rl_normalize_tmp<<<numBlocks1d, blockSize1d>>>(n, tmp, cu_buffer_in);

            // im_deconv *= convolve(tmp, psf_mirror)
            cufftExecR2C(Planfft, (cufftReal*)tmp, (cufftComplex*)fft_tmp);
            rl_convolve_tmp_psf<<<numBlocks1dfft,blockSize1d>>>(n_fft, scale, fft_tmp, fft_psf);
            cufftExecC2R(Planifft, (cufftComplex*)fft_tmp, (cufftReal*)tmp); 

            rl_update_iter<<<numBlocks1d, blockSize1d>>>(n, cu_buffer_out, tmp);
            cudaDeviceSynchronize();

        }
        cudaDeviceSynchronize();
        // free output
        cufftDestroy(Planfft);
        cufftDestroy(Planifft);
        
        cudaMemcpy(buffer_out, cu_buffer_out, n*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(cu_buffer_out);
        cudaFree(cu_buffer_in);

        cudaFree(fft_in);
        cudaFree(fft_out);
        cudaFree(fft_psf);
        cudaFree(fft_psf_mirror);
        cudaFree(fft_tmp);
        cudaFree(tmp);
    }
    
}