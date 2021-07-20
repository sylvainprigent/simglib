/// \file spitfire2d.cu
/// \brief spitfire2d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020


#include "spitfire2d.h"
#include <smanipulate>
#include <score/SMath.h>
#include <score/SException.h>

__global__
void init_2d_buffers_hv(unsigned int N, float* cu_denoised_image, float* cu_noisy_image, float* dual_images0, 
                     float* dual_images1, float* dual_images2, float* dual_images3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_denoised_image[i] = cu_noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
        dual_images3[i] = 0.0;
    }
}

__global__
void init_2d_buffers_sv(unsigned int N, float* cu_denoised_image, float* cu_noisy_image, float* dual_images0, 
                     float* dual_images1, float* dual_images2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_denoised_image[i] = cu_noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
    }
}

__global__
void copy_buffer_2d(float* in_buffer, unsigned int n, float *out_buffer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out_buffer[i] = in_buffer[i];
    }
}

__global__
void sv_2d_primal(unsigned int sx, unsigned int sy, float primal_step, float primal_weight, float primal_weight_comp, float *denoised_image, float *noisy_image,
                  float* dual_images0, float* dual_images1, float* dual_images2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    unsigned int p = sy*x+y;
    unsigned int pxm = p - sy;
    unsigned int pym = p-1;

    float tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);
    
    float dx_adj = dual_images0[pxm] - dual_images0[p];
    float dy_adj = dual_images1[pym] - dual_images1[p];

    tmp -= (primal_weight * (dx_adj + dy_adj)
            + primal_weight_comp * dual_images2[p]);

    if (tmp > 1.0){
        denoised_image[p] = 1.0;
    }
    else if (tmp < 0.0 ){
        denoised_image[p] = 0.0;
    }
    else{
        denoised_image[p] = tmp;
    }    
}

__global__
void hv_2d_primal(unsigned int sx, unsigned int sy, float primal_step, float primal_weight, float primal_weight_comp, float sqrt2, float *denoised_image, float *noisy_image,
                  float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    float tmp, dxx_adj, dyy_adj, dxy_adj;
    int p, pxm, pxp, pym, pyp, pxym;    
    
    p = sy*x+y;
    pxm = p - sy;
    pxp = p + sy;
    pym = p-1;
    pyp = p+1;
    pxym = pxm-1;
    
    tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);
    dxx_adj = dual_images0[pxm] - 2 * dual_images0[p] + dual_images0[pxp];
    dyy_adj = dual_images1[pym] - 2 * dual_images1[p] + dual_images1[pyp];
    dxy_adj = dual_images2[p] - dual_images2[pxm] - dual_images2[pym] + dual_images2[pxym];
    tmp -= (primal_weight * (dxx_adj + dyy_adj + sqrt2 * dxy_adj) + primal_weight_comp * dual_images3[p]);
    
    if (tmp > 1.0){
        denoised_image[p] = 1.0;
    }
    else if (tmp < 0.0 ){
        denoised_image[p] = 0.0;
    }
    else{
        denoised_image[p] = tmp;
    }
}

__global__
void dual_2d_auxiliary(unsigned int N , float* auxiliary_image, float* denoised_image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
    }
}

__global__
void sv_2d_dual(unsigned int sx, unsigned int sy, float dual_weight, float dual_weight_comp, float*auxiliary_image, float* dual_images0, float* dual_images1,
                float* dual_images2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    unsigned int p = sy*x + y;
    unsigned int pxp = p + sy;
    unsigned int pyp = p+1;

    dual_images0[p] += dual_weight * (auxiliary_image[pxp]- auxiliary_image[p]);
    dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
    dual_images2[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void hv_2d_dual(unsigned int sx, unsigned int sy, float dual_weight, float dual_weight_comp, float sqrt2, float*auxiliary_image, float* dual_images0, float* dual_images1,
                float* dual_images2, float* dual_images3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    float dxx, dyy, dxy;
    int p, pxm, pxp, pym, pyp, pxyp; 
    
    p = sy*x+y;
    pxm = p - sy;
    pxp = p + sy;
    pym = p-1;
    pyp = p+1;
    pxyp = pxp+1;
    
    dxx = auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm];
    dual_images0[p] += dual_weight * dxx;
                    
    dyy = auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym];
    dual_images1[p] += dual_weight * dyy;
                    
    dxy = auxiliary_image[pxyp] - auxiliary_image[pxp]  - auxiliary_image[pyp] + auxiliary_image[p];
    dual_images2[p] += sqrt2 * dual_weight * dxy;
                    
    dual_images3[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void hv_dual_2d_normalize(unsigned int N, float inv_reg, float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt( dual_images0[i]*dual_images0[i] + dual_images1[i]*dual_images1[i] + dual_images2[i]*dual_images2[i] + dual_images3[i]*dual_images3[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0/tmp;
            dual_images0[i] *= inv_tmp;
            dual_images1[i] *= inv_tmp;
            dual_images2[i] *= inv_tmp;
            dual_images3[i] *= inv_tmp;
        }
    }
}

__global__
void sv_dual_2d_normalize(unsigned int N, float inv_reg, float* dual_images0, float* dual_images1, float* dual_images2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt( dual_images0[i]*dual_images0[i] + dual_images1[i]*dual_images1[i] + dual_images2[i]*dual_images2[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0/tmp;
            dual_images0[i] *= inv_tmp;
            dual_images1[i] *= inv_tmp;
            dual_images2[i] *= inv_tmp;
        }
    }
}

namespace SImg{

    void cuda_spitfire2d_denoise_sv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable)
    {
        unsigned int N = sx*sy;
    
        // Splitting parameters
        float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));
        float primal_step = 0.99
                / (0.5
                   + (8 * pow(weighting, 2.)
                      + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);
    
        // Initializations
        float* dual_images0;
        float* dual_images1;
        float* dual_images2;
        float* auxiliary_image;
        float* cu_denoised_image;
        float* cu_noisy_image;

        cudaMalloc ( &dual_images0, N*sizeof(float));
        cudaMalloc ( &dual_images1, N*sizeof(float));
        cudaMalloc ( &dual_images2, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_denoised_image, N*sizeof(float));
        cudaMalloc ( &cu_noisy_image, N*sizeof(float));
        cudaMemcpy(cu_noisy_image, noisy_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cida threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16);

        // init in cuda
        init_2d_buffers_sv<<<numBlocks1d, blockSize1d>>>(N, cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, dual_images2);

        // Denoising process
        float inv_reg = 1.0 / regularization;
        for (int iter = 0; iter < niter; iter++) {
    
            // Primal optimization
            copy_buffer_2d<<<numBlocks1d, blockSize1d>>>(cu_denoised_image, N, auxiliary_image);

            sv_2d_primal<<<blockSize2d,gridSize2d>>>(sx, sy, primal_step, primal_weight, primal_weight_comp, 
                                                     cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, 
                                                     dual_images2);
        
            // Stopping criterion
            cudaDeviceSynchronize();
            if (verbose){
                int iter_n = niter / 10;
                if (iter_n < 1) iter_n = 1;
                if (iter % iter_n == 0){
                    observable->notifyProgress(100*(float(iter)/float(niter)));
                }
            }
    
            // Dual optimization
            dual_2d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_denoised_image);

            // dual    
            sv_2d_dual<<<blockSize2d,gridSize2d>>>(sx, sy, dual_weight, dual_weight_comp, 
                                                   auxiliary_image, dual_images0, 
                                                   dual_images1, dual_images2);

    
            // normalize
            sv_dual_2d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_images0, dual_images1, dual_images2);
                                                   
        } // endfor (int iter = 0; iter < nb_iters_max; iter++)
        cudaDeviceSynchronize();
        //timer.toc();
        cudaFree(dual_images0);
        cudaFree(dual_images1);
        cudaFree(dual_images2);
        cudaFree(auxiliary_image);
        
        cudaMemcpy(denoised_image, cu_denoised_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_denoised_image); 
        cudaFree(cu_noisy_image);       
        
        if (verbose){
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire2d_denoise_hv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable)
    {
        unsigned int N = sx*sy;
        float sqrt2 = sqrt(2.);
    
        // Splitting parameters
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (64 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);
    
        // Initializations
        float* dual_images0;
        float* dual_images1;
        float* dual_images2;
        float* dual_images3;
        float* auxiliary_image;
        float* cu_denoised_image;
        float* cu_noisy_image;

        cudaMalloc ( &dual_images0, N*sizeof(float));
        cudaMalloc ( &dual_images1, N*sizeof(float));
        cudaMalloc ( &dual_images2, N*sizeof(float));
        cudaMalloc ( &dual_images3, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_denoised_image, N*sizeof(float));
        cudaMalloc ( &cu_noisy_image, N*sizeof(float));
        cudaMemcpy(cu_noisy_image, noisy_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cuda threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16);

        // init in cuda
        init_2d_buffers_hv<<<numBlocks1d, blockSize1d>>>(N, cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, dual_images2, dual_images3);
        
        // Deconvolution process
        float inv_reg = 1.0 / regularization;
        for (int iter = 0; iter < niter; ++iter) {

            // Primal optimization
            copy_buffer_2d<<<numBlocks1d, blockSize1d>>>(cu_denoised_image, N, auxiliary_image);
    
            hv_2d_primal<<<gridSize2d, blockSize2d>>>(sx, sy, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                     cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, 
                                                     dual_images2, dual_images3);
    
            // Stopping criterion
            cudaDeviceSynchronize();
            if (verbose){
                int iter_n = niter / 10;
                if (iter_n < 1) iter_n = 1;
                if (iter % iter_n == 0){
                    observable->notifyProgress(100*(float(iter)/float(niter)));
                }
            }
    
            // Dual optimization
            // dual_auxilary
            dual_2d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_denoised_image);
    
            // dual    
            hv_2d_dual<<<gridSize2d, blockSize2d>>>(sx, sy, dual_weight, dual_weight_comp, sqrt2, 
                                                   auxiliary_image, dual_images0, 
                                                   dual_images1, dual_images2, dual_images3);
    
            //normlization    
            hv_dual_2d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_images0, dual_images1, 
                                                            dual_images2, dual_images3);    
        } // endfor (int iter = 0; iter < nb_iters_max; iter++)
        cudaDeviceSynchronize();
        //timer.toc();
        cudaFree(dual_images0);
        cudaFree(dual_images1);
        cudaFree(dual_images2);
        cudaFree(dual_images3);
        cudaFree(auxiliary_image);
        
        cudaMemcpy(denoised_image, cu_denoised_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_denoised_image); 
        cudaFree(cu_noisy_image);       
        
        if (verbose){
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire2d_denoise(float *blurry_image, unsigned int sx, unsigned int sy, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable)
    {
        // normalize the input image
        unsigned int bs = sx * sy;
        float imin = blurry_image[0];
        float imax = blurry_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = blurry_image[i];
            if (val > imax)
            {
                imax = val;
            }
            if (val < imin)
            {
                imin = val;
            }
        }

        float *blurry_image_norm = new float[sx * sy];
        normMinMax(blurry_image, sx, sy, 1, 1, 1, blurry_image_norm);

        // run denoising
        if (method == "SV")
        {
            cuda_spitfire2d_denoise_sv(blurry_image_norm, sx, sy, deconv_image, regularization, weighting, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            cuda_spitfire2d_denoise_hv(blurry_image_norm, sx, sy, deconv_image, regularization, weighting, niter, verbose, observable);
        }
        else
        {
            throw SException("spitfire2d: method must be SV or HV");
        }

// normalize back intensities
        float omin = deconv_image[0];
        float omax = deconv_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = deconv_image[i];
            if (val > omax)
            {
                omax = val;
            }
            if (val < omin)
            {
                omin = val;
            }
        }

#pragma omp parallel for
        for (unsigned int i = 0; i < sx * sy; ++i)
        {
           deconv_image[i] = (deconv_image[i] - omin)/(omax-omin);
           deconv_image[i] = deconv_image[i] * (imax - imin) + imin;
        }

        delete[] blurry_image_norm;
    }
}