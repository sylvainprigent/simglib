/// \file spitfire4d.cu
/// \brief spitfire4d definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020


#include "spitfire4d.h"
#include <score/SMath.h>
#include <score/SException.h>
#include <smanipulate>

__global__
void init_4d_buffers_hv(unsigned int N, float* cu_denoised_image, float* cu_noisy_image, float* dual_images0, 
                     float* dual_images1, float* dual_images2, float* dual_images3,
                     float* dual_images4, float* dual_images5, float* dual_images6,
                     float* dual_images7, float* dual_images8, float* dual_images9,
                     float* dual_images10)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_denoised_image[i] = cu_noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
        dual_images3[i] = 0.0;
        dual_images4[i] = 0.0;
        dual_images5[i] = 0.0;
        dual_images6[i] = 0.0;
        dual_images7[i] = 0.0;
        dual_images8[i] = 0.0;
        dual_images9[i] = 0.0;
        dual_images10[i] = 0.0;
    }
}

__global__
void init_4d_buffers_sv(unsigned int N, float* cu_denoised_image, float* cu_noisy_image, float* dual_images0, 
                        float* dual_images1, float* dual_images2, float* dual_images3, float* dual_images4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_denoised_image[i] = cu_noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
        dual_images3[i] = 0.0;
        dual_images4[i] = 0.0;
    }
}

__global__
void copy_buffer(float* in_buffer, unsigned int n, float *out_buffer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out_buffer[i] = in_buffer[i];
    }
}

__global__
void sv_4d_primal(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float primal_step, float primal_weight, 
                  float primal_weight_comp, float deltaz, float deltat, float *denoised_image, float *noisy_image,
                  float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3, float* dual_images4)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N)
    {
        return;
    }

    int t = p % st;
    int pt = (p-t)/st;
    int z = pt % sz;
    int pz = (pt-z)/sz;
    int y = pz % sy;
    int x = (pz-y)/sy;     

    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1 || t < 1 || t >= st-1)
    {
        return;
    }

    //unsigned int p = t + st * (z + sz * (y + sy * x));
    unsigned int pxm = t + st * (z + sz * (y + sy * (x - 1)));
    unsigned int pym = t + st * (z + sz * ((y - 1) + sy * x));
    unsigned int pzm = t + st * ((z - 1) + sz * (y + sy * x));
    unsigned int ptm = t - 1 + st * (z + sz * (y + sy * x));

    float tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);
    float dx_adj = dual_images0[pxm] - dual_images0[p];
    float dy_adj = dual_images1[pym] - dual_images1[p];
    float dz_adj = deltaz * (dual_images2[pzm] - dual_images2[p]);
    float dt_adj = deltat * (dual_images3[ptm] - dual_images3[p]);

    tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj + dt_adj) + primal_weight_comp * dual_images4[p]);
    if (tmp > 1.0)
    {
        denoised_image[p] = 1.0;
    }
    else if (tmp < 0.0)
    {
        denoised_image[p] = 0.0;
    }
    else
    {
        denoised_image[p] = tmp;
    }
}

__global__
void hv_4d_primal(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float primal_step, float primal_weight, 
                  float primal_weight_comp, float sqrt2, float deltaz, float deltat, float *denoised_image, float *noisy_image,
                  float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3,
                  float* dual_images4, float* dual_images5, float* dual_images6,
                  float* dual_images7, float* dual_images8, float* dual_images9, float* dual_images10)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N)
    {
        return;
    }

    int t = p % st;
    int pt = (p-t)/st;
    int z = pt % sz;
    int pz = (pt-z)/sz;
    int y = pz % sy;
    int x = (pz-y)/sy;     

    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1 || t < 1 || t >= st-1)
    {
        return;
    }
    //unsigned int p = t + st * (z + sz * (y + sy * x));
    unsigned int pxm = t + st * (z + sz * (y + sy * (x - 1)));
    unsigned int pym = t + st * (z + sz * (y - 1 + sy * x));
    unsigned int pzm = t + st * (z - 1 + sz * (y + sy * x));
    unsigned int ptm = t - 1 + st * (z + sz * (y + sy * x));
    unsigned int pxp = t + st * (z + sz * (y + sy * (x + 1)));
    unsigned int pyp = t + st * (z + sz * (y + 1 + sy * x));
    unsigned int pzp = t + st * (z + 1 + sz * (y + sy * x));
    unsigned int ptp = t + 1 + st * (z + sz * (y + sy * x));

    float tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);

    // Diagonal terms
    float dxx_adj = dual_images0[pxm] - 2 * dual_images0[p] + dual_images0[pxp];
    float dyy_adj = dual_images1[pym] - 2 * dual_images1[p] + dual_images1[pyp];
    float dzz_adj = (deltaz * deltaz) * (dual_images2[pzm] - 2 * dual_images2[p] + dual_images2[pzp]);
    float dtt_adj = (deltat) * (dual_images3[ptm] - 2 * dual_images3[p] + dual_images3[ptp]);

    // Other terms
    unsigned int pxym = t + st * (z + sz * (y - 1 + sy * (x - 1)));
    float dxy_adj = dual_images4[p] - dual_images4[pxm] - dual_images4[pym] + dual_images4[pxym];

    unsigned int pyzm = t + st * (z - 1 + sz * (y - 1 + sy * x));
    float dyz_adj = deltaz * (dual_images5[p] - dual_images5[pym] - dual_images5[pzm] + dual_images5[pyzm]);

    unsigned int pxzm = t + st * (z - 1 + sz * (y + sy * (x - 1)));
    float dzx_adj = deltaz * (dual_images6[p] - dual_images6[pzm] - dual_images6[pxm] + dual_images6[pxzm]);

    unsigned int pxtm = t - 1 + st * (z + sz * (y + sy * (x - 1)));
    float dtx_adj = deltat * (dual_images7[p] - dual_images7[ptm] - dual_images7[pxm] + dual_images7[pxtm]);

    unsigned int pytm = t - 1 + st * (z + sz * (y - 1 + sy * x));
    float dty_adj = deltat * (dual_images8[p] - dual_images8[ptm] - dual_images8[pym] + dual_images8[pytm]);

    unsigned int pztm = t - 1 + st * (z - 1 + sz * (y + sy * x));
    float dtz_adj = deltat * (dual_images9[p] - dual_images9[ptm] - dual_images9[pzm] + dual_images9[pztm]);

    tmp -= (primal_weight * (dxx_adj + dyy_adj + dzz_adj + dtt_adj + sqrt2 * (dxy_adj + dyz_adj + dzx_adj) + sqrt2 * (dtx_adj + dty_adj + dtz_adj)) + primal_weight_comp * dual_images10[p]);

    if (tmp > 1.0)
    {
        denoised_image[p] = 1.0;
    }
    else if (tmp < 0.0)
    {
        denoised_image[p] = 0.0;
    }
    else
    {
        denoised_image[p] = tmp;
    }
}

__global__
void dual_4d_auxiliary(unsigned int N , float* auxiliary_image, float* denoised_image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
    }
}

__global__
void sv_4d_dual(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float dual_weight, 
                float dual_weight_comp, float deltaz, float deltat, float*auxiliary_image, 
                float* dual_images0, float* dual_images1,
                float* dual_images2, float* dual_images3, float* dual_images4)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N)
    {
        return;
    }

    int t = p % st;
    int pt = (p-t)/st;
    int z = pt % sz;
    int pz = (pt-z)/sz;
    int y = pz % sy;
    int x = (pz-y)/sy;     

    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1 || t < 1 || t >= st-1)
    {
        return;
    }
    //unsigned int p = t + st * (z + sz * (y + sy * x));
    unsigned int pxp = t + st * (z + sz * (y + sy * (x + 1)));
    unsigned int pyp = t + st * (z + sz * (y + 1 + sy * x));
    unsigned int pzp = t + st * (z + 1 + sz * (y + sy * x));
    unsigned int ptp = t + 1 + st * (z + sz * (y + sy * x));

    dual_images0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
    dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
    dual_images2[p] += dual_weight * (deltaz * (auxiliary_image[pzp] - auxiliary_image[p]));
    dual_images3[p] += dual_weight * (deltat * (auxiliary_image[ptp] - auxiliary_image[p]));
    dual_images4[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void hv_4d_dual(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float dual_weight, 
                float dual_weight_comp, float sqrt2, float deltaz, float deltat, 
                float*auxiliary_image, float* dual_images0, float* dual_images1,
                float* dual_images2, float* dual_images3, float* dual_images4, float* dual_images5, float* dual_images6,
                float* dual_images7, float* dual_images8, float* dual_images9, float* dual_images10)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N)
    {
        return;
    }

    int t = p % st;
    int pt = (p-t)/st;
    int z = pt % sz;
    int pz = (pt-z)/sz;
    int y = pz % sy;
    int x = (pz-y)/sy;     

    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1 || t < 1 || t >= st-1)
    {
        return;
    }
    //unsigned int p = t + st * (z + sz * (y + sy * x));
    unsigned int pxp = t + st * (z + sz * (y + sy * (x + 1)));
    unsigned int pyp = t + st * (z + sz * (y + 1 + sy * x));
    unsigned int pzp = t + st * (z + 1 + sz * (y + sy * x));
    unsigned int ptp = t + 1 + st * (z + sz * (y + sy * x));
    unsigned int pxm = t + st * (z + sz * (y + sy * (x - 1)));
    unsigned int pym = t + st * (z + sz * (y - 1 + sy * x));
    unsigned int pzm = t + st * (z - 1 + sz * (y + sy * x));
    unsigned int ptm = t - 1 + st * (z + sz * (y + sy * x));

    float dxx = auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm];
    dual_images0[p] += dual_weight * dxx;
    float dyy = auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym];
    dual_images1[p] += dual_weight * dyy;
    float dzz = (deltaz * deltaz) * (auxiliary_image[pzp] - 2 * auxiliary_image[p] + auxiliary_image[pzm]);
    dual_images2[p] += dual_weight * dzz;
    float dtt = (deltat * deltat) * (auxiliary_image[ptp] - 2 * auxiliary_image[p] + auxiliary_image[ptm]);
    dual_images3[p] += dual_weight * dtt;
                            
    unsigned int pxyp = t + st * (z + sz * (y + 1 + sy * (x + 1)));
    float dxy = auxiliary_image[pxyp] - auxiliary_image[pxp] - auxiliary_image[pyp] + auxiliary_image[p];
    dual_images4[p] += sqrt2 * dual_weight * dxy;
                            
    unsigned int pyzp = t + st * (z + 1 + sz * (y + 1 + sy * x));
    float dyz = deltaz * (auxiliary_image[pyzp] - auxiliary_image[pyp] - auxiliary_image[pzp] + auxiliary_image[p]);
    dual_images5[p] += sqrt2 * dual_weight * dyz;
                            
    unsigned int pxzp = t + st * (z + 1 + sz * (y + sy * (x + 1)));
    float dzx = deltaz * (auxiliary_image[pxzp] - auxiliary_image[pxp] - auxiliary_image[pzp] + auxiliary_image[p]);
    dual_images6[p] += sqrt2 * dual_weight * dzx;
                            
    unsigned int pxtp = t + 1 + st * (z + sz * (y + sy * (x + 1)));
    float dtx = deltat * (auxiliary_image[pxtp] - auxiliary_image[pxp] - auxiliary_image[ptp] + auxiliary_image[p]);
    dual_images7[p] += sqrt2 * dual_weight * dtx;
                            
    unsigned int pytp = t + 1 + st * (z + sz * (y + 1 + sy * x));
    float dty = deltat * (auxiliary_image[pytp] - auxiliary_image[pyp] - auxiliary_image[ptp] + auxiliary_image[p]);
    dual_images8[p] += sqrt2 * dual_weight * dty;
                            
    unsigned int pztp = t + 1 + st * (z + 1 + sz * (y + sy * x));
    float dtz = deltat * (auxiliary_image[pztp] - auxiliary_image[pzp] - auxiliary_image[ptp] + auxiliary_image[p]);
    dual_images9[p] += sqrt2 * dual_weight * dtz;
 
    dual_images10[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void hv_dual_4d_normalize(unsigned int N, float inv_reg, float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3,
                          float* dual_images4, float* dual_images5, float* dual_images6,
                          float* dual_images7, float* dual_images8, float* dual_images9, float* dual_images10)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt( dual_images0[i]*dual_images0[i] + dual_images1[i]*dual_images1[i] + dual_images2[i]*dual_images2[i] + dual_images3[i]*dual_images3[i]
                                   + dual_images4[i] * dual_images4[i] + dual_images4[i] * dual_images4[i] + dual_images6[i] * dual_images6[i]
                                   + dual_images7[i] * dual_images7[i]+ dual_images8[i] * dual_images8[i]
                                   + dual_images9[i] * dual_images9[i]+ dual_images10[i] * dual_images10[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0/tmp;
            dual_images0[i] *= inv_tmp;
            dual_images1[i] *= inv_tmp;
            dual_images2[i] *= inv_tmp;
            dual_images3[i] *= inv_tmp;
            dual_images4[i] *= inv_tmp;
            dual_images5[i] *= inv_tmp;
            dual_images6[i] *= inv_tmp;
            dual_images7[i] *= inv_tmp;
            dual_images8[i] *= inv_tmp;
            dual_images9[i] *= inv_tmp;
            dual_images10[i] *= inv_tmp;
        }
    }
}

__global__
void sv_dual_4d_normalize(unsigned int N, float inv_reg, float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3, float* dual_images4)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt( dual_images0[i]*dual_images0[i] + dual_images1[i]*dual_images1[i] + dual_images2[i]*dual_images2[i] + dual_images3[i]*dual_images3[i] + dual_images4[i]*dual_images4[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0/tmp;
            dual_images0[i] *= inv_tmp;
            dual_images1[i] *= inv_tmp;
            dual_images2[i] *= inv_tmp;
            dual_images3[i] *= inv_tmp;
            dual_images4[i] *= inv_tmp;
        }
    }
}

namespace SImg{

    void cuda_spitfire4d_denoise_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const float& deltaz, const float& deltat, const unsigned int& niter, bool verbose, SObservable* observable)
    {
        unsigned int N = sx * sy * sz * st;
    
        // Splitting parameters
        double dual_step = SMath::max(0.01, SMath::min(0.1, regularization));

        double primal_step = 0.99 / (0.5 + (16 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);

        double primal_weight = primal_step * weighting;
        double primal_weight_comp = primal_step * (1 - weighting);
        double dual_weight = dual_step * weighting;
        double dual_weight_comp = dual_step * (1 - weighting);
        float inv_reg = 1.0 / regularization;
    
        // Initializations
        float* dual_images0;
        float* dual_images1;
        float* dual_images2;
        float* dual_images3;
        float* dual_images4;
        float* auxiliary_image;
        float* cu_denoised_image;
        float* cu_noisy_image;

        cudaMalloc ( &dual_images0, N*sizeof(float));
        cudaMalloc ( &dual_images1, N*sizeof(float));
        cudaMalloc ( &dual_images2, N*sizeof(float));
        cudaMalloc ( &dual_images3, N*sizeof(float));
        cudaMalloc ( &dual_images4, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_denoised_image, N*sizeof(float));
        cudaMalloc ( &cu_noisy_image, N*sizeof(float));
        cudaMemcpy(cu_noisy_image, noisy_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cida threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;

        // init in cuda
        init_4d_buffers_sv<<<numBlocks1d, blockSize1d>>>(N, cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, dual_images2, dual_images3, dual_images4);
        cudaDeviceSynchronize();
        // Denoising process
        for (int iter = 0; iter < niter; iter++) {
    
            // Primal optimization
            copy_buffer<<<numBlocks1d, blockSize1d>>>(cu_denoised_image, N, auxiliary_image);

            sv_4d_primal<<<numBlocks1d, blockSize1d>>>(N, sx, sy, sz, st, primal_step, primal_weight, primal_weight_comp, 
                                                     deltaz, deltat,
                                                     cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, 
                                                     dual_images2, dual_images3, dual_images4);
            cudaDeviceSynchronize();    
            // Stopping criterion
            if (verbose){
                int iter_n = niter / 10;
                if (iter_n < 1) iter_n = 1;
                if (iter % iter_n == 0){
                    observable->notifyProgress(100*(float(iter)/float(niter)));
                }
            }
    
            // Dual optimization
            dual_4d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_denoised_image);

            // dual    
            sv_4d_dual<<<numBlocks1d, blockSize1d>>>(N, sx, sy, sz, st, dual_weight, dual_weight_comp, 
                                                   deltaz, deltat,
                                                   auxiliary_image, dual_images0, 
                                                   dual_images1, dual_images2, dual_images3, dual_images4);

    
            // normalize
            sv_dual_4d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_images0, dual_images1, dual_images2, dual_images3, dual_images4);
                                                   
        } // endfor (int iter = 0; iter < nb_iters_max; iter++)
        cudaDeviceSynchronize();
        cudaFree(dual_images0);
        cudaFree(dual_images1);
        cudaFree(dual_images2);
        cudaFree(dual_images3);
        cudaFree(dual_images4);
        cudaFree(auxiliary_image);
        
        cudaMemcpy(denoised_image, cu_denoised_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_denoised_image); 
        cudaFree(cu_noisy_image);       
        
        if (verbose){
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire4d_denoise_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const float& deltaz, const float& deltat, const unsigned int& niter, bool verbose, SObservable* observable)
    {
        unsigned int N = sx * sy * sz * st;
    
        // Splitting parameters
        float sqrt2 = sqrt(2.);
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (256 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);
        float inv_reg = 1.0 / regularization;
    
        // Initializations
        float* dual_images0;
        float* dual_images1;
        float* dual_images2;
        float* dual_images3;
        float* dual_images4;
        float* dual_images5;
        float* dual_images6;
        float* dual_images7;
        float* dual_images8;
        float* dual_images9;
        float* dual_images10;
        float* auxiliary_image;
        float* cu_denoised_image;
        float* cu_noisy_image;

        cudaMalloc ( &dual_images0, N*sizeof(float));
        cudaMalloc ( &dual_images1, N*sizeof(float));
        cudaMalloc ( &dual_images2, N*sizeof(float));
        cudaMalloc ( &dual_images3, N*sizeof(float));
        cudaMalloc ( &dual_images4, N*sizeof(float));
        cudaMalloc ( &dual_images5, N*sizeof(float));
        cudaMalloc ( &dual_images6, N*sizeof(float));
        cudaMalloc ( &dual_images7, N*sizeof(float));
        cudaMalloc ( &dual_images8, N*sizeof(float));
        cudaMalloc ( &dual_images9, N*sizeof(float));
        cudaMalloc ( &dual_images10, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_denoised_image, N*sizeof(float));
        cudaMalloc ( &cu_noisy_image, N*sizeof(float));
        cudaMemcpy(cu_noisy_image, noisy_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cuda threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;

        // init in cuda
        init_4d_buffers_hv<<<numBlocks1d, blockSize1d>>>(N, cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, 
                                                         dual_images2, dual_images3, dual_images4, dual_images5, dual_images6,
                                                         dual_images7, dual_images8, dual_images9, dual_images10);
        
        // Deconvolution process
        cudaDeviceSynchronize();
        for (int iter = 0; iter < niter; ++iter) {

            // Primal optimization
            copy_buffer<<<numBlocks1d, blockSize1d>>>(cu_denoised_image, N, auxiliary_image);
    
            hv_4d_primal<<<numBlocks1d, blockSize1d>>>(N, sx, sy, sz, st, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                      deltaz, deltat,
                                                      cu_denoised_image, cu_noisy_image, dual_images0, dual_images1, 
                                                      dual_images2, dual_images3, dual_images4, dual_images5, dual_images6,
                                                      dual_images7, dual_images8, dual_images9, dual_images10);
            cudaDeviceSynchronize();    
            // Stopping criterion
            if (verbose){
                int iter_n = niter / 10;
                if (iter_n < 1) iter_n = 1;
                if (iter % iter_n == 0){
                    observable->notifyProgress(100*(float(iter)/float(niter)));
                }
            }
    
            // Dual optimization
            // dual_auxilary
            dual_4d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_denoised_image);
    
            // dual    
            hv_4d_dual<<<numBlocks1d, blockSize1d>>>(N, sx, sy, sz, st, dual_weight, dual_weight_comp, sqrt2, deltaz, deltat,
                                                    auxiliary_image, dual_images0, 
                                                    dual_images1, dual_images2, dual_images3,
                                                    dual_images4, dual_images5, dual_images6, 
                                                    dual_images7, dual_images8, dual_images9,
                                                    dual_images10);
    
            //normlization    
            hv_dual_4d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_images0, dual_images1, 
                                                               dual_images2, dual_images3, dual_images4, 
                                                               dual_images5, dual_images6,
                                                               dual_images7, dual_images8,
                                                               dual_images9, dual_images10);    
        } // endfor (int iter = 0; iter < nb_iters_max; iter++)
        cudaDeviceSynchronize();
        //timer.toc();
        cudaFree(dual_images0);
        cudaFree(dual_images1);
        cudaFree(dual_images2);
        cudaFree(dual_images3);
        cudaFree(dual_images4);
        cudaFree(dual_images5);
        cudaFree(dual_images6);
        cudaFree(dual_images7);
        cudaFree(dual_images8);
        cudaFree(dual_images9);
        cudaFree(dual_images10);
        cudaFree(auxiliary_image);
        
        cudaMemcpy(denoised_image, cu_denoised_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_denoised_image); 
        cudaFree(cu_noisy_image);       
        
        if (verbose){
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire4d_denoise(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const float& deltaz, const float& deltat, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable)
    {
        // normalize the input image
        unsigned int bs = sx * sy * sz * st;
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

        float *blurry_image_norm = new float[sx * sy * sz * st];
        normMinMax(blurry_image, sx, sy, sz, st, 1, blurry_image_norm);

        // run denoising
        if (method == "SV")
        {
            cuda_spitfire4d_denoise_sv(blurry_image_norm, sx, sy, sz, st, denoised_image, regularization, weighting, deltaz, deltat, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            cuda_spitfire4d_denoise_hv(blurry_image_norm, sx, sy, sz, st, denoised_image, regularization, weighting, deltaz, deltat, niter, verbose, observable);
        }
        else
        {
            throw SException("spitfire4d: method must be SV or HV");
        }

        // normalize back intensities
        float omin = denoised_image[0];
        float omax = denoised_image[0];
        for (unsigned int i = 1; i < bs; ++i)
        {
            float val = denoised_image[i];
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
        for (unsigned int i = 0; i < bs; ++i)
        {
            denoised_image[i] = (denoised_image[i] - omin)/(omax-omin);
            denoised_image[i] = denoised_image[i] * (imax - imin) + imin;
        }

        delete[] blurry_image_norm;
    }
}