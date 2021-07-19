/// \file spitfire3d.cpp
/// \brief spitfire3d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire3d.h"

//#include <simage>
//#include <simageio>

#include <smanipulate>
#include <score/SMath.h>
#include <score/SException.h>
#include <sfft/SFFT.h>
#include <sfft/SFFTConvolutionFilter.h>

#include <scli>

#include <cufft.h>

#include "math.h"
#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

__global__
void d_hv_init_3d_buffers(unsigned int N, float* cu_deconv_image, float* cu_blury_image, float* dual_image0, 
                          float* dual_image1, float* dual_image2, float* dual_image3,
                          float* dual_image4, float* dual_image5, float* dual_image6)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_deconv_image[i] = cu_blury_image[i];
        dual_image0[i] = 0.0;
        dual_image1[i] = 0.0;
        dual_image2[i] = 0.0;
        dual_image3[i] = 0.0;
        dual_image4[i] = 0.0;
        dual_image5[i] = 0.0;
        dual_image6[i] = 0.0;
    }
}

__global__
void d_sv_init_3d_buffers(unsigned int N, float* cu_deconv_image, float* cu_blury_image, float* dual_image0, 
                          float* dual_image1, float* dual_image2, float* dual_image3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_deconv_image[i] = cu_blury_image[i];
        dual_image0[i] = 0.0;
        dual_image1[i] = 0.0;
        dual_image2[i] = 0.0;
        dual_image3[i] = 0.0;
    }
}

__global__
void d_init_deconv_residu_3d(unsigned int Nfft, cufftComplex* deconv_image_FT, cufftComplex* residue_image_FT, cufftComplex* blurry_image_FT)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nfft)
    {
        deconv_image_FT[i].x = blurry_image_FT[i].x;
        deconv_image_FT[i].y = blurry_image_FT[i].y;
        residue_image_FT[i].x = blurry_image_FT[i].x;
        residue_image_FT[i].y = blurry_image_FT[i].y;
    }
}

__global__
void d_copy_3d(unsigned int N, float* source, float* destination)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        destination[i] = source[i];
    }
}

__global__
void d_copy_complex_3d(unsigned int N, cufftComplex* source, cufftComplex* destination)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        destination[i].x = source[i].x;
        destination[i].y = source[i].y;
    }
}

__global__
void d_dataterm_3d(unsigned int Nfft, cufftComplex* OTF, cufftComplex* deconv_image_FT, cufftComplex* blurry_image_FT, cufftComplex* residue_image_FT, cufftComplex* adjoint_OTF)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < Nfft)
    {
        float real_tmp = OTF[i].x * deconv_image_FT[i].x - OTF[i].y * deconv_image_FT[i].y - blurry_image_FT[i].x;
        float imag_tmp = OTF[i].x * deconv_image_FT[i].y + OTF[i].y * deconv_image_FT[i].x - blurry_image_FT[i].y;

        residue_image_FT[i].x = (adjoint_OTF[i].x * real_tmp - adjoint_OTF[i].y * imag_tmp);
        residue_image_FT[i].y = (adjoint_OTF[i].x * imag_tmp + adjoint_OTF[i].y * real_tmp);
    }
}

__global__
void d_sv_primal_3d(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, float primal_step, float primal_weight, 
                    float primal_weight_comp, float sqrt2, float* deconv_image, float delta, float* residue_image, 
                    float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1)
    {
        return;
    }

    unsigned int p = z + sz * (y + sy * x);
    unsigned int pxm = p - sz * sy;
    unsigned int pym = p - sz;
    unsigned int pzm = p - 1;

    float tmp = deconv_image[p] - primal_step * residue_image[p] / float(N);

    float dx_adj = dual_images0[pxm] - dual_images0[p];
    float dy_adj = dual_images1[pym] - dual_images1[p];
    float dz_adj = delta * (dual_images2[pzm] - dual_images2[p]);

    tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj) + primal_weight_comp * dual_images3[p]);

    if (tmp > 1.0)
    {
        deconv_image[p] = 1.0;
    }
    else if (tmp < 0.0)
    {
        deconv_image[p] = 0.0;
    }
    else
    {
        deconv_image[p] = tmp;
    }
}

__global__
void d_hv_primal_3d(unsigned int N, unsigned int sx, unsigned int sy, unsigned int sz, float primal_step, float primal_weight, 
                    float primal_weight_comp, float sqrt2, float* deconv_image, float delta, float* residue_image, 
                    float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3,
                    float* dual_images4, float* dual_images5, float* dual_images6)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1)
    {
        return;
    }
    unsigned int p = z + sz * (y + sy * x);
    unsigned int pxm = p - sz * sy;
    unsigned int pym = p - sz;
    unsigned int pzm = p - 1;
    unsigned int pxp = p + sz * sy;
    unsigned int pyp = p + sz;
    unsigned int pzp = p + 1;

    float tmp = deconv_image[p] - primal_step * residue_image[p] / float(N);

    float dxx_adj = dual_images0[pxm] - 2 * dual_images0[p] + dual_images0[pxp];
    float dyy_adj = dual_images1[pym] - 2 * dual_images1[p] + dual_images1[pyp];
    float dzz_adj = (delta * delta) * (dual_images2[pzm] - 2 * dual_images2[p] + dual_images2[pzp]);

    // Other terms
    float dxy_adj = dual_images3[p] - dual_images3[pxm] - dual_images3[pym] + dual_images3[z + sz * (y - 1 + sy * (x - 1))];
    float dyz_adj = delta * (dual_images4[p] - dual_images4[pym] - dual_images4[pzm] + dual_images4[z - 1 + sz * (y - 1 + sy * x)]);
    float dzx_adj = delta * (dual_images5[p] - dual_images5[pzm] - dual_images5[pxm] + dual_images5[z - 1 + sz * (y + sy * (x - 1))]);

    tmp -= (primal_weight * (dxx_adj + dyy_adj + dzz_adj + sqrt2 * (dxy_adj + dyz_adj + dzx_adj)) + primal_weight_comp * dual_images6[p]);

    if (tmp > 1.0)
    {
        deconv_image[p] = 1.0;
    }
    else if (tmp < 0.0)
    {
        deconv_image[p] = 0.0;
    }
    else
    {
        deconv_image[p] = tmp;
    }
}

__global__
void d_dual_auxiliary(unsigned int N, float* auxiliary_image, float* deconv_image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auxiliary_image[i] = 2 * deconv_image[i] - auxiliary_image[i];        
    }
}

__global__
void d_sv_3d_dual(unsigned int sx, unsigned int sy, unsigned int sz, float dual_weight, float dual_weight_comp, float sqrt2, 
                  float delta, 
                  float* auxiliary_image, float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1)
    {
        return;
    }
    unsigned int p = z + sz * (y + sy * x);
    unsigned int pxp = p + sz * sy;
    unsigned int pyp = p + sz;
    unsigned int pzp = p + 1;

    dual_images0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
    dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
    dual_images2[p] += dual_weight * (delta * (auxiliary_image[pzp] - auxiliary_image[p]));
    dual_images3[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void d_hv_3d_dual(unsigned int sx, unsigned int sy, unsigned int sz, float dual_weight, float dual_weight_comp, float sqrt2, 
                  float delta,  
                  float* auxiliary_image, float* dual_images0, float* dual_images1, float* dual_images2, float* dual_images3,
                  float* dual_images4, float* dual_images5, float* dual_images6)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1 || z < 1 || z >= sz-1)
    {
        return;
    }
    unsigned int p = z + sz * (y + sy * x);  
    unsigned int pxm = p - sz*sy;
    unsigned int pym = p - sz;  
    unsigned int pzm = p - 1;
    unsigned int pxp = p + sz*sy;
    unsigned int pyp = p + sz;
    unsigned int pzp = p + 1;  

    dual_images0[p] += dual_weight * (auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm]);
    dual_images1[p] += dual_weight * (auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym]);
    dual_images2[p] += dual_weight * ((delta * delta) * (auxiliary_image[pzp] - 2 * auxiliary_image[p] + auxiliary_image[pzm]));
    dual_images3[p] += sqrt2 * dual_weight * (auxiliary_image[z + sz * (y + 1 + sy * (x + 1))] - auxiliary_image[pxp] - auxiliary_image[pyp] + auxiliary_image[p]);
    dual_images4[p] += sqrt2 * dual_weight * (delta * (auxiliary_image[z + 1 + sz * (y + 1 + sy * x)] - auxiliary_image[pyp] - auxiliary_image[pzp] + auxiliary_image[p]));
    dual_images5[p] += sqrt2 * dual_weight * (delta * (auxiliary_image[z + 1 + sz * (y + sy * (x + 1))] - auxiliary_image[pxp] - auxiliary_image[pzp] + auxiliary_image[p]));
    dual_images6[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void d_hv_dual_3d_normalize(unsigned int N, float inv_reg, float* dual_image0, float* dual_image1, 
                            float* dual_image2, float* dual_image3, float* dual_image4, float* dual_image5,
                            float* dual_image6)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt(dual_image0[i] * dual_image0[i] + dual_image1[i] * dual_image1[i] + dual_image2[i] * dual_image2[i] + dual_image3[i] * dual_image3[i]
                                   + dual_image4[i] * dual_image4[i]+ dual_image5[i] * dual_image5[i]+ dual_image6[i] * dual_image6[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0 / tmp;
            dual_image0[i] *= inv_tmp;
            dual_image1[i] *= inv_tmp;
            dual_image2[i] *= inv_tmp;
            dual_image3[i] *= inv_tmp;
            dual_image4[i] *= inv_tmp;
            dual_image5[i] *= inv_tmp;
            dual_image6[i] *= inv_tmp;
        }
    }
}

__global__
void d_sv_dual_3d_normalize(unsigned int N, float inv_reg, float* dual_image0, float* dual_image1, 
                            float* dual_image2, float* dual_image3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt(dual_image0[i] * dual_image0[i] + dual_image1[i] * dual_image1[i] + dual_image2[i] * dual_image2[i] 
                                   + dual_image3[i] * dual_image3[i] );
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0 / tmp;
            dual_image0[i] *= inv_tmp;
            dual_image1[i] *= inv_tmp;
            dual_image2[i] *= inv_tmp;
            dual_image3[i] *= inv_tmp;
        }
    }
}

namespace SImg
{

    void cuda_spitfire3d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, 
                                   const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, 
                                   bool verbose, SObservable *observable)
    {
        int N = sx * sy * sz;
        int Nfft = sx *sy * (sz / 2 + 1);
        float sqrt2 = sqrt(2.0);

       // Optical transfer function and its adjoint
        float *OTFReal = new float[N];
        shift3D(psf, OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));

        float *adjoint_PSF = new float[N];
        for (int x = 0; x < sx; x++)
        {
            for (int y = 0; y < sy; y++)
            {
                for (int z = 0; z < sz; z++)
                {
                    adjoint_PSF[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
                }
            }
        }

        float *adjoint_PSF_shift = new float[N];
        shift3D(adjoint_PSF, adjoint_PSF_shift, sx, sy, sz, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2, -int((sz - 1)) % 2);

        float *adjoint_OTFReal = new float[N];
        shift3D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        delete[] adjoint_PSF_shift;

        // Splitting parameters
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (144 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);

        // Initializations
        float* dual_image0;
        float* dual_image1;
        float* dual_image2;
        float* dual_image3;
        float* auxiliary_image;
        float* residue_image;
        float* cu_deconv_image;
        float* cu_blurry_image;
        
        cudaMalloc ( &dual_image0, N*sizeof(float));
        cudaMalloc ( &dual_image1, N*sizeof(float));
        cudaMalloc ( &dual_image2, N*sizeof(float));
        cudaMalloc ( &dual_image3, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_deconv_image, N*sizeof(float));
        cudaMalloc ( &cu_blurry_image, N*sizeof(float));
        cudaMalloc ( &residue_image, N*sizeof(float));
        cudaMemcpy(cu_blurry_image, blurry_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cuda threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (Nfft + blockSize1d - 1) / blockSize1d;
        dim3 blockSize3d(16, 16, 16);
        dim3 gridSize3d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16, (sz + 16 - 1) / 16);

        d_sv_init_3d_buffers<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, cu_blurry_image, dual_image0, dual_image1, dual_image2, dual_image3);
        //cudaDeviceSynchronize();
        
        cufftComplex *blurry_image_FT;
        cufftComplex *deconv_image_FT;
        cufftComplex *residue_image_FT;
        cufftComplex *OTF;
        cufftComplex *adjoint_OTF;

        cudaMalloc((void**)&blurry_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&deconv_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&residue_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&OTF, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&adjoint_OTF, sizeof(cufftComplex)*Nfft);

        // copy OTFReal and adjoint_OTFReal to cuda
        float* cu_OTFReal;
        float* cu_adjoint_OTFReal;
        cudaMalloc ( &cu_OTFReal, N*sizeof(float));
        cudaMalloc ( &cu_adjoint_OTFReal, N*sizeof(float));
        cudaMemcpy(cu_OTFReal, OTFReal, N*sizeof(float), cudaMemcpyHostToDevice); 
        cudaMemcpy(cu_adjoint_OTFReal, adjoint_OTFReal, N*sizeof(float), cudaMemcpyHostToDevice); 
        delete[] OTFReal;
        delete[] adjoint_OTFReal;

        // fft2d blurry_image -> blurry_image_FT
        cudaDeviceSynchronize();
        cufftHandle Planfft;
        cufftPlan3d(&Planfft, sx, sy, sz, CUFFT_R2C);
        cufftExecR2C(Planfft, (cufftReal*)cu_blurry_image, (cufftComplex*)blurry_image_FT);
        // fft2d OTFReal -> OTF
        cufftExecR2C(Planfft, (cufftReal*)cu_OTFReal, (cufftComplex*)OTF);
        // fft2d adjoint_OTFReal -> adjoint_OTF
        cufftExecR2C(Planfft, (cufftReal*)cu_adjoint_OTFReal, (cufftComplex*)adjoint_OTF);
        cudaDeviceSynchronize();

        cudaFree(cu_OTFReal);
        cudaFree(cu_adjoint_OTFReal);

        // Deconvolution process
        float inv_reg = 1.0 / regularization;

        d_init_deconv_residu_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT, blurry_image_FT);

        cufftHandle Planifft;
        cufftPlan3d(&Planifft, sx, sy, sz, CUFFT_C2R);
        //cudaDeviceSynchronize();
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
            d_copy_3d<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, auxiliary_image);
            
            cudaDeviceSynchronize();
            cufftExecR2C(Planfft, (cufftReal*)cu_deconv_image, (cufftComplex*)deconv_image_FT);
            cudaDeviceSynchronize();

            d_copy_complex_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT);

            // Data term
            d_dataterm_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, OTF, deconv_image_FT, blurry_image_FT, residue_image_FT, adjoint_OTF);
            cudaDeviceSynchronize();
            cufftExecC2R(Planifft, (cufftComplex*)residue_image_FT, (cufftReal*)residue_image);
            cudaDeviceSynchronize(); 

            // primal
            d_sv_primal_3d<<<gridSize3d, blockSize3d>>>(N, sx, sy, sz, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                        cu_deconv_image, delta, residue_image, dual_image0, dual_image1, 
                                                        dual_image2, dual_image3);
            // Stopping criterion
            if (verbose)
            {
                int iter_n = niter / 10;
                if (iter_n < 1)
                    iter_n = 1;
                if (iter % iter_n == 0)
                {
                    observable->notifyProgress(100 * (float(iter) / float(niter)));
                }
            }

            // Dual optimization
            d_dual_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_deconv_image);

            // dual    
            d_sv_3d_dual<<<gridSize3d, blockSize3d>>>(sx, sy, sz, dual_weight, dual_weight_comp, sqrt2, 
                                                      delta, auxiliary_image, dual_image0, 
                                                      dual_image1, dual_image2, dual_image3);                                         

            // normalize
            d_sv_dual_3d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_image0, dual_image1, 
                                                                 dual_image2, dual_image3);   

        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        cudaDeviceSynchronize();
        // free output
        cufftDestroy(Planfft);
        cufftDestroy(Planifft);
        cudaFree(dual_image0);
        cudaFree(dual_image1);
        cudaFree(dual_image2);
        cudaFree(dual_image3);
        cudaFree(auxiliary_image);
        cudaFree(residue_image);

        cudaFree(blurry_image_FT);
        cudaFree(deconv_image_FT);
        cudaFree(residue_image_FT);
        cudaFree(OTF);
        cudaFree(adjoint_OTF);

        cudaMemcpy(deconv_image, cu_deconv_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_deconv_image); 
        cudaFree(cu_blurry_image);   

        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire3d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, 
                                   const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, 
                                   bool verbose, SObservable *observable)
    {
        int N = sx * sy * sz;
        int Nfft = sx *sy * (sz / 2 + 1);
        float sqrt2 = sqrt(2.0);

       // Optical transfer function and its adjoint
        float *OTFReal = new float[N];
        shift3D(psf, OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));

        float *adjoint_PSF = new float[N];
        for (int x = 0; x < sx; x++)
        {
            for (int y = 0; y < sy; y++)
            {
                for (int z = 0; z < sz; z++)
                {
                    adjoint_PSF[z + sz * (y + sy * x)] = psf[(sz - 1 - z) + sz * ((sy - 1 - y) + sy * (sx - 1 - x))];
                }
            }
        }

        float *adjoint_PSF_shift = new float[N];
        shift3D(adjoint_PSF, adjoint_PSF_shift, sx, sy, sz, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2, -int((sz - 1)) % 2);

        float *adjoint_OTFReal = new float[N];
        shift3D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, sz, int(-float(sx) / 2.0), int(-float(sy) / 2.0), int(-float(sz) / 2.0));
        delete[] adjoint_PSF_shift;

        // Splitting parameters
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (144 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);

        // Initializations
        float* dual_image0;
        float* dual_image1;
        float* dual_image2;
        float* dual_image3;
        float* dual_image4;
        float* dual_image5;
        float* dual_image6;
        float* auxiliary_image;
        float* residue_image;
        float* cu_deconv_image;
        float* cu_blurry_image;
        
        cudaMalloc ( &dual_image0, N*sizeof(float));
        cudaMalloc ( &dual_image1, N*sizeof(float));
        cudaMalloc ( &dual_image2, N*sizeof(float));
        cudaMalloc ( &dual_image3, N*sizeof(float));
        cudaMalloc ( &dual_image4, N*sizeof(float));
        cudaMalloc ( &dual_image5, N*sizeof(float));
        cudaMalloc ( &dual_image6, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_deconv_image, N*sizeof(float));
        cudaMalloc ( &cu_blurry_image, N*sizeof(float));
        cudaMalloc ( &residue_image, N*sizeof(float));
        cudaMemcpy(cu_blurry_image, blurry_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        //STimer timer;
        //timer.setObserver(new SObserverConsole());
        //timer.tic();

        // cuda threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (Nfft + blockSize1d - 1) / blockSize1d;
        dim3 blockSize3d(16, 16, 16);
        dim3 gridSize3d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16, (sz + 16 - 1) / 16);

        d_hv_init_3d_buffers<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, cu_blurry_image, dual_image0, dual_image1, dual_image2, dual_image3, dual_image4, dual_image5, dual_image6);
        //cudaDeviceSynchronize();
        
        cufftComplex *blurry_image_FT;
        cufftComplex *deconv_image_FT;
        cufftComplex *residue_image_FT;
        cufftComplex *OTF;
        cufftComplex *adjoint_OTF;

        cudaMalloc((void**)&blurry_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&deconv_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&residue_image_FT, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&OTF, sizeof(cufftComplex)*Nfft);
        cudaMalloc((void**)&adjoint_OTF, sizeof(cufftComplex)*Nfft);

        // copy OTFReal and adjoint_OTFReal to cuda
        float* cu_OTFReal;
        float* cu_adjoint_OTFReal;
        cudaMalloc ( &cu_OTFReal, N*sizeof(float));
        cudaMalloc ( &cu_adjoint_OTFReal, N*sizeof(float));
        cudaMemcpy(cu_OTFReal, OTFReal, N*sizeof(float), cudaMemcpyHostToDevice); 
        cudaMemcpy(cu_adjoint_OTFReal, adjoint_OTFReal, N*sizeof(float), cudaMemcpyHostToDevice); 
        delete[] OTFReal;
        delete[] adjoint_OTFReal;

        // fft2d blurry_image -> blurry_image_FT
        cudaDeviceSynchronize();
        cufftHandle Planfft;
        cufftPlan3d(&Planfft, sx, sy, sz, CUFFT_R2C);
        cufftExecR2C(Planfft, (cufftReal*)cu_blurry_image, (cufftComplex*)blurry_image_FT);
        // fft2d OTFReal -> OTF
        cufftExecR2C(Planfft, (cufftReal*)cu_OTFReal, (cufftComplex*)OTF);
        // fft2d adjoint_OTFReal -> adjoint_OTF
        cufftExecR2C(Planfft, (cufftReal*)cu_adjoint_OTFReal, (cufftComplex*)adjoint_OTF);
        cudaDeviceSynchronize();

        cudaFree(cu_OTFReal);
        cudaFree(cu_adjoint_OTFReal);

        // Deconvolution process
        float inv_reg = 1.0 / regularization;


        d_init_deconv_residu_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT, blurry_image_FT);

        cufftHandle Planifft;
        cufftPlan3d(&Planifft, sx, sy, sz, CUFFT_C2R);
        //cudaDeviceSynchronize();
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
            d_copy_3d<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, auxiliary_image);
            
            cudaDeviceSynchronize();
            cufftExecR2C(Planfft, (cufftReal*)cu_deconv_image, (cufftComplex*)deconv_image_FT);
            cudaDeviceSynchronize();

            d_copy_complex_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT);
            //cudaDeviceSynchronize();

            // Data term
            d_dataterm_3d<<<numBlocks1dfft,blockSize1d>>>(Nfft, OTF, deconv_image_FT, blurry_image_FT, residue_image_FT, adjoint_OTF);
            cudaDeviceSynchronize();
            cufftExecC2R(Planifft, (cufftComplex*)residue_image_FT, (cufftReal*)residue_image);
            cudaDeviceSynchronize(); 

            // primal
            d_hv_primal_3d<<<gridSize3d, blockSize3d>>>(N, sx, sy, sz, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                        cu_deconv_image, delta, residue_image, dual_image0, dual_image1, 
                                                        dual_image2, dual_image3, dual_image4, dual_image5, dual_image6);
            //cudaDeviceSynchronize();
            // Stopping criterion
            if (verbose)
            {
                int iter_n = niter / 10;
                if (iter_n < 1)
                    iter_n = 1;
                if (iter % iter_n == 0)
                {
                    observable->notifyProgress(100 * (float(iter) / float(niter)));
                }
            }

            // Dual optimization
            d_dual_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_deconv_image);
            //cudaDeviceSynchronize();

            // dual    
            d_hv_3d_dual<<<gridSize3d, blockSize3d>>>(sx, sy, sz, dual_weight, dual_weight_comp, sqrt2, 
                                                      delta, auxiliary_image, dual_image0, 
                                                      dual_image1, dual_image2, dual_image3,
                                                      dual_image4, dual_image5, dual_image6);
            //cudaDeviceSynchronize();                                         

            // normalize
            d_hv_dual_3d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_image0, dual_image1, 
                                                                 dual_image2, dual_image3, dual_image4, 
                                                                 dual_image5, dual_image6);
            //cudaDeviceSynchronize();    

        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        cudaDeviceSynchronize();
        //timer.toc();
        // free output
        cufftDestroy(Planfft);
        cufftDestroy(Planifft);
        cudaFree(dual_image0);
        cudaFree(dual_image1);
        cudaFree(dual_image2);
        cudaFree(dual_image3);
        cudaFree(dual_image4);
        cudaFree(dual_image5);
        cudaFree(dual_image6);
        cudaFree(auxiliary_image);
        cudaFree(residue_image);

        cudaFree(blurry_image_FT);
        cudaFree(deconv_image_FT);
        cudaFree(residue_image_FT);
        cudaFree(OTF);
        cudaFree(adjoint_OTF);

        cudaMemcpy(deconv_image, cu_deconv_image, N*sizeof(float), cudaMemcpyDeviceToHost);   
        cudaFree(cu_deconv_image); 
        cudaFree(cu_blurry_image);   

        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void cuda_spitfire3d_deconv(float *blurry_image, unsigned int sx, unsigned int sy, unsigned int sz, float *psf, float *deconv_image, const float &regularization, const float &weighting, const float &delta, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable)
    {
        // normalize the input image
        unsigned int bs = sx * sy *sz;
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

        float *blurry_image_norm = new float[bs];
        normL2(blurry_image, sx, sy, sz, 1, 1, blurry_image_norm);

        // run denoising
        if (method == "SV")
        {
            cuda_spitfire3d_deconv_sv(blurry_image_norm, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            cuda_spitfire3d_deconv_hv(blurry_image_norm, sx, sy, sz, psf, deconv_image, regularization, weighting, delta, niter, verbose, observable);
        }
        else
        {
            throw SException("spitfire3d: method must be SV or HV");
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
        for (unsigned int i = 0; i < bs; ++i)
        {
            deconv_image[i] = (deconv_image[i] - omin)/(omax-omin);
            deconv_image[i] = deconv_image[i] * (imax - imin) + imin;
        }

        delete[] blurry_image_norm;
    }

}