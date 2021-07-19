/// \file spitfire2d.cpp
/// \brief spitfire2d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire2d.h"

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
void d_sv_init_2d_buffers(unsigned int N, float* cu_deconv_image, float* cu_blury_image, float* dual_image0, 
                          float* dual_image1, float* dual_image2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cu_deconv_image[i] = cu_blury_image[i];
        dual_image0[i] = 0.0;
        dual_image1[i] = 0.0;
        dual_image2[i] = 0.0;
    }
}

__global__
void d_hv_init_2d_buffers(unsigned int N, float* cu_deconv_image, float* cu_blury_image, float* dual_image0, 
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
void d_init_deconv_residu(unsigned int Nfft, cufftComplex* deconv_image_FT, cufftComplex* residue_image_FT, cufftComplex* blurry_image_FT)
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
void d_copy(unsigned int N, float* source, float* destination)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        destination[i] = source[i];
    }
}

__global__
void d_copy_complex(unsigned int N, cufftComplex* source, cufftComplex* destination)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        destination[i].x = source[i].x;
        destination[i].y = source[i].y;
    }
}

__global__
void d_dataterm(unsigned int Nfft, cufftComplex* OTF, cufftComplex* deconv_image_FT, cufftComplex* blurry_image_FT, cufftComplex* residue_image_FT, cufftComplex* adjoint_OTF)
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
void d_sv_primal_2d(unsigned int N, unsigned int sx, unsigned int sy, float primal_step, float primal_weight, 
                    float primal_weight_comp, float sqrt2, float* deconv_image, float* residue_image, 
                    float* dual_image0, float* dual_image1, float* dual_image2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    unsigned int p = y + sy * x;
    float tmp = deconv_image[p] - primal_step * residue_image[p] / float(N);

    unsigned int pxm = p - sy;
    unsigned int pym = p - 1;
    float dx_adj = dual_image0[pxm] - dual_image0[p];
    float dy_adj = dual_image1[pym] - dual_image1[p];

    tmp -= (primal_weight * (dx_adj + dy_adj) + primal_weight_comp * dual_image2[p]);

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
void d_hv_primal_2d(unsigned int N, unsigned int sx, unsigned int sy, float primal_step, float primal_weight, 
                    float primal_weight_comp, float sqrt2, float* deconv_image, float* residue_image, 
                    float* dual_image0, float* dual_image1, float* dual_image2, float* dual_image3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    float tmp, dxx_adj, dyy_adj, dxy_adj;
    int p, pxm, pxp, pym, pyp, pxym;

    p = sy * x + y;
    pxm = p - sy;
    pxp = p + sy;
    pym = p - 1;
    pyp = p + 1;
    pxym = pxm - 1;

    //tmp = deconv_image[p] - primal_step * (residue_image[p] / float(float(sx)*(float(sy)/2.0+1)));
    tmp = deconv_image[p] - primal_step * (residue_image[p] / float(N));
    dxx_adj = dual_image0[pxm] - 2 * dual_image0[p] + dual_image0[pxp];
    dyy_adj = dual_image1[pym] - 2 * dual_image1[p] + dual_image1[pyp];
    dxy_adj = dual_image2[p] - dual_image2[pxm] - dual_image2[pym] + dual_image2[pxym];
    tmp -= (primal_weight * (dxx_adj + dyy_adj + sqrt2 * dxy_adj) + primal_weight_comp * dual_image3[p]);

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
void d_dual_2d_auxiliary(unsigned int N, float* auxiliary_image, float* deconv_image)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        auxiliary_image[i] = 2 * deconv_image[i] - auxiliary_image[i];        
    }
}

__global__
void d_sv_2d_dual(unsigned int sx, unsigned int sy, float dual_weight, float dual_weight_comp, float sqrt2, 
                  float* auxiliary_image, float* dual_image0, float* dual_image1, float* dual_image2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    unsigned int p = y + sy * x;
    unsigned int pxp = p + sy;
    unsigned int pyp = p + 1;

    dual_image0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
    dual_image1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
    dual_image2[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void d_hv_2d_dual(unsigned int sx, unsigned int sy, float dual_weight, float dual_weight_comp, float sqrt2, 
                  float* auxiliary_image, float* dual_image0, float* dual_image1, float* dual_image2, float* dual_image3)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= sx-1 || y < 1 || y >= sy-1)
    {
        return;
    }
    float dxx, dyy, dxy;
    int p, pxm, pxp, pym, pyp, pxyp;

    p = sy * x + y;
    pxm = p - sy;
    pxp = p + sy;
    pym = p - 1;
    pyp = p + 1;
    pxyp = pxp + 1;

    dxx = auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm];
    dual_image0[p] += dual_weight * dxx;

    dyy = auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym];
    dual_image1[p] += dual_weight * dyy;

    dxy = auxiliary_image[pxyp] - auxiliary_image[pxp] - auxiliary_image[pyp] + auxiliary_image[p];
    dual_image2[p] += sqrt2 * dual_weight * dxy;

    dual_image3[p] += dual_weight_comp * auxiliary_image[p];
}

__global__
void d_hv_dual_2d_normalize(unsigned int N, float inv_reg, float* dual_image0, float* dual_image1, 
                            float* dual_image2, float* dual_image3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt(dual_image0[i] * dual_image0[i] + dual_image1[i] * dual_image1[i] + dual_image2[i] * dual_image2[i] + dual_image3[i] * dual_image3[i]);
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

__global__
void d_sv_dual_2d_normalize(unsigned int N, float inv_reg, float* dual_image0, float* dual_image1, 
                            float* dual_image2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float tmp = inv_reg * sqrt(dual_image0[i] * dual_image0[i] + dual_image1[i] * dual_image1[i] + dual_image2[i] * dual_image2[i]);
        if (tmp > 1.0)
        {
            float inv_tmp = 1.0 / tmp;
            dual_image0[i] *= inv_tmp;
            dual_image1[i] *= inv_tmp;
            dual_image2[i] *= inv_tmp;
        }
    }
}

namespace SImg
{

    void cuda_spitfire2d_deconv_sv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, bool verbose, SObservable *observable)
    {
        int N = sx * sy;
        int Nfft = sx * (sy / 2 + 1);
        float sqrt2 = sqrt(2.0);

        // Optical transfer function and its adjoint
        float *OTFReal = new float[sx * sy];
        shift2D(psf, OTFReal, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));

        float *adjoint_PSF = new float[sx * sy];
        for (int x = 0; x < sx; ++x)
        {
            for (int y = 0; y < sy; ++y)
            {
                adjoint_PSF[y + sy * x] = psf[(sy - 1 - y) + sy * (sx - 1 - x)];
            }
        }

        float *adjoint_PSF_shift = new float[sx * sy];
        shift2D(adjoint_PSF, adjoint_PSF_shift, sx, sy, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2);
        float *adjoint_OTFReal = new float[sx * sy];
        shift2D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        delete[] adjoint_PSF_shift;
        delete[] adjoint_PSF;

        // Splitting parameters
        float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));
        float primal_step = 0.99 / (0.5 + (8 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);

        // Initializations
        float* dual_image0;
        float* dual_image1;
        float* dual_image2;
        float* auxiliary_image;
        float* residue_image;
        float* cu_deconv_image;
        float* cu_blurry_image;
        
        cudaMalloc ( &dual_image0, N*sizeof(float));
        cudaMalloc ( &dual_image1, N*sizeof(float));
        cudaMalloc ( &dual_image2, N*sizeof(float));
        cudaMalloc ( &auxiliary_image, N*sizeof(float));
        cudaMalloc ( &cu_deconv_image, N*sizeof(float));
        cudaMalloc ( &cu_blurry_image, N*sizeof(float));
        cudaMalloc ( &residue_image, N*sizeof(float));
        cudaMemcpy(cu_blurry_image, blurry_image, N*sizeof(float), cudaMemcpyHostToDevice); 

        // cuda threads blocs
        int blockSize1d = 256;
        int numBlocks1d = (N + blockSize1d - 1) / blockSize1d;
        int numBlocks1dfft = (Nfft + blockSize1d - 1) / blockSize1d;
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16);

        d_sv_init_2d_buffers<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, cu_blurry_image, dual_image0, dual_image1, dual_image2);
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
        cufftPlan2d(&Planfft, sx, sy, CUFFT_R2C);
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


        d_init_deconv_residu<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT, blurry_image_FT);

        cufftHandle Planifft;
        cufftPlan2d(&Planifft, sx, sy, CUFFT_C2R);
        //cudaDeviceSynchronize();
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
            d_copy<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, auxiliary_image);
            
            cudaDeviceSynchronize();
            cufftExecR2C(Planfft, (cufftReal*)cu_deconv_image, (cufftComplex*)deconv_image_FT);
            cudaDeviceSynchronize();

            d_copy_complex<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT);
            
            // Data term
            d_dataterm<<<numBlocks1dfft,blockSize1d>>>(Nfft, OTF, deconv_image_FT, blurry_image_FT, residue_image_FT, adjoint_OTF);
            cudaDeviceSynchronize();
            cufftExecC2R(Planifft, (cufftComplex*)residue_image_FT, (cufftReal*)residue_image);
            cudaDeviceSynchronize(); 

            // primal
            d_sv_primal_2d<<<gridSize2d, blockSize2d>>>(N, sx, sy, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                       cu_deconv_image, residue_image, dual_image0, dual_image1, 
                                                       dual_image2);
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
            d_dual_2d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_deconv_image);

            // dual    
            d_sv_2d_dual<<<gridSize2d, blockSize2d>>>(sx, sy, dual_weight, dual_weight_comp, sqrt2, 
                                                     auxiliary_image, dual_image0, 
                                                     dual_image1, dual_image2);                                        

            // normalize
            d_sv_dual_2d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_image0, dual_image1, 
                                                                 dual_image2);   

        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        cudaDeviceSynchronize();
        // free output
        cufftDestroy(Planfft);
        cufftDestroy(Planifft);
        cudaFree(dual_image0);
        cudaFree(dual_image1);
        cudaFree(dual_image2);
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

    void cuda_spitfire2d_deconv_hv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, bool verbose, SObservable *observable)
    {
        int N = sx * sy;
        int Nfft = sx * (sy / 2 + 1);
        float sqrt2 = sqrt(2.0);

        // Optical transfer function and its adjoint
        float *OTFReal = new float[sx * sy];
        shift2D(psf, OTFReal, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));

        float *adjoint_PSF = new float[sx * sy];
        for (int x = 0; x < sx; ++x)
        {
            for (int y = 0; y < sy; ++y)
            {
                adjoint_PSF[y + sy * x] = psf[(sy - 1 - y) + sy * (sx - 1 - x)];
            }
        }

        float *adjoint_PSF_shift = new float[sx * sy];
        shift2D(adjoint_PSF, adjoint_PSF_shift, sx, sy, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2);
        float *adjoint_OTFReal = new float[sx * sy];
        shift2D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
        delete[] adjoint_PSF_shift;
        delete[] adjoint_PSF;

        // Splitting parameters
        float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));
        float primal_step = 0.99 / (0.5 + (8 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
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
        dim3 blockSize2d(16, 16);
        dim3 gridSize2d = dim3((sx + 16 - 1) / 16, (sy + 16 - 1) / 16);

        d_hv_init_2d_buffers<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, cu_blurry_image, dual_image0, dual_image1, dual_image2, dual_image3);
        
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
        cufftPlan2d(&Planfft, sx, sy, CUFFT_R2C);
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


        d_init_deconv_residu<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT, blurry_image_FT);

        cufftHandle Planifft;
        cufftPlan2d(&Planifft, sx, sy, CUFFT_C2R);
        //cudaDeviceSynchronize();
        for (unsigned int iter = 0; iter < niter; iter++)
        {
            // Primal optimization
            d_copy<<<numBlocks1d, blockSize1d>>>(N, cu_deconv_image, auxiliary_image);
            
            cudaDeviceSynchronize();
            cufftExecR2C(Planfft, (cufftReal*)cu_deconv_image, (cufftComplex*)deconv_image_FT);
            cudaDeviceSynchronize();

            d_copy_complex<<<numBlocks1dfft,blockSize1d>>>(Nfft, deconv_image_FT, residue_image_FT);
            //cudaDeviceSynchronize();

            // Data term
            d_dataterm<<<numBlocks1dfft,blockSize1d>>>(Nfft, OTF, deconv_image_FT, blurry_image_FT, residue_image_FT, adjoint_OTF);
            cudaDeviceSynchronize();
            cufftExecC2R(Planifft, (cufftComplex*)residue_image_FT, (cufftReal*)residue_image);
            cudaDeviceSynchronize(); 

            // primal
            d_hv_primal_2d<<<gridSize2d, blockSize2d>>>(N, sx, sy, primal_step, primal_weight, primal_weight_comp, sqrt2, 
                                                       cu_deconv_image, residue_image, dual_image0, dual_image1, 
                                                       dual_image2, dual_image3);
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
            d_dual_2d_auxiliary<<<numBlocks1d, blockSize1d>>>(N, auxiliary_image, cu_deconv_image);
            //cudaDeviceSynchronize();

            // dual    
            d_hv_2d_dual<<<gridSize2d, blockSize2d>>>(sx, sy, dual_weight, dual_weight_comp, sqrt2, 
                                                     auxiliary_image, dual_image0, 
                                                     dual_image1, dual_image2, dual_image3);
            //cudaDeviceSynchronize();                                         

            // normalize
            d_hv_dual_2d_normalize<<<numBlocks1d, blockSize1d>>>(N, inv_reg, dual_image0, dual_image1, 
                                                                 dual_image2, dual_image3);
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

    void cuda_spitfire2d_deconv(float *blurry_image, unsigned int sx, unsigned int sy, float *psf, float *deconv_image, const float &regularization, const float &weighting, const unsigned int &niter, const std::string &method, bool verbose, SObservable *observable)
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
        normL2(blurry_image, sx, sy, 1, 1, 1, blurry_image_norm);

        // run denoising
        if (method == "SV")
        {
            cuda_spitfire2d_deconv_sv(blurry_image_norm, sx, sy, psf, deconv_image, regularization, weighting, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            cuda_spitfire2d_deconv_hv(blurry_image_norm, sx, sy, psf, deconv_image, regularization, weighting, niter, verbose, observable);
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