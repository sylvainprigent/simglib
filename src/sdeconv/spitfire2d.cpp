/// \file spitfire2d.cpp
/// \brief spitfire2d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire2d.h"

#include <smanipulate>
#include <score/SMath.h>
#include <sfft/SFFT.h>
#include <sfft/SFFTConvolutionFilter.h>

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void spitfire2d_deconv_sv(float* blurry_image, unsigned int sx, unsigned int sy, float* psf, float* deconv_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable)
{
    int N = sx*sy;
    int Nfft = sx*(sy/2+1);

#ifdef SL_USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
    int fftThreads = fftwf_init_threads();
    if (fftThreads == 0){
        this->notify("Cannot initialize parallel fft: error ");
    }
#endif

    // Optical transfer function and its adjoint
    float *OTFReal = new float[sx*sy];
    shift2D(psf, OTFReal, sx, sy, int(- float(sx) / 2.0), int(-float(sy) / 2.0));

    float* adjoint_PSF = new float[sx*sy];
#pragma omp parallel for
    for (int x = 0 ; x < sx ; ++x){
        for (int y = 0 ; y < sy ; ++y){
            adjoint_PSF[y + sy*x] = psf[(sy - 1 - y) + sy*(sx - 1 - x)];
        }
    }

    float* adjoint_PSF_shift = new float[sx*sy];
    shift2D(adjoint_PSF, adjoint_PSF_shift, sx, sy, -(int(sx) - 1) % 2, -(int(sy) - 1) % 2);
    float* adjoint_OTFReal = new float[sx*sy];
    shift2D(adjoint_PSF_shift, adjoint_OTFReal, sx, sy, int(-float(sx) / 2.0), int(-float(sy) / 2.0));
    delete[] adjoint_PSF_shift;

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
    float* deblurred_image = (float*) malloc(sizeof(float) * N);
    float* dual_image0 = (float*) malloc(sizeof(float) * N);
    float* dual_image1 = (float*) malloc(sizeof(float) * N);
    float* dual_image2 = (float*) malloc(sizeof(float) * N);
    float* dual_image3 = (float*) malloc(sizeof(float) * N);
    float* auxiliary_image = (float*) malloc(sizeof(float) * N);
    float* residue_image = (float*) malloc(sizeof(float) * unsigned(N));

#pragma omp parallel for
    for (int i = 0 ; i < N ; i++){
        dual_image0[i] = 0.0;
        dual_image1[i] = 0.0;
        dual_image2[i] = 0.0;
        dual_image3[i] = 0.0;
    }

    // init deblurred_image as input image
#pragma omp parallel for
    for (int i = 0 ; i < N ; i++){
        deblurred_image[i] = blurry_image[i];
    }

    fftwf_complex* blurry_image_FT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
    fftwf_complex* deblurred_image_FT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
    fftwf_complex* residue_image_FT = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
    fftwf_complex* OTF = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));
    fftwf_complex* adjoint_OTF = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * unsigned(Nfft));

    fft2D(blurry_image, blurry_image_FT, sx, sy);
    fft2D(OTFReal, OTF, sx, sy);
    fft2D(adjoint_OTFReal, adjoint_OTF, sx, sy);

    delete[] OTFReal;
    delete[] adjoint_OTFReal;

#pragma omp parallel for
    for (int i = 0 ; i < Nfft ; i++){
        deblurred_image_FT[i][0] = blurry_image_FT[i][0];
        deblurred_image_FT[i][1] = blurry_image_FT[i][1];
        residue_image_FT[i][0] = blurry_image_FT[i][0];
        residue_image_FT[i][1] = blurry_image_FT[i][1];
    }

    // Deconvolution process
    float inv_reg = 1.0 / regularization;

    for (unsigned int iter = 0; iter < niter; iter++) {
        // Primal optimization
#pragma omp parallel for        
        for (int i = 0 ; i < N ; ++i){
            auxiliary_image[i] = deblurred_image[i];
        }
        fft2D(deblurred_image, deblurred_image_FT, sx, sy);

#pragma omp parallel for
        for (int i = 0 ; i < Nfft ; i++){
            residue_image_FT[i][0] = deblurred_image_FT[i][0];
            residue_image_FT[i][1] = deblurred_image_FT[i][1];
        }

        // Data term
#pragma omp parallel for
        for (int i = 0 ; i < Nfft ; i++){
            float real_tmp = OTF[i][0] * deblurred_image_FT[i][0]
                    - OTF[i][1] * deblurred_image_FT[i][1]
                    - blurry_image_FT[i][0];
            float imag_tmp = OTF[i][0] * deblurred_image_FT[i][1]
                    + OTF[i][1] * deblurred_image_FT[i][0]
                    - blurry_image_FT[i][1];

            residue_image_FT[i][0] = adjoint_OTF[i][0] * real_tmp
                    - adjoint_OTF[i][1] * imag_tmp;
            residue_image_FT[i][1] = adjoint_OTF[i][0] * imag_tmp
                    + adjoint_OTF[i][1] * real_tmp;

        }

        ifft2D(residue_image_FT, residue_image, sx, sy);

#pragma omp parallel for
        for (unsigned int x = 1 ; x < sx-1 ; ++x){
            for (unsigned int y = 1 ; y < sy-1 ; ++y){

                unsigned int p = y + sy * x;
                float tmp = deblurred_image[p] - primal_step * residue_image[p]/float(N);

                unsigned int pxm = p - sy;
                unsigned int pym = p-1;
                float dx_adj = dual_image0[pxm] - dual_image0[p];
                float dy_adj = dual_image1[pym] - dual_image1[p];
                
                tmp -= (primal_weight * (dx_adj + dy_adj)
                        + primal_weight_comp * dual_image2[p]);

                if (tmp > 1.0){
                    deblurred_image[p] = 1.0;
                }
                else if (tmp < 0.0 ){
                    deblurred_image[p] = 0.0;
                }
                else{
                    deblurred_image[p] = tmp;
                } 
            }
        }

        // Stopping criterion
        if (verbose){
            int iter_n = niter / 10;
            if (iter_n < 1) iter_n = 1;
            if (iter % iter_n == 0){
                observable->notifyProgress(100*(float(iter)/float(niter)));
            }
        }

        // Dual optimization
#pragma omp parallel for
        for (int i = 0 ; i < N ; i++){
            auxiliary_image[i] = 2 * deblurred_image[i] - auxiliary_image[i];
        }

#pragma omp parallel for
        for (unsigned int x = 1 ; x < sx-1 ; ++x){
            for (unsigned int y = 1 ; y < sy-1 ; ++y){

                unsigned int p = y + sy * x;
                unsigned int pxp = p + sy;
                unsigned int pyp = p+1;

                dual_image0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
                dual_image1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
                dual_image2[p] += dual_weight_comp * auxiliary_image[p];
            }
        }

#pragma omp parallel for
        for( unsigned int i = 0 ; i < N ; ++i) 
        {
            float tmp = inv_reg * sqrt( dual_image0[i]*dual_image0[i] + dual_image1[i]*dual_image1[i] + dual_image2[i]*dual_image2[i]);
            if (tmp > 1.0)
            {
                float inv_tmp = 1.0/tmp;
                dual_image0[i] *= inv_tmp;
                dual_image1[i] *= inv_tmp;
                dual_image2[i] *= inv_tmp;
            }
        }

    } // endfor (int iter = 0; iter < nb_iters_max; iter++)

    // copy output
    free(dual_image0);
    free(dual_image1);
    free(dual_image2);
    free(dual_image3);
    free(auxiliary_image);
    free(residue_image);

    fftwf_free(blurry_image_FT);
    fftwf_free(deblurred_image_FT);
    fftwf_free(residue_image_FT);
    fftwf_free(OTF);
    fftwf_free(adjoint_OTF);

    if (verbose){
        observable->notifyProgress(100);
    }
}

void spitfire2d_deconv_hv(float* blurry_image, unsigned int sx, unsigned int sy, float* psf, float* deconv_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable)
{

}

}