/// \file spitfire2d.cpp
/// \brief spitfire2d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire2d.h"

#include "math.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

#include <score>

namespace SImg{

void spitfire2d_sv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter)
{
    SObservable* observable = new SObservable();
    SObserverConsole* observer = new SObserverConsole();
    observable->addObserver(observer);
    spitfire2d_sv(noisy_image, sx, sy, denoised_image, regularization, weighting, niter, true, observable);
    delete observer;
    delete observable;
}

void spitfire2d_hv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter)
{
    SObservable* observable = new SObservable();
    SObserverConsole* observer = new SObserverConsole();
    observable->addObserver(observer);
    spitfire2d_hv(noisy_image, sx, sy, denoised_image, regularization, weighting, niter, true, observable);
    delete observer;
    delete observable;
}


void spitfire2d_sv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable  )
{
    unsigned int img_width = sx;
    unsigned int img_height = sy;
    unsigned int N = img_width*img_height;

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
    float* dual_images0 = (float*) malloc(sizeof(float) * N);
    float* dual_images1 = (float*) malloc(sizeof(float) * N);
    float* dual_images2 = (float*) malloc(sizeof(float) * N);
    float* auxiliary_image = (float*) malloc(sizeof(float) * N);
    float inv_reg = 1.0/regularization;

#pragma omp parallel for 
    for (unsigned int i = 0 ; i < N ; i++){
        denoised_image[i] = noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
    }

    // Deconvolution process
    for (int iter = 0; iter < niter; iter++) {

        // Primal optimization
#pragma omp parallel for 
        for (unsigned int i = 0 ; i < N ; i++){
            auxiliary_image[i] = denoised_image[i];
        }

#pragma omp parallel for 
        for(unsigned int x = 1 ; x < img_width-1 ; x++){
            for(unsigned int y = 1 ; y < img_height-1 ; y++){

                unsigned int p = img_height*x+y;
                unsigned int pxm = p - img_height;
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
        for(unsigned int i = 0 ; i < N ; i++){
            auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

#pragma omp parallel for 
        for(unsigned int x = 1 ; x < img_width-1 ; x++){
            for(unsigned int y = 1 ; y < img_height-1 ; y++){

                unsigned int p = img_height*x + y;
                unsigned int pxp = p + img_height;
                unsigned int pyp = p+1;

                dual_images0[p] += dual_weight * (auxiliary_image[pxp]- auxiliary_image[p]);
                dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
                dual_images2[p] += dual_weight_comp * auxiliary_image[p];
            }
        }

#pragma omp parallel for
        for( unsigned int i = 0 ; i < N ; ++i) 
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
    } // endfor (int iter = 0; iter < nb_iters_max; iter++)

    free(dual_images0);
    free(dual_images1);
    free(dual_images2);
    free(auxiliary_image);
    if (verbose){
        observable->notifyProgress(100);
    }
}

void spitfire2d_hv(float* noisy_image, unsigned int sx, unsigned int sy, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, bool verbose, SObservable* observable  )
{
#ifdef SL_USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif

    unsigned int img_width = sx;
    unsigned int img_height = sy;
    unsigned int N = img_width*img_height;
    float sqrt2 = sqrt(2.);

    // Splitting parameters
    float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
    float primal_step = 0.99 / (0.5 + (64 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);
    float primal_weight = primal_step * weighting;
    float primal_weight_comp = primal_step * (1 - weighting);
    float dual_weight = dual_step * weighting;
    float dual_weight_comp = dual_step * (1 - weighting);

    // Initializations
    //denoised_image = (float*) malloc(sizeof(float) * N);
    float* dual_images0 = (float*) malloc(sizeof(float) * N);
    float* dual_images1 = (float*) malloc(sizeof(float) * N);
    float* dual_images2 = (float*) malloc(sizeof(float) * N);
    float* dual_images3 = (float*) malloc(sizeof(float) * N);
    float* auxiliary_image = (float*) malloc(sizeof(float) * N);

    for (unsigned int i = 0 ; i < N ; ++i){
        denoised_image[i] = noisy_image[i];
        dual_images0[i] = 0.0;
        dual_images1[i] = 0.0;
        dual_images2[i] = 0.0;
        dual_images3[i] = 0.0;
    }

    // Deconvolution process
    float inv_reg = 1.0 / regularization;
    for (int iter = 0; iter < niter; ++iter) {
        // Primal optimization

#pragma omp parallel for 
        for (unsigned int i = 0 ; i < N ; ++i){
            auxiliary_image[i] = denoised_image[i];
        }

#pragma omp parallel for 
        for (unsigned int x = 1 ; x < img_width-1 ; ++x){
            for (unsigned int y = 1 ; y < img_height-1 ; ++y){

                float tmp, dxx_adj, dyy_adj, dxy_adj;
                int p, pxm, pxp, pym, pyp, pxym;    

                p = img_height*x+y;
                pxm = p - img_height;
                pxp = p + img_height;
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
        for (unsigned int i = 0 ; i < N ; ++i){
            auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

#pragma omp parallel for
        for (unsigned int x = 1 ; x < img_width-1 ; ++x){
            for (unsigned int y = 1 ; y < img_height-1 ; ++y){

                float dxx, dyy, dxy;
                int p, pxm, pxp, pym, pyp, pxyp; 

                p = img_height*x+y;
                pxm = p - img_height;
                pxp = p + img_height;
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
        }

#pragma omp parallel for
        for( unsigned int i = 0 ; i < N ; ++i) 
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
    } // endfor (int iter = 0; iter < nb_iters_max; iter++)

    free(dual_images0);
    free(dual_images1);
    free(dual_images2);
    free(dual_images3);
    free(auxiliary_image);
    observable->notifyProgress(100);
}

}