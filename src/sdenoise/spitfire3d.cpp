/// \file spitfire3d.cpp
/// \brief spitfire3d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire3d.h"

#include <score/SMath.h>
#include "math.h"

namespace SImg{

void spitfire3d_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& delta, bool verbose, SObservable* observable)
{
    unsigned int img_width  = sx;
    unsigned int img_height = sy;
    unsigned int img_depth  = sz;
    unsigned int N = img_width*img_height*img_depth;

    // Splitting parameters
    float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));

    float primal_step = 0.99 / (0.5 + (12 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);

    float primal_weight = primal_step * weighting;
    float primal_weight_comp = primal_step * (1 - weighting);
    float dual_weight = dual_step * weighting;
    float dual_weight_comp = dual_step * (1 - weighting);

    // Initializations
    denoised_image = (float*) malloc(sizeof(float) * N);
    float* dual_images0 = (float*) malloc(sizeof(float) * N);
    float* dual_images1 = (float*) malloc(sizeof(float) * N);
    float* dual_images2 = (float*) malloc(sizeof(float) * N);
    float* dual_images3 = (float*) malloc(sizeof(float) * N);
    float* auxiliary_image = (float*) malloc(sizeof(float) * N);

    // Denoising process
    float tmp, dx, dy, dz, min_val, max_val, dx_adj, dy_adj, dz_adj;
    min_val = 0.0;
    max_val = 1.0;
    int p, pxm, pym, pzm, pxp, pyp, pzp;
    for (int iter = 0; iter < niter; iter++) {
        // Primal optimization

        for (unsigned int i = 0 ; i < N ; i++){
            auxiliary_image[i] = denoised_image[i];
        }

        for(unsigned int x = 0 ; x < img_width ; x++){
            for(unsigned int y = 0 ; y < img_height ; y++){
                for(unsigned int z = 0 ; z < img_depth ; z++){

                    p = z + img_depth*(y + img_height*x);
                    pxm = z + img_depth*(y + img_height*(x-1));
                    pym = z + img_depth*(y-1 + img_height*x);
                    pzm = z-1 + img_depth*(y + img_height*x);

                    tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);

                    if (x > 0)
                        dx_adj = dual_images0[pxm] - dual_images0[p];
                    else
                        dx_adj = dual_images0[p];
                    if (y > 0)
                        dy_adj = dual_images1[pym] - dual_images1[p];
                    else
                        dy_adj = dual_images1[p];
                    if (z > 0)
                        dz_adj = delta*(dual_images2[pzm] - dual_images2[p]);
                    else
                        dz_adj = dual_images2[p];
                    tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj) + primal_weight_comp * dual_images3[p]);
                    denoised_image[p] = SMath::max(min_val, SMath::min(max_val, tmp));
                }
            }
        }

        // Stopping criterion
        if(verbose){
            if (iter % int(SMath::max(1, niter / 10)) == 0){
                observable->notifyProgress(100*(float(iter)/float(niter)));
            }
        }

        // Dual optimization
        for (unsigned int i = 0 ; i < N ; i++){
            auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

        for(unsigned int x = 0 ; x < img_width ; x++){
            for(unsigned int y = 0 ; y < img_height ; y++){
                for(unsigned int z = 0 ; z < img_depth ; z++){

                    p = z + img_depth*(y + img_height*x);
                    pxp = z + img_depth*(y + img_height*(x+1));
                    pyp = z + img_depth*(y+1 + img_height*x);
                    pzp = z+1 + img_depth*(y + img_height*x);

                    if (x < img_width - 1) {
                        dx = auxiliary_image[pxp] - auxiliary_image[p];
                        dual_images0[p] += dual_weight * dx;
                    }
                    if (y < img_height - 1) {
                        dy = auxiliary_image[pyp] - auxiliary_image[p];
                        dual_images1[p] += dual_weight * dy;
                    }
                    if (z < img_depth - 1) {
                        dz = delta*(auxiliary_image[pzp] - auxiliary_image[p]);
                        dual_images2[p] += dual_weight * dz;
                    }
                    dual_images3[p] += dual_weight_comp	* auxiliary_image[p];
                }
            }
        }

        for (unsigned int i = 0 ; i < N ; i++){

            float tmp = SMath::max(1., 1. / regularization * sqrt(pow(dual_images0[i], 2.)
                                                                    + pow(dual_images1[i], 2.)
                                                                    + pow(dual_images2[i], 2.)
                                                                    + pow(dual_images3[i], 2.)));
            dual_images0[i] /= tmp;
            dual_images1[i] /= tmp;
            dual_images2[i] /= tmp;
            dual_images3[i] /= tmp;
        }
    }

    delete dual_images0;
    delete dual_images1;
    delete dual_images2;
    delete dual_images3;
    delete auxiliary_image;
    observable->notifyProgress(100);
}

void spitfire3d_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& delta, bool verbose, SObservable* observable)
{
    unsigned int img_width  = sx;
    unsigned int img_height = sy;
    unsigned int img_depth  = sz;
    unsigned int N = img_width*img_height*img_depth;
    float sqrt2 = sqrt(2.);

    // Splitting parameters
    float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
    float primal_step = 0.99 / (0.5 + (144 * pow(weighting, 2.)
                                        + pow(1 - weighting, 2.)) * dual_step);

    float primal_weight = primal_step * weighting;
    float primal_weight_comp = primal_step * (1 - weighting);
    float dual_weight = dual_step * weighting;
    float dual_weight_comp = dual_step * (1 - weighting);

    // Initializations
    denoised_image = (float*) malloc(sizeof(float) * N);
    float* dual_images0 = (float*) malloc(sizeof(float) * N);
    float* dual_images1 = (float*) malloc(sizeof(float) * N);
    float* dual_images2 = (float*) malloc(sizeof(float) * N);
    float* dual_images3 = (float*) malloc(sizeof(float) * N);
    float* dual_images4 = (float*) malloc(sizeof(float) * N);
    float* dual_images5 = (float*) malloc(sizeof(float) * N);
    float* dual_images6 = (float*) malloc(sizeof(float) * N);
    float* auxiliary_image = (float*) malloc(sizeof(float) * N);

    // Denoising process
    float tmp, dxx, dyy, dzz, dxy, dyz, dzx, min_val, max_val, dxx_adj,
            dyy_adj, dzz_adj, dxy_adj, dyz_adj, dzx_adj;

    min_val = 0.0;
    max_val = 1.0;
    int p, pxm, pym, pzm, pxp, pyp, pzp;

    for (int iter = 0; iter < niter; iter++)
    {
        // Primal optimization
        for (unsigned int i = 0 ; i < N ; i++){
            auxiliary_image[i] = denoised_image[i];
        }

        for(unsigned int x = 0 ; x < img_width ; x++){
            for(unsigned int y = 0 ; y < img_height ; y++){
                for(unsigned int z = 0 ; z < img_depth ; z++){

                    p = z + img_depth*(y + img_height*x);
                    pxm = z + img_depth*(y + img_height*(x-1));
                    pym = z + img_depth*(y-1 + img_height*x);
                    pzm = z-1 + img_depth*(y + img_height*x);
                    pxp = z + img_depth*(y + img_height*(x+1));
                    pyp = z + img_depth*(y+1 + img_height*x);
                    pzp = z+1 + img_depth*(y + img_height*x);
                    tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);

                    dxx_adj = dyy_adj = dzz_adj = dxy_adj = dyz_adj = dzx_adj = 0.;
                    // Diagonal terms
                    if ((x > 0) && (x < img_width - 1))
                        dxx_adj = dual_images0[pxm]
                                - 2 * dual_images0[p]
                                + dual_images0[pxp];

                    if ((y > 0) && (y < img_height - 1))
                        dyy_adj = dual_images1[pym]
                                - 2 * dual_images1[p]
                                + dual_images1[pyp];

                    if ((z > 0) && (z < img_depth - 1))
                        dzz_adj = (delta*delta)*(dual_images2[pzm]
                                                     - 2 * dual_images2[p]
                                                     + dual_images2[pzp]);

                    // Other terms
                    if ((x == 0) && (y == 0))
                        dxy_adj = dual_images3[p];
                    if ((x > 0) && (y == 0))
                        dxy_adj = dual_images3[p] - dual_images3[pxm];
                    if ((x == 0) && (y > 0))
                        dxy_adj = dual_images3[p] - dual_images3[pym];
                    if ((x > 0) && (y > 0))
                        dxy_adj = dual_images3[p] - dual_images3[pxm]
                                - dual_images3[pym]
                                + dual_images3[z + img_depth*(y-1 + img_height*(x-1))];

                    if ((y == 0) && (z == 0))
                        dyz_adj = dual_images4[p];
                    if ((y > 0) && (z == 0))
                        dyz_adj = dual_images4[p] - dual_images4[pym];
                    if ((y == 0) && (z > 0))
                        dyz_adj = dual_images4[p] - dual_images4[pzm];
                    if ((y > 0) && (z > 0))
                        dyz = delta*(dual_images4[p] - dual_images4[pym]
                                       - dual_images4[pzm]
                                       + dual_images4[z-1 + img_depth*(y-1 + img_height*x)]);

                    if ((z == 0) && (x == 0))
                        dzx_adj = dual_images5[p];
                    if ((z > 0) && (x == 0))
                        dzx_adj = dual_images5[p] - dual_images5[pzm];
                    if ((z == 0) && (x > 0))
                        dzx_adj = dual_images5[pxm] - dual_images5[p];
                    if ((z > 0) && (x > 0))
                        dzx_adj = delta*(dual_images5[p] - dual_images5[pzm]
                                           - dual_images5[pxm]
                                           + dual_images5[z-1 + img_depth*(y + img_height*(x-1))]);
                    tmp -= (primal_weight
                            * (dxx_adj + dyy_adj + dzz_adj
                               + sqrt2 * (dxy_adj + dyz_adj + dzx_adj))
                            + primal_weight_comp * dual_images6[p]);
                    denoised_image[p] = SMath::max(min_val, SMath::min(max_val, tmp));
                }
            }
        }

        // Stopping criterion
        if(verbose){
            if (iter % int(SMath::max(1, niter / 10)) == 0){
                observable->notifyProgress(100*(float(iter)/float(niter)));
            }
        }

        // Dual optimization
        for (unsigned int i = 0 ; i  < N ; i++){
            auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

        for(unsigned int x = 0 ; x < img_width ; x++){
            for(unsigned int y = 0 ; y < img_height ; y++){
                for(unsigned int z = 0 ; z < img_depth ; z++){

                    p = z + img_depth*(y + img_height*x);
                    pxm = z + img_depth*(y + img_height*(x-1));
                    pym = z + img_depth*(y-1 + img_height*x);
                    pzm = z-1 + img_depth*(y + img_height*x);
                    pxp = z + img_depth*(y + img_height*(x+1));
                    pyp = z + img_depth*(y+1 + img_height*x);
                    pzp = z+1 + img_depth*(y + img_height*x);

                    if ((x > 0) && (x < img_width - 1)) {
                        dxx = auxiliary_image[pxp]
                                - 2 * auxiliary_image[p]
                                + auxiliary_image[pxm];
                        dual_images0[p] += dual_weight * dxx;
                    }
                    if ((y > 0) && (y < img_height - 1)) {
                        dyy = auxiliary_image[pyp]
                                - 2 * auxiliary_image[p]
                                + auxiliary_image[pym];
                        dual_images1[p] += dual_weight * dyy;
                    }
                    if ((z > 0) && (z < img_depth - 1)) {
                        dzz = (delta*delta)*(auxiliary_image[pzp]
                                             - 2 * auxiliary_image[p]
                                             + auxiliary_image[pzm]);
                        dual_images2[p] += dual_weight * dzz;
                    }
                    if ((x < img_width - 1) && (y < img_height - 1)) {
                        dxy = auxiliary_image[z + img_depth*(y+1 + img_height*(x+1))]
                                - auxiliary_image[pxp]
                                - auxiliary_image[pyp]
                                + auxiliary_image[p];
                        dual_images3[p] += sqrt2 * dual_weight * dxy;
                    }
                    if ((y < img_height - 1) && (z < img_depth - 1)) {
                        dyz = delta*(auxiliary_image[z+1 + img_depth*(y+1 + img_height*x)]
                                     - auxiliary_image[pyp]
                                     - auxiliary_image[pzp]
                                     + auxiliary_image[p]);
                        dual_images4[p] += sqrt2 * dual_weight * dyz;
                    }
                    if ((z < img_depth - 1) && (x < img_width - 1)) {
                        dzx = delta*(auxiliary_image[z+1 + img_depth*(y + img_height*(x+1))]
                                     - auxiliary_image[pxp]
                                     - auxiliary_image[pzp]
                                     + auxiliary_image[p]);
                        dual_images5[p] += sqrt2 * dual_weight * dzx;
                    }
                    dual_images6[p] += dual_weight_comp
                            * auxiliary_image[p];
                }
            }
        }

        for (unsigned int i = 0 ; i  < N ; i++){
            float tmp = SMath::max(1.,
                             1. / regularization
                             * sqrt(
                                 pow(dual_images0[i], 2.)
                             + pow(dual_images1[i], 2.)
                             + pow(dual_images2[i], 2.)
                             + pow(dual_images3[i], 2.)
                             + pow(dual_images4[i], 2.)
                             + pow(dual_images5[i], 2.)
                             + pow(dual_images6[i],
                                   2.)));
            dual_images0[i] /= tmp;
            dual_images1[i] /= tmp;
            dual_images2[i] /= tmp;
            dual_images3[i] /= tmp;
            dual_images4[i] /= tmp;
            dual_images5[i] /= tmp;
            dual_images6[i] /= tmp;
        }
    } // endfor (int iter = 0; iter < nb_iters_max; iter++)


    delete dual_images0;
    delete dual_images1;
    delete dual_images2;
    delete dual_images3;
    delete dual_images4;
    delete dual_images5;
    delete dual_images6;
    delete auxiliary_image;
    observable->notifyProgress(100);
}

}