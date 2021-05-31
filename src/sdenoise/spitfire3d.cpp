/// \file spitfire3d.cpp
/// \brief spitfire3d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire3d.h"

#include <score>
#include "math.h"

namespace SImg
{

    void spitfire3d_sv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &delta)
    {
        SObservable *observable = new SObservable();
        SObserverConsole *observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire3d_sv(noisy_image, sx, sy, sz, denoised_image, regularization, weighting, niter, delta, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire3d_hv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &delta)
    {
        SObservable *observable = new SObservable();
        SObserverConsole *observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire3d_hv(noisy_image, sx, sy, sz, denoised_image, regularization, weighting, niter, delta, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire3d_sv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &delta, bool verbose, SObservable *observable)
    {
        unsigned int N = sx * sy * sz;

        // Splitting parameters
        float dual_step = SMath::max(0.01, SMath::min(0.1, regularization));

        float primal_step = 0.99 / (0.5 + (12 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);

        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);
        float inv_reg = 1.0 / regularization;

        // Initializations
        float *dual_images0 = (float *)malloc(sizeof(float) * N);
        float *dual_images1 = (float *)malloc(sizeof(float) * N);
        float *dual_images2 = (float *)malloc(sizeof(float) * N);
        float *dual_images3 = (float *)malloc(sizeof(float) * N);
        float *auxiliary_image = (float *)malloc(sizeof(float) * N);

#pragma omp parallel for
        for (unsigned int i = 0; i < N; i++)
        {
            denoised_image[i] = noisy_image[i];
        }

        // Denoising process
        for (int iter = 0; iter < niter; iter++)
        {
// Primal optimization
#pragma omp parallel for
            for (unsigned int i = 0; i < N; i++)
            {
                auxiliary_image[i] = denoised_image[i];
            }
#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {

                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxm = p - sz * sy;
                        unsigned int pym = p - sz;
                        unsigned int pzm = p - 1;

                        float tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);

                        float dx_adj = dual_images0[pxm] - dual_images0[p];
                        float dy_adj = dual_images1[pym] - dual_images1[p];
                        float dz_adj = delta * (dual_images2[pzm] - dual_images2[p]);

                        tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj) + primal_weight_comp * dual_images3[p]);

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
                }
            }

            // notify iterations
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
#pragma omp parallel for
            for (unsigned int i = 0; i < N; i++)
            {
                auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
            }

#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {

                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxp = p + sz * sy;
                        unsigned int pyp = p + sz;
                        unsigned int pzp = p + 1;

                        dual_images0[p] += dual_weight * (auxiliary_image[pxp] - auxiliary_image[p]);
                        dual_images1[p] += dual_weight * (auxiliary_image[pyp] - auxiliary_image[p]);
                        dual_images2[p] += dual_weight * (delta * (auxiliary_image[pzp] - auxiliary_image[p]));
                        dual_images3[p] += dual_weight_comp * auxiliary_image[p];
                    }
                }
            }

#pragma omp parallel for
            for (unsigned int i = 0; i < N; i++)
            {
                float tmp = inv_reg * sqrt(dual_images0[i] * dual_images0[i] + dual_images1[i] * dual_images1[i] + dual_images2[i] * dual_images2[i] + dual_images3[i] * dual_images3[i]);
                if (tmp > 1.0)
                {
                    float inv_tmp = 1.0 / tmp;
                    dual_images0[i] *= inv_tmp;
                    dual_images1[i] *= inv_tmp;
                    dual_images2[i] *= inv_tmp;
                    dual_images3[i] *= inv_tmp;
                }
            }
        } // endfor (int iter = 0; iter < nb_iters_max; iter++)

        delete dual_images0;
        delete dual_images1;
        delete dual_images2;
        delete dual_images3;
        delete auxiliary_image;
        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void spitfire3d_hv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &delta, bool verbose, SObservable *observable)
    {
        unsigned int N = sx * sy * sz;
        float sqrt2 = sqrt(2.);

        // Splitting parameters
        float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
        float primal_step = 0.99 / (0.5 + (144 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);

        float primal_weight = primal_step * weighting;
        float primal_weight_comp = primal_step * (1 - weighting);
        float dual_weight = dual_step * weighting;
        float dual_weight_comp = dual_step * (1 - weighting);
        float inv_reg = 1.0 / regularization;

        // Initializations
        float *dual_images0 = (float *)malloc(sizeof(float) * N);
        float *dual_images1 = (float *)malloc(sizeof(float) * N);
        float *dual_images2 = (float *)malloc(sizeof(float) * N);
        float *dual_images3 = (float *)malloc(sizeof(float) * N);
        float *dual_images4 = (float *)malloc(sizeof(float) * N);
        float *dual_images5 = (float *)malloc(sizeof(float) * N);
        float *dual_images6 = (float *)malloc(sizeof(float) * N);
        float *auxiliary_image = (float *)malloc(sizeof(float) * N);

        // Denoising process
        //float tmp, dxx, dyy, dzz, dxy, dyz, dzx, min_val, max_val, dxx_adj,
        //    dyy_adj, dzz_adj, dxy_adj, dyz_adj, dzx_adj;
        //min_val = 0.0;
        //max_val = 1.0;
        //int p, pxm, pym, pzm, pxp, pyp, pzp;

        for (int iter = 0; iter < niter; iter++)
        {
// Primal optimization
#pragma omp parallel for
            for (unsigned int i = 0; i < N; i++)
            {
                auxiliary_image[i] = denoised_image[i];
            }

#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; x++)
            {
                for (unsigned int y = 1; y < sy - 1; y++)
                {
                    for (unsigned int z = 1; z < sz - 1; z++)
                    {

                        unsigned int p = z + sz * (y + sy * x);
                        unsigned int pxm = p - sz * sy;
                        unsigned int pym = p - sz;
                        unsigned int pzm = p - 1;
                        unsigned int pxp = p + sz * sy;
                        unsigned int pyp = p + sz;
                        unsigned int pzp = p + 1;

                        float tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]);

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
                }
            }

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
#pragma omp parallel for            
            for (unsigned int i = 0; i < N; i++)
            {
                auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
            }

#pragma omp parallel for
            for (unsigned int x = 1; x < sx-1; x++)
            {
                for (unsigned int y = 1; y < sy-1; y++)
                {
                    for (unsigned int z = 1; z < sz-1; z++)
                    {

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
                }
            }

#pragma omp parallel for
            for (unsigned int i = 0; i < N; i++)
            {
                float tmp = inv_reg * sqrt(dual_images0[i] * dual_images0[i] + dual_images1[i] * dual_images1[i] + dual_images2[i] * dual_images2[i] + 
                                           dual_images3[i] * dual_images3[i] + dual_images4[i] * dual_images4[i] + dual_images4[i] * dual_images4[i] + 
                                           dual_images6[i] * dual_images6[i]);
                if (tmp > 1.0)
                {
                    float inv_tmp = 1.0 / tmp;
                    dual_images0[i] *= inv_tmp;
                    dual_images1[i] *= inv_tmp;
                    dual_images2[i] *= inv_tmp;
                    dual_images3[i] *= inv_tmp;
                    dual_images4[i] *= inv_tmp;
                    dual_images5[i] *= inv_tmp;
                    dual_images6[i] *= inv_tmp;
                }
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
        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

}