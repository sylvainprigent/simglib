/// \file spitfire4d.cpp
/// \brief spitfire4d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire4d.h"

#include "math.h"
#include <score>

namespace SImg
{

    void spitfire4d_sv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &deltaz, const float &deltat)
    {
        SObservable *observable = new SObservable();
        SObserverConsole *observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire4d_sv(noisy_image, sx, sy, sz, st, denoised_image, regularization, weighting, niter, deltaz, deltat, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire4d_hv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &deltaz, const float &deltat)
    {
        SObservable *observable = new SObservable();
        SObserverConsole *observer = new SObserverConsole();
        observable->addObserver(observer);
        spitfire4d_hv(noisy_image, sx, sy, sz, st, denoised_image, regularization, weighting, niter, deltaz, deltat, true, observable);
        delete observer;
        delete observable;
    }

    void spitfire4d_sv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &deltaz, const float &deltat, bool verbose, SObservable *observable)
    {
        // get the image buffer and size
        unsigned int N = sx * sy * sz * st;

        // get the average intensity
        float Average_IN_i = 0.;
        float Max_i = noisy_image[0];
        for (int ind = 0; ind < N; ind++)
        {
            Average_IN_i += noisy_image[ind];
            if (noisy_image[ind] > Max_i)
            {
                Max_i = noisy_image[ind];
            }
        }
        Average_IN_i /= float(sx * sy * sz * st);

        // Splitting parameters
        double dual_step = SMath::max(0.01, SMath::min(0.1, regularization));

        double primal_step = 0.99 / (0.5 + (16 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step);

        double primal_weight = primal_step * weighting;
        double primal_weight_comp = primal_step * (1 - weighting);
        double dual_weight = dual_step * weighting;
        double dual_weight_comp = dual_step * (1 - weighting);
        float inv_reg = 1.0 / regularization;

        // Initializations
        float *dual_images0 = new float[N];
        float *dual_images1 = new float[N];
        float *dual_images2 = new float[N];
        float *dual_images3 = new float[N];
        float *dual_images4 = new float[N];
        float *auxiliary_image = new float[N];

#pragma omp parallel for
        for (unsigned int i = 0; i < N; ++i)
        {
            denoised_image[i] = noisy_image[i];
        }

        // Denoising process
        for (int iter = 0; iter < niter; iter++)
        {
// Primal optimization
#pragma omp parallel for
            for (unsigned int i = 0; i < N; ++i)
            {
                auxiliary_image[i] = denoised_image[i];
            }

#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; ++x)
            {
                for (unsigned int y = 1; y < sy - 1; ++y)
                {
                    for (unsigned int z = 1; z < sz - 1; ++z)
                    {
                        for (unsigned int t = 1; t < st - 1; ++t)
                        {

                            unsigned int p = t + st * (z + sz * (y + sy * x));
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
            for (unsigned int i = 0; i < N; ++i)
            {
                auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
            }

#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; ++x)
            {
                for (unsigned int y = 1; y < sy - 1; ++y)
                {
                    for (unsigned int z = 1; z < sz - 1; ++z)
                    {
                        for (unsigned int t = 1; t < st - 1; ++t)
                        {

                            unsigned int p = t + st * (z + sz * (y + sy * x));
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
                    }
                }
            }

#pragma omp parallel for
            for (unsigned int i = 0; i < N; ++i)
            {
                float tmp = inv_reg * sqrt(dual_images0[i] * dual_images0[i] + dual_images1[i] * dual_images1[i] + dual_images2[i] * dual_images2[i] +
                                           dual_images3[i] * dual_images3[i] + dual_images4[i] * dual_images4[i]);
                if (tmp > 1.0)
                {
                    float inv_tmp = 1.0 / tmp;
                    dual_images0[i] *= inv_tmp;
                    dual_images1[i] *= inv_tmp;
                    dual_images2[i] *= inv_tmp;
                    dual_images3[i] *= inv_tmp;
                    dual_images4[i] *= inv_tmp;
                }
            }
        } // enditer

        // normalize intensity
        float Average_IN_o = 0.;
        for (unsigned int ind = 0; ind < N; ind++)
        {
            Average_IN_o += denoised_image[ind];
        }
        Average_IN_o /= float(N);

#pragma omp parallel for
        for (int ind = 0; ind < N; ind++)
        {
            denoised_image[ind] += (Average_IN_i - Average_IN_o);
        }

        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

    void spitfire4d_hv(float *noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float *denoised_image, const float &regularization, const float &weighting, const unsigned int &niter, const float &deltaz, const float &deltat, bool verbose, SObservable *observable)
    {
        // get the image buffer and size
        unsigned int N = sx * sy * sz * st;

        // get the average intensity
        float Average_IN_i = 0.;
        float Max_i = noisy_image[0];
        for (unsigned int ind = 0; ind < N; ind++)
        {
            Average_IN_i += noisy_image[ind];
            if (noisy_image[ind] > Max_i)
            {
                Max_i = noisy_image[ind];
            }
        }
        Average_IN_i /= float(N);

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
        float *dual_images0 = new float[N];
        float *dual_images1 = new float[N];
        float *dual_images2 = new float[N];
        float *dual_images3 = new float[N];
        float *dual_images4 = new float[N];
        float *dual_images5 = new float[N];
        float *dual_images6 = new float[N];
        float *dual_images7 = new float[N];
        float *dual_images8 = new float[N];
        float *dual_images9 = new float[N];
        float *dual_images10 = new float[N];
        float *auxiliary_image = new float[N];

#pragma omp parallel for
        for (unsigned int i = 0; i < N; ++i)
        {
            denoised_image[i] = noisy_image[i];
        }

        // Denoising process
        //float tmp, dxx, dyy, dzz, dtt, dxy, dyz, dty, dzx, dtx, dtz, min_val = 0., max_val = (float)(Max_i), dxx_adj,
        //                                                             dyy_adj, dzz_adj, dtt_adj, dxy_adj, dyz_adj, dzx_adj, dtx_adj, dty_adj, dtz_adj;
        //unsigned int p, pxm, pym, pzm, ptm, pxp, pyp, pzp, ptp, pxym, pyzm, pxzm, pxtm, pytm, pztm;
        for (int iter = 0; iter < niter; iter++)
        {
// Primal optimization
//std::cout << "Primal optimization" << std::endl;
#pragma omp parallel for
            for (unsigned int i = 0; i < N; ++i)
            {
                auxiliary_image[i] = denoised_image[i];
            }
#pragma omp parallel for
            for (unsigned int x = 1; x < sx - 1; ++x)
            {
                for (unsigned int y = 1; y < sy - 1; ++y)
                {
                    for (unsigned int z = 1; z < sz - 1; ++z)
                    {
                        for (unsigned int t = 1; t < st - 1; ++t)
                        {

                            unsigned int p = t + st * (z + sz * (y + sy * x));
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
                    }
                }
            }

            // iterations
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
            //std::cout << "Dual optimization" << std::endl;
#pragma omp parallel for
            for (unsigned int i = 0; i < N; ++i)
            {
                auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
            }
#pragma omp parallel for
            for (unsigned int x = 1; x < sx-1; ++x)
            {
                for (unsigned int y = 1; y < sy-1; ++y)
                {
                    for (unsigned int z = 1; z < sz-1; ++z)
                    {
                        for (unsigned int t = 1; t < st-1; ++t)
                        {

                            unsigned int p = t + st * (z + sz * (y + sy * x));
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
                    }
                }
            }
#pragma omp parallel for
            for (unsigned int i = 0; i < N; ++i)
            {
                float tmp = inv_reg * sqrt( dual_images0[i]*dual_images0[i] + dual_images1[i]*dual_images1[i] + dual_images2[i]*dual_images2[i] + 
                                            dual_images3[i]*dual_images3[i] + dual_images4[i]*dual_images4[i] + dual_images5[i]*dual_images5[i] + 
                                            dual_images6[i]*dual_images6[i] + dual_images7[i]*dual_images7[i] + dual_images8[i]*dual_images8[i] + 
                                            dual_images9[i]*dual_images9[i] + dual_images10[i]*dual_images10[i]);
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
        } // end iter

        // normalize intensity
        float Average_IN_o = 0.;
        for (unsigned int ind = 0; ind < N; ind++)
        {
            Average_IN_o += denoised_image[ind];
        }
        Average_IN_o /= float(N);

        for (unsigned int ind = 0; ind < N; ind++)
        {
            denoised_image[ind] += (Average_IN_i - Average_IN_o);
        }
        if (verbose)
        {
            observable->notifyProgress(100);
        }
    }

}
