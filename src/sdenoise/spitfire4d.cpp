/// \file spitfire4d.cpp
/// \brief spitfire4d implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spitfire4d.h"

#include <score/SMath.h>
#include "math.h"

namespace SImg{

void spitfire4d_sv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat, bool verbose, SObservable* observable)
{
    // get the image buffer and size
    unsigned int w = sx;
    unsigned int h = sy;
    unsigned int d = sz;
    unsigned int T = st; 
    unsigned int buffer_size = w*h*d*T;

    // get the average intensity
    float Average_IN_i= 0.;
    float Max_i = noisy_image[0];
	for (int ind=0; ind<w*h*d*T; ind++)
    {
        Average_IN_i += noisy_image[ind];
        if (noisy_image[ind] > Max_i){
            Max_i = noisy_image[ind];  
        }
    }
	Average_IN_i /= float(w*h*d*T);

	// Splitting parameters
	double dual_step = SMath::max(0.01, SMath::min(0.1, regularization));
	
	double primal_step = 0.99 / (0.5 + (16 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step); 
	
	double primal_weight = primal_step * weighting;
	double primal_weight_comp = primal_step * (1 - weighting);
	double dual_weight = dual_step * weighting;
	double dual_weight_comp = dual_step * (1 - weighting);

    // Initializations
    denoised_image = new float[w*h*d*T];
    float* dual_images0 = new float[w*h*d*T];
    float* dual_images1 = new float[w*h*d*T];
    float* dual_images2 = new float[w*h*d*T];
    float* dual_images3 = new float[w*h*d*T];
    float* dual_images4 = new float[w*h*d*T];
    float* auxiliary_image = new float[w*h*d*T];

    // Denoising process
	float tmp, dx, dy, dz, dt, min_val=0., max_val=(float)(Max_i), dx_adj, dy_adj, dz_adj, dt_adj;
	for (int iter = 0; iter < niter; iter++) 
    {
		// Primal optimization
        for (unsigned int i = 0 ; i < buffer_size ; ++i){
		    auxiliary_image[i] = denoised_image[i];
        }

        for (unsigned int x = 0 ; x < w ; ++x){
            for (unsigned int y = 0 ; y < h ; ++y){
                for (unsigned int z = 0 ; z < d ; ++z){
                    for (unsigned int t = 0 ; t < T ; ++t){
                
                        unsigned int p = t + T*(z + d*(y + h*x)); 
                        tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]); 
                        
                        if (x > 0){
                            unsigned int pxm = t + T*(z + d*(y + h*(x-1))); 
                            dx_adj = dual_images0[pxm] - dual_images0[p];
                        }
                        else{
                            dx_adj = dual_images0[p]; 
                        }
                        if (y > 0){
                            unsigned int pym = t + T*(z + d*((y-1) + h*x)); 
                            dy_adj = dual_images1[pym] - dual_images1[p];
                        }
                        else{
                            dy_adj = dual_images1[p];
                        }
                        if (z > 0){
                            unsigned int pzm = t + T*((z-1) + d*(y + h*x));
                            dz_adj = deltaz*(dual_images2[pzm] - dual_images2[p]);
                        }
                        else{
                            dz_adj = dual_images2[p];
                        }
                        if (t > 0){
                            unsigned int ptm = t-1 + T*(z + d*(y + h*x));
                            dt_adj = deltat*(dual_images3[ptm] - dual_images3[p]);
                        }
                        else{
                            dt_adj = dual_images3[p];
                        }
                        
                        tmp -= (primal_weight * (dx_adj + dy_adj + dz_adj + dt_adj) + primal_weight_comp * dual_images4[p]);
                        denoised_image[p] = SMath::max(min_val, SMath::min(max_val, tmp));
		            }
                }
            }
        }

        // iterations
        if(verbose){
            if (iter % int(SMath::max(1, niter / 10)) == 0){
                observable->notifyProgress(100*(float(iter)/float(niter)));
            }
        }

        // Dual optimization
        for (unsigned int i = 0 ; i < buffer_size ; ++i)
        {
            auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

        for (unsigned int x = 0 ; x < w ; ++x){
            for (unsigned int y = 0 ; y < h ; ++y){
                for (unsigned int z = 0 ; z < d ; ++z){
                    for (unsigned int t = 0 ; t < T ; ++t){

                        unsigned int p = t + T*(z + d*(y + h*x));

                        if (x < w - 1) {
                            unsigned int pxp = t + T*(z + d*(y + h*(x+1)));
                            dx = auxiliary_image[pxp] - auxiliary_image[p];
                            dual_images0[p] += dual_weight * dx;
                        }
                        if (y < h - 1) {
                            unsigned int pyp = t + T*(z + d*(y+1 + h*x));
                            dy = auxiliary_image[pyp] - auxiliary_image[p];
                            dual_images1[p] += dual_weight * dy;
                        }
                        if (z < d - 1) {
                            unsigned int pzp = t + T*(z+1 + d*(y + h*x));
                            dz = deltaz*(auxiliary_image[pzp] - auxiliary_image[p]);
                            dual_images2[p] += dual_weight * dz;
                        }
                        if (t < T - 1) {
                            unsigned int ptp = t+1 + T*(z + d*(y + h*x));
                            dt = deltat*(auxiliary_image[ptp] - auxiliary_image[p]);
                            dual_images3[p] += dual_weight * dt;
                        }
                        dual_images4[p] += dual_weight_comp	* auxiliary_image[p];
                    }
                }
            }
		}

        for (unsigned int i = 0 ; i < buffer_size ; ++i)
		{
			double tmp = SMath::max(1., 1. / regularization * sqrt(pow(dual_images0[i], 2.)
											+ pow(dual_images1[i], 2.)
											+ pow(dual_images2[i], 2.)
											+ pow(dual_images3[i], 2.)
											+ pow(dual_images4[i], 2.)
												));
			dual_images0[i] /= tmp;
			dual_images1[i] /= tmp;
			dual_images2[i] /= tmp;
			dual_images3[i] /= tmp;
			dual_images4[i] /= tmp;
		}
    } // enditer

    // normalize intensity
    float Average_IN_o = 0.;
    for (unsigned int ind=0; ind<buffer_size; ind++){
        Average_IN_o += denoised_image[ind];
    }
    Average_IN_o /= float(buffer_size);

    for (int ind=0; ind<w*h*d*T; ind++){
        denoised_image[ind] += (Average_IN_i-Average_IN_o);
    }

    if (verbose){
        observable->notifyProgress(100);
    }
}

void spitfire4d_hv(float* noisy_image, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int st, float* denoised_image, const float& regularization, const float& weighting, const unsigned int& niter, const float& deltaz, const float& deltat, bool verbose, SObservable* observable)
{
        // get the image buffer and size
    unsigned int w = sx;
    unsigned int h = sy;
    unsigned int d = sz;
    unsigned int T = st; 
    unsigned int buffer_size = w*h*d*T;

    // get the average intensity
    float Average_IN_i= 0.;
    float Max_i = noisy_image[0];
	for (unsigned int ind=0; ind<w*h*d*T; ind++)
    {
        Average_IN_i += noisy_image[ind];
        if (noisy_image[ind] > Max_i){
            Max_i = noisy_image[ind];  
        }
    }
	Average_IN_i /= float(w*h*d*T);

    // Splitting parameters
    float sqrt2 = sqrt(2.);
	float dual_step = SMath::max(0.001, SMath::min(0.01, regularization));
	float primal_step = 0.99 / (0.5 + (256 * pow(weighting, 2.) + pow(1 - weighting, 2.)) * dual_step); 
	float primal_weight = primal_step * weighting;
	float primal_weight_comp = primal_step * (1 - weighting);
	float dual_weight = dual_step * weighting;
	float dual_weight_comp = dual_step * (1 - weighting);

    // Initializations
    denoised_image = new float[w*h*d*T];
    float* dual_images0 = new float[w*h*d*T];
    float* dual_images1 = new float[w*h*d*T];
    float* dual_images2 = new float[w*h*d*T];
    float* dual_images3 = new float[w*h*d*T];
    float* dual_images4 = new float[w*h*d*T];
    float* dual_images5 = new float[w*h*d*T];
    float* dual_images6 = new float[w*h*d*T];
    float* dual_images7 = new float[w*h*d*T];
    float* dual_images8 = new float[w*h*d*T];
    float* dual_images9 = new float[w*h*d*T];
    float* dual_images10 = new float[w*h*d*T];
    float* auxiliary_image = new float[w*h*d*T];

    for (unsigned int i = 0 ; i < buffer_size ; ++i)
    {
        denoised_image[i] = noisy_image[i];
    }

    // Denoising process
	float tmp, dxx, dyy, dzz, dtt, dxy, dyz, dty, dzx, dtx, dtz, min_val=0., max_val=(float)(Max_i), dxx_adj,
			dyy_adj, dzz_adj, dtt_adj, dxy_adj, dyz_adj, dzx_adj, dtx_adj, dty_adj, dtz_adj;
	unsigned int p, pxm, pym, pzm, ptm, pxp, pyp, pzp, ptp, pxym, pyzm, pxzm, pxtm, pytm, pztm;		
	for (int iter = 0; iter < niter; iter++) 
	{
        // Primal optimization
        //std::cout << "Primal optimization" << std::endl;
        for (unsigned int i = 0 ; i < buffer_size ; ++i){
		    auxiliary_image[i] = denoised_image[i];
        }

        for (unsigned int x = 0 ; x < w ; ++x){
            for (unsigned int y = 0 ; y < h ; ++y){
                for (unsigned int z = 0 ; z < d ; ++z){
                    for (unsigned int t = 0 ; t < T ; ++t){

                        p = t + T*(z + d*(y + h*x));
                        pxm = t + T*(z + d*(y + h*(x-1)));
                        pym = t + T*(z + d*(y-1 + h*x));
                        pzm = t + T*(z-1 + d*(y + h*x));
                        ptm = t-1 + T*(z + d*(y + h*x));

                        tmp = denoised_image[p] - primal_step * (denoised_image[p] - noisy_image[p]); 

                        dxx_adj = dyy_adj = dzz_adj = dtt_adj = dxy_adj = dyz_adj = dzx_adj = dty_adj = dtx_adj = dtz_adj = 0.;
                        // Diagonal terms
                        if ((x > 0) && (x < w - 1)){
                            pxp = t + T*(z + d*(y + h*(x+1)));
                            dxx_adj = dual_images0[pxm] - 2 * dual_images0[p] + dual_images0[pxp];
                        }

                        if ((y > 0) && (y < h - 1)){
                            pyp = t + T*(z + d*(y+1 + h*x));
                            dyy_adj = dual_images1[pym] - 2 * dual_images1[p] + dual_images1[pyp];
                        }

                        if ((z > 0) && (z < d - 1)){
                            pzp = t + T*(z+1 + d*(y + h*x));
                            dzz_adj = (deltaz*deltaz)*(dual_images2[pzm] - 2 * dual_images2[p] + dual_images2[pzp]);
                        }

                        if ((t > 0) && (t < T - 1)){
                            ptp = t+1 + T*(z + d*(y + h*x));
                            dtt_adj = (deltat)*(dual_images3[ptm] - 2 * dual_images3[p] + dual_images3[ptp]);
                        }

                        // Other terms
                        if ((x == 0) && (y == 0)){
                            dxy_adj = dual_images4[p];
                        }
                        if ((x > 0) && (y == 0)){
                            dxy_adj = dual_images4[p] - dual_images4[pxm];
                        }
                        if ((x == 0) && (y > 0)){
                            dxy_adj = dual_images4[p] - dual_images4[pym];
                        }
                        if ((x > 0) && (y > 0)){
                            pxym = t + T*(z + d*(y-1 + h*(x-1)));
                            dxy_adj = dual_images4[p] - dual_images4[pxm]
                                    - dual_images4[pym]
                                    + dual_images4[pxym];
                        }

                        if ((y == 0) && (z == 0)){
                            dyz_adj = dual_images5[p];
                        }
                        if ((y > 0) && (z == 0)){
                            dyz_adj = dual_images5[p] - dual_images5[pym];
                        }
                        if ((y == 0) && (z > 0)){
                            dyz_adj = dual_images5[p] - dual_images5[pzm];
                        }
                        if ((y > 0) && (z > 0)){
                            pyzm = t + T*(z-1 + d*(y-1 + h*x));
                            dyz = deltaz*(dual_images5[p] - dual_images5[pym]
                                    - dual_images5[pzm]
                                    + dual_images5[pyzm]);
                        }

                        if ((z == 0) && (x == 0)){
                            dzx_adj = dual_images6[p];
                        }
                        if ((z > 0) && (x == 0)){
                            dzx_adj = dual_images6[p] - dual_images5[pzm];
                        }
                        if ((z == 0) && (x > 0)){
                            dzx_adj = dual_images6[pxm] - dual_images6[p];
                        }
                        if ((z > 0) && (x > 0)){
                            pxzm = t + T*(z-1 + d*(y + h*(x-1)));
                            dzx_adj = deltaz*(dual_images6[p] - dual_images6[pzm]
                                    - dual_images6[pxm]
                                    + dual_images6[pxzm]);
                        }

                        if ((t == 0) && (x == 0)){
                            dtx_adj = dual_images7[p];
                        }
                        if ((t > 0) && (x == 0)){
                            dtx_adj = dual_images7[p] - dual_images7[ptm];
                        }
                        if ((t == 0) && (x > 0)){
                            dtx_adj = dual_images7[pxm] - dual_images7[p];
                        }
                        if ((t > 0) && (x > 0)){
                            pxtm = t-1 + T*(z + d*(y + h*(x-1)));
                            dzx_adj = deltat*(dual_images7[p] - dual_images7[ptm]
                                    - dual_images7[pxm]
                                    + dual_images7[pxtm]);
                        }

                        if ((t == 0) && (y == 0)){
                            dty_adj = dual_images8[p];
                        }
                        if ((t > 0) && (y == 0)){
                            dty_adj = dual_images8[p] - dual_images8[ptm];
                        }
                        if ((t == 0) && (y > 0)){
                            dty_adj = dual_images8[pym] - dual_images8[p];
                        }
                        if ((t > 0) && (y > 0)){
                            pytm = t-1 + T*(z + d*(y-1 + h*x));
                            dtx_adj = deltat*(dual_images8[p] - dual_images8[ptm]
                                    - dual_images8[pym]
                                    + dual_images8[pytm]);
                        }
                        
                        if ((t == 0) && (z == 0)){
                            dtz_adj = dual_images9[p];
                        }
                        if ((t > 0) && (z == 0)){
                            dtz_adj = dual_images9[p] - dual_images9[ptm];
                        }
                        if ((t == 0) && (z > 0)){
                            dtz_adj = dual_images9[pzm] - dual_images9[p];
                        }
                        if ((t > 0) && (z > 0)){
                            pztm = t-1 + T*(z-1 + d*(y + h*x));
                            dtz_adj = deltat*(dual_images9[p] - dual_images9[ptm]
                                    - dual_images9[pzm]
                                    + dual_images9[pztm]);
                        }
                        
                        tmp -= (primal_weight
                                * (dxx_adj + dyy_adj + dzz_adj + dtt_adj 
                                    + sqrt2 * (dxy_adj + dyz_adj + dzx_adj) + sqrt2 * (dtx_adj + dty_adj + dtz_adj))
                                + primal_weight_comp * dual_images10[p]);
                        denoised_image[p] = SMath::max(min_val, SMath::min(max_val, tmp));
                    }
                }
            }
		}

        // iterations
        //std::cout << "iterations" << std::endl;
        if(verbose){
            if (iter % int(SMath::max(1, niter / 10)) == 0){
                observable->notifyProgress(100*(float(iter)/float(niter)));
            }
        }

        // Dual optimization
        //std::cout << "Dual optimization" << std::endl;
        for (unsigned int i = 0 ; i < buffer_size ; ++i){
		    auxiliary_image[i] = 2 * denoised_image[i] - auxiliary_image[i];
        }

        for (unsigned int x = 0 ; x < w ; ++x){
            for (unsigned int y = 0 ; y < h ; ++y){
                for (unsigned int z = 0 ; z < d ; ++z){
                    for (unsigned int t = 0 ; t < T ; ++t){

                        p = t + T*(z + d*(y + h*x));
                        pxp = t + T*(z + d*(y + h*(x+1)));
                        pyp = t + T*(z + d*(y+1 + h*x));
                        pzp = t + T*(z+1 + d*(y + h*x));
                        ptp = t+1 + T*(z + d*(y + h*x));
                        pxm = t + T*(z + d*(y + h*(x-1)));
                        pym = t + T*(z + d*(y-1 + h*x));
                        pzm = t + T*(z-1 + d*(y + h*x));
                        ptm = t-1 + T*(z + d*(y + h*x));

                        if ((x > 0) && (x < w - 1)) {
                            dxx = auxiliary_image[pxp] - 2 * auxiliary_image[p] + auxiliary_image[pxm];
                            dual_images0[p] += dual_weight * dxx;
                        }
                        if ((y > 0) && (y < h - 1)) {
                            dyy = auxiliary_image[pyp] - 2 * auxiliary_image[p] + auxiliary_image[pym];
                            dual_images1[p] += dual_weight * dyy;
                        }
                        if ((z > 0) && (z < d - 1)) {
                            dzz = (deltaz*deltaz)*(auxiliary_image[pzp] - 2 * auxiliary_image[p] + auxiliary_image[pzm]);
                            dual_images2[p] += dual_weight * dzz;
                        }
                        if ((t > 0) && (t < T - 1)) {
                            dtt = (deltat*deltat)*(auxiliary_image[ptp]
                                    - 2 * auxiliary_image[p]
                                    + auxiliary_image[ptm]);
                            dual_images3[p] += dual_weight * dtt;
                        }						
                        if ((x < w - 1) && (y < w - 1)) {
                            unsigned int pxyp = t + T*(z + d*(y+1 + h*(x+1)));
                            dxy = auxiliary_image[pxyp]
                                    - auxiliary_image[pxp]
                                    - auxiliary_image[pyp]
                                    + auxiliary_image[p];
                            dual_images4[p] += sqrt2 * dual_weight * dxy;
                        }
                        if ((y < h - 1) && (z < d - 1)) {
                            unsigned int pyzp = t + T*(z+1 + d*(y+1 + h*x));
                            dyz = deltaz*(auxiliary_image[pyzp]
                                    - auxiliary_image[pyp]
                                    - auxiliary_image[pzp]
                                    + auxiliary_image[p]);
                            dual_images5[p] += sqrt2 * dual_weight * dyz;
                        }
                        if ((z < d - 1) && (x < w - 1)) {
                            unsigned int pxzp = t + T*(z+1 + d*(y + h*(x+1)));
                            dzx = deltaz*(auxiliary_image[pxzp]
                                    - auxiliary_image[pxp]
                                    - auxiliary_image[pzp]
                                    + auxiliary_image[p]);
                            dual_images6[p] += sqrt2 * dual_weight * dzx;
                        }		
                        if ((t < T - 1) && (x < w - 1)) {
                            unsigned int pxtp = t+1 + T*(z + d*(y + h*(x+1)));
                            dtx = deltat*(auxiliary_image[pxtp]
                                    - auxiliary_image[pxp]
                                    - auxiliary_image[ptp]
                                    + auxiliary_image[p]);
                            dual_images7[p] += sqrt2 * dual_weight * dtx;
                        }			
                        if ((t < T - 1) && (y < h - 1)) {
                            unsigned int pytp = t+1 + T*(z + d*(y+1 + h*x));
                            dty = deltat*(auxiliary_image[pytp]
                                    - auxiliary_image[pyp]
                                    - auxiliary_image[ptp]
                                    + auxiliary_image[p]);
                            dual_images8[p] += sqrt2 * dual_weight * dty;
                        }
                        if ((t < T - 1) && (z < d - 1)) {
                            unsigned int pztp = t+1 + T*(z+1 + d*(y + h*x));
                            dtz = deltat*(auxiliary_image[pztp]
                                    - auxiliary_image[pzp]
                                    - auxiliary_image[ptp]
                                    + auxiliary_image[p]);
                            dual_images9[p] += sqrt2 * dual_weight * dtz;
                        }
                                
                        dual_images10[p] += dual_weight_comp * auxiliary_image[p];
                    }
                }
            }
		}

        for (unsigned int i = 0 ; i < buffer_size ; ++i)
		{
			double tmp = SMath::max(1.,
					1. / regularization
							* sqrt(
  									  pow(dual_images0[i], 2.)
									+ pow(dual_images1[i], 2.)
									+ pow(dual_images2[i], 2.)
									+ pow(dual_images3[i], 2.)
									+ pow(dual_images4[i], 2.)
									+ pow(dual_images5[i], 2.)
									+ pow(dual_images6[i], 2.)
									+ pow(dual_images7[i], 2.)
									+ pow(dual_images8[i], 2.)
									+ pow(dual_images9[i], 2.)
										));
					
			dual_images0[i] /= tmp;
			dual_images1[i] /= tmp;
			dual_images2[i] /= tmp;
			dual_images3[i] /= tmp;
			dual_images4[i] /= tmp;
			dual_images5[i] /= tmp;
			dual_images6[i] /= tmp;
			dual_images7[i] /= tmp;
			dual_images8[i] /= tmp;
			dual_images9[i] /= tmp;
			
		}
    } // end iter

    // normalize intensity
    float Average_IN_o = 0.;
    for (unsigned int ind=0; ind<buffer_size; ind++){
        Average_IN_o += denoised_image[ind];
    }
    Average_IN_o /= float(buffer_size);

    for (unsigned int ind=0; ind<w*h*d*T; ind++){
        denoised_image[ind] += (Average_IN_i-Average_IN_o);
    }
    if (verbose){
        observable->notifyProgress(100);
    }

}

}
