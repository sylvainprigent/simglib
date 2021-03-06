/// \file sutils.cpp
/// \brief sutils definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "sutils.h"
#include "math.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

namespace SImg{

void normalize_intensities(float* buffer, unsigned int buffer_length, std::string method)
{
    if (method == "max")
    {
        float max_val = buffer[0];
        for (int p = 1 ; p < buffer_length ; p++){
            if (buffer[p] > max_val){
                max_val = buffer[p]; 
            }
        }
        #pragma omp parallel for
        for (int p = 0 ; p < buffer_length ; p++){
            buffer[p] /= max_val;
        }
    }
    else if (method == "sum")
    {
        float sum_val = buffer[0];
        for (int p = 1 ; p < buffer_length ; p++){
            sum_val += buffer[p];
        }
        #pragma omp parallel for
        for (int p = 0 ; p < buffer_length ; p++){
            buffer[p] /= sum_val;
        }
    }
    else if (method == "L2")
    {
        float norm = 0.0;
        for (unsigned int i = 0 ; i < buffer_length ; i++){
            norm += buffer[i]*buffer[i];
        }
        norm = sqrt(norm);

        #pragma omp parallel for
        for (unsigned int i = 0 ; i < buffer_length ; i++){
            buffer[i] = buffer[i] / norm;
        }
    }
}

void remove_min(float* buffer, int buffer_length)
{
    float min_val = buffer[0];
    for (int p = 1 ; p < buffer_length ; p++){
        if (buffer[p] < min_val){
            min_val = buffer[p]; 
        }
    }
    #pragma omp parallel for
    for (int p = 0 ; p < buffer_length ; p++){
        buffer[p] -= min_val;
    }
}

void laplacian_2d(float* buffer_in, float* buffer_out, unsigned int sx, unsigned int sy, int connectivity)
{
    // caluculate the filter
    if (connectivity == 4){
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                buffer_out[sy*(x)+y] =  4*buffer_in[sy*x+y]
                                        -buffer_in[sy*(x-1)+y]-buffer_in[sy*(x+1)+y]
                                        -buffer_in[sy*(x)+y-1]-buffer_in[sy*(x)+y+1];
            }
        }
    }
    else if (connectivity == 8){
        for (int x = 1 ; x < sx-1 ; x++){
            for (int y = 1 ; y < sy-1 ; y++){
                buffer_out[sy*(x)+y] =  8*buffer_in[sy*x+y]
                                        -buffer_in[sy*(x-1)+y]-buffer_in[sy*(x+1)+y]
                                        -buffer_in[sy*(x)+y-1]-buffer_in[sy*(x)+y+1]
                                        -buffer_in[sy*(x-1)+(y-1)]-buffer_in[sy*(x-1)+(y+1)]
                                        -buffer_in[sy*(x+1)+(y-1)]-buffer_in[sy*(x+1)+(y+1)];   
            }
        }
    }
    // set borders to 0
    for (int x = 0 ; x < sx ; x++){
        buffer_out[sy*(x)+0] = 0;
        buffer_out[sy*(x)+sy-1] = 0;
    } 
    for (int y = 0 ; y < sy ; y++){
        buffer_out[sy*(0)+y] = 0;
        buffer_out[sy*(sx-1)+y] = 0;
    } 
}

}