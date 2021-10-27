/// \file spadding.h
/// \brief spadding implementation
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#include "spadding.h"

#include <iostream>
#include "math.h"

namespace SImg{

int padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out)
{
    if (sx >= sx_out || sy >= sy_out)
    {
        return 1;
    }
    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;

    int x, y;
    for (x = 0 ; x < sx_out ; x++){
        for (y = 0 ; y < padding_y ; y++){
            buffer_out[sy_out*x+y] = 0;    
        }
        for (y = sy ; y < sy_out ; y++){
            buffer_out[sy_out*x+y] = 0;    
        }
    }
    for (y = 0 ; y < sy_out ; y++){
        for (x = 0 ; x < padding_x ; x++){
            buffer_out[sy_out*x+y] = 0;    
        }
        for (x = sx ; x < sx_out ; x++){
            buffer_out[sy_out*x+y] = 0;    
        }
    }
    for(x = 0 ; x < sx ; x++){
        for(y = 0 ; y < sy ; y++){
            buffer_out[sy_out*(x+padding_x)+y+padding_y] = buffer_in[sy*x+y]; 
        }
    }

    return 0;
}

int padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    if (sx >= sx_out || sy >= sy_out || sz >= sz_out)
    {
        return 1;
    }
    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;

    for (int i = 0 ; i < sx_out*sy_out*sz_out ; i++){
        buffer_out[i] = 0.0;
    }

    int x, y, z;
    for(x = 0 ; x < sx ; x++){
        for(y = 0 ; y < sy ; y++){
            for(z = 0 ; z < sz ; z++){
                buffer_out[z+padding_z + sz_out*(sy_out*(x+padding_x)+y+padding_y)] = buffer_in[z + sz*(sy*x+y)]; 
            }
        }
    }
    return 0;
}

int padding_4d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int st, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    if (sx >= sx_out || sy >= sy_out || sz >= sz_out)
    {
        return 1;
    }
    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;

    for (int i = 0 ; i < sx_out*sy_out*sz_out*st ; i++){
        buffer_out[i] = 0.0;
    }

    int x, y, z, t;
    for (t=0 ; t < st ; t++ )
    {
        for(x = 0 ; x < sx ; x++){
            for(y = 0 ; y < sy ; y++){
                for(z = 0 ; z < sz ; z++){
                    buffer_out[t + st*(z+padding_z + sz_out*(sy_out*(x+padding_x)+y+padding_y))] = buffer_in[t + st*(z + sz*(sy*x+y))]; 
                }
            }
        }
    }
    return 0;
}

int mirror_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out)
{
    int ctrl = padding_2d(buffer_in, buffer_out, sx, sy, sx_out, sy_out);
    if (ctrl > 0){
        return 1;
    }

    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int x, y;

    for (y = 0 ; y < sy ; y++){
        // vertical right
        for (x = 0 ; x < padding_x ; x++){
            buffer_out[sy_out*(padding_x+sx+x)+y+padding_y] = buffer_in[sy*(sx-x-1)+y];
        }
    }
    for (y = 0 ; y < sy ; y++){
        // vertical left
        for (x = 0 ; x <= padding_x ; x++){
            buffer_out[sy_out*(padding_x-x)+y+padding_y] = buffer_in[sy*(x)+y];
        }
    }
    for (x = 0 ; x < sx ; x++){
        for (y = 0 ; y <= padding_y ; y++){
            buffer_out[sy_out*(padding_x+x)+y+padding_y+sy] = buffer_in[sy*(x)+sy-y-1];
            buffer_out[sy_out*(padding_x+x)-y+padding_y] = buffer_in[sy*(x)+y];
        }
    }
    for (x = 0 ; x < padding_x; x++){
        // top left corner
        for (y = 0 ; y < padding_y; y++){
            buffer_out[(sy_out*x + y)] = buffer_out[(sy_out*(2*padding_x-x) + y)];       
        }
        // bottom left corner
        for (y = sy + padding_y ; y < sy_out; y++){
            buffer_out[(sy_out*x + y)] = buffer_out[(sy_out*(2*padding_x-x) + y)];       
        }
    }

    for (x = 0; x < padding_x; x++){
        // top right corner
        for (y = 0 ; y < padding_y; y++){
            buffer_out[(sy_out*(x+sx+padding_x) + y)] = buffer_out[(sy_out*(sx+padding_x-x-1) + y)];       
        }
        // bottom right corner
        for (y = sy + padding_y ; y < sy_out; y++){
            buffer_out[(sy_out*(x+sx+padding_x) + y)] = buffer_out[(sy_out*(sx+padding_x-x-1) + y)];         
        }
    }
    return 0;
}

int mirror_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    int ctrl = padding_3d(buffer_in, buffer_out, sx, sy, sz, sx_out, sy_out, sz_out);
    if (ctrl > 0){
        return 1;
    }

    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;
    unsigned int x, y, z, z_;

    for (z = padding_z ; z < sz_out-padding_z ; z++){
        z_ = z - padding_z;
        for (y = 0 ; y < sy ; y++){
            // vertical right
            for (x = 0 ; x < padding_x ; x++){
                buffer_out[z+sz_out*(sy_out*(padding_x+sx+x)+y+padding_y)] = buffer_in[z_+ sz*(sy*(sx-x-1)+y)];
            }
        }
        for (y = 0 ; y < sy ; y++){
            // vertical left
            for (x = 0 ; x <= padding_x ; x++){
                buffer_out[z+sz_out*(sy_out*(padding_x-x)+y+padding_y)] = buffer_in[z_+sz*(sy*(x)+y)];
            }
        }
        for (x = 0 ; x < sx ; x++){
            for (y = 0 ; y <= padding_y ; y++){
                buffer_out[z+sz_out*(sy_out*(padding_x+x)+y+padding_y+sy)] = buffer_in[z_+sz*(sy*(x)+sy-y-1)];
                buffer_out[z+sz_out*(sy_out*(padding_x+x)-y+padding_y)] = buffer_in[z_+sz*(sy*(x)+y)];
            }
        }
        for (x = 0 ; x < padding_x; x++){
            // top left corner
            for (y = 0 ; y < padding_y; y++){
                buffer_out[z+sz_out*(sy_out*x + y)] = buffer_out[z+sz_out*((sy_out*(2*padding_x-x) + y))];       
            }
            // bottom left corner
            for (y = sy + padding_y ; y < sy_out; y++){
                buffer_out[z+sz_out*(sy_out*x + y)] = buffer_out[z+sz_out*((sy_out*(2*padding_x-x) + y))];       
            }
        }

        for (x = 0; x < padding_x; x++){
            // top right corner
            for (y = 0 ; y < padding_y; y++){
                buffer_out[z+sz_out*(sy_out*(x+sx+padding_x) + y)] = buffer_out[z+sz_out*(sy_out*(sx+padding_x-x-1) + y)];       
            }
            // bottom right corner
            for (y = sy + padding_y ; y < sy_out; y++){
                buffer_out[z+sz_out*(sy_out*(x+sx+padding_x) + y)] = buffer_out[z+sz_out*(sy_out*(sx+padding_x-x-1) + y)];         
            }
        }
    }

    int dz;
    int xy = 0;
    int ref = padding_z + sz -1;
    for (z=0 ; z <= padding_z ; z++)
    {
        for (x = 0 ; x < sx_out ; x++)
        {
            for (y = 0 ; y < sy_out ; y++)
            {
                xy = sy_out*x+y;
                buffer_out[ref+z + sz_out*xy] = buffer_out[ref-z + sz_out*xy];
                buffer_out[padding_z-z + sz_out*xy] = buffer_out[padding_z+z + sz_out*xy];   
            } 
        }
    }

    return 0;
}

int mirror_padding_4d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int st, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    int ctrl = padding_4d(buffer_in, buffer_out, sx, sy, sz, st, sx_out, sy_out, sz_out);
    if (ctrl > 0){
        return 1;
    }

    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;
    unsigned int x, y, z, z_, t;

    for (t = 0 ; t < st ; t++)
    {

        for (z = padding_z ; z < sz_out-padding_z ; z++){
            z_ = z - padding_z;
            for (y = 0 ; y < sy ; y++){
                // vertical right
                for (x = 0 ; x < padding_x ; x++){
                    buffer_out[t+st*(z+sz_out*(sy_out*(padding_x+sx+x)+y+padding_y))] = buffer_in[t+st*(z_+ sz*(sy*(sx-x-1)+y))];
                }
            }
            for (y = 0 ; y < sy ; y++){
                // vertical left
                for (x = 0 ; x <= padding_x ; x++){
                    buffer_out[t+st*(z+sz_out*(sy_out*(padding_x-x)+y+padding_y))] = buffer_in[t+st*(z_+sz*(sy*(x)+y))];
                }
            }
            for (x = 0 ; x < sx ; x++){
                for (y = 0 ; y <= padding_y ; y++){
                    buffer_out[t+st*(z+sz_out*(sy_out*(padding_x+x)+y+padding_y+sy))] = buffer_in[t+st*(z_+sz*(sy*(x)+sy-y-1))];
                    buffer_out[t+st*(z+sz_out*(sy_out*(padding_x+x)-y+padding_y))] = buffer_in[t+st*(z_+sz*(sy*(x)+y))];
                }
            }
            for (x = 0 ; x < padding_x; x++){
                // top left corner
                for (y = 0 ; y < padding_y; y++){
                    buffer_out[t+st*(z+sz_out*(sy_out*x + y))] = buffer_out[t+st*(z+sz_out*((sy_out*(2*padding_x-x) + y)))];       
                }
                // bottom left corner
                for (y = sy + padding_y ; y < sy_out; y++){
                    buffer_out[t+st*(z+sz_out*(sy_out*x + y))] = buffer_out[t+st*(z+sz_out*((sy_out*(2*padding_x-x) + y)))];       
                }
            }

            for (x = 0; x < padding_x; x++){
                // top right corner
                for (y = 0 ; y < padding_y; y++){
                    buffer_out[t+st*(z+sz_out*(sy_out*(x+sx+padding_x) + y))] = buffer_out[t+st*(z+sz_out*(sy_out*(sx+padding_x-x-1) + y))];       
                }
                // bottom right corner
                for (y = sy + padding_y ; y < sy_out; y++){
                    buffer_out[t+st*(z+sz_out*(sy_out*(x+sx+padding_x) + y))] = buffer_out[t+st*(z+sz_out*(sy_out*(sx+padding_x-x-1) + y))];         
                }
            }
        }

        int dz;
        for (z=0 ; z <= padding_z ; z++){
            for (x = 0 ; x < sx_out ; x++){
                for (y = 0 ; y < sy_out ; y++){
                    buffer_out[t+st*(padding_z+sz+z + sz_out*(sy_out*x+y))] = buffer_out[t+st*(padding_z+sz-z-2 + sz_out*(sy_out*x+y))];
                    buffer_out[t+st*(padding_z-z + sz_out*(sy_out*x+y))] = buffer_out[t+st*(padding_z+z + sz_out*(sy_out*x+y))];
                }
            }
        }
    }

    return 0;
}

int hanning_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out)
{
    //int ctrl = padding_2d(buffer_in, buffer_out, sx, sy, sx_out, sy_out);
    int ctrl = mirror_padding_2d(buffer_in, buffer_out, sx, sy, sx_out, sy_out);
    if (ctrl > 0){
        return 1;
    }

    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;

    unsigned int hann_N_x = 2*padding_x; 
    unsigned int hann_N_y = 2*padding_y; 

    for (unsigned int y = 0 ; y < sy_out ; y++)
    {
        // vertical left    
        for (unsigned int x = 0 ; x < padding_x-1 ; x++)
        {
            float coef = 0.5*(1- cos(2*3.14*(x)/hann_N_x));
            buffer_out[sy_out*x+y] = buffer_out[sy_out*x+y]*coef;
        }
        // vertical right
        for (unsigned int x = 1 ; x < padding_x ; x++)
        {
            float coef = 0.5*(1- cos(2*3.14*(x-1)/hann_N_x));
            buffer_out[sy_out*(sx_out-x)+y] = buffer_out[sy_out*(sx_out-x)+y]*coef;
        }
    }

    for (unsigned int x = 0 ; x < sx_out ; x++)
    {
        // horizontal top
        for (unsigned int y = 0 ; y < padding_y-1 ; y++)
        {
            float coef = 0.5*(1- cos(2*3.14*(y)/hann_N_y));
            buffer_out[sy_out*x+y] = buffer_out[sy_out*x+y]*coef;
        }
        // hortizontal bottom
        for (unsigned int y = 1 ; y < padding_y ; y++)
        {
            float coef = 0.5*(1- cos(2*3.14*(y-1)/hann_N_y));
            buffer_out[sy_out*x+sy_out-y] = buffer_out[sy_out*x+sy_out-y]*coef;           
        }
    }
    return 0;
}

int hanning_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    int ctrl = mirror_padding_3d(buffer_in, buffer_out, sx, sy, sz, sx_out, sy_out, sz_out);
    if (ctrl > 0){
        return 1;
    }
    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;

    unsigned int hann_N_x = 2*padding_x; 
    unsigned int hann_N_y = 2*padding_y; 
    unsigned int hann_N_z = 2*padding_z; 

    unsigned int x, y, z;
    // XY padding
    for (z = 0 ; z < sz_out ; z++){
        for (unsigned int y = 0 ; y < sy_out ; y++)
        {
            // vertical left   
            for (unsigned int x = 0 ; x < padding_x-1 ; x++)
            {
                float coef = 0.5*(1- cos(2*3.14*(x)/hann_N_x));
                buffer_out[z + sz_out*(sy_out*x+y)] = buffer_out[z + sz_out*(sy_out*x+y)]*coef;
            }
            // vertical right
            for (unsigned int x = 1 ; x < padding_x ; x++)
            {
                float coef = 0.5*(1- cos(2*3.14*(x-1)/hann_N_x));
                buffer_out[z + sz_out*(sy_out*(sx_out-x)+y)] = buffer_out[z + sz_out*(sy_out*(sx_out-x)+y)]*coef;
            }
        }

        for (unsigned int x = 0 ; x < sx_out ; x++)
        {
            // horizontal top
            for (unsigned int y = 0 ; y < padding_y-1 ; y++)
            {
                float coef = 0.5*(1- cos(2*3.14*(y)/hann_N_y));
                buffer_out[z + sz_out*(sy_out*x+y)] = buffer_out[z + sz_out*(sy_out*x+y)]*coef;
            }
            // hortizontal bottom
            for (unsigned int y = 1 ; y < padding_y ; y++)
            {
                float coef = 0.5*(1- cos(2*3.14*(y-1)/hann_N_y));
                buffer_out[z + sz_out*(sy_out*x+sy_out-y)] = buffer_out[z + sz_out*(sy_out*x+sy_out-y)]*coef;           
            }
        }
    }

    // Z hanning
    int dz;
    int xy = 0;
    int ref = padding_z + sz;
    for (z=0 ; z < padding_z ; z++){
        float coef = 0.5*(1- cos(2*3.14*(padding_z-z-1)/hann_N_z));
        for (x = 0 ; x < sx_out ; x++){
            for (y = 0 ; y < sy_out ; y++){
                xy = sy_out*x+y;
                // last frames
                buffer_out[ref+z + sz_out*xy] = buffer_out[ref+z + sz_out*xy]*coef;
                // first frames
                buffer_out[padding_z-z-1 + sz_out*xy] = buffer_out[padding_z-z-1 + sz_out*xy]*coef;
            }
        }
    }
    return 0;
}

int remove_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out)
{
    if (sx <= sx_out || sy <= sy_out)
    {
        return 1;
    }
    unsigned int padding_x = (int(sx) - int(sx_out))/2;
    unsigned int padding_y = (int(sy) - int(sy_out))/2;

    int x, y;
    for(x = padding_x ; x < sx-padding_x ; x++)
    {
        for(y = padding_y ; y < sy-padding_y ; y++)
        {
            buffer_out[sy_out*(x-padding_x)+y-padding_y] = buffer_in[sy*x+y]; 
        }
    }

    return 0;
}

int remove_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    if (sx <= sx_out || sy <= sy_out || sz <= sz_out )
    {
        return 1;
    }
    unsigned int padding_x = (int(sx) - int(sx_out))/2;
    unsigned int padding_y = (int(sy) - int(sy_out))/2;
    unsigned int padding_z = (int(sz) - int(sz_out))/2;

    int x, y, z;
    for (z = padding_z ; z < sz-padding_z ; z ++){
        for(x = padding_x ; x < sx-padding_x ; x++)
        {
            for(y = padding_y ; y < sy-padding_y ; y++)
            {
                buffer_out[z-padding_z + sz_out*(sy_out*(x-padding_x)+y-padding_y)] = buffer_in[z+sz*(sy*x+y)]; 
            }
        }
    }

    return 0;
}

int remove_padding_4d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int st, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{
    if (sx <= sx_out || sy <= sy_out || sz <= sz_out )
    {
        return 1;
    }
    unsigned int padding_x = (int(sx) - int(sx_out))/2;
    unsigned int padding_y = (int(sy) - int(sy_out))/2;
    unsigned int padding_z = (int(sz) - int(sz_out))/2;

    int x, y, z, t;
    for (t = 0 ; t < st ; t++)
    {
        for (z = padding_z ; z < sz-padding_z ; z ++)
        {
            for(x = padding_x ; x < sx-padding_x ; x++)
            {
                for(y = padding_y ; y < sy-padding_y ; y++)
                {
                    buffer_out[t+st*(z-padding_z + sz_out*(sy_out*(x-padding_x)+y-padding_y))] = buffer_in[t+st*(z+sz*(sy*x+y))]; 
                }
            }
        }
    }

    return 0;
}


}