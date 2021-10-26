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
    for (z=0 ; z <= padding_z ; z++){
        for (x = 0 ; x < sx_out ; x++){
            for (y = 0 ; y < sy_out ; y++){
                buffer_out[padding_z+sz+z + sz_out*(sy_out*x+y)] = buffer_out[padding_z+sz-z-2 + sz_out*(sy_out*x+y)];
                buffer_out[padding_z-z + sz_out*(sy_out*x+y)] = buffer_out[padding_z+z + sz_out*(sy_out*x+y)];
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

    unsigned int mirror_pad_x = 4*int(padding_x)/5;
    unsigned int mirror_pad_y = 4*int(padding_y)/5;

    unsigned int hann_N_x = 2*mirror_pad_x; 
    unsigned int hann_N_y = 2*mirror_pad_y; 

    
    for (unsigned int y = 0 ; y < sy_out ; y++)
    {
        // vertical left    
        for (unsigned int x = 0 ; x < padding_x-1 ; x++)
        {
            buffer_out[sy_out*x+y] = buffer_out[sy_out*x+y]*pow(sin(3.14*(x)/hann_N_x), 2);
        }
        // vertical right
        for (unsigned int x = 1 ; x < padding_x ; x++)
        {
            buffer_out[sy_out*(sx_out-x)+y] = buffer_out[sy_out*(sx_out-x)+y]*pow(sin(3.14*(x-1)/hann_N_x), 2);
        }
    }

    for (unsigned int x = 0 ; x < sx_out ; x++)
    {
        // hortizontal top
        for (unsigned int y = 0 ; y < padding_y-1 ; y++)
        {
            buffer_out[sy_out*x+y] = buffer_out[sy_out*x+y]*pow(sin(3.14*(y)/hann_N_y), 2);
        }
        // hortizontal bottom
        for (unsigned int y = 1 ; y < padding_y ; y++)
        {
            buffer_out[sy_out*x+sy_out-y] = buffer_out[sy_out*x+sy_out-y]*pow(sin(3.14*(y-1)/hann_N_y), 2);           
        }
    }
    return 0;
}

int hanning_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out)
{

    int ctrl = padding_3d(buffer_in, buffer_out, sx, sy, sz, sx_out, sy_out, sz_out);
    if (ctrl > 0){
        return 1;
    }

    unsigned int padding_x = (int(sx_out) - int(sx))/2;
    unsigned int padding_y = (int(sy_out) - int(sy))/2;
    unsigned int padding_z = (int(sz_out) - int(sz))/2;
    unsigned int mirror_pad_x = 4*int(padding_x)/5;
    unsigned int mirror_pad_y = 4*int(padding_y)/5;
    unsigned int mirror_pad_z = 4*int(padding_z)/5;

    unsigned int hann_N_x = 2*mirror_pad_x; 
    unsigned int hann_N_y = 2*mirror_pad_y; 
    unsigned int hann_N_z = 2*mirror_pad_z; 

    unsigned int x, y, z;
    for (z = 0 ; z < sz ; z++){
        for (y = 0 ; y < sy ; y++){
            for (x = 0 ; x < mirror_pad_x ; x++){
                buffer_out[z+padding_z + sz_out*(sy_out*(padding_x+sx+x)+y+padding_y)] = buffer_in[z + sz*(sy*(sx-x-1)+y)]*pow(sin(3.14*(x+mirror_pad_x)/hann_N_x), 2);
                buffer_out[z+padding_z + sz_out*(sy_out*(padding_x-x)+y+padding_y)] = buffer_in[z + sz*(sy*(x)+y)]*pow(sin(3.14*(mirror_pad_x-x)/hann_N_x), 2);
            }
        }
        for (x = 0 ; x < sx ; x++){
            for (y = 0 ; y < mirror_pad_y ; y++){
                buffer_out[z+padding_z + sz_out*(sy_out*(padding_x+x)+y+padding_y+sy)] = buffer_in[z+sz*(sy*(x)+sy-y-1)]*pow(sin(3.14*(y+mirror_pad_y)/hann_N_y), 2);
                buffer_out[z+padding_z + sz_out*(sy_out*(padding_x+x)-y+padding_y)] = buffer_in[z+sz*(sy*(x)+y)]*pow(sin(3.14*(mirror_pad_y-y)/hann_N_y), 2);
            }
        }
    }

    int dz;
    for (z=0 ; z <= padding_z ; z++){
        for (x = 0 ; x < sx_out ; x++){
            for (y = 0 ; y < sy_out ; y++){
                buffer_out[padding_z+sz+z + sz_out*(sy_out*x+y)] = buffer_out[padding_z+sz-z-2 + sz_out*(sy_out*x+y)]*pow(sin(3.14*(z+mirror_pad_z)/hann_N_z), 2);
                buffer_out[padding_z-z + sz_out*(sy_out*x+y)] = buffer_out[padding_z+z + sz_out*(sy_out*x+y)]*pow(sin(3.14*(mirror_pad_z+1-z)/hann_N_z), 2);
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