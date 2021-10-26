/// \file sl_padding.h
/// \brief sl_padding definitions
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2020

#ifndef sl_padding_H
#define sl_padding_H

namespace SImg{

int padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out);
int padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out);

int mirror_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out);
int mirror_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out);
int mirror_padding_4d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int st, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out, const unsigned int st_out);

int hanning_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out);
int hanning_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out);

int remove_padding_2d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sx_out, const unsigned int sy_out);
int remove_padding_3d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out);
int remove_padding_4d(float* buffer_in, float* buffer_out, const unsigned int sx, const unsigned int sy, const unsigned int sz, const unsigned int st, const unsigned int sx_out, const unsigned int sy_out, const unsigned int sz_out, const unsigned int st_out);

}

#endif /* !sl_padding_H */