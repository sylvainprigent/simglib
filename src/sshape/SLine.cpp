/// \file SLine.cpp
/// \brief SLine class
/// \author Sylvain Prigent
/// \version 0.1
/// \date 2021

#include "math.h"

#include "simage/SImageUInt.h"
#include "score/SException.h"

#include "SLine.h"

SLine::SLine() : SShape(){

}

SLine::SLine(float x_start, float y_start, float x_end, float y_end) : SShape()
{
    m_x_start = x_start;
    m_y_start = y_start;
    m_x_end = x_end;
    m_y_end = y_end;
}

SLine::SLine(float x_start, float y_start, float z_start, float x_end, float y_end, float z_end) : SShape()
{
    m_x_start = x_start;
    m_y_start = y_start;
    m_z_start = z_start;
    m_x_end = x_end;
    m_y_end = y_end;
    m_z_end = z_end;
}

SLine::~SLine()
{

}

void SLine::draw(SImage* image)
{
    // 2D case    
    if (image->getSizeZ() == 1){

        std::vector<int> px;
        std::vector<int> py;
        calculate2DLineCoordinates( int(m_x_start), int(m_y_start), int(m_x_end), int(m_y_end), px, py);

        unsigned int* buffer = dynamic_cast<SImageUInt*>(image)->getBuffer();
        unsigned int sx = image->getSizeX(); 
        unsigned int sy = image->getSizeY(); 
        int x, y;
        for (int i = 0 ; i < px.size() ; i++)
        {
            x = px[i];
            y = py[i];    
            buffer[3*(sy*x+y)+0] = unsigned(m_color->red()); 
            buffer[3*(sy*x+y)+1] = unsigned(m_color->green()); 
            buffer[3*(sy*x+y)+2] = unsigned(m_color->blue()); 
        }

    } 
    else{
        throw SException("SLine::draw does not yet support 3D draw");
    }

}

void calculate2DLineCoordinates( int x1, int y1, int x2, int y2, std::vector<int> &px, std::vector<int> &py){
    float maxi = 0; float val;
    maxi = fabs(float(x2 - x1));
    val =  fabs(float(y2 - y1));
    if (val > maxi){
        maxi = val;
    }
    int m = int(maxi + 1);
    px = linSpaceRound(x1, x2, m);
    py = linSpaceRound(y1, y2, m);
}

std::vector<float> linSpace(float d1, float d2, int n){

    unsigned int m = n;
    float n1 = floor(m)-1;

    std::vector<float> vect; vect.resize(m-1);
    std::vector<float> z; z.resize(m);
    for(unsigned int i = 0 ; i < m-1 ; ++i){
        vect[i] = i;
    }
    for (unsigned int i = 0 ; i < m-1 ; ++i){
        z[i] =  d1 + vect[i]*(d2-d1)/n1;
    }
    z[m-1] = d2;
    return z;
}

std::vector<int> linSpaceRound(float d1, float d2, int n){
    std::vector<float> lin = linSpace(d1, d2, n);
    std::vector<int> linFloor; linFloor.resize(lin.size());
    for (unsigned int i = 0 ; i < lin.size() ; ++i){
        linFloor[i] = int(floor(lin[i]+0.5));
    }
    return linFloor;
}