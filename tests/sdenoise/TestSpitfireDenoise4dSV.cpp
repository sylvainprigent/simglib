#include <score>
#include <scli>
#include <simageio>
#include <sdenoise>
#include <smanipulate>

#include "sdenoiseTestConfig.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

#include <iostream>
#include "math.h"

int main(int argc, char *argv[])
{
    std::cout << "input image = " << SAMPLE4D << std::endl;
    std::cout << "ref image = " << SAMPLE4DDENOISEDSV << std::endl;

    SImageFloat *inputImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE4D, 32));
    float imax = inputImage->getMax();
    float imin = inputImage->getMin();

    // Run process
    float *noisy_image = inputImage->getBuffer();
    unsigned int sx = inputImage->getSizeX();
    unsigned int sy = inputImage->getSizeY();
    unsigned int sz = inputImage->getSizeZ();
    unsigned int st = inputImage->getSizeT();

    std::cout << "image size = (" << sx << ", " << sy << ", " << sz << ", " << st << ")" << std::endl; 

    SObserverConsole *observer = new SObserverConsole();
    SObservable *observable = new SObservable();
    observable->addObserver(observer);

    float regularization = pow(2, -2);
    float weighting = 0.6; 
    float niter = 200;
    float deltaz = 1;
    float deltat = 1;
    
    SImg::tic();

    // min max normalize intensities
    float* noisy_image_norm = new float[sx*sy*sz*st];
    SImg::normMinMax(noisy_image, sx, sy, sz, st, 1, noisy_image_norm);
    delete inputImage;

    // run denoising
    float *denoised_image = new float[sx*sy*sz*st];
    SImg::spitfire4d_sv(noisy_image_norm, sx, sy, sz, st, denoised_image, regularization, weighting, niter, deltaz, deltat, true, observable);
    
    // normalize back intensities
    #pragma omp parallel for
    for (unsigned int i = 0 ; i < sx*sy*sz*st ; ++i)
    {
        denoised_image[i] = denoised_image[i]*(imax-imin) + imin;
    }
    SImg::toc();
    delete STimerAccess::instance();

    SImageReader::write(new SImageFloat(denoised_image, sx, sy, sz, st), "./bin/denoise4d/denoised_4d_sv.txt");

    // calculate error with the reference image
    /*
    std::cout << "calculate error:" << std::endl;
    SImageFloat *resultImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE4DDENOISEDSV, 32));
    float *b1 = denoised_image;
    float *b2 = resultImage->getBuffer();
    unsigned int p = 0;
    double error = 0.0;

    for (unsigned int x = 1; x < sx - 1; ++x)
    {
        for (unsigned int y = 1; y < sy - 1 ; ++y)
        {
            for (unsigned int z = 1; z < sz - 1; ++z)
            {
                for (unsigned int t = 1; t < st - 1; ++t)
                {
                    p = t+ st*(z + sz*(sy * x + y));
                    error += pow(b1[p] - b2[p], 2);
                }
            }
        }
    }
    delete[] noisy_image_norm;
    delete resultImage;
    delete[] denoised_image;
    delete observable;
    delete observer;
    //error /= float(outputImage->getBufferSize());
    std::cout << "error = " << error << std::endl;
    if (error > 10)
    {
        return 1;
    }
    */
    return 0;
}
