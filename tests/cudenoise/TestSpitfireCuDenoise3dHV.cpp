#include <score>
#include <scli>
#include <simageio>
#include <cudenoise>
#include <smanipulate>

#include "cudenoiseTestConfig.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

#include <iostream>
#include "math.h"

int main(int argc, char *argv[])
{
    std::cout << "input image = " << SAMPLE3D << std::endl;
    std::cout << "ref image = " << SAMPLE3DDENOISED << std::endl;

    SImageFloat *inputImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE3D, 32));
    float imax = inputImage->getMax();
    float imin = inputImage->getMin();

    // Run process
    float *noisy_image = inputImage->getBuffer();
    unsigned int sx = inputImage->getSizeX();
    unsigned int sy = inputImage->getSizeY();
    unsigned int sz = inputImage->getSizeZ();

    SObserverConsole *observer = new SObserverConsole();
    SObservable *observable = new SObservable();
    observable->addObserver(observer);

    float regularization = pow(2, -2);
    float weighting = 0.6; 
    float niter = 200;
    float deltaz = 1.0;
    
    SImg::tic();

    // run denoising
    float *denoised_image = new float[sx*sy*sz];
    SImg::cuda_spitfire3d_denoise(noisy_image, sx, sy, sz, denoised_image, regularization, weighting, niter, deltaz, "HV", true, observable);

    SImg::toc();
    delete STimerAccess::instance();

    //SImageReader::write(new SImageFloat(denoised_image, sx, sy, sz), SAMPLE3DDENOISED);

    // calculate error with the reference image
    std::cout << "calculate error:" << std::endl;
    SImageFloat *resultImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE3DDENOISED, 32));
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
                p = z + sz*(sy * x + y);
                error += pow(b1[p] - b2[p], 2);
            }
        }
    }
    delete[] noisy_image;
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
    return 0;
}
