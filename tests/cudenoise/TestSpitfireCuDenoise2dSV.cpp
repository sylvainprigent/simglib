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
    std::cout << "input image = " << SAMPLE2D << std::endl;
    std::cout << "ref image = " << SAMPLE2DDENOISEDSV << std::endl;

    SImageFloat *inputImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE2D, 32));
    float imax = inputImage->getMax();
    float imin = inputImage->getMin();

    // Run process
    float *noisy_image = inputImage->getBuffer();
    unsigned int sx = inputImage->getSizeX();
    unsigned int sy = inputImage->getSizeY();


    SObserverConsole *observer = new SObserverConsole();
    SObservable *observable = new SObservable();
    observable->addObserver(observer);

    float regularization = pow(2, -2);
    float weighting = 0.6; 
    float niter = 200;
    
    SImg::tic();

    // run denoising
    float *denoised_image = new float[sx*sy];
    SImg::cuda_spitfire2d_denoise(noisy_image, sx, sy, denoised_image, regularization, weighting, niter, "SV", true, observable);
    
    SImg::toc();
    delete STimerAccess::instance();

    //SImageReader::write(new SImageFloat(denoised_image, sx, sy), SAMPLE2DDENOISEDSV);

    // calculate error with the reference image
    std::cout << "calculate error:" << std::endl;
    SImageFloat *resultImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE2DDENOISEDSV, 32));
    float *b1 = denoised_image;
    float *b2 = resultImage->getBuffer();
    unsigned int p = 0;
    double error = 0.0;

    for (unsigned int x = 10; x < sx - 10; ++x)
    {
        for (unsigned int y = 10; y < sy - 10; ++y)
        {
            p = sy * x + y;
            error += pow(b1[p] - b2[p], 2);
        }
    }
    delete noisy_image;
    delete[] denoised_image;
    delete resultImage;
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
