#include <score>
#include <scli>
#include <simageio>
#include <sdeconv>
#include <smanipulate>

#include "sdeconvTestConfig.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

#include <iostream>
#include "math.h"

int main(int argc, char *argv[])
{
    std::cout << "input image = " << SAMPLE2D << std::endl;
    std::cout << "ref image = " << SAMPLE2DDECONVW << std::endl;

    SImageFloat *inputImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE2D, 32));
    float imax = inputImage->getMax();
    float imin = inputImage->getMin();

    // Run process
    float *blurry_image = inputImage->getBuffer();
    unsigned int sx = inputImage->getSizeX();
    unsigned int sy = inputImage->getSizeY();


    SObserverConsole *observer = new SObserverConsole();
    SObservable *observable = new SObservable();
    observable->addObserver(observer);

    float sigma = 1.5;
    float lambda = 0.005;
    int connectivity = 8; // 4

    // create the PSF
    float* psf = new float[sx*sy];
    SImg::gaussian_psf_2d(psf, sx, sy, sigma, sigma);
    float psf_sum = 0.0;
    for (unsigned int i = 0 ; i < sx*sy ; ++i){
        psf_sum += psf[i]; 
    }
    for (unsigned int i = 0 ; i < sx*sy ; ++i){
        psf[i] /= psf_sum;
    }

    float *blurry_image_norm = new float[sx * sy];
    SImg::normL2(blurry_image, sx, sy, 1, 1, 1, blurry_image_norm);
    
    SImg::tic();

    // run denoising
    float *deconv_image = new float[sx*sy];
    SImg::wiener_deconv_2d(blurry_image_norm, psf, deconv_image, sx, sy, lambda, connectivity );
    delete inputImage;

    SImg::toc();
    delete STimerAccess::instance();

    //SImageReader::write(new SImageFloat(deconv_image, sx, sy), SAMPLE2DDECONVW);

    // calculate error with the reference image
    std::cout << "calculate error:" << std::endl;
    SImageFloat *resultImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE2DDECONVW, 32));
    float *b1 = deconv_image;
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
    delete[] deconv_image;
    delete resultImage;
    delete[] blurry_image_norm;
    delete observable;
    delete observer;
    //error /= float(outputImage->getBufferSize());
    std::cout << "error =" << error << std::endl;
    if (error > 10)
    {
        return 1;
    }
    return 0;
}
