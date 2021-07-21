#include <score>
#include <scli>
#include <simageio>
#include <cudeconv>
#include <smanipulate>

#include "cudeconvTestConfig.h"

#ifdef SL_USE_OPENMP
#include "omp.h"
#endif

#include <iostream>
#include "math.h"

int main(int argc, char *argv[])
{
    std::cout << "input image = " << SAMPLE3D << std::endl;
    std::cout << "ref image = " << SAMPLE3DDECONVW << std::endl;

    SImageFloat *inputImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE3D, 32));
    float imax = inputImage->getMax();
    float imin = inputImage->getMin();

    // Run process
    float *blurry_image = inputImage->getBuffer();
    unsigned int sx = inputImage->getSizeX();
    unsigned int sy = inputImage->getSizeY();
    unsigned int sz = inputImage->getSizeZ();


    SObserverConsole *observer = new SObserverConsole();
    SObservable *observable = new SObservable();
    observable->addObserver(observer);

    float lambda = 0.002;
    int connectivity = 4; // 4

    // create the PSF
    SImageFloat *psfImage = dynamic_cast<SImageFloat *>(SImageReader::read(PSF3D, 32));
    float* psf = psfImage->getBuffer();
    float psf_sum = 0.0;
    for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
        psf_sum += psf[i]; 
    }
    for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
        psf[i] /= psf_sum;
    }

    float *blurry_image_norm = new float[sx * sy * sz];
    SImg::normL2(blurry_image, sx, sy, sz, 1, 1, blurry_image_norm);
    
    SImg::tic();

    // run denoising
    float *deconv_image = new float[sx*sy*sz];
    SImg::cuda_wiener_deconv_3d(blurry_image_norm, psf, deconv_image, sx, sy, sz, lambda, connectivity );
    delete inputImage;

    SImg::toc();
    delete STimerAccess::instance();

    //SImageReader::write(new SImageFloat(deconv_image, sx, sy, sz), SAMPLE3DDECONVW);

    // calculate error with the reference image
    std::cout << "calculate error:" << std::endl;
    SImageFloat *resultImage = dynamic_cast<SImageFloat *>(SImageReader::read(SAMPLE3DDECONVW, 32));
    float *b1 = deconv_image;
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
