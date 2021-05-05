#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <sdeconv>
#include <cudeconv>
#include "math.h"

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        // Parse inputs
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-sigma", "PSF sigma (gaussian)", 2);
        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 2);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Deconv a 2D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire2d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire2d: input image: " + inputImageFile);
            observer->message("spitfire2d: output image: " + outputImageFile);
            observer->message("spitfire2d: method: " + method);
            observer->message("spitfire2d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire2d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire2d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* blurry_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire2d can process only 2D gray scale images");
        }

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

        // run deconvolution
        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        SImg::tic();
        float *deconv_image = (float *)malloc(sizeof(float) * (sx*sy));
        SImg::cuda_spitfire2d_deconv(blurry_image, sx, sy, psf, deconv_image, pow(2, -regularization), weighting, niter, method, verbose, observable);
        SImg::toc();

        SImageReader::write(new SImageFloat(deconv_image, sx, sy), outputImageFile);

        delete[] blurry_image;
        delete[] deconv_image;
        delete observable;
    }
    catch (SException &e)
    {
        observer->message(e.what(), SObserver::MessageTypeError);
        return 1;
    }
    catch (std::exception &e)
    {
        observer->message(e.what(), SObserver::MessageTypeError);
        return 1;
    }

    delete observer;
    return 0;
}
