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
        cmdParser.addInputData("-psf", "PSF image file");

        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 11);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterFloat("-delta", "Scale delta in Z", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Deconv a 3D image with the SPITFIR(e) cuda algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        std::string psfImageFile = cmdParser.getDataURI("-psf");
        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const float delta = cmdParser.getParameterFloat("-delta");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire3d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire3d: input image: " + inputImageFile);
            observer->message("spitfire3d: output image: " + outputImageFile);
            observer->message("spitfire3d: psf image: " + psfImageFile);
            observer->message("spitfire3d: method: " + method);
            observer->message("spitfire3d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire3d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire3d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* blurry_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        if (inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire3d can process only 3D gray scale images");
        }

                // create the PSF
        SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));
        if (psfImage->getSizeX() != sx || psfImage->getSizeY() != sy || psfImage->getSizeZ() != sz)
        {
            throw SException("spitfire3d the PSF image size must be the same as the input image size");
        } 
        float* psf = psfImage->getBuffer();
        float psf_sum = 0.0;
        for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
            psf_sum += psf[i]; 
        }
        for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
            psf[i] /= psf_sum;
        }

        // run deconvolution
        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        SImg::tic();
        float *deconv_image = (float *)malloc(sizeof(float) * (sx*sy*sz));
        SImg::cuda_spitfire3d_deconv(blurry_image, sx, sy, sz, psf, deconv_image, pow(2, -regularization), weighting, delta, niter, method, verbose, observable);
        SImg::toc();

        SImageReader::write(new SImageFloat(deconv_image, sx, sy, sz), outputImageFile);

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
