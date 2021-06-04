#include <score>
#include <scli>
#include <simageio>
#include <sdenoise>
#include <smanipulate>
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

        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 2);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterFloat("-delta", "Delta resolution between xy and z", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Denoise a 3D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

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
            observer->message("spitfire3d: method: " + method);
            observer->message("spitfire3d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire3d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire3d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        observer->message("spitfire3d: input image size: " + std::to_string(sx) + ", " + std::to_string(sy) + ", " + std::to_string(sz));
        if (inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire3d can process only 3D gray scale images");
        }
        float imax = inputImage->getMax();
        float imin = inputImage->getMin();

        SObservable * observable = new SObservable();
        observable->addObserver(observer);

        // min max normalize intensities
        float* noisy_image_norm = new float[sx*sy*sz];
        SImg::normMinMax(noisy_image, sx, sy, sz, 1, 1, noisy_image_norm);
        delete inputImage;

        float* denoised_image = new float[sx*sy*sz];
        SImg::tic();
        if (method == "SV"){
            SImg::spitfire3d_sv(noisy_image_norm, sx, sy, sz, denoised_image, pow(2, -regularization), weighting, niter, delta, verbose, observable);
        }
        else if (method == "HV")
        {
            observer->message("spitfire3d: use method HV");
            SImg::spitfire3d_hv(noisy_image_norm, sx, sy, sz, denoised_image, pow(2, -regularization), weighting, niter, delta, verbose, observable);
        }
        else{
            throw SException("spitfire3d: method must be SV or HV");
        }

        // normalize back intensities
        #pragma omp parallel for
        for (unsigned int i = 0 ; i < sx*sy*sz ; ++i)
        {
            denoised_image[i] = denoised_image[i]*(imax-imin) + imin;
        }
        SImg::toc();

        SImageReader::write(new SImageFloat(denoised_image, sx, sy, sz), outputImageFile);

        delete[] noisy_image_norm;
        delete denoised_image;
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
