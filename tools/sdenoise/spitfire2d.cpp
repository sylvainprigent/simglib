#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <sdenoise>
#include <spadding>
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
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", false);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Denoise a 2D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("spitfire2d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire2d: denoise" + inputImageFile);
            observer->message("spitfire2d: input image: " + inputImageFile);
            observer->message("spitfire2d: output image: " + outputImageFile);
            observer->message("spitfire2d: method: " + method);
            observer->message("spitfire2d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire2d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire2d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire2d can process only 2D gray scale images");
        }
        float imax = inputImage->getMax();
        float imin = inputImage->getMin();

        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        SImg::tic();

        if (padding)
        {
            if (verbose){
                observer->message("spitfire2d: use padding");
            }
            // padding 
            unsigned int sx_pad = sx + 12;
            unsigned int sy_pad = sy + 12;
            float* noisy_padded_image = new float[sx_pad*sy_pad];
            SImg::mirror_padding_2d(noisy_image, noisy_padded_image, sx, sy, sx_pad, sy_pad);   

            // min max normalize intensities
            float* noisy_image_norm = new float[sx_pad*sy_pad];
            SImg::normMinMax(noisy_padded_image, sx_pad, sy_pad, 1, 1, 1, noisy_image_norm);
            delete inputImage;

            // run denoising
            float* denoised_image = new float[sx_pad*sy_pad];
            if (method == "SV"){
                SImg::spitfire2d_sv(noisy_image_norm, sx_pad, sy_pad, denoised_image, pow(2, -regularization), weighting, niter, verbose, observable);
            }
            else if (method == "HV")
            {
                SImg::spitfire2d_hv(noisy_image_norm, sx_pad, sy_pad, denoised_image, pow(2, -regularization), weighting, niter, verbose, observable);
            }
            else{
                throw SException("spitfire2d: method must be SV or HV");
            }

            // normalize back intensities
            SImg::normalize_back_intensities(denoised_image, sx_pad*sy_pad, imin, imax);

            // remove padding
            float* output = new float[sx_pad*sy_pad];
            SImg::remove_padding_2d(denoised_image, output, sx_pad, sy_pad, sx, sy);
            delete[] denoised_image;

            SImg::toc();
            delete STimerAccess::instance();

            SImageFloat* denImage = new SImageFloat(output, sx, sy);
            SImageReader::write(denImage, outputImageFile);
            delete denImage;
        }
        else
        {
            // min max normalize intensities
            float* noisy_image_norm = new float[sx*sy];
            SImg::normMinMax(noisy_image, sx, sy, 1, 1, 1, noisy_image_norm);
            delete inputImage;

            // run denoising
            float* denoised_image = new float[sx*sy];
            if (method == "SV"){
                SImg::spitfire2d_sv(noisy_image_norm, sx, sy, denoised_image, pow(2, -regularization), weighting, niter, verbose, observable);
            }
            else if (method == "HV")
            {
                SImg::spitfire2d_hv(noisy_image_norm, sx, sy, denoised_image, pow(2, -regularization), weighting, niter, verbose, observable);
            }
            else{
                throw SException("spitfire2d: method must be SV or HV");
            }

            // normalize back intensities
            delete[] noisy_image_norm;
            SImg::normalize_back_intensities(denoised_image, sx*sy, imin, imax);

            SImg::toc();
            delete STimerAccess::instance();

            SImageFloat* denImage = new SImageFloat(denoised_image, sx, sy);
            SImageReader::write(denImage, outputImageFile);
            delete denImage;
        }

        
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
