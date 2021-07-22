#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <cudenoise>
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
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 11);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterFloat("-delta", "Scale delta in Z", 1.0);
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
            observer->message("cuda spitfire3d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("cuda spitfire3d: input image: " + inputImageFile);
            observer->message("cuda spitfire3d: output image: " + outputImageFile);
            observer->message("cuda spitfire3d: method: " + method);
            observer->message("cuda spitfire3d: regularization parameter: " + std::to_string(regularization));
            observer->message("cuda spitfire3d: weighting parameter: " + std::to_string(weighting));
            observer->message("cuda spitfire3d: delta parameter: " + std::to_string(delta));
            observer->message("cuda spitfire3d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        if (inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("cuda spitfire3d can process only 3D gray scale images");
        }
        
        // run deconvolution
        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        float *denoised_image = (float *)malloc(sizeof(float) * (sx*sy*sz));
        SImg::tic();
        SImg::cuda_spitfire3d_denoise(inputImage->getBuffer(), sx, sy, sz, denoised_image, pow(2, -regularization), weighting, niter, delta, method, verbose, observable);
        SImg::toc();

        SImageFloat* denImage = new SImageFloat(denoised_image, sx, sy, sz);
        SImageReader::write(denImage, outputImageFile);

        delete denImage;
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
