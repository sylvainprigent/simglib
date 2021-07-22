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
        cmdParser.addParameterFloat("-deltaz", "Scale delta in Z", 1.0);
        cmdParser.addParameterFloat("-deltat", "Scale delta in t", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Denoise a 4D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const float deltaz = cmdParser.getParameterFloat("-deltaz");
        const float deltat = cmdParser.getParameterFloat("-deltat");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("cuda spitfire4d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("cuda spitfire4d: input image: " + inputImageFile);
            observer->message("cuda spitfire4d: output image: " + outputImageFile);
            observer->message("cuda spitfire4d: method: " + method);
            observer->message("cuda spitfire4d: regularization parameter: " + std::to_string(regularization));
            observer->message("cuda spitfire4d: weighting parameter: " + std::to_string(weighting));
            observer->message("cuda spitfire4d: delta parameter: " + std::to_string(deltaz));
            observer->message("cuda spitfire4d: delta parameter: " + std::to_string(deltat));
            observer->message("cuda spitfire4d: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        unsigned int st = inputImage->getSizeT();
        if (inputImage->getSizeC() > 1)
        {
            throw SException("cuda spitfire4d can process only 4D gray scale images");
        }
        
        // run deconvolution
        SObservable * observable = new SObservable();
        observable->addObserver(observer);
        float *denoised_image = (float *)malloc(sizeof(float) * (sx*sy*sz*st));
        SImg::tic();
        SImg::cuda_spitfire4d_denoise(inputImage->getBuffer(), sx, sy, sz, st, denoised_image, regularization, weighting, deltaz, deltat, niter, method, verbose, observable);
        SImg::toc();

        SImageFloat* denImage = new SImageFloat(denoised_image, sx, sy, sz, st);
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
