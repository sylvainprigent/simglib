#include <score>
#include <scli>
#include <simageio>
#include <sdenoise>


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
        cmdParser.addParameterFloat("-deltaz", "Delta resolution between xy and z", 1.0);
        cmdParser.addParameterFloat("-deltat", "Delta resolution int t", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Denoise a 3D+t image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire4d: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire4d: input image: " + inputImageFile);
            observer->message("spitfire4d: output image: " + outputImageFile);
            observer->message("spitfire4d: method: " + method);
            observer->message("spitfire4d: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire4d: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire4d: nb iterations: " + std::to_string(niter));
        }

        // Run process

        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float* noisy_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT(), inputImage->getSizeC())
        {
            throw SException("spitfire2d can process only 3D gray scale images");
        }

        SObservable * observable = new SObservable();
        observable->addObserver(observer);

        float* denoised_image;
        SImg::tic();
        if (method == "SV"){
            SImg::spitfire2d_sv(noisy_image, sx, sy, denoised_image, regularization, weighting, niter, verbose, observable);
        }
        else if (method == "HV")
        {
            SImg::spitfire2d_hv(noisy_image, sx, sy, denoised_image, regularization, weighting, niter, verbose, observable);
        }
        else{
            throw SException("spitfire2d: method must be SV or HV");
        }
        SImg::toc();

        SImageReader::write(new SImageFloat(denoised_image, sx, sy), outputImageFile);

        delete inputImage;
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
