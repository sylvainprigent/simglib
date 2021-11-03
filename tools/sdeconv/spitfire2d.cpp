#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <sdeconv>
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
        cmdParser.addOutputData("-psf", "PSF image");

        cmdParser.addParameterFloat("-sigma", "PSF sigma (gaussian)", 1.5);
        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 11);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", true);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Deconv a 2D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");
        std::string psfImageFile = cmdParser.getDataURI("-psf");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool padding = cmdParser.getParameterBool("-padding");
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

        std::cout << "image min = " << inputImage->getMin() << std::endl;
        std::cout << "image max = " << inputImage->getMax() << std::endl;

        if (inputImage->getSizeZ() > 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire2d can process only 2D gray scale images");
        }

        if (padding)
        {
            if (verbose){
                observer->message("spitfire2d: use padding");
            }
            // padding 
            unsigned int sx_pad = sx + 2*24;
            unsigned int sy_pad = sy + 2*24;
            float* blurry_padded_image = new float[sx_pad*sy_pad];
            SImg::hanning_padding_2d(blurry_image, blurry_padded_image, sx, sy, sx_pad, sy_pad); 
            delete inputImage; 

            // create the PSF
            float* psf = new float[sx_pad*sy_pad];
            if (psfImageFile == ""){   
                SImg::gaussian_psf_2d(psf, sx_pad, sy_pad, sigma, sigma);
            }
            else{
                SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));
                if (psfImage->getSizeX() != sx || psfImage->getSizeY() != sy)
                {
                    throw SException("spitfire2d: The PSF image size is different from the input image size");
                }
                SImg::padding_2d(psfImage->getBuffer(), psf, sx, sy, sx_pad, sy_pad);
            }
            float psf_sum = 0.0;
            for (unsigned int i = 0 ; i < sx_pad*sy_pad ; ++i){
                psf_sum += psf[i]; 
            }
            for (unsigned int i = 0 ; i < sx_pad*sy_pad ; ++i){
                psf[i] /= psf_sum;
            }

            // run deconvolution
            SObservable * observable = new SObservable();
            observable->addObserver(observer);
            SImg::tic();
            float *deconv_image = (float *)malloc(sizeof(float) * (sx_pad*sy_pad));
            SImg::spitfire2d_deconv(blurry_padded_image, sx_pad, sy_pad, psf, deconv_image, pow(2, -regularization), weighting, niter, method, verbose, observable);
            delete[] psf;
            delete[] blurry_padded_image;

            // remove padding
            float* output = new float[sx_pad*sy_pad];
            SImg::remove_padding_2d(deconv_image, output, sx_pad, sy_pad, sx, sy);
            delete[] deconv_image;

            SImg::toc();
            SImageReader::write(new SImageFloat(output, sx, sy), outputImageFile);

            delete[] output;
            delete observable;
        }
        else
        {
            // create the PSF
            float* psf = nullptr;
            if (psfImageFile == ""){   
                psf = new float[sx*sy]; 
                SImg::gaussian_psf_2d(psf, sx, sy, sigma, sigma);
            }
            else{
                SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));
                if (psfImage->getSizeX() != sx || psfImage->getSizeY() != sy)
                {
                    throw SException("spitfire2d: The PSF image size is different from the input image size");
                }
                psf = psfImage->getBuffer();
            }
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
            SImg::spitfire2d_deconv(blurry_image, sx, sy, psf, deconv_image, pow(2, -regularization), weighting, niter, method, verbose, observable);
            SImg::toc();

            SImageReader::write(new SImageFloat(deconv_image, sx, sy), outputImageFile);

            delete[] blurry_image;
            delete[] deconv_image;
            delete observable;
        }
        
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
