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
        cmdParser.addInputData("-psf", "PSF image");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 11);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterFloat("-delta", "Scale delta in Z", 1.0);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", false);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Deconv a 2D image with the SPITFIR(e) algotithm");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string psfImageFile = cmdParser.getDataURI("-psf");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const float delta = cmdParser.getParameterFloat("-delta");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool verbose = cmdParser.getParameterBool("-verbose");
        const bool padding = cmdParser.getParameterBool("-padding");

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
            observer->message("spitfire3d: delta parameter: " + std::to_string(delta));
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
        SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));

        if (padding)
        {
            if (verbose){
                observer->message("spitfire3d: use padding");
            }
            // padding 
            unsigned int sx_pad = sx + 2*24;
            unsigned int sy_pad = sy + 2*24;
            unsigned int sz_pad = sz + 6;
            float* blurry_padded_image = new float[sx_pad*sy_pad*sz_pad];
            SImg::hanning_padding_3d(blurry_image, blurry_padded_image, sx, sy, sz, sx_pad, sy_pad, sz_pad);

            // create the PSF
            SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));
            if (psfImage->getSizeX() != sx || psfImage->getSizeY() != sy || psfImage->getSizeZ() != sz)
            {
                throw SException("spitfire3d the PSF image size must be the same as the input image size");
            } 
            float* psf = psfImage->getBuffer();
            float* psf_pad = new float[sx_pad*sy_pad*sz_pad]; 
            SImg::padding_3d(psf, psf_pad, sx, sy, sz, sx_pad, sy_pad, sz_pad);
            
            float psf_sum = 0.0;
            for (unsigned int i = 0 ; i < sx_pad*sy_pad*sz_pad ; ++i){
                psf_sum += psf_pad[i]; 
            }
            for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
                psf_pad[i] /= psf_sum;
            }
            delete psfImage;

            // run deconvolution
            SObservable * observable = new SObservable();
            observable->addObserver(observer);
            SImg::tic();
            float *deconv_image = (float *)malloc(sizeof(float) * (sx_pad*sy_pad*sz_pad));
            SImg::spitfire3d_deconv(blurry_padded_image, sx_pad, sy_pad, sz_pad, psf_pad, deconv_image, pow(2, -regularization), weighting, delta, niter, method, verbose, observable);
            SImg::toc();

            // remove padding
            float* output = new float[sx*sy*sz];
            SImg::remove_padding_3d(deconv_image, output, sx_pad, sy_pad, sz_pad, sx, sy, sz);
            delete[] deconv_image; 

            SImageReader::write(new SImageFloat(output, sx, sy, sz), outputImageFile);
            
            delete[] output; 
            delete observable;
        }
        else
        {
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
            SImg::spitfire3d_deconv(blurry_image, sx, sy, sz, psf, deconv_image, pow(2, -regularization), weighting, delta, niter, method, verbose, observable);
            SImg::toc();

            SImageReader::write(new SImageFloat(deconv_image, sx, sy, sz), outputImageFile);

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
