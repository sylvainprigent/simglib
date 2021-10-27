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

        cmdParser.addParameterFloat("-sigma", "PSF sigma (gaussian)", 1.5);
        cmdParser.addParameterSelect("-method", "Deconvolution method 'SV' or 'HV", "HV");
        cmdParser.addParameterFloat("-regularization", "Regularization parameter as pow(2,-x)", 11);
        cmdParser.addParameterFloat("-weighting", "Weighting parameter", 0.6);
        cmdParser.addParameterInt("-niter", "Nb iterations", 200);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", true);

        cmdParser.addParameterBoolean("-verbose", "Print iterations to console", true);
        cmdParser.setMan("Deconv a 2D image with the SPITFIR(e) algotithm slice per slice");
        cmdParser.parse(2);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const std::string method = cmdParser.getParameterString("-method");
        const float regularization = cmdParser.getParameterFloat("-regularization");
        const float weighting = cmdParser.getParameterFloat("-weighting");
        const int niter = cmdParser.getParameterInt("-niter");
        const bool padding = cmdParser.getParameterBool("-padding");
        const bool verbose = cmdParser.getParameterBool("-verbose");

        if (inputImageFile == ""){
            observer->message("spitfire2dslice: Input image path is empty");
            return 1;
        }

        if (verbose){
            observer->message("spitfire2dslice: input image: " + inputImageFile);
            observer->message("spitfire2dslice: output image: " + outputImageFile);
            observer->message("spitfire2dslice: method: " + method);
            observer->message("spitfire2dslice: regularization parameter: " + std::to_string(regularization));
            observer->message("spitfire2dslice: weighting parameter: " + std::to_string(weighting));
            observer->message("spitfire2dslice: nb iterations: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        if (sz <= 1 || inputImage->getSizeT() > 1 || inputImage->getSizeC() > 1)
        {
            throw SException("spitfire2dslice can process only 3D gray scale images");
        }

        // observable for console log
        SObservable * observable = new SObservable();
        observable->addObserver(observer);

        if (padding)
        {
            if (verbose){
                observer->message("spitfire2dslice: use padding");
            }
            // padding
            unsigned int r_pad = 24;
            unsigned int sx_pad = sx + 2*r_pad;
            unsigned int sy_pad = sy + 2*r_pad;

            // create the PSF
            float* psf = new float[sx_pad*sy_pad];
            SImg::gaussian_psf_2d(psf, sx_pad, sy_pad, sigma, sigma);
            float psf_sum = 0.0;
            for (unsigned int i = 0 ; i < sx_pad*sy_pad ; ++i){
                psf_sum += psf[i]; 
            }
            for (unsigned int i = 0 ; i < sx_pad*sy_pad ; ++i){
                psf[i] /= psf_sum;
            }

            // run 2D deconv on each Z slice
            SImageFloat* outputImage = new SImageFloat(sx, sy, sz);
            float * outputImageBuffer = outputImage->getBuffer();
            SImg::tic();
            for (unsigned int z = 0 ; z < inputImage->getSizeZ() ; ++z)
            {
                // get the slice
                SImageFloat* slice = inputImage->getSlice(z);
                float* sliceBuffer = slice->getBuffer();
                float imin = slice->getMin();
                float imax = slice->getMax();

                // padding
                float* blurry_padded_slice = new float[sx_pad*sy_pad];
                SImg::hanning_padding_2d(sliceBuffer, blurry_padded_slice, sx, sy, sx_pad, sy_pad); 
                delete sliceBuffer; 

                // run deconv
                float *outputSliceBuffer = (float *)malloc(sizeof(float) * (sx_pad*sy_pad));
                SImg::spitfire2d_deconv(blurry_padded_slice, sx_pad, sy_pad, psf, outputSliceBuffer, pow(2, -regularization), weighting, niter, method, verbose, observable);
                delete[] blurry_padded_slice;

                // copy deblured slice to output image
                for (unsigned int x = 0 ; x < sx ; ++x)
                {
                    for (unsigned int y = 0 ; y < sy ; ++y)
                    {
                        outputImageBuffer[ z + sz*(y + sy*x)] = outputSliceBuffer[y+r_pad + sy_pad*(x+r_pad)];
                    }
                }
                delete outputSliceBuffer;
            }
            SImg::toc();

            SImageReader::write(outputImage, outputImageFile);

            delete outputImage;
            delete inputImage;
            delete observable;
        }
        else
        {
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

            // run 2D deconv on each Z slice
            SImageFloat* outputImage = new SImageFloat(sx, sy, sz);
            float * outputImageBuffer = outputImage->getBuffer();
            SImg::tic();
            for (unsigned int z = 0 ; z < inputImage->getSizeZ() ; ++z)
            {
                // get the slice
                SImageFloat* slice = inputImage->getSlice(z);
                float* sliceBuffer = slice->getBuffer();

                // run deconv
                float *outputSliceBuffer = (float *)malloc(sizeof(float) * (sx*sy));
                SImg::spitfire2d_deconv(sliceBuffer, sx, sy, psf, outputSliceBuffer, pow(2, -regularization), weighting, niter, method, verbose, observable);
            
                // copy deblured slice to output image
                for (unsigned int x = 0 ; x < sx ; ++x)
                {
                    for (unsigned int y = 0 ; y < sy ; ++y)
                    {
                        outputImageBuffer[ z + sz*(y + sy*x)] = outputSliceBuffer[y + sy*x];
                    }
                }
                delete outputSliceBuffer;
            }
            SImg::toc();

            SImageReader::write(outputImage, outputImageFile);

            delete outputImage;
            delete inputImage;
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
