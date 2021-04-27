#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <spreprocess>
#include <sdeconv>
#include <spsf>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-sigma", "Sigma of the PSF", 2);
        cmdParser.addParameterFloat("-lambda", "Regularization parameter", 0);
        cmdParser.setMan("Deconvolve an image using the Wiener algorithm");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const float lambda = cmdParser.getParameterFloat("-lambda");

        if (inputImageFile == ""){
            observer->message("Wiener: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Wiener: input image: " + inputImageFile);
            observer->message("Wiener: output image: " + outputImageFile);
            observer->message("Wiener: sigma: " + std::to_string(sigma));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));

        // 1- create a padding around the input image
        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sx_out = sx+2*24;
        unsigned int sy_out = sy+2*24;
        float* buffer_in_padding = new float[sx_out*sy_out];

        int ctrl = sl_hanning_padding_2d(buffer_in, buffer_in_padding, sx, sy, sx_out, sy_out);
        if (ctrl > 0){
            observer->message("Wiener: padding dimensions missmatch", SObserver::MessageTypeError);
            return 1;
        }
        delete inputImage;

        // 2- create the psf
        float* buffer_psf = new float[sx_out*sy_out];
        sl_gaussian_psf_2d(buffer_psf, sx_out, sy_out, sigma, sigma);

        // 3- normalize inputs
        sl_normalize_intensities(buffer_psf, sx_out*sy_out, "max");
        sl_normalize_intensities(buffer_in_padding, sx_out*sy_out, "max");

        // 4- compute the deconvolution
        float* buffer_out = new float[sx_out*sy_out];
        SImg::tic();
        sl_wiener_deconv_2d(buffer_in_padding, buffer_psf, buffer_out, sx_out, sy_out, lambda);
        SImg::toc();

        float* buffer_out_crop = new float[sx*sy];
        sl_remove_padding_2d(buffer_out, buffer_out_crop, sx_out, sy_out, sx, sy);
        delete[] buffer_out;

        // save outputs
        SImageFloat* outputImage = new SImageFloat(buffer_out_crop, sx, sy);
        SImageReader::write(outputImage, outputImageFile);

        delete[] buffer_psf;
        delete[] buffer_in_padding;
        delete outputImage;
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
