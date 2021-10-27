#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <spadding>
#include <sdeconv>
#include <smanipulate>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-sigma", "Sigma of the PSF", 2);
        cmdParser.addParameterInt("-niter", "Number of iterations", 40);
        cmdParser.addParameterFloat("-lambda", "Regularization parameter", 0);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", true);

        cmdParser.setMan("Deconvolve a 2D image using the Richardson Lucy algorithm");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const int niter = cmdParser.getParameterFloat("-niter");
        const float lambda = cmdParser.getParameterFloat("-lambda");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("RichardsonLucy2D: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("RichardsonLucy2D: input image: " + inputImageFile);
            observer->message("RichardsonLucy2D: output image: " + outputImageFile);
            observer->message("RichardsonLucy2D: sigma: " + std::to_string(sigma));
            observer->message("RichardsonLucy2D: niter: " + std::to_string(niter));
        }

        // Load input image
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));
        float imin = inputImage->getMin();
        float imax = inputImage->getMax();
        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();

        if (padding)
        {
            if (verbose){
                observer->message("RichardsonLucy2D: use padding");
            }
            unsigned int sx_out = sx+2*24;
            unsigned int sy_out = sy+2*24;
            float* buffer_in_padding = new float[sx_out*sy_out];

            int ctrl = SImg::hanning_padding_2d(buffer_in, buffer_in_padding, sx, sy, sx_out, sy_out);
            if (ctrl > 0){
                observer->message("RichardsonLucy2D: padding dimensions missmatch", SObserver::MessageTypeError);
                return 1;
            }
            delete inputImage;

            // Create the psf
            float* buffer_psf = new float[sx_out*sy_out];
            SImg::gaussian_psf_2d(buffer_psf, sx_out, sy_out, sigma, sigma);

            // Normalize inputs
            SImg::normalize_intensities(buffer_psf, sx_out*sy_out, "sum");
            SImg::normalize_intensities(buffer_in_padding, sx_out*sy_out, "L2");

            // Compute the deconvolution
            float* buffer_out = new float[sx_out*sy_out];
            SImg::tic();
            if (lambda > 1e-9){
                SImg::richardsonlucy_tv_2d(buffer_in_padding, buffer_psf, buffer_out, sx_out, sy_out, niter, lambda);
            }
            else{
                SImg::richardsonlucy_2d(buffer_in_padding, buffer_psf, buffer_out, sx_out, sy_out, niter);
            }
            SImg::toc();

            float* buffer_out_crop = new float[sx*sy];
            SImg::remove_padding_2d(buffer_out, buffer_out_crop, sx_out, sy_out, sx, sy);
            delete[] buffer_out;
            SImg::normalize_back_intensities(buffer_out_crop, sx*sy, imin, imax);

            // save outputs
            SImageFloat* outputImage = new SImageFloat(buffer_out_crop, sx, sy);
            SImageReader::write(outputImage, outputImageFile);

            delete[] buffer_psf;
            delete[] buffer_in_padding;
            delete outputImage;
        }
        else
        {
            // Create the psf
            float* buffer_psf = new float[sx*sy];
            SImg::gaussian_psf_2d(buffer_psf, sx, sy, sigma, sigma);

            // Normalize inputs
            SImg::normalize_intensities(buffer_psf, sx*sy, "sum");
            SImg::normalize_intensities(buffer_in, sx*sy, "L2");

            // Compute the deconvolution
            float* buffer_out = new float[sx*sy];
            SImg::tic();
            if (lambda > 1e-9){
                SImg::richardsonlucy_tv_2d(buffer_in, buffer_psf, buffer_out, sx, sy, niter, lambda);
            }
            else{
                SImg::richardsonlucy_2d(buffer_in, buffer_psf, buffer_out, sx, sy, niter);
            }
            SImg::normalize_back_intensities(buffer_out, sx*sy, imin, imax);
            SImg::toc();

            // save outputs
            SImageFloat* outputImage = new SImageFloat(buffer_out, sx, sy);
            SImageReader::write(outputImage, outputImageFile);

            delete[] buffer_psf;
            delete inputImage;
            delete outputImage;
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
