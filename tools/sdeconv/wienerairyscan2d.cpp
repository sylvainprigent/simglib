#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <sdeconv>

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
            observer->message("Wiener airyscan: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Wiener airyscan: input image: " + inputImageFile);
            observer->message("Wiener airyscan: output image: " + outputImageFile);
            observer->message("Wiener airyscan: sigma: " + std::to_string(sigma));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));

        // 1- copy the image in float
        float* buffer_in = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sd = inputImage->getSizeZ();

        float** buffer_in_as = new float*[sd];
        for (unsigned int d = 0 ; d < sd ; d++)
        {
            buffer_in_as[d] = new float[sx*sy];
            for (int p = 0 ; p < sx*sy ; p++){
                buffer_in_as[d][p] = buffer_in[sd*(p) + d]; 
            }       
        }
        delete inputImage;

        // 2- create the psf
        float* buffer_psf = new float[sx*sy];
        SImg::gaussian_psf_2d(buffer_psf, sx, sy, sigma, sigma);

        // 3- normalize inputs
        //sl_normalize_intensities(buffer_psf, sx_out*sy_out, "max");
        //sl_normalize_intensities(buffer_in_padding, sx_out*sy_out, "max");

        // 4- compute the deconvolution
        float* buffer_out = new float[sx*sy];
        SImg::tic();
        SImg::wiener_deconv_airyscan_2d(buffer_in_as, buffer_psf, buffer_out, sx, sy, sd, lambda, 8);
        SImg::toc();

        // save outputs
        SImageFloat* outputImage = new SImageFloat(buffer_out, sx, sy);
        SImageReader::write(outputImage, outputImageFile);

        delete[] buffer_psf;
        delete[] buffer_in_as;
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
