#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <spadding>
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
        cmdParser.addParameterInt("-niter", "Number of iterations", 40);
        cmdParser.addParameterFloat("-lambda", "Regularization parameter", 0);
        cmdParser.setMan("Deconvolve an image using the Richardson Lucy algorithm");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float sigma = cmdParser.getParameterFloat("-sigma");
        const int niter = cmdParser.getParameterFloat("-niter");
        const float lambda = cmdParser.getParameterFloat("-lambda");

        if (inputImageFile == ""){
            observer->message("RichardsonLucy: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("RichardsonLucy: input image: " + inputImageFile);
            observer->message("RichardsonLucy: output image: " + outputImageFile);
            observer->message("RichardsonLucy: sigma: " + std::to_string(sigma));
            observer->message("RichardsonLucy: niter: " + std::to_string(niter));
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

        int ctrl = SImg::hanning_padding_2d(buffer_in, buffer_in_padding, sx, sy, sx_out, sy_out);
        if (ctrl > 0){
            observer->message("RichardsonLucy: padding dimensions missmatch", SObserver::MessageTypeError);
            return 1;
        }
        delete inputImage;

        // 2- create the psf
        float* buffer_psf = new float[sx_out*sy_out];
        SImg::gaussian_psf_2d(buffer_psf, sx_out, sy_out, sigma, sigma);

        // 3- normalize inputs
        float max_in = 0;
        float sum_psf = 0;
        float max_psf = 0;
        for (int i = 0 ; i < sx_out*sy_out ; i++){
            if (buffer_psf[i] > max_psf){
                max_psf = buffer_psf[i];   
            }
            sum_psf += buffer_psf[i]; 
            if (buffer_in_padding[i] > max_in){
                max_in = buffer_in_padding[i];   
            }
        }
        for (int i = 0 ; i < sx_out*sy_out ; i++){
            buffer_psf[i] /= sum_psf;     
            buffer_in_padding[i] /= max_in;   
        }

        // 4- compute the deconvolution
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
