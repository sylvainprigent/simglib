#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <smanipulate>
#include <sdeconv>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addInputData("-psf", "PSF image");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterInt("-niter", "Number of iterations", 40);
        cmdParser.setMan("Deconvolve a 3D image using the Richardson Lucy algorithm");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string psfImageFile = cmdParser.getDataURI("-psf");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const int niter = cmdParser.getParameterFloat("-niter");

        if (inputImageFile == ""){
            observer->message("RichardsonLucy: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("RichardsonLucy: input image: " + inputImageFile);
            observer->message("RichardsonLucy: output image: " + outputImageFile);
            observer->message("RichardsonLucy: niter: " + std::to_string(niter));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));

        // 1- create a padding around the input image
        float* blurry_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        
        // 2- create the psf
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

        // 3- Normalize the image
        float *blurry_image_norm = new float[sx*sy*sz];
        SImg::normL2(blurry_image, sx, sy, sz, 1, 1, blurry_image_norm);
        delete inputImage; 

        // 4- compute the deconvolution
        SImg::tic();
        float *deconv_image = new float[sx*sy*sz];
        SImg::richardson_lucy_3d(blurry_image_norm, psf, deconv_image, sx, sy, sz, niter);
        SImg::toc();

        // save outputs
        SImageFloat* outputImage = new SImageFloat(deconv_image, sx, sy, sz);
        SImageReader::write(outputImage, outputImageFile);

        delete psfImage;
        delete[] deconv_image;
        delete[] blurry_image_norm;
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
