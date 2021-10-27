#include <iostream>
#include <score>
#include <scli>
#include <simageio>
#include <sdeconv>
#include <smanipulate>
#include <spadding>

int main(int argc, char *argv[])
{
    SObserverConsole* observer = new SObserverConsole();
    try
    {
        SCliParser cmdParser(argc, argv);
        cmdParser.addInputData("-i", "Input image file");
        cmdParser.addInputData("-psf", "PSF image");
        cmdParser.addOutputData("-o", "Output image file");

        cmdParser.addParameterFloat("-lambda", "Regularization parameter", 0.005);
        cmdParser.addParameterBoolean("-padding", "True to use mirror padding for border", false);
        cmdParser.setMan("Deconvolve an image using the Wiener algorithm");
        cmdParser.parse(4);

        std::string inputImageFile = cmdParser.getDataURI("-i");
        std::string psfImageFile = cmdParser.getDataURI("-psf");
        std::string outputImageFile = cmdParser.getDataURI("-o");

        const float lambda = cmdParser.getParameterFloat("-lambda");
        const bool padding = cmdParser.getParameterBool("-padding");

        if (inputImageFile == ""){
            observer->message("Wiener 3D: Input image path is empty");
            return 1;
        }

        bool verbose = true;
        if (verbose){
            observer->message("Wiener 3D: input image: " + inputImageFile);
            observer->message("Wiener 3D: output image: " + outputImageFile);
            observer->message("Wiener 3D: lambda: " + std::to_string(lambda));
        }

        // Run process
        SImageFloat* inputImage = dynamic_cast<SImageFloat*>(SImageReader::read(inputImageFile, 32));

        // 1- create a padding around the input image
        float* blurry_image = inputImage->getBuffer();
        unsigned int sx = inputImage->getSizeX();
        unsigned int sy = inputImage->getSizeY();
        unsigned int sz = inputImage->getSizeZ();
        float imin = inputImage->getMin();
        float imax = inputImage->getMax();

        SImageFloat* psfImage = dynamic_cast<SImageFloat*>(SImageReader::read(psfImageFile, 32));
        if (psfImage->getSizeX() != sx || psfImage->getSizeY() != sy || psfImage->getSizeZ() != sz)
        {
            throw SException("Wiener 3D the PSF image size must be the same as the input image size");
        } 
        float* psf = psfImage->getBuffer();

        if (padding)
        {
            if (verbose){
                observer->message("Wiener 3D: use padding");
            }
            unsigned int sx_pad = sx + 2*24;
            unsigned int sy_pad = sy + 2*24;
            unsigned int sz_pad = sz + 2*3;

            // Pad and normalize the PSF
            float* psf_pad = new float[sx_pad*sy_pad*sz_pad]; 
            SImg::padding_3d(psf, psf_pad, sx, sy, sz, sx_pad, sy_pad, sz_pad);
            float psf_sum = 0.0;
            for (unsigned int i = 0 ; i < sx_pad*sy_pad*sz_pad ; ++i){
                psf_sum += psf_pad[i]; 
            }
            for (unsigned int i = 0 ; i < sx_pad*sy_pad*sz_pad ; ++i){
                psf_pad[i] /= psf_sum;
            }
            delete psfImage;

            // Pad image
            float* blurry_padded_image = new float[sx_pad*sy_pad*sz_pad];
            SImg::hanning_padding_3d(blurry_image, blurry_padded_image, sx, sy, sz, sx_pad, sy_pad, sz_pad);
            delete inputImage;

            // Normalize the image
            float *blurry_image_norm = new float[sx_pad*sy_pad*sz_pad];
            SImg::normL2(blurry_padded_image, sx_pad, sy_pad, sz_pad, 1, 1, blurry_image_norm);
            delete[] blurry_padded_image; 

            // Compute the deconvolution
            float* buffer_out = new float[sx_pad*sy_pad*sz_pad];
            SImg::tic();
            SImg::wiener_deconv_3d(blurry_image_norm, psf_pad, buffer_out, sx_pad, sy_pad, sz_pad, lambda, 8);
            SImg::toc();
            delete[] blurry_image_norm;
            delete[] psf_pad;

            // Remove padding
            float* output = new float[sx*sy*sz];
            SImg::remove_padding_3d(buffer_out, output, sx_pad, sy_pad, sz_pad, sx, sy, sz);
            delete[] buffer_out; 

            SImg::normalize_back_intensities(output, sx*sy*sz, imin, imax);

            // save outputs
            SImageFloat* outputImage = new SImageFloat(output, sx, sy, sz);
            SImageReader::write(outputImage, outputImageFile);

            delete outputImage;
        }
        else
        {
            // Create the PSF
            float psf_sum = 0.0;
            for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
                psf_sum += psf[i]; 
            }
            for (unsigned int i = 0 ; i < sx*sy*sz ; ++i){
                psf[i] /= psf_sum;
            }

            // Normalize the image
            float *blurry_image_norm = new float[sx*sy*sz];
            SImg::normL2(blurry_image, sx, sy, sz, 1, 1, blurry_image_norm);
            delete inputImage; 

            // Compute the deconvolution
            float* buffer_out = new float[sx*sy*sz];
            SImg::tic();
            SImg::wiener_deconv_3d(blurry_image_norm, psf, buffer_out, sx, sy, sz, lambda, 8);
            SImg::normalize_back_intensities(buffer_out, sx*sy*sz, imin, imax);

            SImg::toc();

            // save outputs
            SImageFloat* outputImage = new SImageFloat(buffer_out, sx, sy, sz);
            SImageReader::write(outputImage, outputImageFile);

            delete psfImage;
            //delete[] buffer_out;
            delete[] blurry_image_norm;
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
